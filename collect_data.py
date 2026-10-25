"""Preview one hand or collect labeled world-landmark features.

Examples:
  python collect_data.py --preview --camera 0
  python collect_data.py --session debug01 --label neutral --samples 10 \
      --hand right --output data/debug_samples.csv
  python collect_data.py --session s01 --label neutral --samples 60 --hand right

Capture starts only after SPACE and a three-second countdown. SPACE pauses;
Q/Esc exits when the preview has focus. Ctrl+C also stops and releases resources.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

from hand_features import CLASS_LABELS, FEATURE_COUNT, FEATURE_VERSION, features_from_result

ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = [f"f{i:02d}" for i in range(FEATURE_COUNT)]
COLUMNS = ["sample_id", "session_id", "label_id", "timestamp_utc", "feature_version", *FEATURE_COLUMNS]
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


class SampleStore:
    """Validate an append-only CSV and make resume explicit for each label."""

    def __init__(self, path: Path, *, session: str, label: str, hand: str,
                 camera: int, resume: bool = False):
        self.path = Path(path)
        self.session = session
        self.label_id = CLASS_LABELS.index(label)
        self.hand = hand
        self.camera = camera
        self.counts = Counter()
        self.ids = set()
        self.handle = None
        self.meta_path = self.path.with_suffix(".metadata.json")
        expected = {
            "feature_version": FEATURE_VERSION,
            "class_labels": list(CLASS_LABELS),
            "physical_hand": hand,
            "input_orientation": "unflipped_camera",
            "camera_index": camera,
        }
        self.metadata = expected
        if self.meta_path.is_file():
            actual = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError("Dataset metadata differs (hand, camera, features, or classes). Use matching settings or a different CSV.")
        if self.path.exists():
            if not self.meta_path.is_file():
                raise ValueError(f"Missing {self.meta_path}; cannot verify an existing dataset. Share this error before changing files.")
            with self.path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != COLUMNS:
                    raise ValueError("CSV header does not match the 63-feature schema")
                for line, row in enumerate(reader, start=2):
                    try:
                        sample_id = row["sample_id"]
                        label_id = int(row["label_id"])
                        features = np.asarray([row[column] for column in FEATURE_COLUMNS], dtype=np.float32)
                        if (None in row or not sample_id or sample_id in self.ids
                                or label_id not in range(len(CLASS_LABELS))
                                or not row["session_id"] or not row["timestamp_utc"]
                                or row["feature_version"] != FEATURE_VERSION
                                or not np.isfinite(features).all()):
                            raise ValueError("invalid values or duplicate sample ID")
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"Invalid CSV row {line}: {exc}") from exc
                    self.ids.add(sample_id)
                    self.counts[(row["session_id"], label_id)] += 1
        if self.current_count and not resume:
            raise ValueError(f"{session}/{label} already has {self.current_count} rows. Add --resume to fill up to --samples; nothing was overwritten.")

    @property
    def current_count(self):
        return self.counts[(self.session, self.label_id)]

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            with self.meta_path.open("x", encoding="utf-8") as handle:
                json.dump(self.metadata, handle, indent=2)
                handle.write("\n")
        self.handle = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        if self.path.stat().st_size == 0:
            self.writer.writerow(COLUMNS)
            self.handle.flush()
        return self

    def append(self, features, timestamp=None):
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
            raise ValueError("Only a finite, already-normalized (63,) vector can be saved")
        if self.handle is None:
            raise RuntimeError("Open SampleStore with a with-block before saving")
        sample_id = uuid.uuid4().hex
        while sample_id in self.ids:
            sample_id = uuid.uuid4().hex
        self.writer.writerow([
            sample_id, self.session, self.label_id,
            timestamp or datetime.now(timezone.utc).isoformat(), FEATURE_VERSION,
            *[format(float(value), ".9g") for value in values],
        ])
        self.handle.flush()
        self.ids.add(sample_id)
        self.counts[(self.session, self.label_id)] += 1
        return sample_id

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def draw_hand(cv2, frame, landmarks):
    height, width = frame.shape[:2]
    if len(landmarks) != 21:
        return
    points = []
    for point in landmarks:
        if not math.isfinite(point.x) or not math.isfinite(point.y):
            return
        x = int(np.clip(point.x, 0, 1) * (width - 1))
        y = int(np.clip(point.y, 0, 1) * (height - 1))
        points.append((x, y))
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (40, 220, 40), 2)
    for point in points:
        cv2.circle(frame, point, 4, (40, 80, 255), -1)


def run_camera(args, store):
    # These imports remain local so CSV/feature checks can run without a webcam stack.
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("Run with the project's .venv Python; verify dependencies with check_setup.py.") from exc

    model_path = project_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path}. Run the official model download command in README.md; no automatic download was attempted.")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera = None
    state = "PREVIEW" if args.preview else "PAUSED"
    countdown_until = 0.0
    next_sample = 0.0
    last_timestamp = -1
    frames = 0
    started = time.monotonic()
    failures = 0
    window = "GesturePilot - hand tracking"
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(f"Camera {args.camera} did not open. Close other camera apps and check Windows permissions.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            print("Preview ready. Q/Esc quits. In capture mode, SPACE starts a 3-second countdown or pauses.", flush=True)
            print("Overlay label is the manual collection target; no gesture classifier is trained yet.", flush=True)
            while True:
                ok, frame = camera.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Camera stopped returning frames. Saved rows are retained; close other camera apps and retry.")
                    time.sleep(0.05)
                    continue
                failures = 0
                now = time.monotonic()
                timestamp = max(last_timestamp + 1, int((now - started) * 1000))
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                features = features_from_result(result)
                now = time.monotonic()
                if state == "COUNTDOWN" and now >= countdown_until:
                    state = "RECORDING"
                    next_sample = now
                if store is not None and state == "RECORDING" and features is not None and now >= next_sample:
                    store.append(features)
                    next_sample = now + 0.2
                    if store.current_count >= args.samples:
                        state = "DONE"
                if result.hand_landmarks:
                    draw_hand(cv2, frame, result.hand_landmarks[0])
                # Mirror only the completed overlay; inference always receives unflipped input.
                display = cv2.flip(frame, 1)
                frames += 1
                fps = frames / max(time.monotonic() - started, 1e-6)
                target = "none (preview only)" if store is None else f"{args.label} (manual)"
                status = f"COUNTDOWN {max(1, math.ceil(countdown_until - now))}" if state == "COUNTDOWN" else state
                lines = [f"State: {status} | Target: {target}",
                         f"Session: {args.session or '-'} | Rows: {store.current_count if store else 0}/{args.samples if store else 0}",
                         "Features: 63 finite values" if features is not None else "No usable hand - no sample saved",
                         f"Preview loop: {fps:.1f} FPS | Input unflipped, display mirrored",
                         "SPACE: countdown/pause | Q/Esc: exit"]
                for index, text in enumerate(lines):
                    y = 25 + 25 * index
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if state == "DONE":
                    print(f"Target reached: {args.session}/{args.label} has {store.current_count} rows.", flush=True)
                    break
                if key == ord(" ") and store is not None:
                    if state in ("RECORDING", "COUNTDOWN"):
                        state = "PAUSED"
                    else:
                        countdown_until = time.monotonic() + 3.0
                        state = "COUNTDOWN"
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera released; preview closed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="Show landmarks without writing any data.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--session", help="Recording session, e.g. s01 or debug01.")
    parser.add_argument("--label", choices=CLASS_LABELS)
    parser.add_argument("--samples", type=int, default=60, help="Desired TOTAL for this session/label (default 60).")
    parser.add_argument("--hand", choices=("right", "left"), help="Physical hand you chose; required for capture.")
    parser.add_argument("--output", default="data/samples.csv")
    parser.add_argument("--resume", action="store_true", help="Explicitly append until the existing session/label reaches --samples.")
    args = parser.parse_args()
    if args.camera < 0 or args.samples <= 0:
        parser.error("--camera must be non-negative and --samples must be positive")
    if args.preview:
        if args.session or args.label or args.hand or args.resume:
            parser.error("Use --preview on its own with optional camera/model settings; it never captures data.")
        run_camera(args, None)
        return
    if not args.session or not args.label or not args.hand:
        parser.error("Capture requires --session, --label, and --hand. Try --preview first.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", args.session):
        parser.error("Use a session ID containing 1-32 letters, digits, underscores, or hyphens")
    output = project_path(args.output)
    if output.resolve() == (ROOT / "data/samples.csv").resolve() and args.session not in {"s01", "s02", "s03", "s04", "s05"}:
        parser.error("Use --output data/debug_samples.csv for debug recordings; data/samples.csv is reserved for s01-s05.")
    store = SampleStore(output, session=args.session, label=args.label, hand=args.hand, camera=args.camera, resume=args.resume)
    if store.current_count >= args.samples:
        print(f"Already have {store.current_count}/{args.samples}; no camera opened or rows changed.")
        return
    print(f"Saving to {output}; existing target rows: {store.current_count}. Use one collector at a time.", flush=True)
    try:
        with store:
            run_camera(args, store)
    finally:
        print(f"Saved target total: {store.current_count}/{args.samples}; rows are append-only.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. Saved samples are retained.")
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
