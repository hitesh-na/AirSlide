"""P5 live classifier preview: dry run only, no presentation keys are sent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import platform
import sys
import time

import numpy as np

from collect_data import draw_hand
from dataset import file_sha256, load_dataset, read_config
from hand_features import FEATURE_COUNT
from live_utils import PerformanceWindow, classify_result, prediction_from_scores, video_timestamp
from training_utils import package_versions, project_path, require_manifest, training_settings, write_json


class Classifier:
    def __init__(self, model, threshold):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.threshold = threshold

    def predict(self, features):
        values = np.asarray(features, dtype=np.float32)
        if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
            raise ValueError("Classifier requires one finite normalized 63-feature vector")
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0))
            scores = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
        return prediction_from_scores(scores, self.threshold)


def check_live_metadata(config, metadata):
    if metadata.get("physical_hand") != config["working_hand"]:
        raise ValueError("Checkpoint and configuration physical hands differ")
    if metadata.get("input_orientation") != config["input_orientation"]:
        raise ValueError("Checkpoint and configuration camera orientations differ")
    return training_settings({"training": metadata.get("training_settings", {})})


def load_inputs(args):
    import torch
    from model import load_gesture_model
    config = read_config(project_path(args.config))
    data = load_dataset(project_path(config["dataset_path"]), config)
    require_manifest(data, config)
    checkpoint = project_path(args.checkpoint)
    model, metadata = load_gesture_model(checkpoint, expected_dataset_sha256=data.sha256)
    settings = check_live_metadata(config, metadata)
    torch.set_num_threads(settings["num_threads"])
    threshold = config["controller"]["score_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Configured score threshold must be numeric")
    classifier = Classifier(model, threshold)
    classifier.predict(np.zeros(FEATURE_COUNT, dtype=np.float32))
    print(f"Checkpoint loaded: {checkpoint}", flush=True)
    print(f"Dataset SHA-256 matched: {data.sha256}", flush=True)
    print(f"Feature/class/hand metadata matched; use your {config['working_hand']} hand.", flush=True)
    return config, classifier, metadata, file_sha256(checkpoint)


def run_camera(args, config, classifier, metadata, checkpoint_hash):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("Run this camera preview in the laptop's existing full .venv.") from exc
    camera_index = config["camera_index"] if args.camera is None else args.camera
    landmark_path = project_path(config["model_path"])
    if not landmark_path.is_file():
        raise FileNotFoundError(f"Missing {landmark_path}; keep the laptop's existing hand_landmarker.task.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = project_path(args.report or f"reports/live_{stamp}.json")
    if report_path.exists():
        raise ValueError("Live report already exists; use a new --report filename")
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(landmark_path)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    camera, stats, reason = None, None, "user_exit"
    last_timestamp, failures, last_state = -1, 0, None
    last_latency = None
    width = height = None
    window = "GesturePilot - live classifier (DRY RUN)"
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError(f"Camera {camera_index} did not open. Close other camera apps and check Windows permissions.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            ready = time.perf_counter()
            stats = PerformanceWindow(ready, args.warmup, args.duration)
            print(f"DRY RUN: predictions only; no keys sent. Warm-up {args.warmup:g}s, then measure {args.duration:g}s.", flush=True)
            print("Show all five gestures, remove your hand, and check labels. Q/Esc exits; window closes after the timed measurement.", flush=True)
            while True:
                frame_started = time.perf_counter()
                ok, frame = camera.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Camera stopped returning frames; preview stopped.")
                    time.sleep(0.05)
                    continue
                failures = 0
                height, width = frame.shape[:2]
                timestamp = video_timestamp(last_timestamp, frame_started - ready)
                last_timestamp = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                prediction = classify_result(result, classifier)
                if result.hand_landmarks:
                    draw_hand(cv2, frame, result.hand_landmarks[0])
                # Only skeleton/camera pixels are mirrored; all text is drawn afterward.
                display = cv2.flip(frame, 1)
                state = "No usable hand" if prediction is None else prediction.label
                if state != last_state:
                    print(f"Preview: {state}", flush=True)
                    last_state = state
                elapsed = max(0.0, frame_started - ready)
                timing = (f"Warm-up: {max(0, args.warmup - elapsed):.1f}s" if elapsed < args.warmup
                          else f"Measurement: {stats.elapsed:.1f}/{args.duration:g}s")
                fps = len(stats.latencies_ms) / stats.elapsed if stats.elapsed > 0 else 0.0
                label_line = "No usable hand | no action" if prediction is None else f"Prediction: {prediction.label} | score {prediction.score:.3f}"
                quality = ("No usable hand" if prediction is None else
                           "Score above configured threshold" if prediction.above_threshold else "Low score - uncertain prediction")
                latency_text = "-" if last_latency is None else f"{last_latency:.1f}"
                lines = ["DRY RUN - prediction only; no keys sent", label_line,
                         f"{quality} | threshold {classifier.threshold:.2f}", timing,
                         f"Processing FPS: {fps:.1f} | previous frame: {latency_text} ms",
                         f"Use {config['working_hand']} hand | Q/Esc exits"]
                for index, line in enumerate(lines):
                    y = 24 + index * 24
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                processed = time.perf_counter()
                last_latency = (processed - frame_started) * 1000
                stats.record(frame_started, processed, prediction is not None)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if stats.complete:
                    reason = "duration_reached"
                    break
                if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("Camera released; live preview closed.", flush=True)
    summary = stats.summary()
    summary.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "dry-run prediction only",
        "key_events_sent": 0, "exit_reason": reason, "device_label": args.device_label,
        "host_platform": platform.platform(), "python_version": platform.python_version(),
        "camera_index": camera_index, "frame_width": width, "frame_height": height,
        "checkpoint_sha256": checkpoint_hash, "dataset_sha256": metadata["dataset_sha256"],
        "feature_version": metadata["feature_version"], "selected_epoch": metadata["selected_epoch"],
        "physical_hand": config["working_hand"], "package_versions": package_versions(),
        "score_threshold": classifier.threshold, "score_note": "Uncalibrated softmax model score; no accuracy measured by this preview.",
    })
    write_json(report_path, summary)
    print(f"Live report: {report_path}", flush=True)
    print(f"Measured frames: {summary['measured_frames']} | usable hand: {summary['usable_hand_frames']} | no usable hand: {summary['no_usable_hand_frames']}")
    for key in ("average_processing_fps", "median_frame_processing_ms", "p95_frame_processing_ms"):
        value = summary[key]
        print(f"{key}: {value:.3f}" if value is not None else f"{key}: NOT_MEASURED")
    print("MEASUREMENT COMPLETE" if summary["completed_requested_interval"] else "PREVIEW ENDED EARLY - full interval not completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default="models/gesture_mlp.pt")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Prediction-only mode (also the default).")
    parser.add_argument("--check", action="store_true", help="Check dataset/model/finite inference without opening a camera.")
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=60.0, help="Measured seconds after warm-up; exits automatically.")
    parser.add_argument("--device-label", choices=("laptop", "desktop", "unspecified"), default="laptop")
    parser.add_argument("--report", help="New JSON report filename; default uses a unique UTC timestamp.")
    args = parser.parse_args()
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be non-negative")
    if not math.isfinite(args.warmup) or args.warmup < 0 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("Use finite --warmup >= 0 and --duration > 0")
    config, classifier, metadata, digest = load_inputs(args)
    if args.check:
        print("MODEL CHECK PASSED - finite synthetic inference only; no camera opened or keys sent.")
        return
    run_camera(args, config, classifier, metadata, digest)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPreview interrupted. No completed measurement is claimed.", file=sys.stderr)
        sys.exit(130)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"LIVE PREVIEW FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
