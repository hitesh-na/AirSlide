"""Pack the evaluated classifier, reports and live-preview source for the laptop."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from dataset import file_sha256, load_dataset, make_manifest, read_config
from hand_features import CLASS_LABELS, FEATURE_VERSION
from make_training_bundle import TRAINING_FILES, VERIFIED_DATA_SHA256

ROOT = Path(__file__).resolve().parent
DEMO_SOURCE_FILES = tuple(name for name in TRAINING_FILES if not name.startswith("data/")) + (
    "app.py", "live_utils.py", "make_demo_bundle.py", "P5_LIVE.md",
    "tests/test_live.py", "tests/test_demo_bundle.py",
)
REPORT_FILES = (
    "reports/training_summary.json", "reports/training_history.csv", "reports/training_curves.png",
    "reports/validation_metrics.json", "reports/validation_classification_report.txt",
    "reports/validation_confusion_matrix.png", "reports/validation_predictions.csv",
    "reports/metrics.json", "reports/classification_report.txt",
    "reports/confusion_matrix.png", "reports/test_predictions.csv",
)
DEMO_FILES = DEMO_SOURCE_FILES + (
    "models/gesture_mlp.pt", "data/samples.metadata.json", "data/split_manifest.json",
) + REPORT_FILES


def verify_reports(root, dataset_digest):
    root = Path(root)
    checkpoint_digest = file_sha256(root / "models/gesture_mlp.pt")
    training = json.loads((root / "reports/training_summary.json").read_text(encoding="utf-8-sig"))
    if (training.get("dataset_sha256") != dataset_digest or training.get("feature_version") != FEATURE_VERSION
            or training.get("class_labels") != list(CLASS_LABELS) or training.get("physical_hand") != "right"
            or training.get("device_label") != "desktop" or type(training.get("selected_epoch")) is not int
            or training["selected_epoch"] < 1):
        raise ValueError("Training summary does not describe the verified desktop model/data")
    summaries = {}
    for split, filename, session in (("validation", "validation_metrics.json", "s04"), ("test", "metrics.json", "s05")):
        report = json.loads((root / "reports" / filename).read_text(encoding="utf-8-sig"))
        expected = {"dataset_sha256": dataset_digest, "checkpoint_sha256": checkpoint_digest,
                    "feature_version": FEATURE_VERSION, "class_labels": list(CLASS_LABELS),
                    "physical_hand": "right", "split": split, "session_ids": [session],
                    "sample_count": 300, "session_count": 1, "selected_epoch": training["selected_epoch"]}
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{split} report and checkpoint/data fingerprints or split metadata differ")
        for name in ("accuracy", "macro_f1"):
            value = report.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {split} {name}")
        summaries[split] = {key: report[key] for key in ("accuracy", "macro_f1", "sample_count", "session_ids")}
    return checkpoint_digest, summaries


def build_demo_bundle(root, output, *, expected_digest=VERIFIED_DATA_SHA256):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"{output} already exists; preserve it or choose a new --output filename")
    missing = [name for name in DEMO_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))
    config = read_config(root / "config.json")
    if config["dataset_path"] != "data/samples.csv":
        raise ValueError("Keep the portable dataset_path data/samples.csv")
    data = load_dataset(root / config["dataset_path"], config)
    if data.sha256 != expected_digest:
        raise ValueError("Dataset differs from the previously verified snapshot; preserve files and report this")
    manifest = json.loads((root / "data/split_manifest.json").read_text(encoding="utf-8-sig"))
    stable = lambda value: {key: item for key, item in value.items() if key != "created_at_utc"}
    if not isinstance(manifest, dict) or stable(manifest) != stable(make_manifest(data, config, project_root=root)):
        raise ValueError("Split manifest differs from the verified dataset/configuration")
    names = list(DEMO_FILES)
    for name in ("requirements.desktop.lock.txt", "requirements.lock.txt"):
        if (root / name).is_file():
            names.append(name)
    files = {name: {"sha256": file_sha256(root / name), "bytes": (root / name).stat().st_size} for name in names}
    checkpoint_digest, summaries = verify_reports(root, data.sha256)
    info = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "destination": "existing camera laptop project", "dataset_sha256": data.sha256,
            "checkpoint_sha256": checkpoint_digest, "evaluation": summaries, "files": files,
            "existing_laptop_file_required": "data/samples.csv with the recorded dataset hash",
            "excluded": [".venv", "camera landmark model", "raw CSV", "debug samples", "images/video", "package/temp caches"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(root / name, arcname=name)
            archive.writestr("DEMO_TRANSFER_INFO.json", json.dumps(info, indent=2) + "\n")
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Demo ZIP failed CRC verification")
            for name, expected in files.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected["sha256"]:
                    raise RuntimeError(f"File changed during packing: {name}; stop editing/training and retry")
        if output.exists():
            raise ValueError("Output appeared during packing; choose a new filename")
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New ZIP path, e.g. V:\\GesturePilot_Laptop.zip")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    info = build_demo_bundle(ROOT, output)
    print(f"Verified checkpoint SHA-256: {info['checkpoint_sha256']}")
    print(f"Verified dataset SHA-256: {info['dataset_sha256']}")
    print(f"Laptop ZIP saved: {output}")
    print(f"ZIP SHA-256: {file_sha256(output)}")
    print("Copy this ZIP to the laptop Downloads folder. Its original CSV, .venv and hand_landmarker.task stay on the laptop.")
    print("LAPTOP PACKAGE READY")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Source model/data/reports were not changed.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"LAPTOP PACKAGE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
