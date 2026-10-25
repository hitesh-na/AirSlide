"""Build a portable desktop-training ZIP from the laptop's verified project/data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from dataset import file_sha256, load_dataset, make_manifest, read_config

ROOT = Path(__file__).resolve().parent
VERIFIED_DATA_SHA256 = "a10f2967aedcdcb37801aded40a3504151a8b447e1f937f72ff72e96f8f00d6d"
TRAINING_FILES = (
    "model.py", "train.py", "evaluate.py", "training_utils.py", "reporting.py",
    "dataset.py", "prepare_data.py", "hand_features.py", "collect_data.py",
    "make_training_bundle.py", "config.json", "requirements-training.txt",
    "README.md", "P4_TRAINING.md", "DESKTOP_TRAINING.md", "GESTUREPILOT_MOTHER.md",
    "tests/test_behavior.py", "tests/test_dataset.py", "tests/test_training.py", "tests/test_bundle.py",
    "data/samples.csv", "data/samples.metadata.json", "data/split_manifest.json",
)


def build_training_bundle(root: Path, output: Path, *, expected_digest=VERIFIED_DATA_SHA256):
    """Read and verify the source data; write only a new archive."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f"{output} already exists. Keep it or choose another --output filename.")
    missing = [name for name in TRAINING_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing project files: " + ", ".join(missing) + ". Extract the complete Desktop Prep package into the laptop project first.")
    config = read_config(root / "config.json")
    csv_path = Path(config["dataset_path"])
    if csv_path.is_absolute():
        raise ValueError("Use the portable relative dataset_path data/samples.csv before transferring.")
    if csv_path.as_posix() != "data/samples.csv":
        raise ValueError("This transfer expects the verified data/samples.csv dataset")
    data = load_dataset(root / csv_path, config)
    if data.sha256 != expected_digest:
        raise ValueError("Dataset fingerprint differs from the verified 1,500-row snapshot. Preserve the files and report this before transferring.")
    manifest = json.loads((root / "data/split_manifest.json").read_text(encoding="utf-8-sig"))
    expected = make_manifest(data, config, project_root=root)
    stable = lambda value: {key: item for key, item in value.items() if key != "created_at_utc"}
    if not isinstance(manifest, dict) or stable(manifest) != stable(expected):
        raise ValueError("Saved split manifest does not match the verified dataset/configuration")
    names = list(TRAINING_FILES)
    # Keep the laptop dependency snapshot for provenance, not as the PC environment.
    if (root / "requirements.lock.txt").is_file():
        names.append("requirements.lock.txt")
    file_info = {name: {"sha256": file_sha256(root / name), "bytes": (root / name).stat().st_size} for name in names}
    transfer_info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_training_device": "desktop CPU",
        "dataset_sha256": data.sha256, "total_rows": len(data.sample_ids),
        "physical_hand": data.metadata["physical_hand"],
        "split_sessions": config["split_sessions"], "files": file_info,
        "excluded": [".venv", "debug recordings", "camera model asset", "images/video", "caches", "old reports/checkpoints"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(root / name, arcname=name)
            archive.writestr("TRANSFER_INFO.json", json.dumps(transfer_info, indent=2) + "\n")
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Transfer ZIP failed its integrity check")
            for name, expected_file in file_info.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_file["sha256"]:
                    raise RuntimeError(f"File changed while packing: {name}. Close the collector/editor and try again.")
        if output.exists():
            raise ValueError("Output appeared during packing; keep it and choose another filename")
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return transfer_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT.parent / "GesturePilot_Training.zip"), help="New ZIP path; existing ZIPs are never overwritten.")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    info = build_training_bundle(ROOT, output)
    print(f"Verified dataset: {info['total_rows']} rows; matching source and split manifest")
    print(f"Dataset SHA-256: {info['dataset_sha256']}")
    print(f"Training ZIP saved: {output}")
    print(f"ZIP SHA-256: {file_sha256(output)}")
    print("Copy this ZIP to the desktop PC. No virtual environment or camera recording is included.")
    print("On the PC, check its Python installation with: py --list")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user. The source project/data were not changed.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"TRANSFER PACKAGE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
