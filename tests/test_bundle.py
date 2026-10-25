"""Verify desktop-transfer archives using synthetic temporary project data."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import CSV_COLUMNS, file_sha256, load_dataset, make_manifest
from hand_features import CLASS_LABELS, FEATURE_VERSION
from make_training_bundle import TRAINING_FILES, build_training_bundle
from test_dataset import synthetic_rows


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.root = self.folder / "project"
        for name in TRAINING_FILES:
            if name.startswith("data/"):
                continue
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.data_path = self.root / "data/samples.csv"
        self.data_path.parent.mkdir(parents=True)
        with self.data_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(synthetic_rows(60))
        metadata = {"feature_version": FEATURE_VERSION, "class_labels": list(CLASS_LABELS),
                    "physical_hand": "right", "camera_index": 0, "input_orientation": "unflipped_camera"}
        self.data_path.with_suffix(".metadata.json").write_text(json.dumps(metadata))
        config = json.loads((self.root / "config.json").read_text())
        data = load_dataset(self.data_path, config)
        (self.root / "data/split_manifest.json").write_text(json.dumps(make_manifest(data, config, project_root=self.root)))
        self.digest = file_sha256(self.data_path)
        self.output = self.folder / "training.zip"

    def test_bundle_preserves_hashes_and_excludes_environment_debug_and_camera(self):
        for name in (".venv/excluded.txt", "data/debug_samples.csv", "models/hand_landmarker.task"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("excluded synthetic test file")
        (self.root / "requirements.lock.txt").write_text("synthetic provenance snapshot")
        original = self.data_path.read_bytes()
        info = build_training_bundle(self.root, self.output, expected_digest=self.digest)
        self.assertEqual(info["total_rows"], 1500)
        self.assertEqual(self.data_path.read_bytes(), original)
        with ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(hashlib.sha256(archive.read("data/samples.csv")).hexdigest(), self.digest)
            self.assertFalse(any(name.startswith((".venv/", "models/")) or "debug_samples" in name for name in archive.namelist()))
            info = json.loads(archive.read("TRANSFER_INFO.json"))
            for name, values in info["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), values["sha256"])

    def test_wrong_snapshot_and_existing_archive_are_refused(self):
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            build_training_bundle(self.root, self.output, expected_digest="b"*64)
        self.assertFalse(self.output.exists())
        build_training_bundle(self.root, self.output, expected_digest=self.digest)
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_training_bundle(self.root, self.output, expected_digest=self.digest)
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
