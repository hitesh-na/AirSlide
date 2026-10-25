"""Return ZIP provenance and exclusions from temporary synthetic artifacts."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hand_features import CLASS_LABELS, FEATURE_VERSION
from make_demo_bundle import DEMO_FILES, build_demo_bundle
import test_bundle


class DemoBundleTests(unittest.TestCase):
    def setUp(self):
        # Reuse the real dataset-validator fixture, never the student's dataset.
        fixture = test_bundle.BundleTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.digest = fixture.root, fixture.digest
        self.output = fixture.folder / "demo.zip"
        for name in (".venv/excluded.txt", "models/hand_landmarker.task", "data/debug_samples.csv"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("excluded synthetic file")
        for name in DEMO_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if name.startswith(("reports/", "models/")):
                target.write_bytes(b"synthetic test artifact, not a real model or report")
            elif not target.exists():
                shutil.copy2(ROOT / name, target)
        digest = hashlib.sha256((self.root / "models/gesture_mlp.pt").read_bytes()).hexdigest()
        common = {"dataset_sha256": self.digest, "checkpoint_sha256": digest,
                  "feature_version": FEATURE_VERSION, "class_labels": list(CLASS_LABELS),
                  "physical_hand": "right", "selected_epoch": 1, "sample_count": 300,
                  "session_count": 1, "accuracy": 0.5, "macro_f1": 0.5, "device_label": "desktop"}
        (self.root / "reports/training_summary.json").write_text(json.dumps(common))
        for name, split, session in (("validation_metrics.json", "validation", "s04"), ("metrics.json", "test", "s05")):
            (self.root / "reports" / name).write_text(json.dumps(dict(common, split=split, session_ids=[session])))

    def test_packs_model_reports_but_preserves_laptop_csv_environment_and_asset(self):
        info = build_demo_bundle(self.root, self.output, expected_digest=self.digest)
        with ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn("models/gesture_mlp.pt", archive.namelist())
            self.assertIn("reports/metrics.json", archive.namelist())
            for excluded in ("data/samples.csv", "models/hand_landmarker.task", ".venv"):
                self.assertNotIn(excluded, archive.namelist())
            self.assertFalse(any(name.startswith(".venv/") or "debug_samples" in name for name in archive.namelist()))
            for name, expected in info["files"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected["sha256"])
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_demo_bundle(self.root, self.output, expected_digest=self.digest)
        self.assertEqual(self.output.read_bytes(), original)

    def test_mismatched_checkpoint_or_wrong_split_report_is_refused(self):
        path = self.root / "reports/metrics.json"
        report = json.loads(path.read_text())
        for field, value in (("checkpoint_sha256", "a" * 64), ("session_ids", ["s04"]), ("accuracy", float("nan"))):
            path.write_text(json.dumps(dict(report, **{field: value})))
            with self.assertRaises(ValueError):
                build_demo_bundle(self.root, self.output, expected_digest=self.digest)
            self.assertFalse(self.output.exists())
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "snapshot"):
            build_demo_bundle(self.root, self.output, expected_digest="b" * 64)


if __name__ == "__main__":
    unittest.main()
