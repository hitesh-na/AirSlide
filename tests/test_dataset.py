"""Dataset/split integrity checks with synthetic temporary CSVs only."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collect_data import COLUMNS
from dataset import CSV_COLUMNS, EXPECTED_SPLITS, load_dataset, make_manifest, read_config, save_manifest
from hand_features import CLASS_LABELS, FEATURE_VERSION, normalize_world_landmarks


def synthetic_rows(per_class):
    rng = np.random.default_rng(314)
    rows = []
    for session in ("s01", "s02", "s03", "s04", "s05"):
        for label in range(5):
            for _ in range(per_class):
                vector = normalize_world_landmarks(rng.normal(size=(21, 3)))
                row = dict(zip(CSV_COLUMNS, [
                    f"sample-{len(rows):06d}", session, str(label),
                    "2026-10-22T22:00:00+00:00", FEATURE_VERSION,
                    *[format(float(v), ".9g") for v in vector],
                ]))
                rows.append(row)
    return rows


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "samples.csv"
        self.config = json.loads((ROOT / "config.json").read_text())
        self.config["dataset_path"] = str(self.path)
        self.metadata = {
            "feature_version": FEATURE_VERSION,
            "class_labels": list(CLASS_LABELS),
            "physical_hand": "right",
            "camera_index": 0,
            "input_orientation": "unflipped_camera",
        }
        self.path.with_suffix(".metadata.json").write_text(json.dumps(self.metadata))
        self.rows = synthetic_rows(2)
        self.write_rows()

    def write_rows(self, rows=None):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def load(self, config=None):
        return load_dataset(self.path, self.config if config is None else config, expected_per_class=2)

    def test_collector_schema_and_no_second_normalization(self):
        self.assertEqual(CSV_COLUMNS, COLUMNS)
        data = self.load()
        original = np.array([self.rows[0][f"f{i:02d}"] for i in range(63)], dtype=np.float32)
        np.testing.assert_array_equal(data.features[0], original)
        self.assertEqual(data.features.shape, (50, 63))
        self.assertEqual(data.features.dtype, np.float32)

    def test_splits_are_disjoint_with_all_classes_and_ids(self):
        data = self.load()
        self.assertEqual([len(data.split_indices[k]) for k in EXPECTED_SPLITS], [30, 10, 10])
        sets = [set(indices.tolist()) for indices in data.split_indices.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        manifest = make_manifest(data, self.config, project_root=self.folder)
        self.assertEqual(manifest["total_rows"], 50)
        self.assertEqual(manifest["unique_sample_ids"], 50)
        self.assertEqual(manifest["dataset_csv"], "samples.csv")
        for split in manifest["splits"].values():
            self.assertEqual(set(split["class_counts"]), set(CLASS_LABELS))
            self.assertEqual(len(split["sample_ids"]), split["row_count"])

    def test_overlapping_session_split_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["split_sessions"]["train"].append("s04")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.load(config)

    def test_incomplete_class_and_unknown_session_are_rejected(self):
        self.write_rows(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "s05/start: found 1"):
            self.load()
        changed = copy.deepcopy(self.rows)
        changed[-1]["session_id"] = "debug01"
        self.write_rows(changed)
        with self.assertRaisesRegex(ValueError, "unexpected session"):
            self.load()

    def test_duplicate_id_and_exact_cross_split_vector_are_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed[-1]["sample_id"] = changed[0]["sample_id"]
        self.write_rows(changed)
        with self.assertRaisesRegex(ValueError, "duplicate sample ID"):
            self.load()
        changed = copy.deepcopy(self.rows)
        for i in range(63):
            changed[-1][f"f{i:02d}"] = changed[0][f"f{i:02d}"]
        self.write_rows(changed)
        with self.assertRaisesRegex(ValueError, "across splits"):
            self.load()

    def test_invalid_features_versions_and_timestamps_are_rejected(self):
        corruptions = (
            ("f01", "nan", "non-finite"),
            ("f00", "0.25", "wrist is not centered"),
            ("f27", "20", "palm scale"),
            ("feature_version", "wrong_version", "feature version"),
            ("timestamp_utc", "2026-10-22T22:00:00", "UTC timestamp"),
            ("label_id", "5", "label ID"),
        )
        for field, value, message in corruptions:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = value
                self.write_rows(changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_metadata_hand_mismatch_and_bad_configuration_are_rejected(self):
        changed = dict(self.metadata, physical_hand="left")
        self.path.with_suffix(".metadata.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "metadata and configuration disagree"):
            self.load()
        config_path = self.folder / "config.json"
        changed_config = dict(self.config, class_labels=list(reversed(CLASS_LABELS)))
        config_path.write_text(json.dumps(changed_config))
        with self.assertRaisesRegex(ValueError, "class order"):
            read_config(config_path)

    def test_manifest_is_idempotent_and_stale_manifest_is_not_overwritten(self):
        manifest = make_manifest(self.load(), self.config, project_root=self.folder)
        output = self.folder / "split_manifest.json"
        self.assertTrue(save_manifest(output, manifest))
        original = output.read_bytes()
        refreshed = dict(manifest, created_at_utc="another-time")
        self.assertFalse(save_manifest(output, refreshed))
        self.assertEqual(output.read_bytes(), original)
        stale = dict(manifest, dataset_sha256="different-source-hash")
        with self.assertRaisesRegex(ValueError, "different data/settings"):
            save_manifest(output, stale)
        self.assertEqual(output.read_bytes(), original)

    def test_cli_validates_full_synthetic_dataset_without_changing_csv(self):
        self.write_rows(synthetic_rows(60))
        original = self.path.read_bytes()
        config_path = self.folder / "config.json"
        config_path.write_text(json.dumps(self.config))
        manifest_path = self.folder / "split_manifest.json"
        command = [sys.executable, str(ROOT / "prepare_data.py"), "--config", str(config_path), "--output", str(manifest_path)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Total rows: 1500", result.stdout)
        self.assertIn("Unique IDs: 1500", result.stdout)
        self.assertIn("DATASET CHECK PASSED", result.stdout)
        self.assertEqual(self.path.read_bytes(), original)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual([manifest["splits"][k]["row_count"] for k in EXPECTED_SPLITS], [900, 300, 300])


if __name__ == "__main__":
    unittest.main()
