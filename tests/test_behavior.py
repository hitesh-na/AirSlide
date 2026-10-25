"""Numerical and CSV behavior checks. No webcam or real key sender is used."""

import csv
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_features import FEATURE_VERSION, features_from_result, normalize_world_landmarks
from collect_data import SampleStore


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(42).normal(size=(21, 3)) * 0.02

    def test_translation_and_positive_scale_invariance(self):
        original = normalize_world_landmarks(self.points)
        moved = normalize_world_landmarks(self.points * 3.7 + [2, -3, 0.5])
        np.testing.assert_allclose(original, moved, rtol=1e-5, atol=1e-6)
        self.assertEqual(original.shape, (63,))
        self.assertEqual(original.dtype, np.float32)
        np.testing.assert_array_equal(original[:3], [0, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(original.reshape(21, 3)[9])), 1, places=5)

    def test_rejects_missing_nonfinite_shape_and_degenerate_scale(self):
        nan_points = self.points.copy()
        nan_points[2, 1] = np.nan
        for bad in (None, [], np.zeros((20, 3)), np.zeros((21, 3)), nan_points):
            with self.subTest(value=type(bad).__name__):
                with self.assertRaises(ValueError):
                    normalize_world_landmarks(bad)

    def test_world_result_and_no_hand(self):
        landmarks = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in self.points]
        actual = features_from_result(SimpleNamespace(hand_world_landmarks=[landmarks]))
        np.testing.assert_array_equal(actual, normalize_world_landmarks(self.points))
        self.assertIsNone(features_from_result(SimpleNamespace(hand_world_landmarks=[])))
        self.assertIsNone(features_from_result(SimpleNamespace(hand_world_landmarks=[landmarks[:20]])))


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "samples.csv"
        self.features = normalize_world_landmarks(np.random.default_rng(7).normal(size=(21, 3)))

    def store(self, **changes):
        options = dict(session="s01", label="neutral", hand="right", camera=0)
        options.update(changes)
        return SampleStore(self.path, **options)

    def add_rows(self, count=2):
        with self.store() as store:
            for _ in range(count):
                store.append(self.features)

    def test_explicit_resume_preserves_rows_and_unique_ids(self):
        self.add_rows()
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "--resume"):
            self.store()
        self.assertEqual(self.path.read_bytes(), original)
        with self.store(resume=True) as store:
            self.assertEqual(store.current_count, 2)
            store.append(self.features)
        self.assertTrue(self.path.read_bytes().startswith(original))
        with self.path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["sample_id"] for row in rows}), 3)
        self.assertTrue(all(row["feature_version"] == FEATURE_VERSION for row in rows))
        restored = np.array([rows[-1][f"f{i:02d}"] for i in range(63)], dtype=np.float32)
        np.testing.assert_array_equal(restored, self.features)

    def test_separate_label_and_session_counts(self):
        self.add_rows()
        with self.store(label="next") as store:
            self.assertEqual(store.current_count, 0)
            store.append(self.features)
        with self.store(session="s02") as store:
            store.append(self.features)
        counts = self.store(resume=True).counts
        self.assertEqual(counts[("s01", 0)], 2)
        self.assertEqual(counts[("s01", 1)], 1)
        self.assertEqual(counts[("s02", 0)], 1)

    def test_changed_hand_camera_or_missing_metadata_is_rejected(self):
        self.add_rows()
        for changes in ({"hand": "left"}, {"camera": 1}):
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                self.store(resume=True, **changes)
        self.path.with_suffix(".metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.store(resume=True)

    def test_duplicate_id_and_nonfinite_csv_are_rejected(self):
        self.add_rows()
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            rows = list(reader)
        for corrupt in ("duplicate", "nan"):
            changed = [dict(row) for row in rows]
            if corrupt == "duplicate":
                changed[1]["sample_id"] = changed[0]["sample_id"]
            else:
                changed[1]["f00"] = "nan"
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(changed)
            with self.assertRaisesRegex(ValueError, "Invalid CSV row"):
                self.store(resume=True)

    def test_invalid_feature_is_not_appended(self):
        with self.store() as store:
            for bad in (np.zeros(62), np.full(63, np.nan)):
                with self.assertRaises(ValueError):
                    store.append(bad)
            self.assertEqual(store.current_count, 0)

    def test_orphan_metadata_still_protects_physical_hand(self):
        self.add_rows()
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            self.store(hand="left")


if __name__ == "__main__":
    unittest.main()
