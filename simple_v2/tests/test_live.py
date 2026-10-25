"""Live prediction/measurement checks; no real camera or key delivery."""

import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import check_live_metadata
from hand_features import normalize_world_landmarks
from live_utils import PerformanceWindow, classify_result, prediction_from_scores, video_timestamp


class LiveTests(unittest.TestCase):
    def test_no_hand_clears_prediction_and_features_are_normalized_once(self):
        points = np.arange(63, dtype=float).reshape(21, 3) / 100
        calls = []
        class FakeClassifier:
            def predict(self, features):
                calls.append(features)
                return prediction_from_scores([0.01, 0.95, 0.02, 0.02], 0.85)
        classifier = FakeClassifier()
        result = SimpleNamespace(hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]])
        self.assertEqual(classify_result(result, classifier).label, "next")
        np.testing.assert_array_equal(calls[0], normalize_world_landmarks(points))
        self.assertIsNone(classify_result(SimpleNamespace(hand_world_landmarks=[]), classifier))
        self.assertEqual(len(calls), 1)

    def test_invalid_hand_is_not_classified(self):
        classifier = SimpleNamespace(predict=lambda _: self.fail("Invalid/no hand must not reach classifier"))
        result = SimpleNamespace(hand_world_landmarks=[[SimpleNamespace(x=0, y=0, z=0)] * 21])
        self.assertIsNone(classify_result(result, classifier))

    def test_score_threshold_is_display_only_and_malformed_scores_fail(self):
        p = prediction_from_scores([0.1, 0.4, 0.2, 0.3], 0.85)
        self.assertEqual(p.label, "next")
        self.assertFalse(p.above_threshold)
        self.assertTrue(prediction_from_scores([0, 0, 0, 1], 0.85).above_threshold)
        for values in ([1, 0], [float("nan"), 0, 0, 0, 1], [0, 0, 0, 0, 0], [-1, 0, 0, 0, 2]):
            with self.assertRaises(ValueError):
                prediction_from_scores(values, 0.85)

    def test_video_timestamps_remain_strictly_increasing(self):
        last = -1
        values = []
        for elapsed in (0, 0, 0.0001, 0.002, 0.001, 0.1):
            last = video_timestamp(last, elapsed)
            values.append(last)
        self.assertTrue(all(a < b for a, b in zip(values, values[1:])))

    def test_warmup_is_excluded_and_wall_fps_and_percentiles_are_correct(self):
        stats = PerformanceWindow(0, 5, 1)
        stats.record(4.99, 5.01, True)
        self.assertEqual(stats.summary()["measured_frames"], 0)
        stats.record(5.0, 5.02, True)
        stats.record(6.0, 6.04, False)
        result = stats.summary()
        self.assertTrue(result["completed_requested_interval"])
        self.assertAlmostEqual(result["average_processing_fps"], 2 / 1.04)
        self.assertAlmostEqual(result["median_frame_processing_ms"], 30)
        self.assertAlmostEqual(result["p95_frame_processing_ms"], 39)
        self.assertEqual((result["usable_hand_frames"], result["no_usable_hand_frames"]), (1, 1))

    def test_early_exit_empty_metrics_and_invalid_timers(self):
        stats = PerformanceWindow(0, 5, 60)
        self.assertIsNone(stats.summary()["average_processing_fps"])
        stats.record(5, 5.01, True)
        self.assertFalse(stats.complete)
        with self.assertRaises(ValueError):
            stats.record(5.005, 5.02, True)
        for warmup, duration in ((-1, 60), (5, 0), (float("nan"), 60), (5, float("inf"))):
            with self.assertRaises(ValueError):
                PerformanceWindow(0, warmup, duration)

    def test_live_metadata_requires_matching_hand_and_orientation(self):
        config = {"working_hand": "right", "input_orientation": "unflipped_camera"}
        valid = {"physical_hand": "right", "input_orientation": "unflipped_camera"}
        self.assertEqual(check_live_metadata(config, valid)["device"], "cpu")
        for key, value in (("physical_hand", "left"), ("input_orientation", "mirrored")):
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.assertRaises(ValueError):
                check_live_metadata(config, bad)


class FakeCameraTests(unittest.TestCase):
    def run_fake_camera(self, folder, fail=False):
        from app import run_camera
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None) if fail else (True, np.zeros((8, 8, 3), dtype=np.uint8))
        cv2 = SimpleNamespace(VideoCapture=lambda _: camera, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                              WINDOW_AUTOSIZE=1, COLOR_BGR2RGB=1, FONT_HERSHEY_SIMPLEX=0,
                              WND_PROP_VISIBLE=1, namedWindow=Mock(), cvtColor=lambda frame, _: frame,
                              flip=lambda frame, _: frame[:, ::-1], putText=Mock(), imshow=Mock(),
                              waitKey=lambda _: ord("q"), getWindowProperty=lambda *_: 1,
                              destroyAllWindows=Mock())
        landmarker = Mock()
        landmarker.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[])
        context = Mock(__enter__=Mock(return_value=landmarker), __exit__=Mock(return_value=False))
        vision = SimpleNamespace(HandLandmarkerOptions=lambda **_: None, RunningMode=SimpleNamespace(VIDEO=1),
                                 HandLandmarker=SimpleNamespace(create_from_options=lambda _: context))
        python = SimpleNamespace(BaseOptions=lambda **_: None, vision=vision)
        tasks = SimpleNamespace(python=python)
        mp = SimpleNamespace(tasks=tasks, Image=lambda **_: None, ImageFormat=SimpleNamespace(SRGB=1))
        modules = {"cv2": cv2, "mediapipe": mp, "mediapipe.tasks": tasks,
                   "mediapipe.tasks.python": python, "mediapipe.tasks.python.vision": vision}
        asset = folder / "synthetic.task"
        asset.write_bytes(b"fake API test asset")
        report = folder / "fake_live.json"
        config = {"camera_index": 0, "model_path": str(asset), "working_hand": "right"}
        args = SimpleNamespace(camera=None, report=str(report), warmup=0, duration=60, device_label="synthetic API test")
        classifier = SimpleNamespace(threshold=0.85, predict=lambda _: self.fail("No-hand frame reached model"))
        metadata = {"dataset_sha256": "a" * 64, "feature_version": "world_wrist_palm_v1", "selected_epoch": 1, "gesture_set_version": "simple_index_v2", "gesture_poses": {}}
        with patch.dict(sys.modules, modules), patch("app.time.sleep"), contextlib.redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
                    run_camera(args, config, classifier, metadata, "b" * 64)
                self.assertFalse(report.exists())
            else:
                run_camera(args, config, classifier, metadata, "b" * 64)
                result = json.loads(report.read_text())
                self.assertEqual(result["key_events_sent"], 0)
                self.assertEqual(result["no_usable_hand_frames"], 1)
                self.assertFalse(result["completed_requested_interval"])
        camera.release.assert_called_once()
        cv2.destroyAllWindows.assert_called_once()
        context.__exit__.assert_called_once()

    def test_q_exit_releases_camera_clears_no_hand_and_marks_early_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_fake_camera(Path(temporary))

    def test_camera_read_failure_releases_resources_without_completion_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.run_fake_camera(Path(temporary), fail=True)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable here; run native live check on PC/laptop")
class NativeLiveTests(unittest.TestCase):
    def test_live_classifier_matches_native_model_and_rejects_invalid_features(self):
        import torch
        from app import Classifier
        from model import GestureMLP
        torch.manual_seed(42)
        model = GestureMLP()
        classifier = Classifier(model, 0.85)
        features = np.linspace(-1, 1, 63, dtype=np.float32)
        prediction = classifier.predict(features)
        with torch.inference_mode():
            expected = torch.softmax(model(torch.from_numpy(features)[None]), dim=1).numpy()[0]
        self.assertEqual(prediction, prediction_from_scores(expected, 0.85))
        self.assertFalse(model.training)
        for invalid in (np.zeros(62), np.full(63, np.nan)):
            with self.assertRaises(ValueError):
                classifier.predict(invalid)


if __name__ == "__main__":
    unittest.main()
