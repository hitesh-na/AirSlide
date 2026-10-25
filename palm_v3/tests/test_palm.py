import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import file_sha256, load_dataset, read_config
from evaluate_diagnostic import DEVELOPMENT_SHA256, load_development_capture, score_summary
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from live_utils import classify_result_with_features, prediction_from_scores
from palm_features import DegeneratePalmError, MODEL_INPUT_VERSION, to_model_input
from training_utils import ARCHITECTURE, DEFAULT_TRAINING, require_manifest, validate_checkpoint_metadata

EXPECTED_DATA = '4ca0275f8bc4aaf1c171ca1ed5dda1f4a629113602ade0fd2637f575b2708dc4'


def sample_metadata(data, config):
    return {'feature_version': FEATURE_VERSION, 'model_input_version': MODEL_INPUT_VERSION,
            'class_labels': list(CLASS_LABELS), 'gesture_set_version': GESTURE_SET_VERSION,
            'gesture_poses': GESTURE_POSES, 'architecture': ARCHITECTURE,
            'split_sessions': config['split_sessions'], 'dataset_sha256': data.sha256,
            'physical_hand': 'right', 'input_orientation': 'unflipped_camera',
            'training_settings': dict(DEFAULT_TRAINING), 'selected_epoch': 1}


class DataCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_config(ROOT / 'config.json')
        cls.data = load_dataset(ROOT / 'data/samples.csv', cls.config)
        tr = cls.data.split_indices['train']
        cls.x = np.concatenate([cls.data.features[tr][cls.data.labels[tr] == label][:3] for label in range(4)])


class PalmGeometryTests(DataCase):
    def test_rigid_rotations_preserve_model_inputs_for_every_class(self):
        reference = to_model_input(self.x)
        rng = np.random.default_rng(8103)
        for _ in range(30):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            rotated = (self.x.reshape(-1, 21, 3) @ rotation).reshape(-1, 63)
            np.testing.assert_allclose(to_model_input(rotated), reference, rtol=0, atol=2e-6)

    def test_rotation_preserves_all_pairwise_landmark_distances(self):
        before = self.x.reshape(-1, 21, 3)
        after = to_model_input(self.x).reshape(-1, 21, 3)
        raw = np.linalg.norm(before[:, :, None] - before[:, None, :], axis=-1)
        transformed = np.linalg.norm(after[:, :, None] - after[:, None, :], axis=-1)
        np.testing.assert_allclose(raw, transformed, rtol=1e-6, atol=5e-7)
        np.testing.assert_allclose(after[:, 0], 0, atol=1e-7)
        np.testing.assert_allclose(after[:, 9], np.tile([0, 1, 0], (len(after), 1)), atol=2e-7)

    def test_single_batch_parity_and_input_is_not_modified(self):
        source = self.x.copy()
        batch = to_model_input(source)
        for i in range(len(source)):
            np.testing.assert_array_equal(to_model_input(source[i]), batch[i])
        np.testing.assert_array_equal(source, self.x)
        self.assertEqual(batch.dtype, np.float32)

    def test_bad_shape_nonfinite_and_wrong_normalization_are_rejected(self):
        for bad in ([], np.zeros(62), np.zeros((1, 1, 63)), np.full(63, np.nan)):
            with self.assertRaises(ValueError):
                to_model_input(bad)
        bad = self.x[0].copy(); bad[0] = .01
        with self.assertRaisesRegex(ValueError, 'wrist-centered'):
            to_model_input(bad)
        with self.assertRaisesRegex(ValueError, 'palm-scaled'):
            to_model_input(self.x[0] * 2)

    def test_degenerate_palm_is_rejected(self):
        with self.assertRaises(DegeneratePalmError):
            to_model_input(np.zeros(63))
        bad = self.x[0].reshape(21, 3).copy()
        bad[5] = bad[17]
        with self.assertRaises(DegeneratePalmError):
            to_model_input(bad.reshape(63))

    def test_degenerate_live_hand_clears_prediction_and_logged_features(self):
        points = self.x[0].reshape(21, 3).copy(); points[5] = points[17]
        result = SimpleNamespace(hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]])
        class Classifier:
            def predict(self, features):
                to_model_input(features)
                raise AssertionError('Degenerate hand must be rejected first')
        self.assertEqual(classify_result_with_features(result, Classifier()), (None, None))

    def test_live_diagnostic_keeps_raw_coordinates_before_alignment(self):
        points = self.x[6].reshape(21, 3)
        result = SimpleNamespace(hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]])
        class Classifier:
            def predict(self, features):
                to_model_input(features)
                return prediction_from_scores([0, 0, 1, 0], .85)
        prediction, recorded = classify_result_with_features(result, Classifier())
        self.assertEqual(prediction.label, 'previous')
        np.testing.assert_allclose(recorded, self.x[6], atol=3e-7)
        self.assertGreater(np.linalg.norm(recorded - to_model_input(recorded)), .1)
        self.assertEqual(classify_result_with_features(SimpleNamespace(hand_world_landmarks=[]), Classifier()), (None, None))


class DataAndMetadataTests(DataCase):
    def test_source_dataset_and_session_splits_are_preserved(self):
        self.assertEqual(self.data.sha256, EXPECTED_DATA)
        self.assertEqual({k: len(v) for k, v in self.data.split_indices.items()}, {'train': 720, 'validation': 240, 'test': 240})
        require_manifest(self.data, self.config)
        self.assertNotIn('simple_test01', self.config['split_sessions']['train'])
        self.assertNotIn('simple_test01', self.config['split_sessions']['validation'])

    def test_input_version_architecture_and_dataset_mismatches_are_rejected(self):
        metadata = sample_metadata(self.data, self.config)
        validate_checkpoint_metadata(metadata, self.data.sha256)
        for key in ('model_input_version', 'architecture', 'dataset_sha256'):
            bad = copy.deepcopy(metadata); bad[key] = 'wrong'
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(bad, self.data.sha256)

    def test_development_capture_matches_the_original_console_counts(self):
        values, scores = load_development_capture(ROOT / 'diagnostic_inputs/vsign_development.csv')
        result = score_summary(scores, .85)
        self.assertEqual(result['raw_label_counts'], dict(neutral=0, next=61, previous=389, blackout=1))
        self.assertEqual(result['wrong_above_threshold_frames'], 38)
        self.assertEqual(result['below_threshold_frames'], 69)
        self.assertEqual(values.shape, (451, 63))
        to_model_input(values)

    def test_changed_development_recording_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'changed.csv'; path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_development_capture(path)


class PackagingTests(DataCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'palm_v3'; self.root.mkdir()
        shutil.copytree(ROOT / 'data', self.root / 'data')
        shutil.copy2(ROOT / 'config.json', self.root / 'config.json')
        (self.root / 'README.md').write_text('test readme')
        (self.root / 'app.py').write_text('test source')
        for name in ('models', 'reports', 'tests', 'diagnostic_inputs'):
            (self.root / name).mkdir()
        shutil.copy2(ROOT / 'diagnostic_inputs/vsign_development.csv', self.root / 'diagnostic_inputs/vsign_development.csv')
        (self.root / 'models/gesture_mlp.pt').write_bytes(b'fixture checkpoint')
        self.digest = file_sha256(self.root / 'models/gesture_mlp.pt')
        shared = {'checkpoint_sha256': self.digest, 'dataset_sha256': EXPECTED_DATA, 'model_input_version': MODEL_INPUT_VERSION}
        (self.root / 'reports/validation_metrics.json').write_text(json.dumps(dict(shared, split='validation', sample_count=240)))
        (self.root / 'reports/diagnostic_replay.json').write_text(json.dumps(dict(shared, source_sha256=DEVELOPMENT_SHA256, frames=451)))
        self.output = Path(self.tmp.name) / 'return.zip'

    def call_bundle(self):
        import run_training
        fake_model = SimpleNamespace(load_gesture_model=lambda *a, **k: (object(), sample_metadata(self.data, self.config)))
        with mock.patch.object(run_training, 'ROOT', self.root), mock.patch.dict(sys.modules, {'model': fake_model}):
            with contextlib.redirect_stdout(io.StringIO()):
                run_training.make_bundle(self.output)

    def test_return_archive_is_isolated_complete_and_refuses_overwrite(self):
        self.call_bundle()
        before = self.output.read_bytes()
        with zipfile.ZipFile(self.output) as archive:
            self.assertTrue(all(n.startswith('palm_v3/') for n in archive.namelist()))
            self.assertEqual(archive.read('palm_v3/data/samples.csv'), (ROOT / 'data/samples.csv').read_bytes())
            self.assertIn('palm_v3/models/gesture_mlp.pt', archive.namelist())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.call_bundle()
        self.assertEqual(self.output.read_bytes(), before)

    def test_stale_report_cannot_be_packaged_as_current_model(self):
        path = self.root / 'reports/diagnostic_replay.json'
        report = json.loads(path.read_text()); report['checkpoint_sha256'] = '0' * 64
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'does not describe'):
            self.call_bundle()
        self.assertFalse(self.output.exists())


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'native PyTorch checks run in the PC project environment')
class NativeTorchTests(DataCase):
    def test_architecture_and_finite_backward_pass(self):
        import torch
        from model import GestureMLP
        model = GestureMLP()
        inputs = torch.from_numpy(to_model_input(self.x))
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (12, 4))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6308)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]*3+[1]*3+[2]*3+[3]*3))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_live_classifier_matches_aligned_native_inference(self):
        import torch
        from app import Classifier
        from model import GestureMLP
        model = GestureMLP().eval(); classifier = Classifier(model, .85)
        with torch.inference_mode():
            expected = torch.softmax(model(torch.from_numpy(to_model_input(self.x))), dim=1).numpy()
        for values, row in zip(self.x, expected):
            prediction = classifier.predict(values)
            np.testing.assert_allclose(prediction.scores, row, atol=1e-7)
        with self.assertRaises(DegeneratePalmError):
            classifier.predict(np.zeros(63))

    def test_checkpoint_round_trip_in_fresh_process_and_old_version_rejected(self):
        import torch
        from model import GestureMLP, load_gesture_model
        model = GestureMLP()
        payload = {'checkpoint_version': 2, 'metadata': sample_metadata(self.data, self.config), 'state_dict': model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'palm.pt'; torch.save(payload, path)
            restored, _ = load_gesture_model(path, expected_dataset_sha256=self.data.sha256)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            result = subprocess.run([sys.executable, '-c', 'from model import load_gesture_model; import sys; load_gesture_model(sys.argv[1], expected_dataset_sha256=sys.argv[2]); print("RELOADED")', str(path), self.data.sha256], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('RELOADED', result.stdout)
            payload['checkpoint_version'] = 1; torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'version 2'):
                load_gesture_model(path)
            payload['checkpoint_version'] = 2; payload['metadata']['model_input_version'] = 'raw'
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'model_input_version'):
                load_gesture_model(path)

    def test_fit_restores_validation_selected_weights(self):
        import torch
        from model import GestureMLP
        from train import fit
        torch.set_num_threads(1); torch.manual_seed(42)
        model = GestureMLP()
        features = torch.from_numpy(to_model_input(self.x))
        labels = torch.tensor([0]*3+[1]*3+[2]*3+[3]*3)
        settings = dict(DEFAULT_TRAINING, max_epochs=3, patience=2)
        _, history, stopping = fit(model, features, labels, features, labels, settings)
        with torch.inference_mode():
            loss = float(torch.nn.functional.cross_entropy(model(features), labels))
        self.assertAlmostEqual(loss, min(r['validation_loss'] for r in history), places=6)
        self.assertEqual(stopping.best_loss, min(r['validation_loss'] for r in history))


if __name__ == '__main__':
    unittest.main()
