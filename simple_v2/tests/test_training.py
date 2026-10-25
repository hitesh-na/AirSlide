"""Training/report checks; native model tests run when PyTorch is installed."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import EXPECTED_SPLITS
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from reporting import confusion_plot, summarize_predictions, training_plot
from training_utils import ARCHITECTURE, EarlyStopping, training_settings, validate_checkpoint_metadata, write_json


def example_metadata():
    return {"feature_version": FEATURE_VERSION, "class_labels": list(CLASS_LABELS),
            "gesture_set_version": GESTURE_SET_VERSION, "gesture_poses": dict(GESTURE_POSES),
            "architecture": copy.deepcopy(ARCHITECTURE), "dataset_sha256": "a" * 64,
            "split_sessions": copy.deepcopy(EXPECTED_SPLITS), "training_settings": training_settings({}),
            "selected_epoch": 1, "device_label": "synthetic test"}


class TrainingUtilityTests(unittest.TestCase):
    def test_defaults_and_invalid_settings(self):
        settings = training_settings({})
        self.assertEqual((settings["seed"], settings["batch_size"], settings["max_epochs"], settings["patience"]), (42, 32, 100, 10))
        for change in ({"device": "cuda"}, {"learning_rate": -1}, {"max_epochs": 101}, {"batch_size": 0}, {"seed": -1}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    training_settings({"training": change})

    def test_early_stopping_keeps_lowest_loss_and_resets_patience(self):
        stopping = EarlyStopping(2)
        self.assertTrue(stopping.update(1.0, 1))
        self.assertTrue(stopping.update(0.8, 2))
        self.assertFalse(stopping.update(0.81, 3))
        self.assertFalse(stopping.should_stop)
        self.assertTrue(stopping.update(0.7, 4))
        self.assertEqual(stopping.bad_epochs, 0)
        self.assertFalse(stopping.update(0.7, 5))
        self.assertFalse(stopping.update(0.71, 6))
        self.assertTrue(stopping.should_stop)
        self.assertEqual((stopping.best_epoch, stopping.best_loss), (4, 0.7))

    def test_checkpoint_rejects_feature_class_architecture_and_source_mismatch(self):
        valid = example_metadata()
        validate_checkpoint_metadata(valid, "a" * 64)
        for key, value in (("gesture_set_version", "baseline"), ("gesture_poses", {}), ("feature_version", "other"), ("class_labels", list(reversed(CLASS_LABELS))),
                           ("architecture", {}), ("split_sessions", {})):
            bad = copy.deepcopy(valid)
            bad[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_checkpoint_metadata(bad)
        with self.assertRaisesRegex(ValueError, "fingerprints differ"):
            validate_checkpoint_metadata(valid, "b" * 64)

    def test_metric_values_and_full_class_support(self):
        summary, text = summarize_predictions([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 0, 3, 3])
        self.assertAlmostEqual(summary["accuracy"], 0.75)
        self.assertAlmostEqual(summary["macro_f1"], (0.5 + 0.8 + 2/3 + 1) / 4)
        self.assertEqual(np.asarray(summary["confusion_matrix"]).sum(), 8)
        self.assertTrue(all(summary["classification_report"][label]["support"] == 2 for label in CLASS_LABELS))
        self.assertIn("blackout", text)

    def test_exported_curves_matrix_and_metrics_are_valid_files(self):
        history = [dict(epoch=i, train_loss=1/i, validation_loss=1.2/i,
                        train_accuracy=i/4, validation_accuracy=i/5) for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            training_plot(history, 3, folder / "curves.png")
            confusion_plot(np.eye(4, dtype=int), "synthetic validation", folder / "matrix.png")
            for filename in ("curves.png", "matrix.png"):
                self.assertTrue((folder / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            summary, _ = summarize_predictions(range(4), range(4))
            write_json(folder / "metrics.json", summary)
            self.assertEqual(json.loads((folder / "metrics.json").read_text())["accuracy"], 1.0)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch unavailable here; run these native checks in the project's .venv")
class NativeTorchTests(unittest.TestCase):
    def test_architecture_logits_and_backward_pass(self):
        import torch
        from model import GestureMLP
        model = GestureMLP()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 6308)
        logits = model(torch.zeros(4, 63))
        self.assertEqual(tuple(logits.shape), (4, 4))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpoint_round_trip_in_fresh_process(self):
        import torch
        from model import GestureMLP
        torch.manual_seed(42)
        model = GestureMLP().eval()
        with torch.inference_mode():
            original = model(torch.zeros(2, 63)).numpy()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic.pt"
            torch.save({"checkpoint_version": 1, "state_dict": model.state_dict(), "metadata": example_metadata()}, path)
            code = 'import json,sys,torch; from model import load_gesture_model; m,_=load_gesture_model(sys.argv[1]); print(json.dumps(m(torch.zeros(2,63)).detach().tolist()))'
            result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            np.testing.assert_allclose(np.array(json.loads(result.stdout)), original, rtol=0, atol=0)

    def test_fit_restores_lowest_validation_loss_checkpoint(self):
        import torch
        from model import GestureMLP
        from train import fit
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = GestureMLP()
        settings = training_settings({"training": {"max_epochs": 3, "patience": 2}})
        x, y = torch.randn(30, 63), torch.arange(30) % 4
        state, history, stopping = fit(model, x[:20], y[:20], x[20:], y[20:], settings)
        expected = min(history, key=lambda row: row["validation_loss"])
        self.assertEqual(stopping.best_epoch, expected["epoch"])
        self.assertAlmostEqual(stopping.best_loss, expected["validation_loss"])
        self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in state.items()))


if __name__ == "__main__":
    unittest.main()
