"""Training settings, checkpoint metadata and manifest checks (no PyTorch import)."""

from dataclasses import dataclass
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile

from dataset import EXPECTED_SPLITS, ROOT, make_manifest
from hand_features import CLASS_LABELS, FEATURE_VERSION

ARCHITECTURE = {"name": "GestureMLP", "input_features": 63, "hidden_sizes": [64, 32], "output_classes": 5, "activation": "ReLU"}
DEFAULT_TRAINING = {"seed": 42, "learning_rate": 0.001, "batch_size": 32, "max_epochs": 100, "patience": 10, "num_threads": 1, "device": "cpu"}


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def training_settings(config):
    supplied = config.get("training", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_TRAINING):
        raise ValueError("Unknown training configuration fields")
    settings = dict(DEFAULT_TRAINING, **supplied)
    for name in ("seed", "batch_size", "max_epochs", "patience", "num_threads"):
        value = settings[name]
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"training.{name} must be a valid integer")
    if settings["seed"] >= 2**32 or settings["max_epochs"] > 100:
        raise ValueError("Use seed below 2**32 and at most 100 epochs")
    lr = settings["learning_rate"]
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if settings["device"] != "cpu":
        raise ValueError("This project uses CPU training")
    return settings


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, loss, epoch):
        if not math.isfinite(loss):
            raise ValueError("Validation loss is not finite; training stopped")
        improved = loss < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.bad_epochs = float(loss), int(epoch), 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


def require_manifest(data, config):
    path = data.path.with_name("split_manifest.json")
    if not path.is_file():
        raise FileNotFoundError("Missing split_manifest.json. Run prepare_data.py --config config.json first.")
    existing = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = make_manifest(data, config)
    if not isinstance(existing, dict):
        raise ValueError("Saved split manifest is invalid")
    stable = lambda value: {key: item for key, item in value.items() if key != "created_at_utc"}
    if stable(existing) != stable(expected):
        raise ValueError("CSV/configuration does not match its saved split manifest. Preserve the files and report this mismatch.")
    return existing


def validate_checkpoint_metadata(metadata, expected_dataset_sha256=None):
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata is missing")
    for key, expected in (("feature_version", FEATURE_VERSION), ("class_labels", list(CLASS_LABELS)), ("architecture", ARCHITECTURE), ("split_sessions", EXPECTED_SPLITS)):
        if metadata.get(key) != expected:
            raise ValueError(f"Incompatible checkpoint {key}")
    digest = metadata.get("dataset_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Checkpoint has no valid dataset fingerprint")
    if expected_dataset_sha256 is not None and digest != expected_dataset_sha256:
        raise ValueError("Checkpoint and current dataset fingerprints differ")


def package_versions():
    result = {}
    for name in ("torch", "numpy", "scikit-learn", "matplotlib", "mediapipe", "opencv-contrib-python", "pyautogui"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed on this device"
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
