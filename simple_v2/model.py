"""The fixed 63 -> 64 -> 32 -> 4 gesture MLP and validated CPU checkpoint loader."""

from pathlib import Path
import pickle

import torch
from torch import nn

from training_utils import validate_checkpoint_metadata


class GestureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(63, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, features):
        return self.network(features)


def load_gesture_model(path, *, expected_dataset_sha256=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run train.py first; no training was started automatically.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Cannot read this project's weights-only checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != 1:
        raise ValueError("Unrecognized checkpoint format; expected version 1")
    metadata = payload.get("metadata")
    validate_checkpoint_metadata(metadata, expected_dataset_sha256)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict is missing")
    model = GestureMLP().to("cpu")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match the fixed MLP: {exc}") from exc
    model.eval()
    return model, metadata
