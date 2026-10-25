"""Shared world-landmark preprocessing for collection and live prediction."""

from __future__ import annotations

import numpy as np

FEATURE_VERSION = "world_wrist_palm_v1"
CLASS_LABELS = ("neutral", "next", "previous", "blackout", "start")
FEATURE_COUNT = 63


def normalize_world_landmarks(points) -> np.ndarray:
    """Convert (21, 3) world coordinates to wrist-centered float32 features.

    Raises ValueError for missing, malformed, non-finite, or degenerate input.
    Training reads these already-normalized vectors directly; do not repeat
    normalization when loading a saved CSV.
    """
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected 21 world landmarks with x/y/z coordinates") from exc
    if values.shape != (21, 3) or not np.isfinite(values).all():
        raise ValueError("World landmarks must be finite and have shape (21, 3)")
    centered = values - values[0]
    scale = float(np.linalg.norm(centered[9]))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Wrist-to-middle-finger-base scale is too small")
    with np.errstate(over="ignore", invalid="ignore"):
        features = (centered / scale).reshape(FEATURE_COUNT).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Normalized features are not finite")
    return features


def features_from_result(result) -> np.ndarray | None:
    """Return the first hand's 63 features, or None when no usable hand exists."""
    hands = getattr(result, "hand_world_landmarks", None)
    if not hands:
        return None
    try:
        points = [(point.x, point.y, point.z) for point in hands[0]]
        return normalize_world_landmarks(points)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
