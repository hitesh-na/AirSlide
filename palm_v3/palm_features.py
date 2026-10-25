"""Rotate wrist-centered, palm-scaled landmarks into the hand's own axes.

Stored CSVs retain world_wrist_palm_v1. This transform is applied once to
training/evaluation/live inputs before the neural network receives them.
"""
import numpy as np

MODEL_INPUT_VERSION = 'palm_frame_v1'


class DegeneratePalmError(ValueError):
    """The estimated palm cannot define a stable coordinate frame."""


def to_model_input(features):
    values = np.asarray(features, dtype=np.float64)
    if (values.ndim not in (1, 2) or values.shape[-1] != 63
            or values.size == 0 or not np.isfinite(values).all()):
        raise ValueError('Expected one or more finite 63-feature hand vectors')
    points = values.reshape(-1, 21, 3)
    if not np.allclose(points[:, 0], 0, rtol=0, atol=1e-6):
        raise ValueError('Palm alignment requires wrist-centered input')
    along = points[:, 9].copy()
    lengths = np.linalg.norm(along, axis=1, keepdims=True)
    if np.any(lengths < 1e-6):
        raise DegeneratePalmError('Wrist and middle-finger base are too close')
    if not np.allclose(lengths, 1, rtol=1e-4, atol=1e-5):
        raise ValueError('Palm alignment requires already palm-scaled input')
    along /= lengths
    across = points[:, 5] - points[:, 17]
    across -= np.sum(across * along, axis=1, keepdims=True) * along
    widths = np.linalg.norm(across, axis=1, keepdims=True)
    if np.any(widths < 1e-4):
        raise DegeneratePalmError('Palm bases cannot define a stable side axis')
    across /= widths
    normal = np.cross(across, along)
    basis = np.stack((across, along, normal), axis=2)
    aligned = np.einsum('npi,nij->npj', points, basis)
    result = aligned.astype(np.float32).reshape(values.shape)
    if not np.isfinite(result).all():
        raise ValueError('Aligned hand features are not finite')
    return result
