"""Prediction display and timed measurements; no camera or key-sending imports."""

from dataclasses import dataclass, field
import math

import numpy as np

from hand_features import CLASS_LABELS, features_from_result


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    above_threshold: bool


def prediction_from_scores(scores, threshold):
    values = np.asarray(scores, dtype=np.float64)
    if (values.shape != (len(CLASS_LABELS),) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.isclose(values.sum(), 1.0, atol=1e-5)):
        raise ValueError("Expected five finite softmax scores summing to one")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Score threshold must be between zero and one")
    index = int(values.argmax())
    score = float(values[index])
    return Prediction(CLASS_LABELS[index], score, score >= threshold)


def classify_result(result, classifier):
    # A fresh return value every frame prevents a stale label on no-hand frames.
    features = features_from_result(result)
    return None if features is None else classifier.predict(features)


def video_timestamp(previous, seconds_since_start):
    return max(previous + 1, int(seconds_since_start * 1000))


@dataclass
class PerformanceWindow:
    ready_at: float
    warmup_seconds: float
    duration_seconds: float
    latencies_ms: list = field(default_factory=list)
    first_frame_at: float | None = None
    last_processed_at: float | None = None
    usable_hand_frames: int = 0

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in (self.ready_at, self.warmup_seconds, self.duration_seconds))
                or self.warmup_seconds < 0 or self.duration_seconds <= 0):
            raise ValueError("Use finite warm-up >= 0 and measurement duration > 0")

    def record(self, frame_started, processed_at, usable_hand):
        if not all(math.isfinite(v) for v in (frame_started, processed_at)) or processed_at < frame_started:
            raise ValueError("Invalid frame processing interval")
        if frame_started < self.ready_at + self.warmup_seconds:
            return
        if self.last_processed_at is not None and frame_started < self.last_processed_at:
            raise ValueError("Frames must be recorded in sequential time order")
        if self.first_frame_at is None:
            self.first_frame_at = frame_started
        self.last_processed_at = processed_at
        self.latencies_ms.append((processed_at - frame_started) * 1000)
        self.usable_hand_frames += int(bool(usable_hand))

    @property
    def elapsed(self):
        if self.first_frame_at is None:
            return 0.0
        return max(0.0, self.last_processed_at - self.first_frame_at)

    @property
    def complete(self):
        return bool(self.latencies_ms) and self.elapsed >= self.duration_seconds

    def summary(self):
        count = len(self.latencies_ms)
        return {
            "warmup_seconds_excluded": self.warmup_seconds,
            "requested_measurement_seconds": self.duration_seconds,
            "measured_seconds": self.elapsed, "measured_frames": count,
            "usable_hand_frames": self.usable_hand_frames,
            "no_usable_hand_frames": count - self.usable_hand_frames,
            "average_processing_fps": count / self.elapsed if self.elapsed > 0 else None,
            "median_frame_processing_ms": float(np.median(self.latencies_ms)) if count else None,
            "p95_frame_processing_ms": float(np.percentile(self.latencies_ms, 95)) if count else None,
            "completed_requested_interval": self.complete,
            "latency_definition": "Immediately before camera.read() through RGB conversion, synchronous landmarks, classifier, skeleton and text overlay; excludes imshow/waitKey and key delivery.",
            "fps_definition": "Measured frames / wall time from first post-warmup frame-read start to last completed overlay; includes display/event-loop gaps between frames.",
            "scope": "Local processing measurement, not camera-to-slide response latency; usable-hand and no-hand frame counts reported separately.",
        }
