"""Single-pose numerical diagnostics; never write data/ or alter a model."""

from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np

from hand_features import CLASS_LABELS, FEATURE_COUNT

CSV_COLUMNS = ('elapsed_seconds', 'expected_label', 'has_usable_hand', 'raw_label',
               'raw_score', 'above_threshold', 'qualified_label',
               *(f'score_{label}' for label in CLASS_LABELS),
               *(f'f{i:02d}' for i in range(FEATURE_COUNT)))


class PoseDiagnostic:
    """Count every measured frame, including uncertainty and no-hand frames.

    The target comes from the user's instruction, not an independent annotator.
    Agreement is therefore conditional on holding the requested pose.
    """

    def __init__(self, expected_label, threshold):
        if expected_label not in CLASS_LABELS:
            raise ValueError('Choose one of the four trained labels for the diagnostic')
        if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Diagnostic score threshold must be between zero and one')
        self.expected_label = expected_label
        self.threshold = float(threshold)
        self.rows = []

    def record(self, elapsed_seconds, prediction, features):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('Diagnostic elapsed time must be finite and non-negative')
        if self.rows and elapsed_seconds < self.rows[-1]['elapsed_seconds']:
            raise ValueError('Diagnostic frames must be in time order')
        row = dict(elapsed_seconds=float(elapsed_seconds), expected_label=self.expected_label,
                   has_usable_hand=prediction is not None)
        if prediction is None:
            row.update(raw_label='no_usable_hand', raw_score=None, above_threshold=False,
                       qualified_label='no_usable_hand')
            row.update({f'score_{name}': None for name in CLASS_LABELS})
            row.update({f'f{i:02d}': None for i in range(FEATURE_COUNT)})
        else:
            scores = np.asarray(prediction.scores, dtype=np.float64)
            values = np.asarray(features, dtype=np.float32)
            if (scores.shape != (len(CLASS_LABELS),) or not np.isfinite(scores).all()
                    or (scores < 0).any() or (scores > 1).any()
                    or not np.isclose(scores.sum(), 1, atol=1e-5)):
                raise ValueError('Diagnostic requires all four finite softmax scores')
            if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
                raise ValueError('Diagnostic requires the finite 63-feature raw normalized input before palm alignment')
            index = int(scores.argmax())
            label, score = CLASS_LABELS[index], float(scores[index])
            if prediction.label != label or not math.isclose(prediction.score, score, abs_tol=1e-7):
                raise ValueError('Prediction label/score disagrees with its score vector')
            above = score >= self.threshold
            row.update(raw_label=label, raw_score=score, above_threshold=above,
                       qualified_label=label if above else 'uncertain')
            row.update({f'score_{name}': float(scores[i]) for i, name in enumerate(CLASS_LABELS)})
            # Copy scalar values so later model buffers cannot alter this record.
            row.update({f'f{i:02d}': float(value) for i, value in enumerate(values)})
        self.rows.append(row)

    def summary(self):
        scored = [row for row in self.rows if row['has_usable_hand']]
        raw = Counter(row['raw_label'] for row in scored)
        matched = sum(row['raw_label'] == self.expected_label for row in scored)
        wrong_high = sum(row['raw_label'] != self.expected_label and row['above_threshold'] for row in scored)
        per_label = {}
        for name in CLASS_LABELS:
            scores = [row['raw_score'] for row in scored if row['raw_label'] == name]
            per_label[name] = {'frames': len(scores),
                               'median_score': float(np.median(scores)) if scores else None,
                               'max_score': float(max(scores)) if scores else None}
        return {
            'expected_label': self.expected_label, 'score_threshold': self.threshold,
            'measured_frames': len(self.rows), 'usable_hand_frames': len(scored),
            'no_usable_hand_frames': len(self.rows) - len(scored),
            'raw_label_counts': {name: raw[name] for name in CLASS_LABELS},
            'raw_matches': matched, 'raw_disagreements': len(scored) - matched,
            'raw_target_agreement_percent': matched * 100 / len(scored) if scored else None,
            'low_score_frames': sum(not row['above_threshold'] for row in scored),
            'wrong_above_threshold_frames': wrong_high,
            'correct_above_threshold_frames': sum(row['raw_label'] == self.expected_label
                                                  and row['above_threshold'] for row in scored),
            'by_raw_label': per_label,
            'scope': 'Frame agreement with a user-declared held pose; not independent labels or general live accuracy. Correlated video frames are not independent test examples.',
            'model_changed': False, 'training_data_appended': False, 'images_saved': False,
        }

    def write_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)


def print_diagnostic(summary, csv_path):
    print(f"POSE DIAGNOSTIC | manual target: {summary['expected_label']}")
    print(f"Usable hand frames: {summary['usable_hand_frames']} | no usable hand: {summary['no_usable_hand_frames']}")
    for name in CLASS_LABELS:
        values = summary['by_raw_label'][name]
        median = '-' if values['median_score'] is None else f"{values['median_score']:.3f}"
        maximum = '-' if values['max_score'] is None else f"{values['max_score']:.3f}"
        print(f"Raw {name}: {values['frames']} frames | median score {median} | max {maximum}")
    rate = summary['raw_target_agreement_percent']
    rate_text = 'NOT_MEASURED' if rate is None else f'{rate:.2f}%'
    print(f"Target agreement: {summary['raw_matches']}/{summary['usable_hand_frames']} usable-hand frames ({rate_text})")
    print(f"Below score threshold: {summary['low_score_frames']} frames")
    print(f"Wrong label at/above threshold: {summary['wrong_above_threshold_frames']} frames")
    print(f'Numerical diagnostic CSV: {csv_path}')
    print('Diagnostic only: model and training data unchanged; scores are not calibrated probabilities.')
