"""Replay the supplied V-sign development recording using a fixed checkpoint."""
import argparse
import csv
import sys

import numpy as np

from dataset import file_sha256, load_dataset, read_config
from hand_features import CLASS_LABELS
from palm_features import MODEL_INPUT_VERSION, to_model_input
from training_utils import project_path, require_manifest, write_json

DEVELOPMENT_SHA256 = '93be90a5618ec652cdc25c8d38d181568a1ead2b447249a5580e25c6b8d94947'


def load_development_capture(path):
    if file_sha256(path) != DEVELOPMENT_SHA256:
        raise ValueError('Development recording differs from the reviewed original CSV')
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 451 or any(r['expected_label'] != 'previous' or r['has_usable_hand'] != 'True' for r in rows):
        raise ValueError('Expected the reviewed 451-frame held-V development recording')
    values = np.asarray([[float(r[f'f{i:02d}']) for i in range(63)] for r in rows], dtype=np.float32)
    original = np.asarray([[float(r['score_' + label]) for label in CLASS_LABELS] for r in rows])
    if not np.isfinite(values).all() or not np.isfinite(original).all():
        raise ValueError('Development recording contains nonfinite values')
    if not np.allclose(original.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Development scores do not sum to one')
    return values, original


def score_summary(scores, threshold):
    labels = scores.argmax(axis=1)
    above = scores.max(axis=1) >= threshold
    return {'frames': len(scores),
            'raw_label_counts': {name: int((labels == i).sum()) for i, name in enumerate(CLASS_LABELS)},
            'target_agreement_frames': int((labels == 2).sum()),
            'below_threshold_frames': int((~above).sum()),
            'wrong_above_threshold_frames': int(((labels != 2) & above).sum()),
            'minimum_previous_score': float(scores[:, 2].min()),
            'median_previous_score': float(np.median(scores[:, 2]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--checkpoint', default='models/gesture_mlp.pt')
    parser.add_argument('--input', default='diagnostic_inputs/vsign_development.csv')
    args = parser.parse_args()
    import torch
    from model import load_gesture_model
    config = read_config(project_path(args.config))
    data = load_dataset(project_path(config['dataset_path']), config)
    require_manifest(data, config)
    source = project_path(args.input)
    values, original = load_development_capture(source)
    checkpoint = project_path(args.checkpoint)
    model, metadata = load_gesture_model(checkpoint, expected_dataset_sha256=data.sha256)
    torch.set_num_threads(metadata['training_settings']['num_threads'])
    with torch.inference_mode():
        scores = torch.softmax(model(torch.from_numpy(to_model_input(values))), dim=1).numpy()
    threshold = config['controller']['score_threshold']
    summary = score_summary(scores, threshold)
    summary.update(model_input_version=MODEL_INPUT_VERSION, dataset_sha256=data.sha256,
                   checkpoint_sha256=file_sha256(checkpoint), source_sha256=file_sha256(source),
                   expected_label='previous', score_threshold=threshold,
                   original_recorded_model=score_summary(original, threshold),
                   scope='Replay of known development examples used to investigate this issue; not a fresh live test or independent accuracy estimate.')
    output = project_path('reports/diagnostic_replay.json')
    write_json(output, summary)
    print('DEVELOPMENT REPLAY — known V-sign recording, not a new live test')
    print(f"Previous: {summary['target_agreement_frames']}/{summary['frames']}")
    print(f"Raw labels: {summary['raw_label_counts']}")
    print(f"Below threshold: {summary['below_threshold_frames']} | Wrong at/above threshold: {summary['wrong_above_threshold_frames']}")
    print(f"Report: {output}")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        print(f'DEVELOPMENT REPLAY FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
