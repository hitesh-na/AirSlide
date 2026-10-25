"""Return the evaluated four-class model to the existing camera laptop."""

import argparse
import json
import math
from pathlib import Path
import sys

from dataset import EXPECTED_SPLITS, file_sha256, load_dataset, read_config
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from package_utils import SOURCE_FILES, REPORT_FILES, build_zip
from training_utils import require_manifest

ROOT = Path(__file__).resolve().parent


def verify_reports(root, data):
    root = Path(root)
    checkpoint_hash = file_sha256(root / 'models/gesture_mlp.pt')
    training = json.loads((root / 'reports/training_summary.json').read_text(encoding='utf-8-sig'))
    shared = {'dataset_sha256': data.sha256, 'feature_version': FEATURE_VERSION,
              'class_labels': list(CLASS_LABELS), 'gesture_set_version': GESTURE_SET_VERSION,
              'gesture_poses': GESTURE_POSES, 'physical_hand': 'right'}
    if (any(training.get(k) != v for k, v in shared.items())
            or training.get('split_sessions') != EXPECTED_SPLITS
            or training.get('device_label') != 'desktop'
            or type(training.get('selected_epoch')) is not int or training['selected_epoch'] < 1):
        raise ValueError('Training summary does not describe the four-class desktop snapshot')
    summaries = {}
    for split, name in (('validation', 'validation_metrics.json'), ('test', 'metrics.json')):
        report = json.loads((root / 'reports' / name).read_text(encoding='utf-8-sig'))
        expected = dict(shared, checkpoint_sha256=checkpoint_hash, split=split,
                        session_ids=EXPECTED_SPLITS[split], sample_count=len(data.split_indices[split]),
                        session_count=len(EXPECTED_SPLITS[split]), selected_epoch=training['selected_epoch'])
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError(f'{split} report does not match this checkpoint/dataset/split')
        for key in ('accuracy', 'macro_f1'):
            value = report.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'Invalid {split} metric: {key}')
        summaries[split] = {k: report[k] for k in ('accuracy', 'macro_f1', 'sample_count', 'session_ids')}
    return checkpoint_hash, summaries


def build_demo_bundle(root, output):
    root = Path(root)
    config = read_config(root / 'config.json')
    data = load_dataset(root / 'data/samples.csv', config)
    require_manifest(data, config)
    checkpoint_hash, summaries = verify_reports(root, data)
    names = SOURCE_FILES + ('models/gesture_mlp.pt', 'data/samples.metadata.json',
                            'data/lineage.json', 'data/split_manifest.json') + REPORT_FILES
    return build_zip(root, output, names,
                     {'destination': 'existing laptop GesturePilot project / simple_v2',
                      'dataset_sha256': data.sha256, 'checkpoint_sha256': checkpoint_hash,
                      'evaluation': summaries,
                      'required_existing_csv': 'simple_v2/data/samples.csv from laptop preparation'},
                     info_name='DEMO_TRANSFER_INFO.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT.parent.parent / 'GesturePilot_Simple_Laptop.zip'))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    info = build_demo_bundle(ROOT, output)
    print(f"Checkpoint SHA-256: {info['checkpoint_sha256']}")
    print(f'Laptop ZIP: {output}')
    print('SIMPLE LAPTOP PACKAGE READY')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f'SIMPLE LAPTOP PACKAGE FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
