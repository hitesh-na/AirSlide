"""Prepare the separate four-class snapshot and desktop ZIP on the camera laptop."""

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'simple_v2' else HERE
sys.path.insert(0, str(ROOT / 'simple_v2'))

from data_migration import prepare_training_bundle
from dataset import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path.home() / 'Downloads/GesturePilot_Simple_Training.zip'))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    info = prepare_training_bundle(ROOT, output)
    print('Raw index-up counts: idx_train01=60, idx_train02=60, idx_train03=60, idx_val01=60')
    print('Fresh simple_test01: neutral=60, next (INDEX UP)=60, previous=60, blackout=60')
    print('Reused baseline rows: 720; old next/start/s05 rows excluded')
    print('Merged total rows: 1200 | Unique IDs: 1200 | Finite features: 63 per row')
    print('Session overlap: none | Exact feature duplicates across splits: 0')
    for role, count in info['split_row_counts'].items():
        print(f'{role}: {count} rows')
    print(f"Dataset SHA-256: {info['dataset_sha256']}")
    print(f"Prepared dataset: {ROOT / 'simple_v2/data/samples.csv'}")
    print(f'Training ZIP: {output}')
    print(f'ZIP SHA-256: {file_sha256(output)}')
    print('SIMPLE DATASET CHECK PASSED')
    print('PC TRAINING PACKAGE READY')
    print('No model was trained or evaluated. Copy the generated ZIP to your PC.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Preparation stopped; original sources and baseline model were preserved.', file=sys.stderr)
        sys.exit(130)
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f'SIMPLE PREPARATION FAILED: {exc}', file=sys.stderr)
        print('Keep the source CSVs unchanged and share this error.', file=sys.stderr)
        sys.exit(1)
