"""Run native checks, train once, validate, replay known V inputs and package."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from dataset import ROOT, file_sha256, load_dataset, read_config
from evaluate_diagnostic import DEVELOPMENT_SHA256
from palm_features import MODEL_INPUT_VERSION
from training_utils import require_manifest


def checked_step(arguments):
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'Stopped after failed step: {arguments[0]} (exit {result.returncode})')


def make_bundle(output):
    from model import load_gesture_model
    config = read_config(ROOT / 'config.json')
    data = load_dataset(ROOT / config['dataset_path'], config)
    require_manifest(data, config)
    checkpoint = ROOT / 'models/gesture_mlp.pt'
    _, metadata = load_gesture_model(checkpoint, expected_dataset_sha256=data.sha256)
    checkpoint_hash = file_sha256(checkpoint)
    if file_sha256(ROOT / 'diagnostic_inputs/vsign_development.csv') != DEVELOPMENT_SHA256:
        raise ValueError('The packaged development recording differs from the replayed input')
    for filename in ('validation_metrics.json', 'diagnostic_replay.json'):
        report = json.loads((ROOT / 'reports' / filename).read_text())
        if (report.get('checkpoint_sha256') != checkpoint_hash or report.get('dataset_sha256') != data.sha256
                or report.get('model_input_version') != MODEL_INPUT_VERSION):
            raise ValueError(f'{filename} does not describe the newly trained palm model')
        if filename == 'validation_metrics.json' and (report.get('split') != 'validation' or report.get('sample_count') != 240):
            raise ValueError('Expected evaluation of the original 240 validation rows')
        if filename == 'diagnostic_replay.json' and (report.get('source_sha256') != DEVELOPMENT_SHA256 or report.get('frames') != 451):
            raise ValueError('Expected the reviewed 451-frame development capture')
    readme = ROOT.parent / 'README.md'
    if not readme.is_file():
        raise FileNotFoundError(f'Missing project README: {readme}')
    sources = [p for p in ROOT.glob('*.py')] + [ROOT / 'config.json']
    for directory in ('data', 'models', 'reports', 'tests', 'diagnostic_inputs'):
        sources.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    entries = [(path, 'palm_v3/' + path.relative_to(ROOT).as_posix()) for path in sources]
    entries.append((readme, 'README.md'))
    # The packaged Git portability tests also need the project's newline rules.
    attributes = ROOT.parent / '.gitattributes'
    if attributes.is_file():
        entries.append((attributes, '.gitattributes'))
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError('Return ZIP already exists; choose a new --output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for path, archive_name in sorted(entries, key=lambda item: item[1]):
                archive.write(path, archive_name)
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:
                raise ValueError('Return ZIP verification failed')
    except Exception:
        if output.exists():
            output.unlink()
        raise
    print(f'PALM V3 LAPTOP PACKAGE: {output}')
    print(f'ZIP SHA-256: {file_sha256(output)}')
    print('Training and offline checks completed. Live recognition remains to be checked on the laptop.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-label', choices=('desktop', 'laptop'), default='desktop')
    parser.add_argument('--output', required=True, help='New return ZIP filename')
    args = parser.parse_args()
    if (ROOT / 'models/gesture_mlp.pt').exists():
        raise ValueError('Palm checkpoint already exists. Keep it and report the current results before retraining.')
    if Path(args.output).expanduser().exists():
        raise ValueError('Return ZIP already exists; choose a new --output filename')
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Use the existing project .venv on the PC, which already has CPU PyTorch.') from exc
    print('PALM V3: using existing data, CPU PyTorch and a separate checkpoint folder.', flush=True)
    checked_step(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    checked_step(['prepare_data.py', '--config', 'config.json'])
    checked_step(['train.py', '--config', 'config.json', '--device-label', args.device_label])
    checked_step(['evaluate.py', '--split', 'validation'])
    checked_step(['evaluate_diagnostic.py'])
    make_bundle(args.output)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Training workflow stopped by user.', file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'PALM V3 WORKFLOW FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
