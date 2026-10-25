"""Explicit portable file lists and verified, no-overwrite ZIP creation."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

from dataset import file_sha256
from hand_features import GESTURE_SET_VERSION

SOURCE_FILES = (
    'config.json', 'hand_features.py', 'camera_drawing.py', 'dataset.py',
    'model.py', 'train.py', 'evaluate.py', 'training_utils.py', 'reporting.py',
    'app.py', 'live_utils.py', 'prepare_data.py', 'prepare_simple_v2.py', 'data_migration.py',
    'package_utils.py', 'make_demo_bundle.py', 'requirements-training.txt',
    'README.md', 'GESTUREPILOT_MOTHER.md',
    'tests/test_pipeline.py', 'tests/test_training.py', 'tests/test_live.py',
)
DATA_FILES = ('data/samples.csv', 'data/samples.metadata.json',
              'data/lineage.json', 'data/split_manifest.json')
REPORT_FILES = (
    'reports/training_summary.json', 'reports/training_history.csv',
    'reports/training_curves.png', 'reports/validation_metrics.json',
    'reports/validation_classification_report.txt', 'reports/validation_confusion_matrix.png',
    'reports/validation_predictions.csv', 'reports/metrics.json',
    'reports/classification_report.txt', 'reports/confusion_matrix.png',
    'reports/test_predictions.csv',
)


def build_zip(root, output, names, details, *, info_name):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'{output} already exists. Preserve it or choose a new --output name.')
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError('Archive file list contains duplicate paths')
    for name in names:
        candidate = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not candidate.is_file():
            raise FileNotFoundError(f'Missing or invalid portable source: {candidate}')
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError('Archive files must belong to the version folder')
        if candidate.resolve() == output:
            raise ValueError('Archive output cannot replace a source file')
    files = {name: {'sha256': file_sha256(root / name), 'bytes': (root / name).stat().st_size}
             for name in names}
    info = dict(details, created_at_utc=datetime.now(timezone.utc).isoformat(),
                gesture_set_version=GESTURE_SET_VERSION, files=files,
                excluded=['.venv', 'camera model', 'raw capture source CSVs',
                          'old models/reports', 'debug data', 'images/video', 'caches'])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.zip', delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(root / name, arcname='simple_v2/' + name)
            archive.writestr('simple_v2/' + info_name, json.dumps(info, indent=2) + '\n')
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('ZIP integrity check failed')
            for name, expected in files.items():
                if hashlib.sha256(archive.read('simple_v2/' + name)).hexdigest() != expected['sha256']:
                    raise RuntimeError(f'File changed during packaging: {name}')
        # Publish without overwriting an existing output, including on Windows.
        try:
            with output.open('xb') as destination, temporary.open('rb') as source:
                import shutil
                try:
                    shutil.copyfileobj(source, destination)
                except BaseException:
                    destination.close()
                    output.unlink(missing_ok=True)
                    raise
        except FileExistsError as exc:
            raise ValueError('Output appeared during packing; choose another filename') from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return info
