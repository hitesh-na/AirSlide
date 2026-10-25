"""Validate raw captures and prepare a separate four-class training snapshot.

No raw source file, baseline model, session ID or saved feature is changed.
The final test is checked structurally but never used for fitting/selection.
"""

from collections import Counter
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from dataset import (CSV_COLUMNS, FEATURE_COLUMNS, EXPECTED_SPLITS, file_sha256,
                     load_dataset, make_manifest, read_config, row_fingerprint, save_manifest)
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from package_utils import SOURCE_FILES, DATA_FILES, build_zip

VERIFIED_BASELINE_SHA256 = 'a10f2967aedcdcb37801aded40a3504151a8b447e1f937f72ff72e96f8f00d6d'
RAW_LABELS = ['neutral', 'next', 'previous', 'blackout', 'start']
RAW_METADATA = {'feature_version': FEATURE_VERSION, 'class_labels': RAW_LABELS,
                'physical_hand': 'right', 'camera_index': 0, 'input_orientation': 'unflipped_camera'}
RAW_PLANS = {
    'baseline': {s: tuple(range(5)) for s in ('s01', 's02', 's03', 's04', 's05')},
    'index_up': {s: (1,) for s in ('idx_train01', 'idx_train02', 'idx_train03', 'idx_val01')},
    'fresh_test': {'simple_test01': (0, 1, 2, 3)},
}
RAW_PATHS = {'baseline': 'data/samples.csv', 'index_up': 'data/index_up_v2.csv',
             'fresh_test': 'data/simple_test_v2.csv'}


def read_raw(path, source_key):
    path = Path(path)
    metadata_path = path.with_suffix('.metadata.json')
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'Missing capture CSV or its metadata: {path}. Finish the matching recordings first.')
    initial = {'csv_path': RAW_PATHS[source_key], 'csv_sha256': file_sha256(path),
               'metadata_sha256': file_sha256(metadata_path)}
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    if not isinstance(metadata, dict) or any(metadata.get(k) != v for k, v in RAW_METADATA.items()):
        raise ValueError(f'{path.name}: raw capture metadata differs in hand/camera/features/classes/orientation')
    plan, rows, ids, counts = RAW_PLANS[source_key], [], set(), Counter()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError(f'{path.name}: CSV columns do not match the original collector')
        for line_number, row in enumerate(reader, 2):
            try:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('wrong number of columns')
                sample_id, session = row['sample_id'], row['session_id']
                if not sample_id or sample_id != sample_id.strip() or sample_id in ids:
                    raise ValueError('missing/invalid or duplicate sample ID')
                label = int(row['label_id'])
                if session not in plan or label not in plan[session]:
                    raise ValueError('unexpected session/label for this raw capture file')
                if row['feature_version'] != FEATURE_VERSION:
                    raise ValueError('incompatible feature version')
                stamp = datetime.fromisoformat(row['timestamp_utc'])
                if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
                    raise ValueError('timestamp must be timezone-aware UTC')
                values64 = np.asarray([row[c] for c in FEATURE_COLUMNS], dtype=np.float64)
                with np.errstate(over='ignore', invalid='ignore'):
                    values = values64.astype(np.float32)
                if not np.isfinite(values64).all() or not np.isfinite(values).all():
                    raise ValueError('non-finite float32 features')
                if not np.allclose(values[:3], 0, rtol=0, atol=1e-6):
                    raise ValueError('wrist is not centered')
                if not np.isclose(np.linalg.norm(values[27:30]), 1, rtol=1e-4, atol=1e-5):
                    raise ValueError('palm scale is not normalized')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'{path.name} row {line_number}: {exc}') from exc
            ids.add(sample_id)
            counts[(session, label)] += 1
            rows.append(row)
    for session, labels in plan.items():
        for label in labels:
            if counts[(session, label)] != 60:
                raise ValueError(f'{path.name}: {session}/{RAW_LABELS[label]} has {counts[(session, label)]} rows; expected 60')
    if (file_sha256(path) != initial['csv_sha256']
            or file_sha256(metadata_path) != initial['metadata_sha256']):
        raise ValueError('Capture changed during checking. Close all collectors and retry.')
    initial['row_count'] = len(rows)
    return rows, initial


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def verify_raw_unchanged(original_root, sources):
    for key, info in sources.items():
        path = Path(original_root) / RAW_PATHS[key]
        if (file_sha256(path) != info['csv_sha256']
                or file_sha256(path.with_suffix('.metadata.json')) != info['metadata_sha256']):
            raise ValueError('A source capture changed during preparation. Stop the collector and retry.')


def prepare_training_bundle(original_root, output, *, expected_baseline_sha256=VERIFIED_BASELINE_SHA256):
    original_root, output = Path(original_root).resolve(), Path(output).resolve()
    version_root = original_root / 'simple_v2'
    if output.exists():
        raise ValueError(f'{output} already exists; keep it or choose a new --output filename')
    for name in SOURCE_FILES:
        if not (version_root / name).is_file():
            raise FileNotFoundError(f'Missing {version_root / name}; extract the complete Simple V2 source package first')
    config = read_config(version_root / 'config.json')
    if config['working_hand'] != 'right' or config['camera_index'] != 0:
        raise ValueError('These verified captures use the right hand and camera 0')
    baseline = original_root / RAW_PATHS['baseline']
    if not baseline.is_file() or file_sha256(baseline) != expected_baseline_sha256:
        raise ValueError('Original 1,500-row dataset differs from its previously verified fingerprint; preserve it and report this mismatch')
    all_raw, sources, all_ids = {}, {}, set()
    for key, relative_path in RAW_PATHS.items():
        rows, info = read_raw(original_root / relative_path, key)
        these_ids = {row['sample_id'] for row in rows}
        if all_ids & these_ids:
            raise ValueError('A sample ID is reused across raw source files')
        all_ids.update(these_ids)
        all_raw[key], sources[key] = rows, info
    # Copy cells exactly. Retain only eligible baseline labels and sessions;
    # never rename old thumbs-up rows or change their session IDs.
    selected = [(row, 'baseline') for row in all_raw['baseline']
                if row['session_id'] in {'s01', 's02', 's03', 's04'} and int(row['label_id']) in {0, 2, 3}]
    selected += [(row, 'index_up') for row in all_raw['index_up']]
    selected += [(row, 'fresh_test') for row in all_raw['fresh_test']]
    lineage = {'gesture_set_version': GESTURE_SET_VERSION, 'raw_sources': sources,
               'samples': {row['sample_id']: {'source_key': key, 'row_sha256': row_fingerprint(row)}
                           for row, key in selected}}
    with tempfile.TemporaryDirectory(prefix='prepare_', dir=version_root) as temporary:
        staged_root = Path(temporary)
        staged = staged_root / 'data'
        staged.mkdir()
        with (staged / 'samples.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(row for row, _ in selected)
        write_json(staged / 'lineage.json', lineage)
        metadata = dict(RAW_METADATA, class_labels=list(CLASS_LABELS),
                        gesture_set_version=GESTURE_SET_VERSION, gesture_poses=GESTURE_POSES,
                        raw_sources=sources, lineage_sha256=file_sha256(staged / 'lineage.json'))
        write_json(staged / 'samples.metadata.json', metadata)
        data = load_dataset(staged / 'samples.csv', config)
        manifest = make_manifest(data, config, project_root=staged_root)
        save_manifest(staged / 'split_manifest.json', manifest)
        verify_raw_unchanged(original_root, sources)
        target = version_root / 'data'
        if target.exists():
            if not target.is_dir():
                raise ValueError('simple_v2/data already exists as a non-directory; preserve it and report this')
            for name in ('samples.csv', 'samples.metadata.json', 'lineage.json'):
                if not (target / name).is_file() or file_sha256(target / name) != file_sha256(staged / name):
                    raise ValueError('Existing simple_v2 data describes different sources. It was preserved; report this mismatch.')
            save_manifest(target / 'split_manifest.json', manifest)
        else:
            os.rename(staged, target)
    data = load_dataset(version_root / 'data/samples.csv', config)
    verify_raw_unchanged(original_root, sources)
    counts = {role: len(indices) for role, indices in data.split_indices.items()}
    info = build_zip(version_root, output, SOURCE_FILES + DATA_FILES,
                     {'destination': r'V:\Projects\GesturePilot\simple_v2',
                      'dataset_sha256': data.sha256, 'total_rows': len(data.sample_ids),
                      'split_row_counts': counts, 'split_sessions': EXPECTED_SPLITS,
                      'class_labels': list(CLASS_LABELS), 'gesture_poses': GESTURE_POSES,
                      'raw_sources': sources, 'reused_baseline_rows': 720,
                      'new_index_rows': 240, 'fresh_test_rows': 240},
                     info_name='TRAINING_TRANSFER_INFO.json')
    return info
