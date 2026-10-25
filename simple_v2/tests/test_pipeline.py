"""Synthetic migration/transfer checks, without webcam or real student data."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_migration import (RAW_METADATA, RAW_PATHS, RAW_PLANS, VERIFIED_BASELINE_SHA256,
                            prepare_training_bundle, write_json)
from dataset import (CSV_COLUMNS, EXPECTED_SPLITS, file_sha256, load_dataset,
                     make_manifest, read_config, save_manifest)
from hand_features import CLASS_LABELS, FEATURE_VERSION, GESTURE_SET_VERSION, GESTURE_POSES
from make_demo_bundle import build_demo_bundle
from package_utils import SOURCE_FILES, REPORT_FILES


def write_rows(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.project = self.folder / 'project'
        self.version = self.project / 'simple_v2'
        (self.project / 'data').mkdir(parents=True)
        for name in SOURCE_FILES:
            destination = self.version / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        (self.project / 'models').mkdir()
        (self.project / 'models/gesture_mlp.pt').write_bytes(b'unchanged baseline model')
        (self.project / 'data/split_manifest.json').write_text('unchanged baseline manifest')
        (self.project / 'config.json').write_text('unchanged baseline configuration')
        rng = np.random.default_rng(73)
        self.raw_rows = {}
        for key, plan in RAW_PLANS.items():
            rows = []
            for session, labels in plan.items():
                for label in labels:
                    for sample in range(60):
                        points = rng.normal(0, 0.1, (21, 3)).astype(np.float32)
                        points[0], points[9] = [0, 0, 0], [1, 0, 0]
                        values = [f'{key}_{session}_{label}_{sample}', session, str(label),
                                  '2026-10-23T12:00:00+00:00', FEATURE_VERSION]
                        values += [format(float(v), '.9g') for v in points.ravel()]
                        rows.append(dict(zip(CSV_COLUMNS, values)))
            self.raw_rows[key] = rows
            path = self.project / RAW_PATHS[key]
            write_rows(path, rows)
            write_json(path.with_suffix('.metadata.json'), RAW_METADATA)
        self.digest = file_sha256(self.project / 'data/samples.csv')
        self.output = self.folder / 'training.zip'

    def prepare(self, output=None):
        return prepare_training_bundle(self.project, output or self.output,
                                       expected_baseline_sha256=self.digest)

    def rewrite(self, key):
        write_rows(self.project / RAW_PATHS[key], self.raw_rows[key])

    def source_hashes(self):
        names = list(RAW_PATHS.values()) + [str(Path(p).with_suffix('.metadata.json')) for p in RAW_PATHS.values()]
        names += ['models/gesture_mlp.pt', 'data/split_manifest.json', 'config.json']
        return {name: file_sha256(self.project / name) for name in names}

    def test_merge_preserves_sources_ids_features_sessions_and_balanced_splits(self):
        before = self.source_hashes()
        info = self.prepare()
        self.assertEqual(before, self.source_hashes())
        config = read_config(self.version / 'config.json')
        data = load_dataset(self.version / 'data/samples.csv', config)
        self.assertEqual(data.features.shape, (1200, 63))
        self.assertEqual(len(set(data.sample_ids)), 1200)
        self.assertEqual(info['split_row_counts'], {'train': 720, 'validation': 240, 'test': 240})
        for role, indices in data.split_indices.items():
            expected = 180 if role == 'train' else 60
            self.assertEqual(np.bincount(data.labels[indices]).tolist(), [expected] * 4)
        original_rows = {row['sample_id']: row for rows in self.raw_rows.values() for row in rows}
        with (self.version / 'data/samples.csv').open(newline='') as handle:
            merged = list(csv.DictReader(handle))
        self.assertTrue(all(row == original_rows[row['sample_id']] for row in merged))
        self.assertNotIn('s05', data.session_ids)
        for row in merged:
            if row['session_id'].startswith('s0'):
                self.assertIn(int(row['label_id']), [0, 2, 3])
        manifest = json.loads((self.version / 'data/split_manifest.json').read_text())
        self.assertEqual(manifest['splits']['test']['session_ids'], ['simple_test01'])
        self.assertEqual(manifest['gesture_set_version'], GESTURE_SET_VERSION)
        self.assertTrue(all(info['files'][name]['sha256'] == file_sha256(self.version / name)
                            for name in info['files']))

    def test_training_zip_contains_only_isolated_version_files_and_verified_bytes(self):
        self.prepare()
        with ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertTrue(all(name.startswith('simple_v2/') for name in names))
            self.assertIn('simple_v2/data/samples.csv', names)
            self.assertFalse(any('.venv' in name or 'index_up_v2.csv' in name or 'simple_test_v2.csv' in name
                                 or 'models/' in name for name in names))
            info = json.loads(archive.read('simple_v2/TRAINING_TRANSFER_INFO.json'))
            for name, item in info['files'].items():
                self.assertEqual(hashlib.sha256(archive.read('simple_v2/' + name)).hexdigest(), item['sha256'])

    def test_laptop_cli_then_extracted_pc_dataset_check_in_fresh_processes(self):
        # Adapt only this temporary synthetic fixture's fingerprint. Production
        # has no CLI override and always requires the user's verified baseline.
        source = self.version / 'data_migration.py'
        text = source.read_text()
        self.assertEqual(text.count(VERIFIED_BASELINE_SHA256), 1)
        source.write_text(text.replace(VERIFIED_BASELINE_SHA256, self.digest))
        helper = self.project / 'prepare_simple_v2.py'
        shutil.copyfile(ROOT / 'prepare_simple_v2.py', helper)
        result = subprocess.run([sys.executable, str(helper), '--output', str(self.output)],
                                cwd=self.project, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('SIMPLE DATASET CHECK PASSED', result.stdout)
        self.assertIn('PC TRAINING PACKAGE READY', result.stdout)
        pc_root = self.folder / 'pc_project'
        with ZipFile(self.output) as archive:
            archive.extractall(pc_root)
        result = subprocess.run([sys.executable, 'prepare_data.py', '--config', 'config.json'],
                                cwd=pc_root / 'simple_v2', capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Total rows: 1200', result.stdout)
        self.assertIn('train: 720 rows', result.stdout)
        self.assertIn('validation: 240 rows', result.stdout)
        self.assertIn('test: 240 rows', result.stdout)
        self.assertIn('matches; kept unchanged', result.stdout)

    def test_baseline_hash_mismatch_is_rejected_before_writing_new_dataset(self):
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            prepare_training_bundle(self.project, self.output, expected_baseline_sha256='0' * 64)
        self.assertFalse((self.version / 'data').exists())
        self.assertFalse(self.output.exists())

    def test_missing_test_capture_is_not_treated_as_zero_row_success(self):
        (self.project / RAW_PATHS['fresh_test']).unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Finish the matching recordings'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_incomplete_capture_count_is_rejected(self):
        self.raw_rows['fresh_test'].pop()
        self.rewrite('fresh_test')
        with self.assertRaisesRegex(ValueError, '59 rows; expected 60'):
            self.prepare()

    def test_raw_metadata_mismatch_is_rejected(self):
        path = (self.project / RAW_PATHS['index_up']).with_suffix('.metadata.json')
        write_json(path, dict(RAW_METADATA, physical_hand='left'))
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.prepare()

    def test_nonfinite_unnormalized_and_invalid_timestamp_features_are_rejected(self):
        original = self.raw_rows['index_up'][0].copy()
        for column, value, message in [('f17', 'nan', 'non-finite'), ('f00', '0.1', 'wrist'),
                                        ('f27', '2', 'scale'), ('timestamp_utc', '2026-10-23', 'UTC')]:
            with self.subTest(column=column):
                self.raw_rows['index_up'][0] = dict(original, **{column: value})
                self.rewrite('index_up')
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
        self.raw_rows['index_up'][0] = original

    def test_sample_ids_cannot_overlap_between_raw_sources(self):
        self.raw_rows['index_up'][0]['sample_id'] = self.raw_rows['baseline'][0]['sample_id']
        self.rewrite('index_up')
        with self.assertRaisesRegex(ValueError, 'reused across raw source'):
            self.prepare()

    def test_index_up_file_rejects_old_session_or_additional_gesture(self):
        self.raw_rows['index_up'][0]['session_id'] = 's01'
        self.rewrite('index_up')
        with self.assertRaisesRegex(ValueError, 'unexpected session/label'):
            self.prepare()

    def test_exact_features_copied_from_training_to_test_are_rejected(self):
        first = self.raw_rows['baseline'][0]
        for key in CSV_COLUMNS[5:]:
            self.raw_rows['fresh_test'][0][key] = first[key]
        self.rewrite('fresh_test')
        with self.assertRaisesRegex(ValueError, 'across splits'):
            self.prepare()
        self.assertFalse((self.version / 'data').exists())

    def test_existing_zip_is_not_overwritten_and_identical_data_can_be_reused(self):
        self.prepare()
        first = self.output.read_bytes()
        original_manifest = (self.version / 'data/split_manifest.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()
        self.assertEqual(first, self.output.read_bytes())
        self.prepare(self.folder / 'training_second.zip')
        self.assertEqual(original_manifest, (self.version / 'data/split_manifest.json').read_bytes())

    def test_changed_capture_cannot_overwrite_a_prepared_dataset(self):
        self.prepare()
        original = (self.version / 'data/samples.csv').read_bytes()
        self.raw_rows['index_up'][0]['f06'] = '0.123456789'
        self.rewrite('index_up')
        with self.assertRaisesRegex(ValueError, 'different sources'):
            self.prepare(self.folder / 'changed.zip')
        self.assertEqual(original, (self.version / 'data/samples.csv').read_bytes())
        self.assertFalse((self.folder / 'changed.zip').exists())

    def test_gesture_definition_and_split_changes_are_rejected(self):
        path = self.version / 'config.json'
        valid = json.loads(path.read_text())
        for changes in ({'gesture_poses': dict(GESTURE_POSES, next='thumbs_up')},
                        {'class_labels': ['neutral', 'next', 'previous', 'blackout', 'start']},
                        {'split_sessions': dict(EXPECTED_SPLITS, validation=['s01'])}):
            with self.subTest(changes=changes):
                write_json(path, dict(valid, **changes))
                with self.assertRaises(ValueError):
                    read_config(path)
        write_json(path, valid)

    def test_tampered_merged_row_or_lineage_is_rejected(self):
        self.prepare()
        config = read_config(self.version / 'config.json')
        path = self.version / 'data/samples.csv'
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        rows[0]['f06'] = '0.123456789'
        write_rows(path, rows)
        with self.assertRaisesRegex(ValueError, 'source lineage'):
            load_dataset(path, config)

    def seed_reports(self):
        config = read_config(self.version / 'config.json')
        data = load_dataset(self.version / 'data/samples.csv', config)
        path = self.version / 'models/gesture_mlp.pt'
        path.parent.mkdir()
        path.write_bytes(b'synthetic report/transfer fixture, not a native model')
        for name in REPORT_FILES:
            path_report = self.version / name
            path_report.parent.mkdir(exist_ok=True)
            path_report.write_text('synthetic archive fixture')
        shared = {'dataset_sha256': data.sha256, 'feature_version': FEATURE_VERSION,
                  'class_labels': list(CLASS_LABELS), 'gesture_set_version': GESTURE_SET_VERSION,
                  'gesture_poses': GESTURE_POSES, 'physical_hand': 'right', 'selected_epoch': 1}
        write_json(self.version / 'reports/training_summary.json',
                   dict(shared, split_sessions=EXPECTED_SPLITS, device_label='desktop'))
        for split, filename in [('validation', 'validation_metrics.json'), ('test', 'metrics.json')]:
            write_json(self.version / 'reports' / filename,
                       dict(shared, checkpoint_sha256=file_sha256(path), split=split,
                            session_ids=EXPECTED_SPLITS[split], sample_count=240,
                            session_count=len(EXPECTED_SPLITS[split]), accuracy=0.5, macro_f1=0.5))

    def test_return_package_has_model_reports_but_preserves_existing_laptop_csv(self):
        self.prepare()
        self.seed_reports()
        output = self.folder / 'return.zip'
        build_demo_bundle(self.version, output)
        with ZipFile(output) as archive:
            self.assertIn('simple_v2/models/gesture_mlp.pt', archive.namelist())
            self.assertIn('simple_v2/reports/metrics.json', archive.namelist())
            self.assertNotIn('simple_v2/data/samples.csv', archive.namelist())

    def test_return_package_rejects_old_or_mismatched_test_report(self):
        self.prepare()
        self.seed_reports()
        path = self.version / 'reports/metrics.json'
        report = json.loads(path.read_text())
        report['session_ids'] = ['s05']
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'test report does not match'):
            build_demo_bundle(self.version, self.folder / 'return.zip')


if __name__ == '__main__':
    unittest.main()
