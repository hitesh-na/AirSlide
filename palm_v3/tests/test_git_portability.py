"""Check that Git clones and source ZIPs preserve the fingerprinted runtime data."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


PROJECT = Path(__file__).resolve().parents[2]
PROTECTED = (
    'palm_v3/data/samples.csv',
    'palm_v3/data/samples.metadata.json',
    'palm_v3/data/lineage.json',
    'palm_v3/data/split_manifest.json',
    'palm_v3/diagnostic_inputs/vsign_development.csv',
)
VALIDATE = '''
from pathlib import Path
from dataset import load_dataset, read_config
from evaluate_diagnostic import load_development_capture
from training_utils import require_manifest
config = read_config(Path('config.json'))
data = load_dataset(Path('data/samples.csv').resolve(), config)
require_manifest(data, config)
values, scores = load_development_capture(Path('diagnostic_inputs/vsign_development.csv'))
assert len(data.sample_ids) == 1200
assert values.shape == (451, 63)
print('Dataset, manifest and development-recording fingerprints passed.')
'''


@unittest.skipUnless(shutil.which('git'), 'Git is needed for clone/archive portability checks')
class GitDataPortabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.work = Path(cls.temporary.name)
        cls.source = cls.work / 'source'
        cls.source.mkdir()
        cls.environment = dict(os.environ, GIT_CONFIG_NOSYSTEM='1',
                               GIT_CONFIG_GLOBAL=os.devnull, GIT_ATTR_NOSYSTEM='1')
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR'):
            cls.environment.pop(key, None)
        cls.original = {name: (PROJECT / name).read_bytes() for name in PROTECTED}
        sources = [PROJECT / '.gitattributes', PROJECT / 'palm_v3/config.json']
        sources.extend((PROJECT / 'palm_v3').glob('*.py'))
        sources.extend(PROJECT / name for name in PROTECTED)
        for path in sources:
            destination = cls.source / path.relative_to(PROJECT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        cls.git('init', '--quiet', str(cls.source))
        # Stage with newline conversion enabled, as on many Windows installations.
        cls.git('-C', str(cls.source), '-c', 'core.autocrlf=true', 'add', '.')
        cls.git('-C', str(cls.source), '-c', 'user.name=AirSlide portability test',
                '-c', 'user.email=airslide-test@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'Test fixture')

    @classmethod
    def git(cls, *arguments):
        return subprocess.run(['git', *arguments], check=True, capture_output=True,
                              text=True, env=cls.environment)

    def validate_download(self, folder):
        for name, original in self.original.items():
            self.assertEqual((folder / name).read_bytes(), original, name)
        result = subprocess.run([sys.executable, '-c', VALIDATE],
                                cwd=folder / 'palm_v3', capture_output=True,
                                text=True, env=self.environment)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_clone_with_windows_newline_conversion(self):
        folder = self.work / 'clone_windows'
        self.git('-c', 'core.autocrlf=true', 'clone', '--quiet', '--no-local',
                 str(self.source), str(folder))
        self.validate_download(folder)

    def test_clone_without_newline_conversion(self):
        folder = self.work / 'clone_lf'
        self.git('-c', 'core.autocrlf=false', 'clone', '--quiet', '--no-local',
                 str(self.source), str(folder))
        self.validate_download(folder)

    def test_source_zip_preserves_original_fingerprints(self):
        archive = self.work / 'source.zip'
        self.git('-C', str(self.source), 'archive', '--format=zip',
                 '--output=' + str(archive), 'HEAD')
        folder = self.work / 'source_zip'
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(folder)
        self.validate_download(folder)


if __name__ == '__main__':
    unittest.main()
