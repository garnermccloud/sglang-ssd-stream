import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_bundle', ROOT / 'verify_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class PublicationTests(unittest.TestCase):
    def test_source_and_benchmark(self):
        result = bundle.verify(ROOT)
        self.assertEqual(result['measurements_verified'], 48)
        self.assertAlmostEqual(result['comparison']['code']['change_percent'], 18.163296052492008)

    def test_rejects_wrong_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, 'Incompatible runtime'):
                bundle.verify(ROOT, base_root=Path(directory))

    def test_rejects_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / 'bundle'
            shutil.copytree(ROOT, copy)
            entry = json.loads((copy / 'manifest.json').read_text())['files'][0]
            (copy / entry['file']).write_text('tampered\n')
            with self.assertRaises(AssertionError):
                bundle.verify(copy)

    def test_rejects_benchmark_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / 'bundle'
            shutil.copytree(ROOT, copy)
            path = copy / 'evidence/control-before.json'
            data = json.loads(path.read_text())
            data['code'][0]['wall_tps'] *= 2
            path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                bundle.verify(copy)

    def test_paths_cannot_escape(self):
        for name in ['/etc/passwd', '../outside']:
            with self.assertRaises(ValueError):
                bundle.safe_path(ROOT, name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'link').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError):
                bundle.safe_path(root, 'link/outside')


if __name__ == '__main__':
    unittest.main()
