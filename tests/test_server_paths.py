"""Check isolation, checkpoint discovery/resume, and error logs in subprocesses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RunPathsTest(unittest.TestCase):
    def test_repeated_runs_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, MMCLAST_EXPS=tmp, PYTHONPATH=str(ROOT),
                       PYTHONDONTWRITEBYTECODE='1')
            def call(entry, code):
                return subprocess.run([sys.executable, '-c',
                    'import sys; sys.argv=' + repr([entry, '--tag', 'mnist_probe']) +
                    '; from server_paths import *; ' + code], env=env,
                    cwd=ROOT, text=True, capture_output=True)
            first = call('train.py', "r=Path(experiment_root()); (r/'checkpoints/mnist_probe/stage2.pth').write_text('old'); print('first log')")
            self.assertEqual(first.returncode, 0, first.stderr)
            second = call('train.py', "r=Path(experiment_root()); assert Path(previous_checkpoint(2)).read_text()=='old'; print('second log')")
            self.assertEqual(second.returncode, 0, second.stderr)
            runs = sorted((Path(tmp)/'mnist/mnist_probe').iterdir())
            self.assertEqual(len(runs), 2)
            self.assertIn('first log', (runs[0]/'console.log').read_text())
            self.assertIn('second log', (runs[1]/'console.log').read_text())
            self.assertEqual((Path(tmp)/'mnist/checkpoints/mnist_probe').resolve(), runs[1]/'checkpoints/mnist_probe')
            failure = call('eval.py', "r=Path(experiment_root()); raise RuntimeError('test failure')")
            self.assertNotEqual(failure.returncode, 0)
            evaluation = next((Path(tmp)/'mnist/eval').iterdir())
            self.assertIn('test failure', (evaluation/'console.log').read_text())
            self.assertEqual(json.loads((evaluation/'run.json').read_text())['status'], 'failed')
            self.assertEqual((evaluation/'checkpoints/mnist_probe').resolve(), runs[1]/'checkpoints/mnist_probe')

if __name__ == '__main__':
    unittest.main()
