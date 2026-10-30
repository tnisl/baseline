"""Entry points, package paths and provenance after the directory restructure."""
from pathlib import Path
import shutil
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from helpers import BASE, ROOT
from lifemate_v7 import runner
from lifemate_v7.common import find_root


class LayoutTests(unittest.TestCase):
    def test_both_entry_points_work_outside_project_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in ('run.py', 'run_one.py'):
                proc = subprocess.run([sys.executable, str(BASE / script), '--help'], cwd=directory,
                                      capture_output=True, text=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertIn('usage:', proc.stdout)

    def test_nested_package_finds_data_from_source_ancestors(self):
        with tempfile.TemporaryDirectory() as directory, patch('pathlib.Path.cwd', return_value=Path(directory)):
            self.assertEqual(find_root(), ROOT.resolve())

    def test_fingerprint_includes_prompt_text_assets(self):
        source = Path(runner.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'lifemate_v7'
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__'))
            with patch.object(runner, '__file__', str(target / 'runner.py')):
                before = runner.workflow_sha256()
                for name in ('planner.txt', 'caller.txt', 'summarizer.txt'):
                    before = runner.workflow_sha256()
                    prompt = target / 'prompts' / name
                    prompt.write_text(prompt.read_text() + '\nA changed prompt\n')
                    self.assertNotEqual(before, runner.workflow_sha256())

    def test_seed_restart_preserves_module_entry_point(self):
        from lifemate_v7.reproducibility import prepare_process
        original = ['python', '-m', 'lifemate_v7', '--mode', 'workflow']
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'orig_argv', original), \
                patch('os.execvpe') as execute:
            prepare_process(42, vllm=True)
        args = execute.call_args.args
        self.assertEqual(args[1], [sys.executable, '-m', 'lifemate_v7', '--mode', 'workflow'])
        self.assertEqual(args[2]['PYTHONHASHSEED'], '42')


if __name__ == '__main__':
    unittest.main()
