"""Restore regression tests using temporary data only; no agent/model imports.

Run from the repository root with:
    python -m unittest discover -s tests -p test_backup_restore.py -v
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
from typing import Any, Dict, Optional
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


class BackupRestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backups = self.root / 'backup'
        self.station_id = 'test-station'
        self.station_backup = self.backups / self.station_id
        self.objects = self.station_backup / 'objects'
        self.objects.mkdir(parents=True)
        (self.station_backup / 'snapshots').mkdir()
        self.target = self.root / 'station_data'
        self.entries = []

        # Import the real helper with only its two local dependencies stubbed.
        # restore_backup and _compute_file_hash use no file_io_utils functions.
        package = types.ModuleType('station')
        package.__path__ = []
        package.constants = types.SimpleNamespace(BACKUP_BASE_DIR=str(self.backups))
        package.file_io_utils = types.ModuleType('station.file_io_utils')
        self.modules = patch.dict(sys.modules, {'station': package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        spec = importlib.util.spec_from_file_location(
            '_backup_restore_subject', ROOT / 'station' / 'backup_utils.py')
        self.subject = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.subject
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(self.subject)
        package.backup_utils = self.subject
        sys.modules['station.backup_utils'] = self.subject

    def add_object(self, path, content=b'test bytes\n', stored=None, missing=False):
        digest = hashlib.sha256(content).hexdigest()
        entry = {'path': path, 'hash': digest, 'size': len(content),
                 'mode': 0o100640, 'mtime': 0}
        self.entries.append(entry)
        if not missing:
            obj = self.objects / digest[:2] / digest[2:]
            obj.parent.mkdir(exist_ok=True)
            obj.write_bytes(content if stored is None else stored)
        return entry

    def save_manifest(self):
        path = self.station_backup / 'snapshots' / 'tick_1.json'
        path.write_text(json.dumps({'files': self.entries}))

    def restore(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = self.subject.restore_backup(self.station_id, 1, str(self.target))
        return result, output.getvalue()

    def test_complete_snapshot_preserves_bytes_and_permissions(self):
        self.add_object('agents/example.yaml')
        self.save_manifest()
        ok, _ = self.restore()
        self.assertIs(ok, True)
        restored = self.target / 'agents/example.yaml'
        self.assertEqual(restored.read_bytes(), b'test bytes\n')
        self.assertEqual(restored.stat().st_mode & 0o777, 0o640)

    def test_missing_required_object_returns_false_retains_partial(self):
        self.add_object('present.txt')
        self.add_object('absent.txt', b'absent bytes', missing=True)
        self.save_manifest()
        ok, output = self.restore()
        self.assertIs(ok, False)
        self.assertTrue((self.target / 'present.txt').is_file())
        self.assertFalse((self.target / 'absent.txt').exists())
        self.assertIn('missing object', output)

    def test_corrupt_restored_bytes_return_false(self):
        self.add_object('corrupt.txt', b'expected bytes', stored=b'altered bytes!')
        self.save_manifest()
        ok, output = self.restore()
        self.assertIs(ok, False)
        self.assertIn('SHA-256 mismatch', output)

    def test_existing_temporary_exclusions_remain_successful(self):
        for path in ('rooms/research/storage/tmp/scratch',
                     'rooms/research/storage/shared/tmp/scratch'):
            self.add_object(path, path.encode(), missing=True)
        self.add_object('kept.txt')
        self.save_manifest()
        self.assertIs(self.restore()[0], True)
        self.assertEqual(list(self.target.iterdir()), [self.target / 'kept.txt'])

    def test_existing_target_rejected_without_modification(self):
        self.add_object('new.txt')
        self.save_manifest()
        self.target.mkdir()
        marker = self.target / 'keep.txt'
        marker.write_bytes(b'preserve existing target')
        self.assertIs(self.restore()[0], False)
        self.assertEqual(marker.read_bytes(), b'preserve existing target')
        self.assertFalse((self.target / 'new.txt').exists())

    def test_missing_manifest_returns_false_without_target(self):
        self.assertIs(self.restore()[0], False)
        self.assertFalse(self.target.exists())

    def run_shell(self):
        # A copied package in a disposable working directory, not real station_data.
        package = self.root / 'station'
        package.mkdir()
        (package / '__init__.py').write_text('')
        (package / 'constants.py').write_text("BACKUP_BASE_DIR = './backup'\n")
        (package / 'file_io_utils.py').write_text('')
        shutil.copy2(ROOT / 'station/backup_utils.py', package / 'backup_utils.py')
        script = self.root / 'restore.sh'
        shutil.copy2(ROOT / 'scripts/restore.sh', script)
        bindir = self.root / 'bin'
        bindir.mkdir()
        python = bindir / 'python'
        python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python.chmod(0o700)
        return subprocess.run(
            ['bash', str(script), self.station_id, '1'], cwd=self.root,
            env={'PATH': str(bindir) + os.pathsep + os.defpath,
                 'PYTHONDONTWRITEBYTECODE': '1'},
            text=True, input='', capture_output=True, timeout=10)

    def test_shell_complete_restore_exit_success(self):
        self.add_object('success.txt')
        self.save_manifest()
        result = self.run_shell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Restore completed successfully', result.stdout)

    def test_shell_missing_object_exit_failure(self):
        self.add_object('absent.txt', missing=True)
        self.save_manifest()
        result = self.run_shell()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Restore failed', result.stdout)
        self.assertNotIn('Restore completed successfully', result.stdout)

    def test_shell_corrupt_object_exit_failure(self):
        self.add_object('corrupt.txt', stored=b'bad object')
        self.save_manifest()
        result = self.run_shell()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Restore failed', result.stdout)
        self.assertNotIn('Restore completed successfully', result.stdout)

    def test_token_usage_caller_stops_after_failed_restore(self):
        self.add_object('absent.txt', missing=True)
        self.save_manifest()
        source = ROOT / 'scripts/estimate_station_tokens.py'
        # Exercise the existing caller decision, without YAML/provider imports.
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'collect_station_token_usage')
        collect = Mock()
        context = {'Path': Path, 'TemporaryDirectory': TemporaryDirectory,
                   'shutil': shutil, 'redirect_stdout': redirect_stdout, 'sys': sys,
                   'Optional': Optional, 'Dict': Dict, 'Any': Any,
                   '_resolve_backup_station_dir': lambda *args: self.station_backup,
                   '_collect_station_token_usage_from_root': collect}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), context)
        with redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Failed to restore backup'):
                context['collect_station_token_usage'](
                    str(self.target), backup_station_id=self.station_id, backup_tick=1)
        collect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
