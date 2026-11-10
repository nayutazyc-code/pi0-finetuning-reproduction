"""Check server restoration without contacting SSH or stepping physics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'repos/openpi/packages/openpi-client/src')]
spec = importlib.util.spec_from_file_location('release_launcher', ROOT / 'scripts/run_release_candidate.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ReleaseLauncherTest(unittest.TestCase):
    def exercise(self, *, dispatch_error=False, run_error=False, metadata_error=False):
        selections = []

        def dispatch(command, **kwargs):
            model = 'candidate' if '--model candidate' in command[-1] else 'v6'
            selections.append(model)
            if model == 'candidate' and dispatch_error:
                raise subprocess.CalledProcessError(255, command)
            return SimpleNamespace(check_returncode=lambda: None, stdout=json.dumps({'model': model}))

        client = MagicMock()
        client.__enter__.return_value.metadata = {'checkpoint': 'verified'}
        if metadata_error:
            client.__enter__.side_effect = [RuntimeError('Checkpoint mismatch'), client.__enter__.return_value]
        fake_remote = SimpleNamespace(RemotePi0=MagicMock(return_value=client))
        run = MagicMock(return_value=Path('/private/tmp/fake-run'))
        if run_error:
            run.side_effect = RuntimeError('Simulation failed')
        tunnel = MagicMock()
        with patch.object(sys, 'argv', ['run_release_candidate.py']), \
             patch.dict(sys.modules, {'pi0_repro.remote': fake_remote, 'pi0_repro.simulation': SimpleNamespace(run=run)}), \
             patch.object(launcher.subprocess, 'check_output', return_value='interface: en0\n'), \
             patch.object(launcher.subprocess, 'run', side_effect=dispatch), \
             patch.object(launcher.subprocess, 'Popen', return_value=tunnel), \
             patch.object(launcher.socket, 'socket', return_value=MagicMock()):
            if dispatch_error or run_error or metadata_error:
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    launcher.main()
            else:
                launcher.main()
        self.assertEqual(selections, ['candidate', 'v6'])
        tunnel.terminate.assert_called_once()
        tunnel.wait.assert_called_once_with(timeout=5)
        if dispatch_error or metadata_error:
            run.assert_not_called()

    def test_normal_completion_restores_v6(self):
        self.exercise()

    def test_lost_dispatch_response_still_restores(self):
        self.exercise(dispatch_error=True)

    def test_simulation_error_restores(self):
        self.exercise(run_error=True)

    def test_metadata_mismatch_does_not_run_and_restores(self):
        self.exercise(metadata_error=True)


if __name__ == '__main__':
    unittest.main()
