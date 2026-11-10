"""Safety guards only; never launch or stop a real policy in these tests."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'server'))
import select_motion_policy as selector


class MotionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.parent = self.root / selector.OUTPUT
        (self.parent / 'cloud_training').mkdir(parents=True)
        self.config = self.root / 'config.toml'
        self.config.write_text('[model]\nfamily="pi0"\n')
        self.source = self.root / 'source.py'
        self.source.write_text('x = 1\n')
        self.checkpoint = self.root / 'checkpoint/1499'
        (self.checkpoint / 'params/d').mkdir(parents=True)
        (self.checkpoint / '_CHECKPOINT_METADATA').write_text('{}')
        (self.checkpoint / 'params/_METADATA').write_text('{}')
        (self.checkpoint / 'params/manifest.ocdbt').write_bytes(b'manifest')
        (self.checkpoint / 'params/d/part').write_bytes(b'params')
        norm = self.checkpoint / 'assets/local/pi0_banana_train/norm_stats.json'
        norm.parent.mkdir(parents=True)
        norm.write_bytes(b'norm fixture')
        self.norm_digest = hashlib.sha256(norm.read_bytes()).hexdigest()
        self.original_digest = selector.NORM_SHA
        selector.NORM_SHA = self.norm_digest
        self.addCleanup(setattr, selector, 'NORM_SHA', self.original_digest)
        self.plan = {'experiment': 'banana_v9_motion1500', 'final_step': 1499,
                     'config': 'config.toml', 'config_sha256': selector.digest(self.config),
                     'norm_stats_sha256': self.norm_digest,
                     'source_sha256': {'source.py': selector.digest(self.source)}}
        self.status = {'status': 'trained_pending_validation', 'experiment': self.plan['experiment'],
                       'completed_optimizer_updates': 1500, 'final_checkpoint_committed': True,
                       'checkpoint': str(self.checkpoint), 'norm_stats_sha256': self.norm_digest}
        self.write_records()

    def write_records(self):
        (self.parent / 'training_plan.json').write_text(json.dumps(self.plan))
        (self.parent / 'cloud_training/status.json').write_text(json.dumps(self.status))

    def test_complete_candidate_and_source_pin(self):
        self.assertEqual(selector.validate_candidate(self.root, self.checkpoint), self.config)
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Bound source changed'):
            selector.validate_candidate(self.root, self.checkpoint)

    def test_active_training_state_rejected_before_checkpoint_use(self):
        self.status['status'] = 'training'
        self.write_records()
        with self.assertRaisesRegex(ValueError, 'has not completed'):
            selector.validate_candidate(self.root, self.checkpoint)

    def test_commit_markers_without_parameter_data_rejected(self):
        (self.checkpoint / 'params/d/part').unlink()
        with self.assertRaisesRegex(ValueError, 'parameter data is missing'):
            selector.validate_candidate(self.root, self.checkpoint)

    def test_normalization_changed_rejected(self):
        (self.checkpoint / 'assets/local/pi0_banana_train/norm_stats.json').write_text('different')
        with self.assertRaisesRegex(ValueError, 'normalization differs'):
            selector.validate_candidate(self.root, self.checkpoint)

    def test_process_inventory_matches_exact_script_tokens(self):
        proc = self.root / 'fake-proc'
        (proc / '1').mkdir(parents=True)
        script = str(self.root / 'server/serve_pi0.py')
        (proc / '1/cmdline').write_bytes(b'python\0' + script.encode() + b'\0')
        (proc / '2').mkdir()
        (proc / '2/cmdline').write_bytes(('echo\0' + script + ' --help\0').encode())
        self.assertEqual(selector.process_inventory(self.root, proc), [1])
        (proc / '3').mkdir()
        (proc / '3/cmdline').write_bytes(b'python\0' + str(self.root / 'server/banana_finetune.py').encode() + b'\0')
        with self.assertRaisesRegex(ValueError, 'training process remains active'):
            selector.process_inventory(self.root, proc)

    def test_ambiguous_policy_processes_rejected(self):
        proc = self.root / 'fake-proc'
        for pid in (1, 2):
            (proc / str(pid)).mkdir(parents=True)
            (proc / str(pid) / 'cmdline').write_bytes(
                b'python\0' + str(self.root / 'server/serve_pi0.py').encode() + b'\0')
        with self.assertRaisesRegex(ValueError, 'Multiple project policy'):
            selector.process_inventory(self.root, proc)

    def test_client_execution_matches_control(self):
        app = Path(__file__).resolve().parents[1]
        baseline = tomllib.loads((app / 'configs/banana_v8_control32_pi0.toml').read_text())
        candidate = tomllib.loads((app / 'configs/banana_v9_motion_pi0.toml').read_text())
        baseline['model']['expected_checkpoint_suffix'] = 'banana_v9_motion1500/1499'
        self.assertEqual(candidate, baseline)


if __name__ == '__main__':
    unittest.main()
