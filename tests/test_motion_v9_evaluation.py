"""Offline protocol checks; no server dispatch or simulator construction."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import evaluate_motion_v9 as evaluate
import summarize_motion_v9 as summarize
from pi0_repro.config import load_config
from pi0_repro.motion_quality import motion_quality


class EvaluationProtocolTests(unittest.TestCase):
    def test_only_exact_sidecar_backend_refusal_is_retryable_runtime_error(self):
        error = RuntimeError("远端错误：\nImage transport failed: ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 8000)")
        self.assertTrue(evaluate.backend_startup_error(error))
        for error in (RuntimeError('Image transport failed: ConnectionRefusedError'),
                      RuntimeError('远端错误：\nImage transport failed: ValueError: invalid model'),
                      RuntimeError('arbitrary policy failure'), ValueError(str(error))):
            self.assertFalse(evaluate.backend_startup_error(error))

    def select_setup(self):
        metadata = {key: 'fixed' for key in evaluate.CONTRACT_KEYS}
        metadata['transport_proxy'] = {'kind': 'lossless-png'}
        args = SimpleNamespace(remote_root='/root/project', ssh_socket='/tmp/socket',
                               ssh_target='cloud', metadata_contract={key: metadata[key] for key in evaluate.CONTRACT_KEYS})
        config = SimpleNamespace(connection=SimpleNamespace(connect_timeout_seconds=10))
        return metadata, args, config

    def test_startup_retry_is_bounded_and_retained_without_restarting_selector(self):
        metadata, args, config = self.select_setup()
        attempts = []
        class Client:
            def __init__(self, config):
                attempts.append(config.connection.connect_timeout_seconds)
                if len(attempts) == 1:
                    raise RuntimeError('远端错误：\nImage transport failed: ConnectionRefusedError: [Errno 111] Connection refused')
                self.metadata = metadata
            def __enter__(self): return self
            def __exit__(self, *args): pass
        receipt = {}
        with patch.object(evaluate.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'dispatch', '')) as dispatch, patch.object(evaluate.time, 'sleep'):
            self.assertEqual(evaluate.select_policy('v9', args, receipt, config, Client), metadata)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(attempts, [1, 1])
        self.assertEqual(len(receipt['startup_errors']), 1)
        self.assertEqual(config.connection.connect_timeout_seconds, 10)
        def refused(config): raise ConnectionRefusedError('sidecar down')
        with patch.object(evaluate.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')), patch.object(evaluate.time, 'monotonic', side_effect=[0, 91]):
            with self.assertRaises(ConnectionRefusedError):
                evaluate.select_policy('v9', args, {}, config, refused)

    def test_metadata_mismatch_and_unrelated_runtime_failure_never_retry(self):
        metadata, args, config = self.select_setup()
        def broken(config): raise RuntimeError('policy failure')
        with patch.object(evaluate.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')), patch.object(evaluate.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'policy failure'):
                evaluate.select_policy('v9', args, {}, config, broken)
            class WrongMetadata:
                def __init__(self, config):
                    self.metadata = dict(metadata, denoise_steps='changed')
                def __enter__(self): return self
                def __exit__(self, *args): pass
            with self.assertRaisesRegex(ValueError, 'Serving contract differs'):
                evaluate.select_policy('v9', args, {}, config, WrongMetadata)
            sleep.assert_not_called()

    def fixture_baseline(self, root):
        for path in ('scripts', 'src/pi0_repro', 'configs', str(evaluate.BASELINE)):
            (root / path).mkdir(parents=True, exist_ok=True)
        membership = root / evaluate.VALIDATION_MEMBERSHIP
        membership.parent.mkdir(parents=True)
        membership.write_text(json.dumps({'episodes': [{'source_episode': layout, 'split': 'validation'}
                                                      for layout in evaluate.LAYOUTS]}))
        for path in ('scripts/repeat_motion_baselines.py', 'src/pi0_repro/motion_quality.py', 'src/pi0_repro/execution.py'):
            (root / path).write_text('frozen\n')
        config_path = root / 'configs' / evaluate.CONFIGS['control']
        config_path.write_bytes((ROOT / 'configs' / evaluate.CONFIGS['control']).read_bytes())
        config = asdict(load_config(config_path))
        plan = {'layouts': list(evaluate.LAYOUTS), 'repeat_warmup_counts': [2, 3, 4],
                'script_sha256': evaluate.sha(root / 'scripts/repeat_motion_baselines.py'),
                'metric_sha256': evaluate.sha(root / 'src/pi0_repro/motion_quality.py'),
                'config_sha256': {'control': evaluate.sha(config_path)}}
        rows = []
        for index, (layout, repeat) in enumerate((layout, repeat) for layout in evaluate.LAYOUTS for repeat in (1, 2, 3)):
            run = root / f'run{index}'
            run.mkdir()
            scene = run / 'scene.toml'
            scene.write_text('[scene]\n')
            actual = json.loads(json.dumps(config))
            actual['simulation'].update(scene_config=str(scene), seed=100 + evaluate.LAYOUTS.index(layout), display=False)
            (run / 'run.json').write_text(json.dumps({'config': actual, 'source_code_sha256': {
                'execution.py': evaluate.sha(root / 'src/pi0_repro/execution.py')}}))
            (run / 'server.json').write_text(json.dumps({key: 'fixed' for key in evaluate.CONTRACT_KEYS}))
            rows.append({'profile': 'control', 'phase': 'validation', 'repeat': repeat, 'layout': layout, 'run_directory': str(run)})
        base = root / evaluate.BASELINE
        (base / 'baseline_plan.json').write_text(json.dumps(plan))
        (base / 'comparison.json').write_text(json.dumps({'complete': True, 'plan': plan, 'results': rows}))

    def test_bindings_recovered_from_all_baseline_receipts_and_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture_baseline(root)
            binding = evaluate.baseline_binding(root)
            self.assertEqual(len(binding['control_receipts']), 18)
            self.assertEqual(set(binding['layouts']), set(evaluate.LAYOUTS))
            (root / 'src/pi0_repro/execution.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Execution source differs'):
                evaluate.baseline_binding(root)

    def test_transitions_keep_unreached_and_uncompleted_separate(self):
        tasks = [{}, {'near': True}, {}, {'near': True}, {}]
        missing = summarize.transition(tasks, 'inside', 'success')
        self.assertIsNone(missing['seconds'])
        self.assertFalse(missing['reached'])
        censored = summarize.transition(tasks, 'near', 'lift')
        self.assertAlmostEqual(censored['seconds'], .06)
        self.assertTrue(censored['censored_at_recording_end'])
        tasks[-1]['lift'] = True
        self.assertTrue(summarize.transition(tasks, 'near', 'lift')['completed'])

    def test_stationary_failure_never_scores_perfect_active_motion(self):
        states = np.zeros((100, 14))
        tasks = [{'near_grasp_region': True} for _ in states]
        quality = motion_quality(states, states, tasks)
        transition = dict(reached=True, completed=False, censored_at_recording_end=True,
                          seconds=1.98, timeout=True, early_failure_or_abort=False)
        row = {'motion_quality': quality, 'stages': summarize.stage_metrics(states, tasks, quality['thresholds']),
               'near_to_lift_or_recording_end': transition,
               'inside_to_success_or_recording_end': dict(transition, reached=False, seconds=None, timeout=False),
               'near_to_lift_or_timeout_seconds': 1.98, 'inside_to_success_or_timeout_seconds': None,
               'task_result': {'success': False}}
        report = summarize.aggregate([row, {'analysis_error': 'retained failure'}])
        self.assertEqual(report['attempts'], 2)
        self.assertEqual(report['infrastructure_or_analysis_failures'], 1)
        self.assertEqual(report['primary']['active_frame_max_jerk_p99_rad_s3']['n'], 0)
        self.assertIsNone(report['primary']['active_frame_max_jerk_p99_rad_s3']['median'])
        self.assertEqual(report['coverage']['transport_motion_comparable'], 0)
        self.assertGreater(report['primary']['threshold_dwell_total_seconds']['median'], 1)
        self.assertEqual(report['secondary']['attempt_denominator'], 2)


if __name__ == '__main__':
    unittest.main()
