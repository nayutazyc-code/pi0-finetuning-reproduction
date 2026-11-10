"""Guard motion comparisons against stationary failures and shifted spikes."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
spec = importlib.util.spec_from_file_location(
    'ik_analysis', ROOT / 'scripts/analyze_ik_continuous_evaluation.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class EvaluationGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = json.loads((ROOT / 'outputs/ik_continuous_20261029/evaluation/plan.json').read_text())['gate']
        self.reference = dict(
            median_active_jerk_p99=100., covered_median_active_jerk_p99=100.,
            worst_active_jerk_peak=200., worst_joint_increment_rad=.1,
            worst_low_motion_pause_seconds=1., transport_covered=8,
            covered_median_simulated_seconds=20., worst_all_frame_jerk_peak=300.,
            covered_median_wall_seconds=25., median_low_motion_pause_total_seconds=1.)
        self.candidate = copy.deepcopy(self.reference)
        self.candidate.update(median_active_jerk_p99=70., covered_median_active_jerk_p99=70.)

    def test_stationary_failure_has_no_covered_smoothness_credit(self):
        from pi0_repro.motion_quality import motion_quality
        import numpy as np
        states = np.zeros((100, 14))
        quality = motion_quality(states, states, [{} for _ in states])
        summary = analysis.summarize([dict(
            quality=quality, success=False, simulated_seconds=2., control_wall_seconds=2.,
            inference_underruns=0, inference_wait_seconds=0.,
            action_apply_ms={'p99': 20., 'max': 20.})])
        self.assertIsNone(summary['median_active_jerk_p99'])
        self.assertIsNone(summary['covered_median_active_jerk_p99'])
        self.assertEqual(summary['transport_covered'], 0)
        self.assertFalse(analysis.compare(summary, self.reference, self.gate)['passed'])

    def test_typical_improvement_cannot_hide_peak_pause_or_coverage_regression(self):
        self.assertTrue(analysis.compare(self.candidate, self.reference, self.gate)['passed'])
        for key, value in [('worst_all_frame_jerk_peak', 400.),
                           ('worst_active_jerk_peak', 250.),
                           ('transport_covered', 7),
                           ('worst_low_motion_pause_seconds', 1.2),
                           ('median_low_motion_pause_total_seconds', 1.2),
                           ('covered_median_wall_seconds', 30.)]:
            with self.subTest(metric=key):
                candidate = dict(self.candidate, **{key: value})
                self.assertFalse(analysis.compare(candidate, self.reference, self.gate)['passed'])


if __name__ == '__main__':
    unittest.main()
