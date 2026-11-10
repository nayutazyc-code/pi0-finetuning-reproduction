"""Coverage and real motion must prevent favorable scores for stalled failures."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('progress_eval',Path(__file__).resolve().parents[1]/'scripts/analyze_v6_progress_evaluation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ProgressEvaluationTests(unittest.TestCase):
    gate={'median_active_jerk_p99_ratio_max':.9,'covered_median_active_jerk_p99_ratio_max':.9,'worst_peak_ratio_max':1.1,'worst_increment_ratio_max':1.1,'worst_low_motion_pause_add_seconds_max':.1,'covered_duration_ratio_max':1.1}
    def reference(self):return dict(median_active_jerk_p99=100,covered_median_active_jerk_p99=100,worst_active_jerk_peak=1000,worst_joint_increment_rad=.02,worst_low_motion_pause_seconds=.5,transport_covered=6,covered_median_simulated_seconds=25)
    def test_better_typical_score_cannot_hide_coverage_loss_or_peak_regression(self):
        r=self.reference();c=copy.deepcopy(r);c.update(median_active_jerk_p99=70,covered_median_active_jerk_p99=70)
        self.assertTrue(module.compare(c,r,self.gate)['passed'])
        c['transport_covered']=5;self.assertFalse(module.compare(c,r,self.gate)['passed'])
        c['transport_covered']=6;c['worst_active_jerk_peak']=1200;self.assertFalse(module.compare(c,r,self.gate)['passed'])
    def test_unavailable_covered_metrics_are_not_zero_or_a_win(self):
        r=self.reference();c=copy.deepcopy(r);c.update(covered_median_active_jerk_p99=None,covered_median_simulated_seconds=None,transport_covered=0)
        self.assertFalse(module.compare(c,r,self.gate)['passed'])


if __name__=='__main__':unittest.main()
