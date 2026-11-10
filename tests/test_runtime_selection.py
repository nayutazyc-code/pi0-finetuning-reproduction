"""Do not select apparent smoothness caused by stalled/incomplete transport."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('runtime_selector',Path(__file__).resolve().parents[1]/'scripts/select_runtime_candidate.py')
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)


class RuntimeSelectionTests(unittest.TestCase):
    gate={'median_trial_apply_p99_ratio_max':.9,'worst_apply_interval_ratio_max':1.,
          'gap_over50_rate_per1000_steps_ratio_max':1.,'running_underruns_max':0,
          'covered_median_simulated_duration_ratio_max':1.1,'covered_median_active_jerk_p99_ratio_max':1.1}
    def data(self):
        b=dict(profile='legacy_multiview',steps=1000,simulated_seconds=20,action_apply_ms={'p99':30,'max':100},
               gaps_over_50ms=3,inference_underruns=0,motion={'coverage':{'transport_motion_comparable':True},'active_frame_max_jerk_rad_s3':{'p99':100}})
        c=copy.deepcopy(b);c['profile']='worker_multiview';c['action_apply_ms']={'p99':22,'max':50};c['gaps_over_50ms']=1
        return {'audit_passed':True,'trials':[b,c]}
    def test_audited_improvement_is_only_eligible_for_review(self):
        self.assertEqual(selector.evaluate(self.data(),self.gate)['decision'],'eligible_for_review')
    def test_low_latency_without_transport_is_not_promoted(self):
        d=self.data();d['trials'][1]['motion']['coverage']['transport_motion_comparable']=False
        self.assertEqual(selector.evaluate(d,self.gate)['decision'],'insufficient_evidence')
    def test_tail_regression_or_underrun_blocks_default_change(self):
        for key,value in [('tail',101),('underrun',1)]:
            d=self.data();c=d['trials'][1]
            if key=='tail':c['action_apply_ms']['max']=value
            else:c['inference_underruns']=value
            self.assertEqual(selector.evaluate(d,self.gate)['decision'],'retain_existing_default')


if __name__=='__main__':unittest.main()
