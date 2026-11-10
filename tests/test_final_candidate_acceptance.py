"""Acceptance logic tests without GPU, networking or simulation."""
from collections import Counter
import copy
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('final_acceptance',ROOT/'scripts/evaluate_final_candidate.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def models():
    reference=dict(attempts=12,successes=10,scored_active_motion=12,transport_covered=10,
        median_active_jerk_p99=100.,covered_median_active_jerk_p99=100.,worst_active_jerk_peak=1000.,
        worst_all_frame_jerk_peak=1000.,worst_joint_increment_rad=.02,worst_low_motion_pause_seconds=0.,
        median_low_motion_pause_total_seconds=0.,covered_median_simulated_seconds=25.,
        covered_median_wall_seconds=26.,median_trial_apply_p99_ms=21.,worst_apply_interval_ms=50.,
        total_running_inference_underruns=0,total_running_wait_seconds=0.)
    candidate=copy.deepcopy(reference);candidate.update(median_active_jerk_p99=79.,covered_median_active_jerk_p99=79.,worst_active_jerk_peak=900.,worst_all_frame_jerk_peak=900.)
    return candidate,reference

class AcceptanceTest(unittest.TestCase):
    def passes(self,c,r):return all(row['passed'] for row in module.gate_checks(c,r))

    def test_abba_and_each_layout_twice(self):
        cases=module.make_cases();self.assertEqual(len(cases),24)
        self.assertEqual([cases[i*6]['model'] for i in range(4)],list(module.BLOCKS))
        counts=Counter((c['model'],c['layout']) for c in cases)
        self.assertEqual(len(counts),12);self.assertEqual(set(counts.values()),{2})
        self.assertEqual(len({(c['model'],c['layout'],c['repeat']) for c in cases}),24)

    def test_all_gates_required(self):
        c,r=models();self.assertTrue(self.passes(c,r))
        c['worst_apply_interval_ms']=56.;self.assertFalse(self.passes(c,r))

    def test_success_floor_and_reference(self):
        c,r=models();c['successes']=9;self.assertFalse(self.passes(c,r))
        c['successes']=10;r['successes']=11;self.assertFalse(self.passes(c,r))

    def test_missing_nonfinite_or_zero_motion_fails(self):
        for value in (None,float('nan'),float('inf'),0.):
            c,r=models();c['median_active_jerk_p99']=value;self.assertFalse(self.passes(c,r))
        c,r=models();r['median_active_jerk_p99']=0.;self.assertFalse(self.passes(c,r))

    def test_incomplete_or_uncovered_fails(self):
        c,r=models();c['attempts']=11;self.assertFalse(self.passes(c,r))
        c,r=models();c['scored_active_motion']=11;self.assertFalse(self.passes(c,r))
        c,r=models();c['transport_covered']=9;self.assertFalse(self.passes(c,r))

    def test_underrun_cannot_hide_under_pause(self):
        c,r=models();c['total_running_inference_underruns']=1;self.assertFalse(self.passes(c,r))
        c,r=models();c['total_running_wait_seconds']=.10001;self.assertFalse(self.passes(c,r))

    def test_xml_accepts_only_six_damping_changes(self):
        joints=[f'vx300s_{s}/{k}' for s in ('left','right') for k in ('forearm_roll','wrist_angle','wrist_rotate')]
        def xml(damping,extra=''):return '<mujoco><worldbody>'+''.join(f'<joint name="{n}" damping="{damping}" {extra}/>' for n in joints)+'</worldbody></mujoco>'
        self.assertTrue(module.xml_pair_equal(xml(2),xml(4)))
        self.assertFalse(module.xml_pair_equal(xml(2),xml(3)))
        self.assertFalse(module.xml_pair_equal(xml(2),xml(4,'frictionloss="1"')))

    def test_configured_all_real_validation_layouts_and_scene_tamper(self):
        dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'
        manifest=dataset/'dataset_manifest.json'
        if not manifest.exists():self.skipTest('Local real-layout dataset not installed')
        raw=Path(module.read(manifest)['source_path'])
        plan=dict(models=module.MODELS,layouts={n:module.read(raw/n/'layout.json') for n in module.LAYOUTS})
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for name in module.LAYOUTS:
                for model in module.MODELS:
                    scale=module.MODELS[model]['damping_scale']
                    scene=dict(plan['layouts'][name]['scene'],wrist_damping_scale=scale)
                    p=out/f'{name}_damping{int(scale)}.toml'
                    p.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in scene.items() if v is not None)+'\n')
                    cfg=module.configured(plan,out,model,name,18001)
                    self.assertEqual(cfg.simulation.seed,plan['layouts'][name]['reset_seed'])
                    self.assertEqual(cfg.model.expected_checkpoint_suffix,module.MODELS[model]['suffix'])
                    p.write_text(p.read_text().replace('settle_seconds = 0.6','settle_seconds = 0.7'))
                    with self.assertRaisesRegex(AssertionError,'Generated scene changed'):
                        module.configured(plan,out,model,name,18001)
                    # A bad scene must not obstruct rollback metadata verification.
                    rollback=module.connection_config(plan,'v6',18001)
                    self.assertEqual(rollback.model.expected_checkpoint_suffix,module.MODELS['v6']['suffix'])

if __name__=='__main__':unittest.main()
