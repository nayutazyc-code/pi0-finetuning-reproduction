"""Offline prototype: transient demand, hysteresis and actuator-scope checks."""
import os
import unittest
import numpy as np
from pi0_repro.wrist_effort_guard import SustainedEffortState


class GuardStateTests(unittest.TestCase):
    def test_brief_pulses_pass_and_continuous_effort_activates(self):
        state=SustainedEffortState()
        for _ in range(20):
            self.assertFalse(state.update(4.7));self.assertFalse(state.update(.4))
        self.assertFalse(state.update(-1.2));self.assertFalse(state.update(-2.))
        self.assertTrue(state.update(-3.))

    def test_unclipped_demand_holds_until_persistent_low_error(self):
        state=SustainedEffortState()
        for _ in range(3):state.update(2.)
        for _ in range(10):self.assertTrue(state.update(2.))
        self.assertTrue(state.update(.4));self.assertTrue(state.update(.6))
        self.assertTrue(state.update(.4));self.assertTrue(state.update(.4))
        self.assertFalse(state.update(.4))
        state.reset();self.assertEqual((state.active,state.high,state.low),(False,0,0))

    def test_nonfinite_demand_and_invalid_threshold_rejected(self):
        for value in (0.,-1.,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):SustainedEffortState(limit_nm=value)
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):SustainedEffortState().update(value)
        with self.assertRaises(ValueError):SustainedEffortState(high_steps=True)


@unittest.skipUnless(os.environ.get('PI0_TEST_NATIVE_RENDER')=='1','Native MuJoCo required')
class GuardBindingTests(unittest.TestCase):
    def test_only_one_force_limit_changes_and_clipping_does_not_hide_demand(self):
        from pi0_repro.banana_env import BananaBoxEnv
        from pi0_repro.scene import SceneConfig
        from pi0_repro.wrist_effort_guard import WristEffortGuard
        env=BananaBoxEnv(SceneConfig(controller_profile='position_pd'),max_episode_steps=20,render_observations=False)
        try:
            env.reset(seed=7);m=env.physics.model;d=env.physics.data;guard=WristEffortGuard(env.physics);a=guard.actuator
            flags=m.actuator_forcelimited.copy();ranges=m.actuator_forcerange.copy()
            gain=m.actuator_gainprm.copy();bias=m.actuator_biasprm.copy();damping=m.dof_damping.copy()
            d.ctrl[a]=d.actuator_length[a]+.02;env.physics.forward()
            for _ in range(3):receipt=guard.before_step()
            self.assertTrue(receipt['active'])
            expected=flags.copy();expected[a]=1;np.testing.assert_array_equal(expected,m.actuator_forcelimited)
            expected=ranges.copy();expected[a]=[-1.,1.];np.testing.assert_array_equal(expected,m.actuator_forcerange)
            for actual,old in ((m.actuator_gainprm,gain),(m.actuator_biasprm,bias),(m.dof_damping,damping)):
                np.testing.assert_array_equal(actual,old)
            env.physics.forward();self.assertLessEqual(abs(d.actuator_force[a]),1.+1e-9)
            receipt=guard.before_step();self.assertTrue(receipt['active']);self.assertGreater(abs(receipt['unclipped_previous_command_demand_nm']),1.)
            guard.reset();np.testing.assert_array_equal(m.actuator_forcelimited,flags);np.testing.assert_array_equal(m.actuator_forcerange,ranges)
        finally:env.close()


if __name__=='__main__':unittest.main()
