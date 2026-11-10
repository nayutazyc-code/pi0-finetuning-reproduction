from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from pi0_repro.banana_expert import (
    ExpertConfig, ScriptedBananaExpert, interpolate_rotation, load_expert_config,
    trajectory_alpha, trajectory_duration,
)


def rotation(axis, angle):
    quaternion = np.empty(4)
    mujoco.mju_axisAngle2Quat(quaternion, np.asarray(axis, float), angle)
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


class ExpertMotionTests(unittest.TestCase):
    def test_legacy_defaults_and_timing_remain_exact(self):
        config = ExpertConfig()
        self.assertEqual(config.trajectory_shape, 'cubic')
        self.assertEqual(config.ik_position_tolerance, .0015)
        self.assertEqual(config.ik_rotation_tolerance, .025)
        self.assertFalse(config.interpolate_orientation)
        self.assertFalse(config.respect_peak_speed)
        self.assertEqual(config.gripper_trajectory_shape, 'linear')
        self.assertEqual(config.open_seconds, .7)
        self.assertFalse(config.direct_recovery_release)
        self.assertFalse(config.direct_recovery_place)
        for t in (.01, .2, .75, 1.):
            self.assertEqual(trajectory_alpha(t, 'cubic'), t*t*(3-2*t))
        self.assertEqual(trajectory_duration(.4, .7, 1.8, 'cubic', respect_peak=False, cartesian=False),
                         max(1.8, .4/.7))
        self.assertEqual(trajectory_duration(.12, .08, .5, 'cubic', respect_peak=False, cartesian=True),
                         max(.5, .12/.08*1.5))

    def test_quintic_endpoints_derivatives_and_nominal_speed(self):
        t = np.linspace(0, 1, 10001)
        alpha = trajectory_alpha(t, 'quintic')
        self.assertEqual(alpha[0], 0)
        self.assertEqual(alpha[-1], 1)
        self.assertTrue((np.diff(alpha) >= -1e-14).all())
        # Analytic velocity/acceleration vanish at both endpoints.
        velocity = 30*t*t*(1-t)**2
        acceleration = 60*t-180*t*t+120*t*t*t
        np.testing.assert_array_equal(velocity[[0,-1]], 0)
        np.testing.assert_array_equal(acceleration[[0,-1]], 0)
        self.assertAlmostEqual(float(velocity.max()), 1.875)
        self.assertTrue(np.isfinite(np.diff(alpha, n=3)).all())
        for cartesian in (False, True):
            seconds = trajectory_duration(.4, .7, .1, 'quintic', respect_peak=True, cartesian=cartesian)
            self.assertLessEqual(.4/seconds*velocity.max(), .7+1e-12)

    def test_so3_shortest_path_is_continuous_near_pi(self):
        start = rotation([1,0,0], .6)
        target = rotation([0,0,1], np.pi-1e-7) @ start
        samples = [interpolate_rotation(start, target, float(a)) for a in np.linspace(0,1,101)]
        np.testing.assert_array_equal(samples[0], start)
        np.testing.assert_array_equal(samples[-1], target)
        for a,b in zip(samples[:-1],samples[1:]):
            self.assertLess(np.linalg.norm(ScriptedBananaExpert._rotation_error(a,b)), .032)
            np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(b)),1,places=12)

    def test_small_rotation_and_first_recovery_step(self):
        start = rotation([0,1,0], .4)
        target = rotation([1,0,0], .2) @ start
        alpha = trajectory_alpha(1/25, 'quintic')
        first = interpolate_rotation(start,target,alpha)
        self.assertLess(np.linalg.norm(ScriptedBananaExpert._rotation_error(start,first)), .001)
        np.testing.assert_allclose(interpolate_rotation(start,start,.3),start,atol=1e-14)

    def test_gripper_hold_default_and_smooth_profile_speed(self):
        def targets(config, start, end, seconds):
            expert = object.__new__(ScriptedBananaExpert)
            expert.config = config
            expert.env = SimpleNamespace(control_dt=.02)
            expert.action = np.zeros(14)
            expert.action[13] = start
            expert.last_q = np.zeros(6)
            emitted = []
            expert._step = lambda joints, gripper: emitted.append(gripper)
            expert._hold(seconds, end)
            return np.r_[start, emitted]

        ordinary = targets(ExpertConfig(), .35, 1., .7)
        np.testing.assert_allclose(ordinary, np.linspace(.35,1.,36), atol=1e-15, rtol=0)
        profile = load_expert_config(Path(__file__).resolve().parents[1]/'configs/banana_motion_expert.toml')
        for start,end,seconds in ((.35,1.,profile.open_seconds),(1.,.35,profile.close_seconds)):
            q = targets(profile,start,end,seconds)
            velocity = np.diff(q)/.02
            self.assertAlmostEqual(q[-1],end)
            self.assertLessEqual(float(np.abs(velocity).max()),1.5)
            self.assertLess(abs(velocity[0]),.03*np.abs(velocity).max())
            self.assertLess(abs(velocity[-1]),.03*np.abs(velocity).max())
            self.assertTrue(np.isfinite(np.diff(q,n=3)).all())

    def test_invalid_config_and_opt_in_profile(self):
        for field,value in [('trajectory_shape','"linear"'),('ik_position_tolerance','0'),
                            ('ik_rotation_tolerance','-0.01'),('interpolate_orientation','1'),
                            ('respect_peak_speed','"yes"'),('gripper_trajectory_shape','"step"'),
                            ('direct_recovery_release','1'),('direct_recovery_place','"yes"')]:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'invalid.toml'
                path.write_text(f'[expert]\n{field} = {value}\n')
                with self.assertRaises(ValueError):load_expert_config(path)
        profile = Path(__file__).resolve().parents[1]/'configs/banana_motion_expert.toml'
        config = load_expert_config(profile)
        self.assertEqual(config.trajectory_shape,'quintic')
        self.assertTrue(config.respect_peak_speed and config.interpolate_orientation and config.direct_recovery_grasp)
        self.assertEqual(config.gripper_trajectory_shape,'quintic')
        self.assertTrue(config.direct_recovery_release and config.direct_recovery_place)

    @staticmethod
    def recovery_expert(config, *, recovery_mode=True, info=None, banana_xy=(.145,.715)):
        expert = object.__new__(ScriptedBananaExpert)
        expert.config = config
        expert.recovery_mode = recovery_mode
        expert.site = 0
        expert.q_indices = np.arange(8,14)
        expert.action = np.zeros(14)
        expert.last_q = np.zeros(6)
        expert.rotation = np.eye(3)
        expert.phase, expert.target = 'initial', None
        fields = dict(grasped=False, inside_box=True, supported_by_box=True,
                      stable=True, released=False, touching_table=False, ever_grasped=True)
        fields.update(info or {})
        data = SimpleNamespace(ctrl=np.linspace(-.2,.4,16), qpos=np.zeros(16),
                               site_xpos=np.array([[.13,.70,.15]]),site_xmat=np.eye(3).reshape(1,9))
        named = SimpleNamespace(data=SimpleNamespace(xpos={
            'banana':np.array([*banana_xy,.12]),'open_box':np.array([.145,.715,0.])}))
        expert.env = SimpleNamespace(physics=SimpleNamespace(data=data,named=named), control_dt=.02,
            inner_low=np.array([-.1,-.1,.01]), ever_grasped=True, ever_lifted=True,
            _task_info=lambda **kwargs:fields,
            _state_from_physical=lambda q:np.r_[q[:6],.35,q[8:14],.35])
        return expert

    def test_supported_recovery_release_keeps_actuator_arms(self):
        config = ExpertConfig(direct_recovery_release=True,gripper_trajectory_shape='quintic',open_seconds=.9)
        expert = self.recovery_expert(config)
        emitted, moves = [], []
        expert.execute = lambda action,phase,target:emitted.append((phase,action.copy()))
        expert._move = lambda target:moves.append(np.asarray(target).copy())
        expert.run()
        opening = [action for phase,action in emitted if phase=='release']
        self.assertEqual(len(opening),45)
        expected = expert.env._state_from_physical(expert.env.physics.data.ctrl)
        for action in opening:
            np.testing.assert_array_equal(action[[0,1,2,3,4,5,7,8,9,10,11,12]],
                                          expected[[0,1,2,3,4,5,7,8,9,10,11,12]])
        self.assertAlmostEqual(opening[-1][13],1.)
        self.assertEqual(len(moves),1)
        np.testing.assert_array_equal(moves[0],np.array([.13,.70,config.transport_height]))
        self.assertTrue(expert.env.ever_grasped and expert.env.ever_lifted)

    def test_direct_release_requires_recovery_and_all_support_guards(self):
        enabled = ExpertConfig(direct_recovery_release=True)
        for config,recovery_mode,override in (
            (ExpertConfig(),True,{}),(enabled,False,{}),
            (enabled,True,{'inside_box':False}),(enabled,True,{'supported_by_box':False}),
            (enabled,True,{'stable':False}),(enabled,True,{'released':True})):
            expert = self.recovery_expert(config,recovery_mode=recovery_mode,info=override)
            with self.assertRaisesRegex(RuntimeError,'table-resting or physically pinched'):
                expert.run()
            self.assertTrue(expert.env.ever_grasped)

    def test_held_near_box_can_skip_only_transport_hover(self):
        for config,recovery_mode,banana_xy,expected_count in (
            (ExpertConfig(direct_recovery_place=True),True,(.145,.715),1),
            (ExpertConfig(),True,(.145,.715),2),
            (ExpertConfig(direct_recovery_place=True),False,(.145,.715),2),
            (ExpertConfig(direct_recovery_place=True),True,(.19,.715),2)):
            expert = self.recovery_expert(config,recovery_mode=recovery_mode,banana_xy=banana_xy)
            moves, tails = [], []
            expert._move = lambda target:moves.append((expert.phase,np.asarray(target).copy()))
            expert._release_and_retreat = lambda target:tails.append(np.asarray(target).copy())
            expert._place(.015)
            self.assertEqual(len(moves),expected_count)
            self.assertEqual(moves[-1][0],'lower_into_box')
            self.assertEqual(len(tails),1)
            self.assertEqual(tails[0][2],config.transport_height)


if __name__ == '__main__':
    unittest.main()
