"""Seeded pose/reset and opt-in display cameras, without a policy connection."""
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from dm_control import mujoco as dm_mujoco
from gym_aloha.constants import ASSETS_DIR, START_ARM_POSE
import numpy as np

from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.observations import aloha_observation
from pi0_repro.random_pose import RANGE_FIELDS, sample_banana_scene
from pi0_repro.scene import SceneConfig, build_scene_xml, load_scene_config


SEEDS = tuple(range(2026102801, 2026102813))
CAMERAS = ('display_front', 'display_side', 'display_wrist_right')


class RandomPoseTests(unittest.TestCase):
    def test_false_default_preserves_legacy_xml_and_never_draws_rng(self):
        class UnusedRng:
            def uniform(self, *args):
                raise AssertionError('Fixed pose must not consume RNG')
        for profile, digest in (
            ('original', '8c1bd507221c04d860f5add2864cc025e5cf757c0a1e51801c49ebb5277c2b73'),
            ('position_pd', '68afcc225bf3f7e7691fd39aa26afc07045f383dcc9d95581d4c5bbe68551875')):
            settings = SceneConfig(controller_profile=profile)
            sampled = sample_banana_scene(settings, UnusedRng())
            self.assertEqual(asdict(settings), asdict(sampled))
            self.assertIsNot(settings, sampled)
            xml = build_scene_xml(settings)
            self.assertEqual(xml, build_scene_xml(settings, display_cameras=False))
            normalized = xml.replace(str(Path(ASSETS_DIR).resolve()), 'GYM_ALOHA_ASSETS')
            self.assertEqual(hashlib.sha256(normalized.encode()).hexdigest(), digest)

    def test_pure_samples_reproduce_and_only_banana_xy_yaw_change(self):
        settings = SceneConfig(randomize_banana_pose=True, controller_profile='position_pd')
        baseline = asdict(settings)
        samples = []
        for seed in SEEDS:
            left = sample_banana_scene(settings, np.random.default_rng(seed))
            right = sample_banana_scene(settings, np.random.default_rng(seed))
            self.assertEqual(asdict(left), asdict(right))
            x, y, z = left.banana_position
            self.assertTrue(.16 <= x <= .20 and .46 <= y <= .50 and 0 <= left.banana_yaw_degrees <= 20)
            self.assertEqual(z, .045)
            self.assertEqual(left.box_position, (.13, .72, 0.))
            self.assertFalse(left.randomize_banana_pose)
            changed = asdict(left)
            for key in ('banana_position', 'banana_yaw_degrees', 'randomize_banana_pose'):
                changed[key] = baseline[key]
            self.assertEqual(changed, baseline)
            samples.append((x, y, left.banana_yaw_degrees))
        self.assertEqual(len(set(samples)), len(SEEDS))
        self.assertEqual(asdict(settings), baseline)

    def test_validation_and_toml_serialization(self):
        settings = SceneConfig()
        for flag in (1, 'true', None):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'randomize_banana_pose'):
                build_scene_xml(replace(settings, randomize_banana_pose=flag))
        for name in RANGE_FIELDS:
            for bounds in ((1,), (1, 2, 3), (1, 1), (2, 1), (True, 2), (0, float('nan')), (0, float('inf')), '0,1'):
                with self.subTest(name=name, bounds=bounds), self.assertRaisesRegex(ValueError, name):
                    build_scene_xml(replace(settings, **{name: bounds}))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scene.toml'
            path.write_text('[scene]\nrandomize_banana_pose=true\nbanana_x_range=[0.16,0.20]\n')
            parsed = load_scene_config(path)
            self.assertEqual(parsed.banana_x_range, (.16, .20))
            self.assertTrue(parsed.randomize_banana_pose)
            self.assertEqual(json.loads(json.dumps(asdict(parsed)))['banana_yaw_range_degrees'], [0., 20.])

    def test_real_resets_repeat_seed_preserve_box_and_record_sample_settle(self):
        base = SceneConfig(randomize_banana_pose=True, controller_profile='position_pd')
        env = BananaBoxEnv(base, max_episode_steps=10, render_observations=False)
        try:
            previous = None
            for seed in SEEDS:
                observation, _ = env.reset(seed=seed)
                expected = sample_banana_scene(base, np.random.default_rng(seed))
                self.assertEqual(asdict(env.episode_scene), asdict(expected))
                np.testing.assert_array_equal(env.physics.named.data.xpos['open_box'], (.13, .72, 0.))
                self.assertEqual(env.last_reset['seed'], seed)
                self.assertEqual(env.last_reset['sample']['position'], list(expected.banana_position))
                np.testing.assert_array_equal(env.last_reset['settled']['position'], env.physics.named.data.xpos['banana'])
                np.testing.assert_array_equal(env.last_reset['settled']['quaternion_wxyz'], env.physics.named.data.xquat['banana'])
                self.assertLess(env.last_reset['settled']['position'][2], .045)
                json.dumps(env.last_reset)
                if previous is not None:
                    self.assertNotEqual(previous, env.last_reset['sample']['position'])
                previous = env.last_reset['sample']['position']
            env.reset(seed=SEEDS[0])
            before = (env.physics.data.qpos.copy(), env.physics.data.qvel.copy(), json.dumps(env.last_reset, sort_keys=True))
            env.reset(seed=SEEDS[-1])
            env.reset(seed=SEEDS[0])
            np.testing.assert_array_equal(env.physics.data.qpos, before[0])
            np.testing.assert_array_equal(env.physics.data.qvel, before[1])
            self.assertEqual(json.dumps(env.last_reset, sort_keys=True), before[2])
            self.assertIs(env.scene, base)
            self.assertTrue(env.scene.randomize_banana_pose)
            replay = BananaBoxEnv(env.episode_scene, max_episode_steps=10, render_observations=False)
            try:
                replay.reset(seed=999)
                np.testing.assert_array_equal(replay.physics.data.qpos, env.physics.data.qpos)
                np.testing.assert_array_equal(replay.physics.data.qvel, env.physics.data.qvel)
            finally:
                replay.close()
        finally:
            env.close()

    def test_fixed_reset_and_controls_match_legacy_physics_path_exactly(self):
        settings = SceneConfig(controller_profile='position_pd')
        env = BananaBoxEnv(settings, max_episode_steps=20, render_observations=False)
        reference = dm_mujoco.Physics.from_xml_string(build_scene_xml(settings))
        try:
            env.reset(seed=17)
            initial_rng_state = json.dumps(env.np_random.bit_generator.state, sort_keys=True)
            env.reset()
            self.assertEqual(initial_rng_state, json.dumps(env.np_random.bit_generator.state, sort_keys=True))
            half = math.radians(settings.banana_yaw_degrees) / 2
            with reference.reset_context():
                reference.data.qpos[:16] = START_ARM_POSE
                reference.data.ctrl[:] = START_ARM_POSE
                reference.data.qvel[:] = 0
                reference.named.data.qpos['banana_joint'] = (*settings.banana_position, math.cos(half), 0, 0, math.sin(half))
            reference.step(nstep=round(settings.settle_seconds / float(reference.model.opt.timestep)))
            reference.forward()
            for step in range(10):
                np.testing.assert_array_equal(env.physics.data.qpos, reference.data.qpos)
                np.testing.assert_array_equal(env.physics.data.qvel, reference.data.qvel)
                np.testing.assert_array_equal(env.physics.data.ctrl, reference.data.ctrl)
                self.assertEqual(env.physics.data.time, reference.data.time)
                action = env._state_from_physical(START_ARM_POSE)
                action[8] += .01 * math.sin(step / 3)
                reference.data.ctrl[:] = env._action_to_control(action)
                reference.step(nstep=env.substeps)
                reference.forward()
                env.step(action)
        finally:
            env.close()
            reference.free()


class DisplayCameraTests(unittest.TestCase):
    def test_only_camera_nodes_are_added_and_binding_is_unique(self):
        settings = SceneConfig(controller_profile='position_pd')
        base = ET.fromstring(build_scene_xml(settings))
        shown = ET.fromstring(build_scene_xml(settings, display_cameras=True))
        parents = {child: node for node in shown.iter() for child in node}
        for name in CAMERAS:
            camera = shown.findall(f".//camera[@name='{name}']")
            self.assertEqual(len(camera), 1)
            parent = parents[camera[0]]
            if name == 'display_wrist_right':
                self.assertEqual(parent.get('name'), 'vx300s_right/gripper_link')
            else:
                self.assertEqual(parent.tag, 'worldbody')
            parent.remove(camera[0])
        def signature(node):
            return (node.tag, node.attrib, (node.text or '').strip(), tuple(signature(child) for child in node))
        self.assertEqual(signature(base), signature(shown))
        for flag in (1, 'true', None):
            with self.assertRaisesRegex(ValueError, 'display_cameras'):
                build_scene_xml(settings, display_cameras=flag)

    def test_compiled_physics_and_dynamics_unchanged_wrist_moves_with_gripper(self):
        settings = SceneConfig(controller_profile='position_pd')
        baseline = BananaBoxEnv(settings, max_episode_steps=20, render_observations=False)
        shown = BananaBoxEnv(settings, max_episode_steps=20, render_observations=False, display_cameras=True)
        try:
            for field in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'njnt', 'nsite', 'nlight'):
                self.assertEqual(getattr(baseline.physics.model, field), getattr(shown.physics.model, field))
            self.assertEqual(shown.physics.model.ncam, baseline.physics.model.ncam + 3)
            for field in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'body_gravcomp',
                          'geom_pos', 'geom_quat', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp',
                          'dof_damping', 'dof_frictionloss', 'jnt_range', 'actuator_gainprm', 'actuator_biasprm',
                          'actuator_ctrlrange', 'light_pos', 'light_diffuse'):
                np.testing.assert_array_equal(getattr(baseline.physics.model, field), getattr(shown.physics.model, field), err_msg=field)
            baseline.reset(seed=7)
            shown.reset(seed=7)
            for step in range(8):
                np.testing.assert_array_equal(baseline.physics.data.qpos, shown.physics.data.qpos)
                np.testing.assert_array_equal(baseline.physics.data.qvel, shown.physics.data.qvel)
                action = baseline._state_from_physical(START_ARM_POSE)
                action[8] += step * .002
                baseline.step(action)
                shown.step(action)
            model, data = shown.physics.model, shown.physics.data
            camera = model.name2id('display_wrist_right', 'camera')
            body = model.name2id('vx300s_right/gripper_link', 'body')
            self.assertEqual(model.cam_bodyid[camera], body)
            position, rotation = data.cam_xpos[camera].copy(), data.cam_xmat[camera].copy()
            shown.physics.named.data.qpos['vx300s_right/wrist_angle'] += .25
            shown.physics.forward()
            self.assertGreater(np.linalg.norm(data.cam_xpos[camera] - position), .001)
            self.assertGreater(np.linalg.norm(data.cam_xmat[camera] - rotation), .01)
            body_rotation = data.xmat[body].reshape(3, 3)
            expected_position = data.xpos[body] + body_rotation @ model.cam_pos[camera]
            np.testing.assert_allclose(data.cam_xpos[camera], expected_position, atol=1e-12)
            target = data.xpos[body] + body_rotation @ np.array((.14, 0., 0.))
            direction = (target - data.cam_xpos[camera]) / np.linalg.norm(target - data.cam_xpos[camera])
            self.assertGreater(np.dot(direction, -data.cam_xmat[camera].reshape(3, 3)[:, 2]), .999999)
        finally:
            baseline.close()
            shown.close()

    @unittest.skipUnless(os.environ.get('PI0_TEST_NATIVE_RENDER') == '1', 'Enable explicitly for native offscreen pixel checks')
    def test_native_display_views_never_change_top_policy_pixels(self):
        settings = SceneConfig(controller_profile='position_pd', randomize_banana_pose=True)
        baseline = BananaBoxEnv(settings, max_episode_steps=5)
        shown = BananaBoxEnv(settings, max_episode_steps=5, display_cameras=True)
        try:
            left, _ = baseline.reset(seed=SEEDS[0])
            right, _ = shown.reset(seed=SEEDS[0])
            np.testing.assert_array_equal(left['pixels']['top'], right['pixels']['top'])
            expected = right['pixels']['top'].copy()
            for name in CAMERAS:
                view = shown.render_camera(name, height=240, width=320)
                self.assertEqual(view.shape, (240, 320, 3))
                self.assertGreater(float(np.std(view)), 5)
                np.testing.assert_array_equal(shown.render(), expected)
                np.testing.assert_array_equal(shown._observation()['pixels']['top'], expected)
            a = aloha_observation(left, 'same prompt', resize=False)
            b = aloha_observation(right, 'same prompt', resize=False)
            np.testing.assert_array_equal(a['images']['cam_high'], b['images']['cam_high'])
            self.assertEqual(a['images']['cam_high'].shape, (3, 480, 640))
            with self.assertRaises(ValueError):
                shown.render_camera('missing_display_camera')
        finally:
            baseline.close()
            shown.close()


if __name__ == '__main__':
    unittest.main()
