"""Pixel/state/lifetime invariants for the opt-in persistent policy camera."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pi0_repro.config import PROJECT_ROOT, load_config


class CameraReuseConfigTests(unittest.TestCase):
    def test_reuse_is_opt_in_boolean_and_banana_only(self):
        source=(PROJECT_ROOT/'configs/banana_v6_random_multiview_pi0.toml').read_text()
        self.assertFalse(load_config(PROJECT_ROOT/'configs/banana_v6_random_multiview_pi0.toml').simulation.reuse_policy_camera)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.toml'
            p.write_text(source+'\nreuse_policy_camera = true\n')
            self.assertTrue(load_config(p).simulation.reuse_policy_camera)
            p.write_text(source+'\nreuse_policy_camera = "yes"\n')
            with self.assertRaises(ValueError):load_config(p)
            p.write_text('[simulation]\nreuse_policy_camera = true\n')
            with self.assertRaises(ValueError):load_config(p)
            p.write_text(source+'\ndisplay_in_worker = true\n')
            self.assertTrue(load_config(p).simulation.display_in_worker)
            p.write_text(source+'\ndisplay_in_worker = "yes"\n')
            with self.assertRaises(ValueError):load_config(p)
            p.write_text(source.replace('display_backend = "glfw"','display_backend = "matplotlib"')+'\ndisplay_in_worker = true\n')
            with self.assertRaises(ValueError):load_config(p)


@unittest.skipUnless(os.environ.get('PI0_TEST_NATIVE_RENDER')=='1','Requires native macOS rendering')
class PersistentCameraTests(unittest.TestCase):
    def test_pixels_physics_and_frame_ownership_across_steps_reset_and_close(self):
        from pi0_repro.banana_env import BananaBoxEnv
        from pi0_repro.scene import SceneConfig
        env=BananaBoxEnv(SceneConfig(controller_profile='position_pd',randomize_banana_pose=True),
                         max_episode_steps=30,render_observations=False,display_cameras=True,reuse_policy_camera=True)
        try:
            obs,_=env.reset(seed=2026102801)
            self.assertIsNone(env._policy_camera)
            old=env.render();old_copy=old.copy();camera=env._policy_camera
            for i in range(12):
                before=env.physics.get_state().copy()
                legacy=env.physics.render(height=480,width=640,camera_id='top')
                current=env.render()
                np.testing.assert_array_equal(legacy,current)
                np.testing.assert_array_equal(before,env.physics.get_state())
                np.testing.assert_array_equal(old,old_copy)
                self.assertIs(camera,env._policy_camera)
                action=obs['agent_pos'].copy();action[8]+=.1
                obs,*_=env.step(action)
            env.reset(seed=2026102805)
            self.assertIs(camera,env._policy_camera)
            np.testing.assert_array_equal(env.render(),env.physics.render(height=480,width=640,camera_id='top'))
            np.testing.assert_array_equal(old,old_copy)
        finally:env.close()
        self.assertIsNone(env._policy_camera);self.assertIsNone(env.physics)
        env.close()


if __name__=='__main__':unittest.main()
