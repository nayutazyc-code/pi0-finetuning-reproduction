"""Fresh policy inputs, bounded snapshots, native RGB equality and errors."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pi0_repro.config import PROJECT_ROOT, load_config


class DeferredConfigTests(unittest.TestCase):
    def test_opt_in_and_incompatible_flows_rejected(self):
        source=(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml').read_text()
        self.assertFalse(load_config(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml').simulation.defer_video_render)
        good=source.replace('display = true','display = false')+'\ndefer_video_render = true\n'
        bad=[good.replace('defer_video_render = true','defer_video_render = "yes"'),
             good.replace('save_video = true','save_video = false'),
             good.replace('render_on_request = true','render_on_request = false'),
             good.replace('display = false','display = true'),
             good.replace('pi0_repro/BananaBox-v0','gym_aloha/AlohaTransferCube-v0')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.toml';path.write_text(good)
            self.assertTrue(load_config(path).simulation.defer_video_render)
            for text in bad:
                path.write_text(text)
                with self.assertRaises(ValueError):load_config(path)


@unittest.skipUnless(os.environ.get('PI0_TEST_NATIVE_RENDER')=='1','Requires native macOS rendering')
class DeferredNativeTests(unittest.TestCase):
    def test_simulation_uses_fresh_policy_pixels_and_same_video_with_fewer_live_renders(self):
        from unittest.mock import patch
        from pi0_repro import simulation
        import imageio.v2 as imageio
        observations=[]
        class Client:
            metadata={'model_family':'pi0'}
            def __init__(self,config):pass
            def __enter__(self):return self
            def __exit__(self,*a):self.close()
            def close(self):pass
            def infer(self,obs):
                observations.append(obs['images']['cam_high'].copy())
                return np.repeat(obs['state'][None],100,axis=0),.1,{}
        outputs=[]
        with tempfile.TemporaryDirectory() as tmp:
            for deferred in (False,True):
                out=Path(tmp)/str(deferred);out.mkdir();outputs.append(out)
                cfg=load_config(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml')
                s=cfg.simulation;s.max_steps=6;s.display=False;s.async_inference=False
                s.rtc_enabled=False;s.real_time=False;s.control_pacing='after_step';s.async_video=False
                s.video_fps=25;s.defer_video_render=deferred
                with patch.object(simulation,'new_run',return_value=out),patch.object(simulation,'RemotePi0',Client):
                    simulation.run(cfg)
            np.testing.assert_array_equal(observations[0],observations[1])
            with np.load(outputs[0]/'episode_000.npz') as a,np.load(outputs[1]/'episode_000.npz') as b:
                for key in ('state','action','requested_action'):
                    np.testing.assert_array_equal(a[key],b[key])
            with imageio.get_reader(outputs[0]/'episode_000.mp4') as a,imageio.get_reader(outputs[1]/'episode_000.mp4') as b:
                left=[frame for frame in a];right=[frame for frame in b]
                self.assertEqual(len(left),len(right));self.assertEqual(len(left),4)
                for fa,fb in zip(left,right):np.testing.assert_array_equal(fa,fb)
            base=json.loads((outputs[0]/'summary.json').read_text())['episodes'][0]['execution']
            cand=json.loads((outputs[1]/'summary.json').read_text())['episodes'][0]['execution']
            self.assertEqual(base['camera_renders'],4);self.assertEqual(cand['camera_renders'],1)
            self.assertEqual(cand['deferred_top_video']['frames'],4)

    def test_snapshot_pixels_physics_lifetime_and_partial_terminal(self):
        from pi0_repro.banana_env import BananaBoxEnv
        from pi0_repro.scene import SceneConfig
        from pi0_repro.deferred_video import DeferredTopVideo
        env=BananaBoxEnv(SceneConfig(controller_profile='position_pd',randomize_banana_pose=True),
                         max_episode_steps=12,render_observations=False)
        frames=[]
        class Writer:
            closed=False
            def __init__(self,*a,**kw):pass
            def append_data(self,frame):frames.append(frame.copy())
            def close(self):self.closed=True
        try:
            obs,_=env.reset(seed=2026102801)
            video=DeferredTopVideo(env.physics,env.xml,max_steps=12,stride=2)
            expected=[]
            # Stop at odd step7, preserving the terminal frame off the25Hz grid.
            for step in range(8):
                if step%2==0 or step==7:
                    before=env.physics.get_state().copy();ctrl=env.physics.data.ctrl.copy()
                    video.capture(env.physics,step)
                    np.testing.assert_array_equal(before,env.physics.get_state())
                    np.testing.assert_array_equal(ctrl,env.physics.data.ctrl)
                    expected.append(env.render())
                if step<7:
                    action=obs['agent_pos'].copy();action[8]+=.1
                    obs,*_=env.step(action)
            video.capture(env.physics,7)  # same terminal is not duplicated
            final=env.physics.get_state().copy()
            with tempfile.TemporaryDirectory() as tmp:
                receipt=video.finish(Path(tmp)/'episode.mp4',fps=25,writer_factory=Writer)
                self.assertTrue(receipt['complete']);self.assertEqual(receipt['frames'],5)
                self.assertEqual(receipt['frame_control_steps'],[0,2,4,6,7])
                self.assertEqual(len(frames),5)
                for a,b in zip(expected,frames):np.testing.assert_array_equal(a,b)
                self.assertEqual(receipt['pixel_sha256'],[hashlib.sha256(x.tobytes()).hexdigest() for x in expected])
                np.testing.assert_array_equal(final,env.physics.get_state())
                with np.load(Path(tmp)/'episode_video_states.npz',allow_pickle=False) as d:
                    self.assertEqual(len(d['integration_state']),5)
                with self.assertRaises(ValueError):video.capture(env.physics,5)
        finally:env.close()

    def test_encoder_error_retains_snapshots_and_failure_receipt(self):
        from pi0_repro.banana_env import BananaBoxEnv
        from pi0_repro.scene import SceneConfig
        from pi0_repro.deferred_video import DeferredTopVideo
        env=BananaBoxEnv(SceneConfig(controller_profile='position_pd'),max_episode_steps=2,render_observations=False)
        class Writer:
            def __init__(self,*a,**kw):pass
            def append_data(self,frame):raise RuntimeError('deliberate encoder failure')
            def close(self):pass
        try:
            env.reset(seed=7);video=DeferredTopVideo(env.physics,env.xml,max_steps=2,stride=1)
            for step in range(3):video.capture(env.physics,step)
            with self.assertRaises(RuntimeError):video.capture(env.physics,3)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'episode.mp4'
                with self.assertRaisesRegex(RuntimeError,'deliberate encoder failure'):
                    video.finish(path,fps=25,writer_factory=Writer)
                self.assertTrue((Path(tmp)/'episode_video_states.npz').is_file())
                receipt=json.loads((Path(tmp)/'episode_video_receipt.json').read_text())
                self.assertFalse(receipt['complete']);self.assertIn('deliberate',receipt['error'])
        finally:env.close()


if __name__=='__main__':unittest.main()
