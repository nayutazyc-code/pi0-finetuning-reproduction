"""Presentation pixels must never replace policy pixels or seeded reset receipts."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pi0_repro.config import PROJECT_ROOT, load_config
from pi0_repro.scene import SceneConfig


class MultiviewIntegrationTests(unittest.TestCase):
    def test_multiview_rejects_invalid_task_rate_and_recording_settings(self):
        source=(PROJECT_ROOT/'configs/banana_v6_random_multiview_pi0.toml').read_text()
        cases=[source.replace('display_fps = 25.0','display_fps = 17.0'),
               source.replace('save_video = true','save_video = false'),
               source.replace('task = "pi0_repro/BananaBox-v0"','task = "gym_aloha/AlohaTransferCube-v0"'),
               source.replace('record_multiview = true','record_multiview = "yes"')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.toml'
            for text in cases:
                path.write_text(text)
                with self.assertRaises(ValueError):load_config(path)

    def test_policy_pixels_and_original_video_remain_top_with_separate_multiview(self):
        self._verify_policy_isolation(worker_display=False)

    def test_worker_window_never_constructs_parent_viewer_or_changes_policy_pixels(self):
        self._verify_policy_isolation(worker_display=True)

    def test_deferred_top_video_keeps_worker_display_and_policy_pixels_separate(self):
        self._verify_policy_isolation(worker_display=True,defer_video=True)

    def _verify_policy_isolation(self,*,worker_display,defer_video=False):
        from pi0_repro import simulation
        observed=[];encoders=[]
        top=np.full((480,640,3),7,np.uint8)
        class Client:
            metadata={'model_family':'pi0'}
            def __init__(self,config):pass
            def __enter__(self):return self
            def __exit__(self,*args):self.close()
            def close(self):pass
            def infer(self,obs):
                observed.append(obs['images']['cam_high'].copy())
                return np.zeros((100,14),np.float32),.1,{}
        class Encoder:
            def __init__(self,path,**kwargs):
                self.path=path;self.frames=[];self.wait_seconds=0;self.closed=False;encoders.append(self)
            def append_data(self,frame):self.frames.append(frame.copy())
            def close(self):self.closed=True
        class Env:
            action_space=SimpleNamespace(low=np.full(14,-2),high=np.full(14,2))
            xml='<mujoco/>'
            physics=SimpleNamespace(get_state=lambda:np.zeros(45))
            def reset(self,*,seed):
                self.episode_scene=replace(SceneConfig(controller_profile='position_pd'),banana_position=(.18+seed*.0001,.48,.045))
                self.last_reset={'seed':seed,'sample':{'position':list(self.episode_scene.banana_position)}}
                return {'pixels':{},'agent_pos':np.zeros(14,np.float32)},{}
            def render(self):return top.copy()
            def render_camera(self,name,*,height,width):
                return np.full((height,width,3),{'display_front':50,'display_side':100,'display_wrist_right':150}[name],np.uint8)
            def step(self,action):
                return {'pixels':{},'agent_pos':np.zeros(14,np.float32)},0.,False,False,{'executed_action':action.tolist(),'is_success':False}
            def close(self):pass
        class Renderer:
            def __init__(self,xml,state,info,**kwargs):
                self.step=0
                assert kwargs['display']==worker_display
            def check_status(self):pass
            def submit(self,state,*,step,info):self.step=step
            def poll(self,*,current_step=None):
                from pi0_repro.multiview import MultiViewComposer
                frame=MultiViewComposer().compose(np.full((480,640,3),50,np.uint8),
                    np.full((240,320,3),100,np.uint8),np.full((240,320,3),150,np.uint8),step=self.step,info={})
                return frame,{'source_control_step':self.step,'frame_age_control_steps':0}
            def wait_for_step(self,step=0,**kwargs):return self.poll(current_step=step)
            def diagnostics(self):return {'test_renderer':True,'native_display_frames':4,
                                          'native_display_draw_seconds':.01,'native_title_updates':1}
            def close(self):pass
        class Deferred:
            def __init__(self,*a,**kw):self.count=0
            def capture(self,physics,step):self.count+=1
            def finish(self,path,*,fps,writer_factory):
                writer=writer_factory(path,fps=fps)
                for _ in range(self.count):writer.append_data(top)
                writer.close()
                return {'frames':self.count,'complete':True}
        cfg=load_config(PROJECT_ROOT/'configs/banana_v6_random_multiview_pi0.toml')
        cfg.simulation.episodes=2;cfg.simulation.seed=11;cfg.simulation.max_steps=4
        cfg.simulation.async_inference=False;cfg.simulation.rtc_enabled=False
        cfg.simulation.real_time=False;cfg.simulation.control_pacing='after_step';cfg.simulation.display=False
        cfg.simulation.display_in_worker=worker_display;cfg.simulation.display=worker_display
        cfg.simulation.defer_video_render=defer_video
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(simulation,'new_run',return_value=out),patch.object(simulation,'RemotePi0',Client),\
                 patch.object(simulation,'_make_env',return_value=Env()),patch.object(simulation,'BackgroundVideoWriter',Encoder),\
                 patch.object(simulation.imageio,'get_writer',Encoder),\
                 patch('pi0_repro.deferred_video.DeferredTopVideo',Deferred),\
                 patch('pi0_repro.async_multiview.AsyncMultiViewRenderer',Renderer),\
                 patch('pi0_repro.viewer.GlfwViewer',side_effect=AssertionError('Parent must not construct a window')):
                simulation.run(cfg)
            self.assertEqual(len(observed),2)
            for image in observed:np.testing.assert_array_equal(image,top.transpose(2,0,1))
            originals=[w for w in encoders if '_multiview' not in w.path.name]
            composites=[w for w in encoders if '_multiview' in w.path.name]
            self.assertEqual(len(originals),2);self.assertEqual(len(composites),2)
            for writer in originals:
                for image in writer.frames:np.testing.assert_array_equal(image,top)
            for writer in composites:
                self.assertTrue(writer.frames)
                self.assertEqual(writer.frames[0].shape,(720,1280,3))
            self.assertTrue(all(w.closed for w in encoders))
            for episode,seed in enumerate((11,12)):
                receipt=json.loads((out/f'episode_{episode:03d}_scene.json').read_text())
                self.assertEqual(receipt['reset']['seed'],seed)
                self.assertAlmostEqual(receipt['scene']['banana_position'][0],.18+seed*.0001)
                self.assertTrue((out/f'episode_{episode:03d}.npz').exists())


if __name__=='__main__':unittest.main()
