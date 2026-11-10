"""An interrupted control step must remove its process-wide GC callback."""
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from pi0_repro.config import PROJECT_ROOT,load_config
from pi0_repro.scene import SceneConfig


class TraceCleanupTests(unittest.TestCase):
    def test_step_exception_removes_callback_and_retains_partial_record(self):
        from pi0_repro import simulation
        class Client:
            metadata={'model_family':'pi0'}
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):self.close()
            def close(self):pass
            def infer(self,obs):return np.zeros((100,14),np.float32),.1,{}
        class Env:
            action_space=SimpleNamespace(low=np.full(14,-2),high=np.full(14,2))
            episode_scene=SceneConfig(controller_profile='position_pd');last_reset={'seed':7}
            def reset(self,**kw):return {'pixels':{},'agent_pos':np.zeros(14,np.float32)},{}
            def render(self):return np.zeros((480,640,3),np.uint8)
            def step(self,action):
                gc.collect()
                raise RuntimeError('deliberate interrupted step')
            def close(self):pass
        cfg=load_config(PROJECT_ROOT/'configs/banana_v6_bridge20_pi0.toml');s=cfg.simulation
        s.display=False;s.save_video=False;s.async_inference=False;s.rtc_enabled=False
        s.real_time=False;s.control_pacing='after_step';s.trace_control_timing=True;s.max_steps=2
        original=list(gc.callbacks)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(simulation,'new_run',return_value=out),patch.object(simulation,'RemotePi0',Client),patch.object(simulation,'_make_env',return_value=Env()):
                with self.assertRaisesRegex(RuntimeError,'deliberate interrupted step'):simulation.run(cfg)
            self.assertEqual(gc.callbacks,original)
            trace=json.loads((out/'episode_000_runtime_trace.json').read_text())
            self.assertTrue(trace['gc_events'])
            self.assertEqual(trace['unfinished_gc_callbacks'],0)
            result=json.loads((out/'summary.json').read_text())['episodes'][0]
            self.assertIn('deliberate interrupted step',result['error'])
            self.assertTrue((out/'episode_000.npz').is_file())


if __name__=='__main__':unittest.main()
