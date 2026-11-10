"""Small spawned-worker IPC tests; no MuJoCo/OpenGL or policy connection."""
from queue import Empty
import time
import unittest

import numpy as np

from pi0_repro.async_multiview import AsyncMultiViewRenderer


def fake_renderer(xml,state,info,inputs,pixels,metadata,lock,stop,errors,fps):
    frame=np.frombuffer(pixels,np.uint8).reshape(720,1280,3)
    step,counter=0,0
    while not stop.is_set():
        if info.get('error'):
            errors.put('injected display failure');return
        if stop.wait(info.get('delay_seconds',0)):return
        counter+=1
        with lock:
            frame[:]=step%256
            metadata[:]=[step,info.get('simulation_time_seconds',step*.02),time.monotonic(),1,1,counter]
        while not stop.is_set():
            try:step,state,info=inputs.get(timeout=.02);break
            except Empty:continue


def fake_closed_window(xml,state,info,inputs,pixels,metadata,lock,stop,errors,fps,*,display,window_closed,viewer_stats):
    assert display
    window_closed.value=1
    stop.wait(.2)


class AsyncMultiviewTests(unittest.TestCase):
    def test_worker_window_close_uses_normal_abort_signal_and_owned_child_cleanup(self):
        from pi0_repro.execution import WindowClosed
        renderer=AsyncMultiViewRenderer('<unused/>',np.zeros(3),{},display=True,_worker_target=fake_closed_window)
        try:
            with self.assertRaises(WindowClosed):renderer.wait_for_step(0,timeout=8)
        finally:renderer.close()
        self.assertFalse(renderer._process.is_alive())
        self.assertTrue(renderer.diagnostics()['native_window_in_worker'])
    def test_slow_display_does_not_block_submit_or_poll_and_preserves_prior_frames(self):
        renderer=AsyncMultiViewRenderer('<unused/>',np.zeros(3),{},_worker_target=fake_renderer)
        try:
            first,receipt=renderer.wait_for_step(0,timeout=8)
            self.assertEqual(first.shape,(720,1280,3));self.assertEqual(receipt['source_control_step'],0)
            start=time.monotonic()
            renderer.submit(np.ones(3),step=2,info={'delay_seconds':.4,'simulation_time_seconds':.64})
            old,receipt=renderer.poll(current_step=2)
            self.assertLess(time.monotonic()-start,.15)
            self.assertEqual(receipt['source_control_step'],0)
            newer,receipt=renderer.wait_for_step(2,timeout=5)
            self.assertEqual(receipt['source_simulation_time_seconds'],.64)
            self.assertTrue((newer==2).all());self.assertTrue((first==0).all());self.assertTrue((old==0).all())
            for step in range(3,60):renderer.submit(np.zeros(3),step=step,info={'delay_seconds':.4})
            self.assertGreater(renderer.dropped,0)
        finally:renderer.close()
        self.assertFalse(renderer._process.is_alive())

    def test_errors_propagate_and_child_cleanup_is_bounded(self):
        renderer=AsyncMultiViewRenderer('<unused/>',np.zeros(3),{'error':True},_worker_target=fake_renderer)
        try:
            with self.assertRaises(RuntimeError):renderer.wait_for_step(0,timeout=8)
        finally:renderer.close()
        self.assertFalse(renderer._process.is_alive())


if __name__=='__main__':unittest.main()
