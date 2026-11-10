"""Bounded, exact-state top-video capture with rendering after control ends.

Policy images still come from live physics. Captures contain the MuJoCo
integration state; they never step or render the control physics instance.
"""
import hashlib
from pathlib import Path
import time

import mujoco
import numpy as np


class DeferredTopVideo:
    def __init__(self, physics, xml: str, *, max_steps: int, stride: int):
        self.xml = xml
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.stride = stride
        capacity = (max_steps + stride - 1) // stride + 1
        size = mujoco.mj_stateSize(physics.model.ptr, self.spec)
        self.states = np.empty((capacity, size), dtype=np.float64)
        self.steps = np.empty(capacity, dtype=np.int64)
        self.count = 0
        self.capture_seconds = 0.

    def capture(self, physics, step: int):
        if self.count and step == self.steps[self.count-1]:
            return
        if self.count >= len(self.states):
            raise RuntimeError('Deferred video state capacity exceeded')
        if self.count and step < self.steps[self.count-1]:
            raise ValueError('Deferred video steps must be increasing')
        started = time.perf_counter()
        mujoco.mj_getState(physics.model.ptr, physics.data.ptr, self.states[self.count], self.spec)
        self.steps[self.count] = step
        self.count += 1
        self.capture_seconds += time.perf_counter()-started

    def finish(self, path: Path, *, fps: int, writer_factory):
        """Persist captured states before rendering; preserve them on encoder error."""
        from dm_control import mujoco as dm_mujoco
        started = time.perf_counter()
        path = Path(path)
        state_path = path.with_name(path.stem+'_video_states.npz')
        np.savez_compressed(state_path, integration_state=self.states[:self.count],
                            control_step=self.steps[:self.count], state_spec=int(self.spec))
        receipt = dict(schema='deferred-top-video-v1', frames=self.count,
            state_spec='mjSTATE_INTEGRATION', state_file=state_path.name,
            state_file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
            xml_sha256=hashlib.sha256(self.xml.encode()).hexdigest(),
            capture_seconds=self.capture_seconds, allocated_state_bytes=self.states.nbytes+self.steps.nbytes,
            frame_control_steps=self.steps[:self.count].tolist(), pixel_sha256=[],
            scope='Exact integration snapshots rendered after control; simulation-time video, not wall-clock playback.',
            complete=False)
        clone = writer = None
        try:
            clone = dm_mujoco.Physics.from_xml_string(self.xml)
            writer = writer_factory(path, fps=fps)
            for state in self.states[:self.count]:
                mujoco.mj_setState(clone.model.ptr, clone.data.ptr, state, self.spec)
                clone.forward()
                frame = clone.render(height=480, width=640, camera_id='top')
                receipt['pixel_sha256'].append(hashlib.sha256(frame.tobytes()).hexdigest())
                writer.append_data(frame)
            writer.close()
            writer = None
            receipt['complete'] = True
        except BaseException as exc:
            receipt['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            try:
                if writer is not None: writer.close()
            finally:
                if clone is not None: clone.free()
                receipt['post_control_seconds'] = time.perf_counter()-started
                from .recording import write_json
                write_json(path.with_name(path.stem+'_video_receipt.json'), receipt)
        return receipt
