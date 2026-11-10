"""Read-only, spawned display renderer; control never waits for camera rendering.

Only a small physics-state snapshot crosses the input queue. A shared RGB
buffer holds the newest complete display frame. The policy camera and physics
integration remain in the caller. Display frames can lag and carry source-step
receipts; they must never be used as policy observations.
"""
import multiprocessing as mp
from queue import Empty, Full
import time
import traceback

import numpy as np


def _render_loop(xml, initial_state, initial_info, inputs, pixels, metadata, lock, stop, errors, fps,
                 *, display=False, window_closed=None, viewer_stats=None):
    physics=None
    cameras=[]
    viewer=None
    try:
        from dm_control import mujoco as dm_mujoco
        from dm_control.mujoco import engine
        from .multiview import MultiViewComposer
        physics=dm_mujoco.Physics.from_xml_string(xml)
        # Presentation-only quality settings; the caller's model/lighting and
        # policy render are never changed. Avoid expensive shadow passes here.
        physics.model.vis.quality.shadowsize=256
        physics.model.vis.quality.offsamples=0
        for name,height,width in [('display_front',480,640),('display_side',240,320),('display_wrist_right',240,320)]:
            cameras.append(engine.Camera(physics,height=height,width=width,camera_id=name))
        composer=MultiViewComposer()
        if display:
            from .viewer import GlfwViewer
            # A spawned process runs this entry on its own main thread, as
            # required by Cocoa/GLFW. No native window belongs to the controller.
            viewer=GlfwViewer(width=1280,height=720)
        shared=np.frombuffer(pixels,dtype=np.uint8).reshape(720,1280,3)
        payload=(0,initial_state,initial_info)
        next_render=0.
        counter=0
        while not stop.is_set():
            if viewer is not None:viewer.poll()
            if payload is None:
                try:payload=inputs.get(timeout=.05)
                except Empty:continue
            remaining=next_render-time.monotonic()
            if remaining>0 and stop.wait(remaining):break
            # Replace an older pending snapshot, never integrate the clone.
            while True:
                try:payload=inputs.get_nowait()
                except Empty:break
            step,state,info=payload
            physics.set_state(state)
            physics.data.time=float(info.get('simulation_time_seconds',step*.02))
            physics.forward()
            started=time.perf_counter()
            views=[camera.render(render_flag_overrides={'shadow':False}) for camera in cameras]
            render_ms=(time.perf_counter()-started)*1000
            started=time.perf_counter()
            frame=composer.compose(*views,step=step,info=info)
            compose_ms=(time.perf_counter()-started)*1000
            counter+=1
            with lock:
                shared[:]=frame
                metadata[:]=[step,physics.data.time,time.monotonic(),render_ms,compose_ms,counter]
            if viewer is not None:
                viewer.draw(frame,f"pi0 | step={step} | display process")
                viewer_stats[:]=[viewer.frames,viewer.draw_seconds,viewer.title_updates]
            payload=None
            next_render=time.monotonic()+max(0.,1/fps-(render_ms+compose_ms)/1000)
    except BaseException as exc:
        from .execution import WindowClosed
        if isinstance(exc,WindowClosed) and window_closed is not None:
            window_closed.value=1
        else:
            try:errors.put_nowait(traceback.format_exc())
            except Full:pass
    finally:
        if viewer is not None:viewer.close()
        for camera in cameras:
            camera._scene.free()
        if physics is not None:physics.free()


class AsyncMultiViewRenderer:
    width=1280
    height=720

    def __init__(self,xml,initial_state,initial_info,*,fps=25.,display=False,_worker_target=None):
        if not isinstance(fps,(int,float)) or isinstance(fps,bool) or not 0<fps<=50:
            raise ValueError('Renderer fps must be in (0,50]')
        state=np.asarray(initial_state,dtype=np.float64)
        if state.ndim!=1 or not np.isfinite(state).all():raise ValueError('Expected finite physics state')
        self._context=mp.get_context('spawn')
        self._inputs=self._context.Queue(maxsize=1)
        self._errors=self._context.Queue(maxsize=1)
        self._pixels=self._context.RawArray('B',self.height*self.width*3)
        self._metadata=self._context.RawArray('d',6)
        self._metadata[0]=-1
        self._lock=self._context.Lock()
        self._stop=self._context.Event()
        if type(display) is not bool:raise ValueError('Display must be boolean')
        self.display=display
        self._window_closed=self._context.RawValue('b',0)
        self._viewer_stats=self._context.RawArray('d',3)
        self._frame=None
        self._last_frame_id=-1
        self._receipt=None
        self.submitted=self.dropped=0
        self._process=self._context.Process(target=_worker_target or _render_loop,
            args=(xml,state.copy(),dict(initial_info),self._inputs,self._pixels,self._metadata,
                  self._lock,self._stop,self._errors,float(fps)),
            kwargs={'display':True,'window_closed':self._window_closed,'viewer_stats':self._viewer_stats} if display else {},
            daemon=True)
        self._process.start()

    def submit(self,state,*,step,info):
        if not isinstance(step,int) or isinstance(step,bool) or step<0:raise ValueError('Invalid source step')
        state=np.asarray(state,dtype=np.float64)
        if state.ndim!=1 or not np.isfinite(state).all():raise ValueError('Expected finite physics state')
        payload=(step,state.copy(),dict(info))
        try:self._inputs.put_nowait(payload);self.submitted+=1;return True
        except Full:
            try:self._inputs.get_nowait();self.dropped+=1
            except Empty:self.dropped+=1;return False
            try:self._inputs.put_nowait(payload);self.submitted+=1;return True
            except Full:self.dropped+=1;return False

    def check_status(self):
        if self._window_closed.value:
            from .execution import WindowClosed
            raise WindowClosed()
        try:error=self._errors.get_nowait()
        except Empty:error=None
        if error:raise RuntimeError('Display renderer failed: '+error)
        if not self._process.is_alive() and not self._stop.is_set():
            raise RuntimeError('Display renderer exited unexpectedly')

    def poll(self,*,current_step=None):
        self.check_status()
        if self._lock.acquire(False):
            try:
                step,simtime,produced,render_ms,compose_ms,frame_id=self._metadata[:]
                if step>=0 and int(frame_id)!=self._last_frame_id:
                    self._frame=np.frombuffer(self._pixels,dtype=np.uint8).reshape(self.height,self.width,3).copy()
                    self._last_frame_id=int(frame_id)
                    self._receipt={'source_control_step':int(step),'source_simulation_time_seconds':simtime,
                                   'produced_monotonic_seconds':produced,'worker_render_ms':render_ms,
                                   'worker_compose_ms':compose_ms,'worker_frame_id':int(frame_id)}
            finally:self._lock.release()
        receipt=dict(self._receipt) if self._receipt else None
        if receipt:
            receipt['frame_age_wall_seconds']=max(0.,time.monotonic()-receipt['produced_monotonic_seconds'])
            if current_step is not None:receipt['frame_age_control_steps']=current_step-receipt['source_control_step']
        return self._frame,receipt

    def wait_for_step(self,step=0,*,timeout=30.):
        deadline=time.monotonic()+timeout
        while True:
            frame,receipt=self.poll(current_step=step)
            if receipt and receipt['source_control_step']>=step:return frame,receipt
            if time.monotonic()>=deadline:raise TimeoutError('Display renderer did not deliver requested startup/terminal frame')
            time.sleep(.005)

    def diagnostics(self):
        return {'transport':'spawned read-only physics clone; latest-frame shared buffer',
                'submitted_snapshots':self.submitted,'dropped_snapshots':self.dropped,
                'published_frames':max(0,self._last_frame_id),'policy_frame_included':False,
                'display_shadows':False,'control_waits_for_rendering':False,
                'native_window_in_worker':self.display,
                'native_display_frames':int(self._viewer_stats[0]),
                'native_display_draw_seconds':float(self._viewer_stats[1]),
                'native_title_updates':int(self._viewer_stats[2]),
                'scope':'Display/recording may reuse an older snapshot; inspect per-frame source-step receipts.'}

    def close(self):
        self._stop.set()
        self._process.join(timeout=5)
        if self._process.is_alive():self._process.terminate();self._process.join(timeout=2)
        for queue in (self._inputs,self._errors):queue.cancel_join_thread();queue.close()
