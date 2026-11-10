"""Opt-in phase/GC envelopes, not a causal or exclusive CPU decomposition."""
import gc
import threading
import time


def measured_call(timings, name, function, *args, **kwargs):
    wall, cpu, process = time.perf_counter(), time.thread_time(), time.process_time()
    try:
        return function(*args, **kwargs)
    finally:
        end = time.perf_counter()
        timings[name] = dict(start_perf_counter=wall, end_perf_counter=end,
            wall_seconds=end-wall, thread_cpu_seconds=time.thread_time()-cpu,
            process_cpu_seconds=time.process_time()-process)


class ControlTimingTrace:
    def __init__(self, origin):
        self.origin = origin
        self.control_thread = threading.get_ident()
        self.step = 0
        self.events = []
        self.pending = {}
        self._callback = self._on_gc
        gc.callbacks.append(self._callback)

    def _on_gc(self, phase, info):
        identity = threading.get_ident()
        key = (identity, info['generation'])
        if phase == 'start':
            self.pending[key] = (time.perf_counter(),time.thread_time(),self.step)
        elif key in self.pending:
            wall,cpu,step = self.pending.pop(key)
            end = time.perf_counter()
            self.events.append(dict(start_wall_seconds=wall-self.origin,end_wall_seconds=end-self.origin,
                wall_seconds=end-wall,collector_thread_cpu_seconds=time.thread_time()-cpu,
                generation=info['generation'],start_control_step=step,end_control_step=self.step,
                collector_is_control_thread=identity==self.control_thread))

    def close(self):
        if self._callback in gc.callbacks:
            gc.callbacks.remove(self._callback)
        return dict(schema='control-runtime-trace-v1',origin_perf_counter=self.origin,
            gc_events=self.events,unfinished_gc_callbacks=len(self.pending),
            scope='GC callbacks may execute on inference threads. Overlap is not proof of main-thread blocking. Phase/process/thread CPU spans overlap; wall-minus-threadCPU does not identify one OS/GPU/GIL cause.')
