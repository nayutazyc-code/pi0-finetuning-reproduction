"""Time-aligned background inference and bounded joint-target updates.

All MuJoCo and Matplotlib operations stay on the main thread. Only one request
may be in flight; action i from an observation at s belongs to control step s+i.
"""
import copy
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from queue import Full, Queue
import threading
import logging
import time

import imageio.v2 as imageio
import numpy as np

JOINTS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPERS = np.array([6, 13])
CONTROL_DT = 0.02


class WindowClosed(Exception):
    pass


class ActionReference:
    """Pure timestamped reference; quintic weights crossfade valid old/new plans.

    Keeping the previous reference (including any ongoing crossfade) makes
    forecast and execution agree even when a second reply arrives mid-transition.
    Grippers always come from the current raw plan.
    """

    def __init__(self, origin, chunk, switch_step, previous=None, blend_steps=0):
        self.origin, self.chunk, self.switch_step = origin, chunk, switch_step
        available = previous.origin + len(previous.chunk) - switch_step if previous is not None else 0
        self.blend_steps = min(blend_steps, max(0, available))
        if self.blend_steps < 2:
            self.blend_steps = 0
        self.previous = previous if self.blend_steps else None

    def is_blending(self, step):
        return self.previous is not None and 0 <= step - self.switch_step < self.blend_steps

    def target(self, step):
        target = self.chunk[step - self.origin].copy()
        if self.is_blending(step):
            u = (step - self.switch_step) / (self.blend_steps - 1)
            weight = u ** 3 * (10 + u * (-15 + 6 * u))
            old = self.previous.target(step)
            target[JOINTS] = (1 - weight) * old[JOINTS] + weight * target[JOINTS]
        return target


class TemporalEnsembleReference:
    """Pure same-timestamp average of bounded, oldest-first accepted plans.

    ACT-style weights exp(-coefficient * rank) slightly favor older predictions.
    Expired plans never contribute. Grippers come from the newest valid plan.
    This reference is shared by execution and copied-limiter RTC forecasting.
    """

    blend_steps = 0

    def __init__(self, plans, coefficient):
        self.plans = tuple(plans)
        self.coefficient = coefficient

    def contributors(self, step):
        return tuple((origin, chunk) for origin, chunk in self.plans
                     if origin <= step < origin + len(chunk))

    def is_blending(self, step):
        return len(self.contributors(step)) > 1

    def target(self, step):
        plans = self.contributors(step)
        if not plans:
            raise ValueError("No valid temporal-ensemble plan at this timestamp")
        origin, newest = plans[-1]
        target = newest[step - origin].copy()
        if len(plans) > 1:
            weights = np.exp(-self.coefficient * np.arange(len(plans), dtype=np.float64))
            weights /= weights.sum()
            aligned = np.stack([chunk[step - origin, JOINTS] for origin, chunk in plans])
            target[JOINTS] = weights @ aligned
        return target


class ChunkExecutor:
    """Select a timestamped action; never append stale chunk prefixes."""

    def __init__(self, client, settings):
        self.client, self.settings = client, settings
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi0-infer") if settings.async_inference else None
        self.pending = None
        self.limiter = None
        self.project_target = None
        self.recent_delays = []
        self.request_metadata = {}
        self.rtc_prefixes, self.rtc_request_steps, self.rtc_valid_steps = [], [], []
        self.request_prefixes, self.executed_targets = {}, {}
        self.chunk = None
        self.origin = 0
        self.previous = None
        self.reference = None
        self.chunk_transition_steps = []
        self.temporal_ensemble_counts = []
        self.switch_step = 0
        self.last_target = None
        self.chunks, self.prediction_steps, self.timings = [], [], []
        self.stats = dict(requests_started=0, stale_chunks_discarded=0,
                          expired_prefix_steps_skipped=0, inference_underruns=0,
                          underrun_wait_seconds=0.0, startup_wait_seconds=0.0,
                          blended_steps=0)

    @staticmethod
    def _observe(observation):
        # Deferred suppliers are evaluated on the control/main thread, immediately
        # before submitting inference. Their pixels and state share this step.
        return observation() if callable(observation) else observation

    def _request(self, step, observation):
        observation = self._observe(observation)
        self.stats["requests_started"] += 1
        if self.settings.rtc_enabled and self.chunk is not None:
            available = self.origin + self.settings.replan_steps - step
            if available > 0:
                forecast = max([self.settings.rtc_delay_steps, *self.recent_delays[-10:]])
                delay = min(available, forecast)
                simulated = copy.deepcopy(self.limiter)
                prefix = []
                for offset in range(available):
                    target = self._target_for_step(step + offset)[1]
                    submitted = simulated.limit(target)
                    if self.project_target is not None:
                        submitted = self.project_target(submitted)
                    elif simulated.low is not None:
                        submitted = np.clip(submitted, simulated.low, simulated.high)
                    simulated.update_executed(submitted)
                    prefix.append(submitted)
                prefix.extend([prefix[-1]] * (self.settings.replan_steps - available))
                rtc = dict(schema="pi0-rtc-v1", prefix=np.asarray(prefix, np.float32),
                           valid_steps=available, delay_steps=delay,
                           blend_steps=self.settings.rtc_blend_steps,
                           guidance_max=self.settings.rtc_guidance_max)
                observation = {**observation, "rtc": rtc}
                self.request_metadata[step] = {k: v for k, v in rtc.items() if k != "prefix"}
                self.request_metadata[step]["prefix_sha256"] = hashlib.sha256(rtc["prefix"].tobytes()).hexdigest()
                self.request_prefixes[step] = rtc["prefix"].copy()
                self.rtc_prefixes.append(rtc["prefix"].copy())
                self.rtc_request_steps.append(step)
                self.rtc_valid_steps.append(available)
        if self.settings.rtc_enabled and "rtc" not in observation:
            # Startup or an exhausted overlap: reuse the guided graph with zero
            # temporal weights, rather than compiling a second sampler in a stall.
            initial = np.asarray(observation["state"], np.float32).copy()
            initial[GRIPPERS] = np.clip(initial[GRIPPERS], 0, 1)
            observation = {**observation, "rtc": dict(schema="pi0-rtc-v1",
                prefix=np.repeat(initial[None], self.settings.replan_steps, axis=0),
                valid_steps=self.settings.replan_steps, delay_steps=0, blend_steps=0,
                guidance_max=self.settings.rtc_guidance_max)}
        self.pending = (step, self.pool.submit(self.client.infer, observation))

    def _accept(self, origin, result, step):
        chunk, elapsed, response = result
        horizon = self.settings.replan_steps
        if len(chunk) < horizon:
            raise ValueError("服务端动作块长度小于 replan_steps。")
        age = step - origin
        request = self.request_metadata.pop(origin, None)
        usable = age < horizon
        if request is not None:
            prefix = self.request_prefixes.pop(origin)
            observed = [self.executed_targets[s] for s in range(origin, min(step, origin + request["valid_steps"]))]
            request["planned_vs_executed_prefix_max_error"] = (float(np.max(np.abs(np.asarray(observed) - prefix[:len(observed)]))) if observed else 0.0)
            # A guided replacement is meaningful only within the old plan's overlap.
            usable = usable and age < request["valid_steps"]
            if response.get("rtc", {}).get("schema") != "pi0-rtc-v1":
                raise ValueError("服务端没有返回 RTC 引导执行记录")
            self.recent_delays.append(max(age, math.ceil(elapsed / (CONTROL_DT * 1000))))
        self.timings.append(dict(step=origin, received_at_step=step,
                                 round_trip_ms=elapsed, chunk_shape=list(chunk.shape),
                                 skipped_prefix_steps=age, accepted=usable,
                                 prediction_index=len(self.chunks) if usable else None,
                                 rtc=response.get("rtc"), rtc_request=request,
                                 server_timing=response.get("server_timing", {}),
                                 policy_timing=response.get("policy_timing", {}),
                                 transport_timing=response.get("transport_timing", {}),
                                 client_transport=response.get("client_transport", {})))
        logging.info("pi0 request_step=%d received_step=%d request=%.1f ms skipped=%d accepted=%s", origin, step, elapsed, age, usable)
        if not usable:
            self.stats["stale_chunks_discarded"] += 1
            return
        self.previous = (self.origin, self.chunk) if self.chunk is not None else None
        self.origin, self.chunk, self.switch_step = origin, chunk.copy(), step
        if self.settings.temporal_ensemble:
            plans = self.reference.plans if self.reference is not None else ()
            plans = tuple((start, actions) for start, actions in plans if start + len(actions) > step)
            plans += ((origin, self.chunk),)
            self.reference = TemporalEnsembleReference(
                plans[-self.settings.temporal_ensemble_max_chunks:], self.settings.temporal_ensemble_coeff)
        elif self.settings.chunk_blend_shape == "quintic":
            self.reference = ActionReference(origin, self.chunk, step, self.reference,
                                             self.settings.chunk_blend_steps)
        self.chunk_transition_steps.append(self.reference.blend_steps if self.reference is not None else 0)
        self.chunks.append(chunk.copy())
        self.prediction_steps.append(origin)
        self.stats["expired_prefix_steps_skipped"] += age

    def action(self, step, observation, idle):
        blocked = 0.0
        if self.pool is None:
            if self.chunk is None or step >= self.origin + self.settings.replan_steps:
                start = time.perf_counter()
                self._accept(step, self.client.infer(self._observe(observation)), step)
                blocked = time.perf_counter() - start
                self.stats["requests_started"] += 1
                key = "startup_wait_seconds" if step == 0 else "underrun_wait_seconds"
                self.stats[key] += blocked
                if step:
                    self.stats["inference_underruns"] += 1
        else:
            if self.pending is not None and self.pending[1].done():
                origin, future = self.pending
                self.pending = None
                self._accept(origin, future.result(), step)
            if self.chunk is None or step >= self.origin + self.settings.replan_steps:
                start = time.perf_counter()
                if step:
                    self.stats["inference_underruns"] += 1
                while self.chunk is None or step >= self.origin + self.settings.replan_steps:
                    if self.pending is None:
                        self._request(step, observation)
                    origin, future = self.pending
                    while not future.done():
                        idle()
                        time.sleep(0.002)
                    self.pending = None
                    self._accept(origin, future.result(), step)
                blocked = time.perf_counter() - start
                key = "startup_wait_seconds" if step == 0 else "underrun_wait_seconds"
                self.stats[key] += blocked
            remaining = self.origin + self.settings.replan_steps - step
            if self.pending is None and remaining <= self.settings.prefetch_steps:
                self._request(step, observation)
        raw, target, blended = self._target_for_step(step)
        if blended:
            self.stats["blended_steps"] += 1
        self.temporal_ensemble_counts.append(
            len(self.reference.contributors(step)) if self.settings.temporal_ensemble else 0)
        self.last_target = target.copy()
        return raw, target, len(self.chunks) - 1, step - self.origin, blocked

    def _target_for_step(self, step):
        """Shared, side-effect-free reference used by actions and RTC forecasts."""
        offset = step - self.origin
        raw = self.chunk[offset].copy()
        if self.reference is not None:
            return raw, self.reference.target(step), self.reference.is_blending(step)
        target = raw.copy()
        blend = self.settings.chunk_blend_steps
        age = step - self.switch_step
        blended = bool(blend and age < blend and self.previous is not None)
        if blended:
            old_origin, old_chunk = self.previous
            old_offset = step - old_origin
            if 0 <= old_offset < self.settings.replan_steps:
                old = old_chunk[old_offset]
            else:
                old = self.last_target if self.last_target is not None else raw
            weight = (age + 1) / (blend + 1)
            target[JOINTS] = (1 - weight) * old[JOINTS] + weight * raw[JOINTS]
        return raw, target, blended

    def record_executed(self, step, target):
        self.executed_targets[step] = np.asarray(target, np.float32).copy()

    def close(self):
        self.stats["request_in_flight_at_end"] = self.pending is not None
        if self.pool is not None:
            # Closing the socket interrupts a blocked recv before joining its worker.
            self.client.close()
            self.pool.shutdown(wait=True, cancel_futures=True)


class JointTargetLimiter:
    """Velocity limits for all targets; optional acceleration limits for joints.

    Acceleration is defined on commanded joint targets, not physical joint qvel.
    Hard actuator bounds can override the acceleration constraint at a stop.
    """

    def __init__(self, state, settings, low=None, high=None):
        self.last = np.asarray(state, np.float64).copy()
        self.velocity = np.zeros_like(self.last)
        self.settings = settings
        self.low, self.high = low, high
        self.filter_position = self.last.copy()
        self.filter_velocity = np.zeros_like(self.last)
        self.filter_hz = settings.joint_target_filter_hz
        self.jerk_reference = None
        if settings.max_joint_jerk is not None:
            from .jerk_reference import JerkLimitedArmReference
            self.jerk_reference = JerkLimitedArmReference(
                self.last[JOINTS], settings.max_joint_velocity,
                settings.max_joint_acceleration, settings.max_joint_jerk, CONTROL_DT)
        self.max_delta = np.full(14, np.inf)
        if settings.max_joint_velocity is not None:
            self.max_delta[JOINTS] = settings.max_joint_velocity * CONTROL_DT
        if settings.max_gripper_velocity is not None:
            self.max_delta[GRIPPERS] = settings.max_gripper_velocity * CONTROL_DT

    def limit(self, target):
        acceleration = self.settings.max_joint_acceleration
        if acceleration is None:
            # Keep the historical float32 rate-limiter calculation for the baseline.
            last = self.last.astype(np.float32)
            maximum_delta = self.max_delta.astype(np.float32)
            return np.clip(np.asarray(target, np.float32), last - maximum_delta, last + maximum_delta)
        target = np.asarray(target, np.float64)
        if acceleration is not None and self.low is not None:
            target = np.clip(target, self.low, self.high)
        if self.filter_hz is not None:
            # Exact critically damped response to a piecewise-constant target.
            # Only arm joints are filtered; no averaging of gripper decisions.
            omega = 2 * math.pi * self.filter_hz
            decay = math.exp(-omega * CONTROL_DT)
            offset = self.filter_position[JOINTS] - target[JOINTS]
            coefficient = self.filter_velocity[JOINTS] + omega * offset
            self.filter_position[JOINTS] = target[JOINTS] + (offset + coefficient * CONTROL_DT) * decay
            self.filter_velocity[JOINTS] = (self.filter_velocity[JOINTS] - omega * coefficient * CONTROL_DT) * decay
            target = target.copy()
            target[JOINTS] = self.filter_position[JOINTS]
        if self.jerk_reference is not None:
            output = self.last + np.clip(target - self.last, -self.max_delta, self.max_delta)
            reference_velocity = self.filter_velocity[JOINTS] if self.filter_hz is not None else None
            output[JOINTS] = self.jerk_reference.advance(target[JOINTS], reference_velocity)
            if self.low is not None:
                output = np.clip(output, self.low, self.high)
            return output.astype(np.float32)
        delta = np.clip(target - self.last, -self.max_delta, self.max_delta)
        if acceleration is not None:
            dv = acceleration * CONTROL_DT
            velocity = np.clip(delta[JOINTS] / CONTROL_DT,
                               self.velocity[JOINTS] - dv, self.velocity[JOINTS] + dv)
            delta[JOINTS] = velocity * CONTROL_DT
        output = self.last + delta
        if acceleration is not None and self.low is not None:
            output = np.clip(output, self.low, self.high)
        return output.astype(np.float32)

    def update_executed(self, executed):
        executed = np.asarray(executed, np.float64)
        if self.jerk_reference is not None:
            self.jerk_reference.update_executed(executed[JOINTS])
        self.velocity = (executed - self.last) / CONTROL_DT
        if self.filter_hz is not None and self.jerk_reference is None:
            # Anti-windup: re-anchor a constrained filter axis to the command
            # actually submitted, also during copied RTC-prefix forecasting.
            constrained = JOINTS[np.abs(executed[JOINTS] - self.filter_position[JOINTS]) > 1e-6]
            self.filter_position[constrained] = executed[constrained]
            self.filter_velocity[constrained] = self.velocity[constrained]
        self.last = executed.copy()


class BackgroundVideoWriter:
    """Bounded lossless frame queue. Backpressure is explicit; frames never drop."""

    def __init__(self, path, fps=50):
        self.frames = Queue(maxsize=100)
        self.error = None
        self.wait_seconds = 0.0
        self.path, self.fps = path, fps
        self.thread = threading.Thread(target=self._run, name="pi0-video", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with imageio.get_writer(self.path, fps=self.fps) as writer:
                while True:
                    frame = self.frames.get()
                    if frame is None:
                        return
                    writer.append_data(frame)
        except BaseException as exc:
            self.error = exc

    def append_data(self, frame):
        # Copy because the caller may reuse its image buffer on the next step.
        item = np.array(frame, copy=True)
        start = time.perf_counter()
        while True:
            if self.error is not None:
                raise RuntimeError("后台视频编码失败") from self.error
            try:
                self.frames.put(item, timeout=0.02)
                break
            except Full:
                continue
        self.wait_seconds += time.perf_counter() - start

    def close(self):
        while self.thread.is_alive():
            if self.error is not None:
                break
            try:
                self.frames.put(None, timeout=0.05)
                break
            except Full:
                continue
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("后台视频编码失败") from self.error
