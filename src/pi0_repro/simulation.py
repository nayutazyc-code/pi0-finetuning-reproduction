from dataclasses import asdict
import json
import logging
from pathlib import Path
import time

import gym_aloha  # Registers gym_aloha/AlohaTransferCube-v0.
import gymnasium
import imageio.v2 as imageio
import numpy as np

from .config import Config
from .diagnostics import task_progress
from .execution import BackgroundVideoWriter, ChunkExecutor, JointTargetLimiter, WindowClosed
from .observations import aloha_observation
from .recording import new_run, write_json
from .remote import RemotePi0
from .pacing import next_apply_deadline


BANANA_TASK = "pi0_repro/BananaBox-v0"


def _make_env(config: Config, output: Path):
    settings = config.simulation
    if settings.task == BANANA_TASK:
        from .banana_env import BananaBoxEnv
        from .scene import load_scene_config
        if settings.scene_config is None:
            raise ValueError("香蕉任务需要 scene_config，请使用 configs/banana_pi0.toml。")
        scene = load_scene_config(Path(settings.scene_config))
        env = BananaBoxEnv(scene, max_episode_steps=settings.max_steps,
                           render_observations=not settings.render_on_request,
                           display_cameras=settings.display_layout == "multiview",
                           reuse_policy_camera=settings.reuse_policy_camera,
                           collect_step_timings=settings.trace_control_timing)
        try:
            write_json(output / "scene.json", {"scene": asdict(scene), **env.description()})
            (output / "scene.xml").write_text(env.xml + "\n", encoding="utf-8")
        except BaseException:
            env.close()
            raise
        return env
    return gymnasium.make(
        settings.task, obs_type="pixels_agent_pos", render_mode="rgb_array",
        max_episode_steps=settings.max_steps,
    )


def run(config: Config, *, display: bool = False) -> Path:
    if config.model.profile != "aloha_sim":
        raise ValueError("Mac 仿真入口只运行 ALOHA；LIBERO 使用 infer-file 或 Linux 官方仿真。")
    settings = config.simulation
    if settings.defer_video_render and (display or settings.display) and not settings.display_in_worker:
        raise ValueError("Deferred top video requires headless operation or display_in_worker")
    if settings.task not in {"gym_aloha/AlohaTransferCube-v0", BANANA_TASK}:
        raise ValueError("仿真入口仅支持原传方块和本项目 BananaBox 场景。")
    banana_task = settings.task == BANANA_TASK
    if banana_task:
        logging.info("香蕉任务使用远端配置：%s", config.policy_config)
        if config.policy_config == "pi0_aloha_sim":
            logging.info("默认传方块权重尚未针对香蕉任务微调。")
    output = new_run(config, "run")
    plt = None
    figure = None
    artist = None
    viewer = None
    composer = None
    worker_display = bool((display or settings.display) and settings.display_in_worker)
    if settings.display_layout == "multiview":
        from .multiview import MultiViewComposer
        composer = MultiViewComposer()
        write_json(output / "display.json", {**composer.describe(), "pipeline": {
            "renderer": "spawned read-only physics clone", "policy_camera": "original top, main process",
            "display_shadows": False, "may_reuse_older_snapshot": True,
            "native_window_owner": "display process" if worker_display else "control process",
            "frame_receipts": "episode_NNN_multiview.jsonl records target step, source step and frame age",
        }})
    if (display or settings.display) and settings.display_backend == "glfw" and not worker_display:
        from .viewer import GlfwViewer
        viewer = GlfwViewer(width=composer.width, height=composer.height) if composer else GlfwViewer()
    if (display or settings.display) and viewer is None and not worker_display:
        import matplotlib.pyplot as plt
        plt.ion()
        figure, axes = plt.subplots()
        axes.set_title("Original pi0 · ALOHA simulation")
        axes.set_axis_off()

    # ROI policies send the full 640x480 observation; others resize locally.
    # No arbitrary clipping to [-1, 1]: ALOHA uses absolute joint targets.
    results = []
    try:
        with RemotePi0(config) as client:
            write_json(output / "server.json", client.metadata)
            env = _make_env(config, output)
            try:
                for episode in range(settings.episodes):
                    if episode and settings.async_inference:
                        client.close()
                        client = RemotePi0(config)
                        # This client is also explicitly closed by executor.close().
                    gym_obs, latest_info = env.reset(seed=settings.seed + episode)
                    if banana_task:
                        write_json(output / f"episode_{episode:03d}_scene.json", {
                            "scene": asdict(env.episode_scene), "reset": env.last_reset,
                            "scope": "Actual seeded episode reset; top policy camera is independent of display cameras.",
                        })
                    render_bridge = None
                    renderer_startup_seconds = 0.0
                    if composer is not None:
                        from .async_multiview import AsyncMultiViewRenderer
                        started_renderer = time.perf_counter()
                        render_bridge = AsyncMultiViewRenderer(env.xml, env.physics.get_state(), latest_info,
                                                              fps=max(settings.display_fps, settings.video_fps),
                                                              display=worker_display)
                        try:
                            render_bridge.wait_for_step(0)
                        except BaseException:
                            render_bridge.close()
                            raise
                        renderer_startup_seconds = time.perf_counter() - started_renderer
                    executor = ChunkExecutor(client, settings)
                    states, actions, rewards = [], [], []
                    requested_actions, submitted_actions, task_rows = [], [], []
                    blended_actions, source_indices, source_offsets = [], [], []
                    control_rows = []
                    display_rows = []
                    presentation_rows = []
                    stage_totals = dict(camera_seconds=0.0, camera_thread_cpu_seconds=0.0, observation_seconds=0.0,
                                        inference_dispatch_seconds=0.0, environment_step_seconds=0.0,
                                        display_seconds=0.0, video_submit_seconds=0.0,
                                        presentation_exchange_seconds=0.0)
                    camera_renders = 0
                    video_stride = 50 // settings.video_fps
                    frame_cache_step, frame_cache = -1, None
                    def camera_frame(current_step):
                        nonlocal frame_cache_step, frame_cache, camera_renders
                        if frame_cache_step != current_step:
                            start_camera = time.perf_counter()
                            start_camera_cpu = time.thread_time()
                            frame_cache = env.render() if settings.render_on_request else gym_obs["pixels"]["top"]
                            frame_cache_step = current_step
                            stage_totals["camera_seconds"] += time.perf_counter() - start_camera
                            stage_totals["camera_thread_cpu_seconds"] += time.thread_time() - start_camera_cpu
                            if settings.render_on_request:
                                camera_renders += 1
                        return frame_cache
                    presentation_step, presentation_cache, presentation_renders = -1, None, 0
                    presentation_receipt = None
                    def presentation_frame(current_step):
                        nonlocal presentation_step, presentation_cache, presentation_renders, presentation_receipt
                        if composer is None:
                            return camera_frame(current_step)
                        if presentation_step != current_step:
                            started = time.perf_counter()
                            render_bridge.submit(env.physics.get_state(), step=current_step, info=latest_info)
                            presentation_cache, presentation_receipt = render_bridge.poll(current_step=current_step)
                            if presentation_cache is None:
                                raise RuntimeError("Display renderer lost its initial frame")
                            stage_totals["presentation_exchange_seconds"] += time.perf_counter() - started
                            presentation_step = current_step
                            presentation_renders += 1
                        return presentation_cache
                    def policy_observation():
                        start_observation = time.perf_counter()
                        pixels = camera_frame(step)
                        adapted = aloha_observation({"pixels": {"top": pixels}, "agent_pos": gym_obs["agent_pos"]},
                                                    settings.prompt, resize=not config.model.send_full_resolution)
                        stage_totals["observation_seconds"] += time.perf_counter() - start_observation
                        return adapted
                    space = env.action_space
                    limiter = JointTargetLimiter(gym_obs["agent_pos"], settings,
                                                 getattr(space, "low", None), getattr(space, "high", None))
                    executor.limiter = limiter
                    executor.project_target = getattr(env, "project_action", None)
                    wall_start = time.perf_counter()
                    runtime_trace = None
                    if settings.trace_control_timing:
                        from .control_timing import ControlTimingTrace
                        runtime_trace = ControlTimingTrace(wall_start)
                    deadline = wall_start + 0.02
                    next_draw = wall_start
                    display_frames_before = getattr(viewer, "frames", 0)
                    display_seconds_before = getattr(viewer, "draw_seconds", 0.0)
                    title_updates_before = getattr(viewer, "title_updates", 0)
                    def advance_draw_clock():
                        nonlocal next_draw
                        next_draw += 1.0 / settings.display_fps
                        now_draw = time.perf_counter()
                        if next_draw < now_draw - 1.0 / settings.display_fps:
                            next_draw = now_draw + 1.0 / settings.display_fps
                    def draw_due():
                        if composer is not None:
                            return step % round(50 / settings.display_fps) == 0
                        # At the control rate, render each current state once instead
                        # of missing every other draw on a slightly early loop start.
                        return ((settings.control_pacing == "before_step" and settings.display_fps >= 50)
                                or time.perf_counter() >= next_draw)
                    late_steps = 0
                    max_lateness = 0.0
                    def idle():
                        if worker_display:render_bridge.check_status()
                        if viewer is not None:
                            viewer.poll()
                        if figure is not None:
                            if not plt.fignum_exists(figure.number):
                                raise WindowClosed()
                            figure.canvas.flush_events()
                    terminated = truncated = False
                    max_reward = 0.0
                    success = None
                    aborted = False
                    writer = None
                    deferred_video = None
                    deferred_receipt = None
                    multiview_writer = None
                    error = None
                    stem = f"episode_{episode:03d}"
                    try:
                        if settings.save_video:
                            if settings.defer_video_render:
                                from .deferred_video import DeferredTopVideo
                                deferred_video = DeferredTopVideo(env.physics, env.xml, max_steps=settings.max_steps, stride=video_stride)
                            else:
                                writer = (BackgroundVideoWriter(output / f"{stem}.mp4", fps=settings.video_fps)
                                          if settings.async_video else imageio.get_writer(output / f"{stem}.mp4", fps=settings.video_fps))
                            if settings.record_multiview:
                                multiview_writer = (BackgroundVideoWriter(output / f"{stem}_multiview.mp4", fps=settings.video_fps)
                                    if settings.async_video else imageio.get_writer(output / f"{stem}_multiview.mp4", fps=settings.video_fps))
                        for step in range(settings.max_steps):
                            if runtime_trace is not None: runtime_trace.step = step
                            loop_begin = time.perf_counter()
                            if worker_display:render_bridge.check_status()
                            stage_before = stage_totals.copy()
                            if viewer is not None:
                                viewer.poll()
                            if figure is not None and not plt.fignum_exists(figure.number):
                                aborted = True
                                break
                            if (writer is not None or deferred_video is not None) and step % video_stride == 0:
                                frame = camera_frame(step) if writer is not None else None
                                start_video = time.perf_counter()
                                if deferred_video is not None:
                                    deferred_video.capture(env.physics, step)
                                else:
                                    writer.append_data(frame)
                                if multiview_writer is not None:
                                    multiview_writer.append_data(presentation_frame(step))
                                    presentation_rows.append({"video_frame": len(presentation_rows),
                                        "target_control_step": step, **presentation_receipt})
                                stage_totals["video_submit_seconds"] += time.perf_counter() - start_video
                            if worker_display and draw_due():
                                presentation_frame(step)
                            if viewer is not None and draw_due():
                                start_display = time.perf_counter()
                                frame = presentation_frame(step)
                                viewer.draw(frame, f"pi0 | step={step} | grasped={latest_info.get('ever_grasped', False)} | inside={latest_info.get('inside_box', False)}")
                                stage_totals["display_seconds"] += time.perf_counter() - start_display
                                display_rows.append({"step": step, "wall_elapsed_seconds": time.perf_counter() - wall_start,
                                                     "draw_seconds": time.perf_counter() - start_display,
                                                     "viewer_stages": viewer.last_stage_timings.copy()})
                                advance_draw_clock()
                            if plt is not None and draw_due():
                                start_display = time.perf_counter()
                                frame = presentation_frame(step)
                                if banana_task:
                                    axes.set_title(
                                        f"pi0 banana trial | step={step} | "
                                        f"grasped={latest_info.get('ever_grasped', False)} | "
                                        f"inside={latest_info.get('inside_box', False)}"
                                    )
                                if artist is None:
                                    artist = axes.imshow(frame)
                                else:
                                    artist.set_data(frame)
                                plt.pause(0.001)
                                stage_totals["display_seconds"] += time.perf_counter() - start_display
                                display_rows.append({"step": step, "wall_elapsed_seconds": time.perf_counter() - wall_start,
                                                     "draw_seconds": time.perf_counter() - start_display})
                                advance_draw_clock()
                            start_dispatch = time.perf_counter()
                            raw, target, source_index, source_offset, blocked = executor.action(step, policy_observation, idle)
                            dispatch_seconds = time.perf_counter() - start_dispatch
                            stage_totals["inference_dispatch_seconds"] += dispatch_seconds
                            if step == 0:
                                # Initial request/warm-up is not a running-control deadline miss.
                                deadline = time.perf_counter() + 0.02
                            state = gym_obs["agent_pos"].copy()
                            start_limiter = time.perf_counter()
                            submitted = limiter.limit(target)
                            limiter_seconds = time.perf_counter() - start_limiter
                            pacing_wait = 0.0
                            pacing_requested = pacing_cpu = pacing_process_cpu = 0.0
                            if settings.real_time and settings.control_pacing == "before_step":
                                # Prepare rendering/inference first, then apply commands
                                # on the clock. Sleeping only after physics allows variable
                                # render cost to alternate short and long action intervals.
                                start_pacing = time.perf_counter()
                                start_pacing_cpu = time.thread_time()
                                start_pacing_process = time.process_time()
                                remaining = deadline - start_pacing
                                pacing_requested = max(0., remaining)
                                if remaining > 0:
                                    time.sleep(remaining)
                                pacing_wait = time.perf_counter() - start_pacing
                                pacing_cpu = time.thread_time() - start_pacing_cpu
                                pacing_process_cpu = time.process_time() - start_pacing_process
                            start_physics = time.perf_counter()
                            start_physics_cpu = time.thread_time()
                            start_physics_process = time.process_time()
                            gym_obs, reward, terminated, truncated, info = env.step(submitted)
                            physics_cpu_seconds = time.thread_time() - start_physics_cpu
                            physics_process_seconds = time.process_time() - start_physics_process
                            physics_seconds = time.perf_counter() - start_physics
                            stage_totals["environment_step_seconds"] += physics_seconds
                            info["rate_limited"] = bool(np.max(np.abs(submitted - raw)) > 1e-7)
                            executed = np.asarray(info.get("executed_action", submitted), dtype=np.float32)
                            limiter.update_executed(executed)
                            executor.record_executed(step, executed)
                            blended_actions.append(target.copy())
                            source_indices.append(source_index)
                            source_offsets.append(source_offset)
                            now = time.perf_counter()
                            deadline_event = start_physics if settings.control_pacing == "before_step" else now
                            lateness = max(0.0, deadline_event - deadline) if settings.real_time else 0.0
                            if lateness > 0.002:
                                late_steps += 1
                            max_lateness = max(max_lateness, lateness)
                            control_rows.append({"step": step, "wall_elapsed_seconds": now - wall_start,
                                                 "loop_begin_wall_seconds": loop_begin - wall_start,
                                                 "target_limiter_seconds": limiter_seconds,
                                                 "pre_apply_prepare_seconds": start_physics - loop_begin - pacing_wait,
                                                 "inference_blocked_seconds": blocked,
                                                 "inference_dispatch_seconds": dispatch_seconds,
                                                 "environment_step_seconds": physics_seconds,
                                                 "environment_step_thread_cpu_seconds": physics_cpu_seconds,
                                                 "environment_step_process_cpu_seconds": physics_process_seconds,
                                                 "environment_phases": getattr(env, "last_step_timings", None) if settings.trace_control_timing else None,
                                                 "camera_seconds": stage_totals["camera_seconds"] - stage_before["camera_seconds"],
                                                 "camera_thread_cpu_seconds": stage_totals["camera_thread_cpu_seconds"] - stage_before["camera_thread_cpu_seconds"],
                                                 "display_seconds": stage_totals["display_seconds"] - stage_before["display_seconds"],
                                                 "video_submit_seconds": stage_totals["video_submit_seconds"] - stage_before["video_submit_seconds"],
                                                 "presentation_exchange_seconds": stage_totals["presentation_exchange_seconds"] - stage_before["presentation_exchange_seconds"],
                                                 "action_apply_wall_seconds": start_physics - wall_start,
                                                 "scheduled_deadline_wall_seconds": deadline - wall_start if settings.real_time else None,
                                                 "deadline_phase": "action_apply" if settings.control_pacing == "before_step" else "loop_end",
                                                 "pacing_wait_seconds": pacing_wait,
                                                 "pacing_requested_seconds": pacing_requested,
                                                 "pacing_oversleep_seconds": max(0., pacing_wait-pacing_requested) if pacing_requested else 0.,
                                                 "pacing_thread_cpu_seconds": pacing_cpu,
                                                 "pacing_process_cpu_seconds": pacing_process_cpu,
                                                 "control_pacing": settings.control_pacing,
                                                 "deadline_lateness_seconds": lateness,
                                                 "source_prediction_index": source_index,
                                                 "source_prediction_offset": source_offset})
                            states.append(state)
                            requested_actions.append(raw.copy())
                            submitted_actions.append(submitted.copy())
                            executed = np.asarray(info.get("executed_action", submitted), dtype=np.float32)
                            actions.append(executed.copy())
                            rewards.append(float(reward))
                            latest_info = info
                            if banana_task:
                                task_rows.append({key: value for key, value in info.items() if key != "executed_action"})
                                if (step + 1) % 10 == 0 or info["is_success"]:
                                    logging.info(
                                        "banana step=%d ever_grasped=%s inside_box=%s released=%s "
                                        "stable=%s hold=%d/%d success=%s",
                                        step + 1, info["ever_grasped"], info["inside_box"], info["released"],
                                        info["stable"], info["success_hold_steps"], info["required_hold_steps"],
                                        info["is_success"],
                                    )
                            max_reward = max(max_reward, float(reward))
                            if "is_success" in info:
                                success = bool(info["is_success"])
                            if settings.real_time:
                                remaining = deadline - time.perf_counter()
                                if settings.control_pacing == "after_step" and remaining > 0:
                                    time.sleep(remaining)
                                # Keep the periodic clock anchored despite ordinary sleep jitter.
                                # After a whole missed period, reset rather than burst through
                                # multiple expired control deadlines.
                                wake = time.perf_counter()
                                if settings.control_pacing == "before_step":
                                    # A meaningful apply miss shifts the next apply slot;
                                    # do not compensate a long frame with a short action.
                                    deadline = next_apply_deadline(deadline, start_physics, wake, lateness,
                                                                  recover_overrun=settings.recover_control_overruns)
                                else:
                                    deadline = wake + 0.02 if wake >= deadline + 0.02 else deadline + 0.02
                                control_rows[-1]["deadline_update_wall_seconds"] = wake - wall_start
                                control_rows[-1]["next_deadline_wall_seconds"] = deadline - wall_start
                            control_rows[-1]["loop_finish_wall_seconds"] = time.perf_counter() - wall_start
                            control_rows[-1]["post_apply_seconds"] = control_rows[-1]["loop_finish_wall_seconds"] - (start_physics - wall_start)
                            if terminated or truncated:
                                break
                        if (writer is not None or deferred_video is not None) and actions:
                            if deferred_video is not None:
                                deferred_video.capture(env.physics, len(actions))
                            else:
                                writer.append_data(camera_frame(len(actions)))
                            if multiview_writer is not None:
                                # Only after control has ended: let the final
                                # display frame catch up, with a bounded wait.
                                render_bridge.submit(env.physics.get_state(), step=len(actions), info=latest_info)
                                render_bridge.wait_for_step(len(actions), timeout=3.)
                                multiview_writer.append_data(presentation_frame(len(actions)))
                                presentation_rows.append({"video_frame": len(presentation_rows),
                                    "target_control_step": len(actions), **presentation_receipt})
                    except WindowClosed:
                        aborted = True
                    except BaseException as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        control_wall_seconds = time.perf_counter() - wall_start
                        trace_receipt = runtime_trace.close() if runtime_trace is not None else None
                        # Ensure inference and encoder workers stop even when a window closes.
                        # Persist partial trajectories even if worker cleanup itself fails.
                        cleanup_error = None
                        try:
                            executor.close()
                        except BaseException as exc:
                            cleanup_error = exc
                        if render_bridge is not None:
                            try:
                                render_bridge.close()
                            except BaseException as exc:
                                cleanup_error = cleanup_error or exc
                        for encoder in (writer, multiview_writer):
                            try:
                                if encoder is not None:
                                    encoder.close()
                            except BaseException as exc:
                                cleanup_error = cleanup_error or exc
                        if deferred_video is not None:
                            try:
                                # Inference/display workers have stopped. This rendering is
                                # outside the measured active control interval.
                                deferred_receipt = deferred_video.finish(output / f"{stem}.mp4", fps=settings.video_fps,
                                                                         writer_factory=imageio.get_writer)
                            except BaseException as exc:
                                cleanup_error = cleanup_error or exc
                        if cleanup_error is not None and error is None:
                            error = f"{type(cleanup_error).__name__}: {cleanup_error}"
                        if trace_receipt is not None:
                            write_json(output / f"{stem}_runtime_trace.json", trace_receipt)
                        np.savez_compressed(
                            output / f"{stem}.npz",
                            state=np.asarray(states, dtype=np.float32).reshape(-1, 14),
                            action=np.asarray(actions, dtype=np.float32).reshape(-1, 14),
                            temporal_ensemble_counts=np.asarray(executor.temporal_ensemble_counts[:len(actions)], dtype=np.int32),
                            requested_action=np.asarray(requested_actions, dtype=np.float32).reshape(-1, 14),
                            submitted_action=np.asarray(submitted_actions, dtype=np.float32).reshape(-1, 14),
                            blended_action=np.asarray(blended_actions, dtype=np.float32).reshape(-1, 14),
                            action_prediction_index=np.asarray(source_indices, dtype=np.int32),
                            action_prediction_offset=np.asarray(source_offsets, dtype=np.int32),
                            predicted_chunks=np.stack(executor.chunks) if executor.chunks else np.empty((0, settings.replan_steps, 14), dtype=np.float32),
                            prediction_steps=np.asarray(executor.prediction_steps, dtype=np.int32),
                            chunk_transition_steps=np.asarray(executor.chunk_transition_steps, dtype=np.int32),
                            rtc_prefixes=np.stack(executor.rtc_prefixes) if executor.rtc_prefixes else np.empty((0, settings.replan_steps, 14), dtype=np.float32),
                            rtc_request_steps=np.asarray(executor.rtc_request_steps, dtype=np.int32),
                            rtc_valid_steps=np.asarray(executor.rtc_valid_steps, dtype=np.int32),
                            reward=np.asarray(rewards, dtype=np.float32),
                        )
                        (output / f"{stem}_latency.jsonl").write_text(
                            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in executor.timings),
                            encoding="utf-8",
                        )
                        (output / f"{stem}_control.jsonl").write_text(
                            "".join(json.dumps(row) + "\n" for row in control_rows), encoding="utf-8")
                        (output / f"{stem}_display.jsonl").write_text(
                            "".join(json.dumps(row) + "\n" for row in display_rows), encoding="utf-8")
                        if settings.record_multiview:
                            (output / f"{stem}_multiview.jsonl").write_text(
                                "".join(json.dumps(row) + "\n" for row in presentation_rows), encoding="utf-8")
                        if banana_task:
                            (output / f"{stem}_task.jsonl").write_text(
                                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_rows),
                                encoding="utf-8",
                            )
                        result = {
                            "episode": episode, "steps": len(actions), "seed": settings.seed + episode,
                            "success": success, "max_reward": max_reward,
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "local_step_limit_reached": len(actions) >= settings.max_steps,
                            "aborted_by_window_close": aborted, "error": error,
                            "execution": {**executor.stats, "control_wall_seconds": control_wall_seconds,
                                          "jerk_reference_projection_resets": getattr(limiter.jerk_reference, "projection_resets", 0),
                                          "simulated_seconds": len(actions) * 0.02,
                                          "real_time": settings.real_time, "late_steps_over_2ms": late_steps,
                                          "display_enabled": viewer is not None or figure is not None or worker_display,
                                          "display_backend": settings.display_backend,
                                          "display_layout": settings.display_layout,
                                          "presentation_frame_queries": presentation_renders,
                                          "renderer_startup_seconds": renderer_startup_seconds,
                                          "display_renderer": render_bridge.diagnostics() if render_bridge else None,
                                          "multiview_video": f"{stem}_multiview.mp4" if settings.record_multiview else None,
                                          "native_display_frames": (render_bridge.diagnostics()['native_display_frames'] if worker_display else getattr(viewer, "frames", 0) - display_frames_before),
                                          "native_display_draw_seconds": (render_bridge.diagnostics()['native_display_draw_seconds'] if worker_display else getattr(viewer, "draw_seconds", 0.0) - display_seconds_before),
                                          "native_title_updates": (render_bridge.diagnostics()['native_title_updates'] if worker_display else getattr(viewer, "title_updates", 0) - title_updates_before),
                                          "max_deadline_lateness_seconds": max_lateness,
                                          "video_queue_wait_seconds": getattr(writer, "wait_seconds", 0.0),
                                          "deferred_top_video": deferred_receipt,
                                          "stage_totals": stage_totals,
                                          "camera_renders": camera_renders if settings.render_on_request else len(actions) + 1,
                                          "video_fps": settings.video_fps,
                                          "control_pacing": settings.control_pacing,
                                          "request_interval_steps": settings.replan_steps - settings.prefetch_steps if settings.async_inference else settings.replan_steps},
                        }
                        if banana_task:
                            result["task_progress"] = task_progress(task_rows, actions)
                            result.update({
                                "ever_grasped": bool(latest_info.get("ever_grasped", False)),
                                "ever_lifted": bool(latest_info.get("ever_lifted", False)),
                                "final_inside_box": bool(latest_info.get("inside_box", False)),
                                "action_clipped_steps": sum(bool(row.get("action_clipped")) for row in task_rows),
                                "rate_limited_steps": sum(bool(row.get("rate_limited")) for row in task_rows),
                            })
                        results.append(result)
                        write_json(output / "summary.json", {"task": settings.task, "episodes": results})
                        logging.info("Episode result: %s", result)
                    if cleanup_error is not None:
                        raise cleanup_error
                    if aborted:
                        break
            finally:
                client.close()
                env.close()
    finally:
        if viewer is not None:
            viewer.close()
        if plt is not None:
            plt.close("all")
    return output
