"""Replay recorded controls locally to separate command quality from physics.

This is an open-loop diagnostic, never a new-model closed-loop evaluation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "repos/openpi/packages/openpi-client/src")]
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
from pi0_repro.config import Config, load_config
from pi0_repro.diagnostics import task_progress
from pi0_repro.execution import BackgroundVideoWriter, JointTargetLimiter
from pi0_repro.recording import new_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rollout", type=Path)
    source.add_argument("--demo", type=Path)
    parser.add_argument("--config", type=Path, help="Apply this target limiter; omitted means direct recorded-action replay")
    parser.add_argument("--action-source", choices=("executed", "requested"), default="executed")
    parser.add_argument("--video", action="store_true", help="Save a 25 fps video; no live window or cloud connection")
    args = parser.parse_args()
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig

    source_path = (args.rollout or args.demo).resolve()
    if args.rollout:
        scene_path = source_path / "scene.json"
        scene = json.loads(scene_path.read_text())["scene"]
        run_meta = json.loads((source_path / "run.json").read_text())
        seed = run_meta["config"]["simulation"]["seed"]
        trajectory = source_path / "episode_000.npz"
    else:
        scene_path = source_path / "episode.json"
        meta = json.loads(scene_path.read_text())
        scene, seed = meta["scene"], meta.get("reset_seed", 7)
        trajectory = source_path / "trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as data:
        actions = data["requested_action" if args.action_source == "requested" else "action"].copy()
        reference_states = data["state"].copy()
    if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Replay requires nonempty finite 14-dimensional actions")
    if reference_states.shape != actions.shape or not np.isfinite(reference_states).all():
        raise ValueError("Recorded states must be finite and match action shape")
    config = load_config(args.config) if args.config else Config()
    config.simulation.task = "pi0_repro/BananaBox-v0"
    config.simulation.seed = seed
    config.simulation.max_steps = len(actions)
    config.simulation.episodes = 1
    config.simulation.display = False
    config.simulation.save_video = args.video
    config.simulation.render_on_request = True
    config.simulation.video_fps = 25
    output = new_run(config, "replay-actions")
    write_json(output / "replay_source.json", {
        "scope": "Open-loop frozen-action diagnostic; not policy evaluation, training data, or an improvement claim",
        "source": str(source_path), "action_source": args.action_source,
        "limiter_config": str(args.config.resolve()) if args.config else None,
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scene_metadata_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
    })
    env = BananaBoxEnv(SceneConfig(**scene), max_episode_steps=len(actions), terminate_on_success=False,
                       render_observations=False)
    writer = None
    states, submitted, executed, tasks = [], [], [], []
    start = time.perf_counter()
    try:
        observation, _ = env.reset(seed=seed)
        initial_error = float(np.max(np.abs(observation["agent_pos"] - reference_states[0])))
        if initial_error > 1e-5:
            raise ValueError(f"Initial state differs by {initial_error}; recovery segments require their saved physical snapshot and are unsupported here")
        limiter = JointTargetLimiter(observation["agent_pos"], config.simulation,
                                     env.action_space.low, env.action_space.high) if args.config else None
        if args.video:
            writer = BackgroundVideoWriter(output / "episode_000.mp4", fps=25)
        write_json(output / "scene.json", {"scene": asdict(env.scene), **env.description()})
        (output / "scene.xml").write_text(env.xml)
        for step, action in enumerate(actions):
            states.append(observation["agent_pos"].copy())
            if writer and step % 2 == 0:
                writer.append_data(env.render())
            target = limiter.limit(action) if limiter else action
            observation, _, _, _, info = env.step(target)
            command = np.asarray(info.pop("executed_action"), np.float32)
            if limiter:
                limiter.update_executed(command)
            submitted.append(target.copy())
            executed.append(command)
            tasks.append(info)
        if writer:
            writer.append_data(env.render())
        np.savez_compressed(output / "episode_000.npz", state=np.asarray(states), action=np.asarray(executed),
                            requested_action=actions, submitted_action=np.asarray(submitted))
        (output / "episode_000_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in tasks))
        (output / "episode_000_latency.jsonl").write_text("")
        write_json(output / "summary.json", {"task": config.simulation.task, "scope": "open-loop replay diagnostic",
            "episodes": [{"episode": 0, "steps": len(executed), "success": bool(tasks[-1]["is_success"]),
                          "ever_success": any(r["is_success"] for r in tasks),
                          "source_state_max_error": float(np.max(np.abs(np.asarray(states) - reference_states))),
                          "initial_state_max_error": initial_error,
                          "wall_seconds": time.perf_counter() - start,
                          "task_progress": task_progress(tasks, executed)}]})
    finally:
        if writer:
            writer.close()
        env.close()
    print(output)


if __name__ == "__main__":
    main()
