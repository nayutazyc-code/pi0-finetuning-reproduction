"""Expert correction from replayed learner states, with strict train-only lineage."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from .recording import write_json
from .scene import SceneConfig


def restore_learner_state(env, request, output):
    rollout = Path(request["rollout"]).resolve()
    source = Path(request["source_episode"]).resolve()
    step = request["step"]
    meta = json.loads((source / "episode.json").read_text())
    scene = json.loads((rollout / "scene.json").read_text())["scene"]
    if meta.get("split") != "train" or request.get("source_split") != "train":
        raise ValueError("Recovery sources must have preassigned train membership")
    # Exact scene, including controller, appearance and geometry. Replay cannot
    # transplant a learner state into a different simulator or split.
    canonical = lambda value: json.loads(json.dumps(asdict(SceneConfig(**value))))
    if canonical(asdict(env.scene)) != canonical(scene) or canonical(scene) != canonical(meta["scene"]):
        raise ValueError("Recovery source scene differs from the recorded learner rollout")
    server = json.loads((rollout / "server.json").read_text())
    if server.get("model_family") != "pi0":
        raise ValueError("Recovery requires an actual original-pi0 learner rollout")
    archive = rollout / "episode_000.npz"
    with np.load(archive, allow_pickle=False) as data:
        if type(step) is not int or not 1 <= step < len(data["action"]):
            raise ValueError("Recovery step must be inside the recorded rollout")
        observation, info = env._observation(), env._task_info(update=False)
        for action in data["action"][:step]:
            observation, _, _, truncated, info = env.step(action)
            if truncated:
                raise ValueError("Source replay exceeded environment horizon")
        state_error = float(np.max(np.abs(observation["agent_pos"] - data["state"][step])))
        if state_error > 1e-5:
            raise ValueError(f"Learner replay state mismatch {state_error}; refusing correction")
    np.savez_compressed(output / "recovery_start.npz", qpos=env.physics.data.qpos.copy(),
        qvel=env.physics.data.qvel.copy(), ctrl=env.physics.data.ctrl.copy(),
        time=np.asarray(env.physics.data.time), ever_grasped=np.asarray(env.ever_grasped),
        ever_lifted=np.asarray(env.ever_lifted))
    # Keep physical state/velocities and task history. Only segment-local counters
    # restart; timestamps for the new expert segment begin at zero in collector.
    env.steps = env.hold_steps = 0
    env._needs_reset = False
    provenance = dict(**request, source_layout_id=meta["layout_id"],
        learner_checkpoint=server.get("checkpoint"), replay_state_max_error=state_error,
        learner_trajectory_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256((output / "recovery_start.npz").read_bytes()).hexdigest(),
        label_source="scripted_expert_after_learner_replay", independent_layout=False)
    write_json(output / "recovery.json", provenance)
    return observation, info, provenance


def collect_recovery(config_path, rollout, source_episode, step):
    from .demonstrations import collect_demos
    source_episode, rollout = Path(source_episode).resolve(), Path(rollout).resolve()
    meta = json.loads((source_episode / "episode.json").read_text())
    if meta.get("split") != "train":
        raise ValueError("Cannot collect corrections from validation/test layouts")
    request = dict(rollout=str(rollout), source_episode=str(source_episode), step=step, source_split="train")
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    plan = [dict(layout_id="recovery-" + digest, split="train", reset_seed=meta["reset_seed"],
                 scene_overrides=meta["scene"], recovery=request)]
    return collect_demos(Path(config_path), layout_plan=plan,
        dataset_metadata={"kind": "dagger_style_expert_correction", "source_split": "train",
                          "independent_layouts": 0, "learner_labels_used": False})
