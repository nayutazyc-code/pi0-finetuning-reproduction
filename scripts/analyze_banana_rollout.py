"""Inspect a completed banana model rollout and replay its recorded actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.scene import SceneConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    directory = args.run.resolve()
    run = json.loads((directory / "run.json").read_text())
    scene = json.loads((directory / "scene.json").read_text())["scene"]
    summary = json.loads((directory / "summary.json").read_text())
    events = [json.loads(line) for line in (directory / "episode_000_task.jsonl").read_text().splitlines()]
    timings = [json.loads(line) for line in (directory / "episode_000_latency.jsonl").read_text().splitlines()]
    with np.load(directory / "episode_000.npz", allow_pickle=False) as data:
        states = data["state"].copy()
        requested = data["requested_action"].copy()
        executed = data["action"].copy()
        submitted = data["submitted_action"].copy() if "submitted_action" in data else requested.copy()
    if len(states) != len(events) or not len(states):
        raise ValueError("Rollout state/event count mismatch")
    out = directory / "analysis"
    out.mkdir(exist_ok=True)
    selected = sorted(set(np.linspace(0, len(states), 9, dtype=int).tolist()))
    top = imageio.get_reader(directory / "episode_000.mp4")
    top_images = {i: Image.fromarray(top.get_data(i)) for i in selected}
    top.close()
    env = BananaBoxEnv(SceneConfig(**scene), max_episode_steps=len(states), terminate_on_success=False)
    env.render = lambda: np.zeros((480, 640, 3), np.uint8)
    side_images = {}
    max_state_error = max_banana_error = 0.0
    robot_banana_contact_steps = robot_box_contact_steps = 0
    site = env.physics.model.name2id("cali_right_site1", "site")
    distances = []
    try:
        obs, info = env.reset(seed=run["config"]["simulation"]["seed"])
        for i, action in enumerate(submitted):
            max_state_error = max(max_state_error, float(np.max(np.abs(obs["agent_pos"] - states[i]))))
            if i in selected:
                side_images[i] = Image.fromarray(env.physics.render(height=480, width=640, camera_id="angle"))
            obs, _, _, _, info = env.step(action)
            banana = np.array(info["banana_position"])
            max_banana_error = max(max_banana_error, float(np.linalg.norm(banana - events[i]["banana_position"])))
            distances.append(float(np.linalg.norm(env.physics.data.site_xpos[site] - banana)))
            hit_banana = hit_box = False
            for j in range(env.physics.data.ncon):
                c = env.physics.data.contact[j]
                pair = {int(c.geom1), int(c.geom2)}
                robot = bool(pair & env.robot_geoms)
                hit_banana |= robot and bool(pair & env.banana_collision_geoms)
                hit_box |= robot and bool(pair & env.box_geoms)
            robot_banana_contact_steps += int(hit_banana)
            robot_box_contact_steps += int(hit_box)
        side_images[len(states)] = Image.fromarray(env.physics.render(height=480, width=640, camera_id="angle"))
    finally:
        env.close()
    for name, frames in (("top", top_images), ("angle", side_images)):
        sheet = Image.new("RGB", (3 * 320, 3 * 266 + 30), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"pi0 banana rollout | {name} | chronological frames", fill="black")
        for j, (i, frame) in enumerate(sorted(frames.items())):
            x, y = j % 3 * 320, j // 3 * 266 + 30
            sheet.paste(frame.resize((320, 240)), (x, y + 24))
            draw.text((x + 5, y + 4), f"step {i}, simulation t={i/50:.2f}s", fill="black")
        sheet.save(out / f"{name}_overview.jpg", quality=94)
    side_images[len(states)].save(out / "terminal_angle.png")
    latency = np.array([r["round_trip_ms"] for r in timings])
    result = {
        "summary": summary, "frames": len(states), "inference_requests": len(timings),
        "finite_state_action": bool(np.isfinite(states).all() and np.isfinite(requested).all()),
        "requested_executed_max_difference": float(np.max(np.abs(requested - executed))),
        "submitted_executed_max_difference": float(np.max(np.abs(submitted - executed))),
        "rate_limited_steps": sum(bool(e.get("rate_limited", False)) for e in events),
        "maximum_submitted_joint_change_radians": float(np.max(np.abs(np.diff(submitted[:,[0,1,2,3,4,5,7,8,9,10,11,12]],axis=0)))),
        "first_request_ms": float(latency[0]), "warm_round_trip_median_ms": float(np.median(latency[1:])),
        "warm_round_trip_p95_ms": float(np.quantile(latency[1:], .95)),
        "first_action_minus_state": (requested[0] - states[0]).tolist(),
        "maximum_target_change_radians_excluding_grippers": float(np.max(np.abs(np.diff(requested[:, [0,1,2,3,4,5,7,8,9,10,11,12]], axis=0)))),
        "banana_displacement_from_first_event_m": float(np.max(np.linalg.norm(np.array([e["banana_position"] for e in events]) - events[0]["banana_position"], axis=1))),
        "robot_banana_contact_steps": robot_banana_contact_steps, "robot_box_contact_steps": robot_box_contact_steps,
        "minimum_right_tcp_to_banana_center_m": min(distances),
        "maximum_replay_pre_action_state_error": max_state_error, "maximum_replay_banana_position_error_m": max_banana_error,
        "source_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("episode_000.npz", "scene.xml", "server.json", "run.json")},
        "source_code_sha256": {name: hashlib.sha256((ROOT/"src/pi0_repro"/name).read_bytes()).hexdigest() for name in ("scene.py", "banana_env.py", "observations.py", "simulation.py")},
        "scope": "one recorded layout; not a held-out validation/test success-rate estimate; replay is evidence only and uses no policy calls"
    }
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
