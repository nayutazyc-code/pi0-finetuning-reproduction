"""Record real simulated expert rollouts with observation/action alignment."""
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import time
import uuid

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .banana_env import BananaBoxEnv
from .banana_expert import ScriptedBananaExpert, load_expert_config
from .config import PROJECT_ROOT
from .recording import write_json
from .scene import load_scene_config


class EpisodeFinished(Exception):
    pass


def collect_demos(config_path: Path, *, episodes: int = 1, display: bool = False,
                  layout_plan: list[dict] | None = None, dataset_metadata: dict | None = None) -> Path:
    if episodes < 1:
        raise ValueError("episodes 必须大于 0。")
    if layout_plan is not None:
        if not layout_plan:
            raise ValueError("布局计划不能为空。")
        episodes = len(layout_plan)
    config = load_expert_config(config_path)
    scene = load_scene_config(Path(config.scene_config))
    directory = PROJECT_ROOT / "data/banana_demos" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    directory.mkdir(parents=True, exist_ok=False)
    env = None
    plt = figure = artist = None
    summary = {"source": "simulation_scripted_expert_v2", "episodes": []}
    try:
        write_json(directory / "collection.json", {
            "source": summary["source"], "expert_config": asdict(config), "base_scene": asdict(scene),
            "fps": 50, "alignment": "image/state/timestamp before action; action is actual submitted target",
            "policy_model_used": False, "learner_states_used": any(row.get("recovery") for row in (layout_plan or [])),
            "human_review_required": True,
            "dataset": dataset_metadata,
            "layout_plan": layout_plan,
            "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("scene.py", "banana_env.py", "banana_expert.py", "demonstrations.py", "layouts.py")},
        })
        if display:
            import matplotlib.pyplot as plt
            plt.ion()
            figure, axes = plt.subplots(figsize=(8, 6))
            axes.set_axis_off()
        for episode in range(episodes):
            output = directory / f"episode_{episode:03d}"
            (output / "rgb").mkdir(parents=True)
            (output / "phase_views").mkdir()
            layout = None if layout_plan is None else layout_plan[episode]
            episode_scene = scene if layout is None else replace(scene, **layout["scene_overrides"])
            env = BananaBoxEnv(episode_scene, max_episode_steps=config.max_steps, terminate_on_success=False)
            (output / "scene.xml").write_text(env.xml + "\n", encoding="utf-8")
            if layout is None:
                (directory / "scene.xml").write_text(env.xml + "\n", encoding="utf-8")
            seed = 7 + episode if layout is None else layout["reset_seed"]
            observation, info = env.reset(seed=seed)
            recovery = None
            if layout and layout.get("recovery"):
                from .recovery import restore_learner_state
                observation, info, recovery = restore_learner_state(env, layout["recovery"], output)
            initial_pose = {"banana_position": env.physics.named.data.xpos["banana"].tolist(),
                            "banana_quaternion_wxyz": env.physics.named.data.xquat["banana"].tolist(),
                            "box_position": env.physics.named.data.xpos["open_box"].tolist()}
            write_json(output / "layout.json", {
                "layout": layout, "scene": asdict(episode_scene), "reset_seed": seed,
                "settled_initial_pose": initial_pose, "task": env.description(), "recovery": recovery,
                "scene_sha256": hashlib.sha256(env.xml.encode()).hexdigest(),
            })
            start_time = float(env.physics.data.time)
            state_rows, action_rows, requested_rows, next_rows, stamps, phases = [], [], [], [], [], []
            error = None
            interrupted = False
            last_phase = None
            lost_contact_steps = 0
            intentional_release = False
            writer = None
            expert = None
            event_file = None
            try:
                writer = imageio.get_writer(output / "preview.mp4", fps=50)
                event_file = (output / "steps.jsonl").open("w", encoding="utf-8")

                def execute(action, phase, target):
                    nonlocal observation, info, artist, last_phase, lost_contact_steps, intentional_release
                    if figure is not None and not plt.fignum_exists(figure.number):
                        raise KeyboardInterrupt
                    frame = observation["pixels"]["top"].copy()
                    state = observation["agent_pos"].copy()
                    timestamp = float(env.physics.data.time) - start_time
                    following, reward, terminated, truncated, result = env.step(action)
                    index = len(state_rows)
                    Image.fromarray(frame).save(output / "rgb" / f"{index:06d}.png")
                    writer.append_data(frame)
                    state_rows.append(state)
                    action_rows.append(result["executed_action"])
                    requested_rows.append(action.copy())
                    next_rows.append(following["agent_pos"].copy())
                    stamps.append(timestamp)
                    phases.append(phase)
                    row = {key: value for key, value in result.items() if key != "executed_action"}
                    row.update({"index": index, "timestamp": timestamp, "phase": phase,
                                "tcp_target": None if target is None else np.asarray(target).tolist()})
                    event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if last_phase != phase:
                        Image.fromarray(frame).save(output / "phase_views" / f"{index:06d}-{phase}.png")
                        last_phase = phase
                    observation, info = following, result
                    if phase in {"release", "retreat", "settle"}:
                        intentional_release = True
                    if phase in {"transport", "lower_into_box"} and result["released"]:
                        lost_contact_steps += 1
                    else:
                        lost_contact_steps = 0
                    if plt is not None:
                        if artist is None:
                            artist = axes.imshow(frame)
                        else:
                            artist.set_data(frame)
                        axes.set_title(f"Scripted expert | {phase} | step={index + 1}")
                        plt.pause(0.001)
                    if (index + 1) % 100 == 0:
                        logging.info("示范 step=%d phase=%s grasped=%s inside=%s", index + 1, phase,
                                     result["ever_grasped"], result["inside_box"])
                        event_file.flush()
                    if lost_contact_steps >= 3:
                        raise RuntimeError("计划松爪前已失去夹持，轨迹未通过受控放置质量门槛。")
                    if terminated:
                        if not intentional_release:
                            raise RuntimeError("物体在计划释放前已脱手；环境成功，但不作为受控抓放演示。")
                        raise EpisodeFinished
                    if truncated:
                        raise RuntimeError("达到采集步数上限，轨迹未成功。")

                expert = ScriptedBananaExpert(env, config, execute, recovery_mode=recovery is not None)
                expert.run()
                if not info["is_success"]:
                    raise RuntimeError("专家阶段已结束，但未达到完整抓放成功条件。")
            except EpisodeFinished:
                pass
            except KeyboardInterrupt:
                interrupted = True
                error = "interrupted"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logging.error("示范未成功：%s", error)
            finally:
                if expert is not None:
                    expert.close()
                if event_file is not None:
                    event_file.close()
                if writer is not None:
                    writer.append_data(observation["pixels"]["top"])
                    writer.close()
                Image.fromarray(observation["pixels"]["top"]).save(output / "terminal.png")
                Image.fromarray(env.physics.render(height=480, width=640, camera_id="angle")).save(output / "terminal_angle.png")
                np.savez_compressed(
                    output / "trajectory.npz",
                    state=np.asarray(state_rows, np.float32).reshape(-1, 14),
                    action=np.asarray(action_rows, np.float32).reshape(-1, 14),
                    requested_action=np.asarray(requested_rows, np.float32).reshape(-1, 14),
                    next_state=np.asarray(next_rows, np.float32).reshape(-1, 14),
                    timestamp=np.asarray(stamps, np.float64), phase=np.asarray(phases, dtype="U32"),
                )
                automatic_success = bool(info.get("is_success", False)) and error is None
                result = {
                    "episode": episode, "source": summary["source"], "steps": len(state_rows),
                    "layout_id": None if layout is None else layout["layout_id"],
                    "split": None if layout is None else layout["split"],
                    "scene": asdict(episode_scene), "settled_initial_pose": initial_pose,
                    "reset_seed": seed, "scene_sha256": hashlib.sha256(env.xml.encode()).hexdigest(),
                    "prompt": config.prompt, "fps": 50, "recovery": recovery,
                    "automatic_success": automatic_success, "environment_success": bool(info.get("is_success", False)),
                    "intentional_release": intentional_release, "human_review": "pending",
                    "eligible_for_training": False, "error": error,
                    "ever_grasped": bool(info.get("ever_grasped", False)),
                    "final_inside_box": bool(info.get("inside_box", False)),
                    "final_info": {key: value for key, value in info.items() if key != "executed_action"},
                }
                write_json(output / "episode.json", result)
                from .quality import audit_episode
                audit_episode(output)
                summary["episodes"].append(result)
                summary["successful_candidates"] = sum(row["automatic_success"] for row in summary["episodes"])
                write_json(directory / "summary.json", summary)
                env.close()
                env = None
                if layout_plan is not None:
                    from .layouts import write_dataset_index
                    write_dataset_index(directory)
                logging.info("候选演示已保存：%s；automatic_success=%s", output, automatic_success)
            if interrupted:
                break
    finally:
        if plt is not None and figure is not None:
            plt.close(figure)
        if env is not None:
            env.close()
    logging.info("采集结束：%s，成功候选 %d/%d；仍需回看视频确认。", directory,
                 summary.get("successful_candidates", 0), len(summary["episodes"]))
    return directory
