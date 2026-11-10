"""Seeded stratified layout plans and outcome-transparent dataset indexes."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from .banana_expert import load_expert_config
from .recording import write_json
from .scene import load_scene_config


RANGES = ("banana_x", "banana_y", "banana_yaw_degrees", "box_x", "box_y")


def make_layout_plan(path: Path) -> tuple[Path, list[dict], dict]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if set(config) != {"collection"}:
        raise ValueError("数据集配置应仅包含 [collection]。")
    config = config["collection"]
    expected = {"expert_config", "seed", "train_layouts", "validation_layouts", "test_layouts", *RANGES}
    optional = {"table_palette", "light_scale"}
    if not expected <= set(config) or set(config) - expected - optional:
        raise ValueError(f"数据集配置字段不匹配：{set(config) ^ expected}")
    for key in ("seed", "train_layouts", "validation_layouts", "test_layouts"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} 应为非负整数。")
    counts = {split: config[split + "_layouts"] for split in ("train", "validation", "test")}
    count = sum(counts.values())
    if count < 1:
        raise ValueError("至少需要一个布局。")
    for key in RANGES:
        bounds = config[key]
        if (not isinstance(bounds, list) or len(bounds) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
                or bounds[0] >= bounds[1]):
            raise ValueError(f"{key} 应为两个递增有限数值。")
    palette = config.get("table_palette")
    if palette is not None and (not palette or any(len(color) != 4 or any(type(v) not in (int,float) or not 0 <= v <= 1 for v in color) for color in palette)):
        raise ValueError("table_palette must contain RGBA colors")
    light = config.get("light_scale", [1.0, 1.0])
    if len(light) != 2 or not 0.5 <= light[0] <= light[1] <= 1.5:
        raise ValueError("light_scale range must be within [0.5,1.5]")
    expert_path = Path(config["expert_config"]).expanduser()
    if not expert_path.is_absolute():
        expert_path = path.resolve().parent / expert_path
    expert = load_expert_config(expert_path)
    base = load_scene_config(Path(expert.scene_config))
    rng = np.random.default_rng(config["seed"])
    # Each dimension occupies every stratum once. This is not a full Cartesian grid.
    values = {}
    for key in RANGES:
        low, high = config[key]
        values[key] = low + (high - low) * (rng.permutation(count) + rng.uniform(0.1, 0.9, count)) / count
    labels = np.asarray([split for split, n in counts.items() for _ in range(n)])
    np.random.default_rng(config["seed"] + 1).shuffle(labels)
    plan = []
    for index in range(count):
        overrides = {
            "banana_position": [round(float(values["banana_x"][index]), 6),
                                round(float(values["banana_y"][index]), 6), base.banana_position[2]],
            "banana_yaw_degrees": round(float(values["banana_yaw_degrees"][index]), 6),
            "box_position": [round(float(values["box_x"][index]), 6),
                             round(float(values["box_y"][index]), 6), base.box_position[2]],
        }
        if palette is not None:
            overrides["table_rgba"] = palette[index % len(palette)]
        if "light_scale" in config:
            overrides["light_scale"] = round(float(rng.uniform(*light)), 6)
        digest = hashlib.sha256(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:12]
        plan.append({"layout_id": "layout-" + digest, "split": str(labels[index]),
                     "reset_seed": config["seed"] + index, "scene_overrides": overrides})
    if len({row["layout_id"] for row in plan}) != count:
        raise ValueError("发现重复布局，请扩大范围。")
    metadata = {"config_path": str(path.resolve()), "config": config,
                "sampling": "seeded stratified marginal sampling; no success-based resampling",
                "split_policy": "layout assignment fixed before acquisition; human review still required",
                "planned_counts": counts, "base_scene": asdict(base)}
    return expert_path.resolve(), plan, metadata


def write_dataset_index(directory: Path) -> Path:
    manifest = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    plan = manifest.get("layout_plan") or []
    rows = []
    for episode_path in sorted(directory.glob("episode_*/episode.json")):
        row = json.loads(episode_path.read_text(encoding="utf-8"))
        rows.append({"episode": row["episode"], "path": episode_path.parent.relative_to(directory).as_posix(),
                     "layout_id": row.get("layout_id"), "split": row.get("split"),
                     "steps": row["steps"], "automatic_success": row["automatic_success"],
                     "human_review": row["human_review"], "eligible_for_training": row["eligible_for_training"],
                     "review_status": row.get("review_status", row["human_review"]),
                     "reviewed_by": row.get("reviewed_by"), "assistant_review": row.get("assistant_review"),
                     "error": row.get("error")})
    counts = {}
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        counts[split] = {"planned": sum(row["split"] == split for row in plan),
                         "recorded": len(subset),
                         "successful_candidates": sum(row["automatic_success"] for row in subset),
                         "accepted": sum(row["eligible_for_training"] for row in subset),
                         "accepted_by_human": sum(row["eligible_for_training"] and row.get("reviewed_by") == "human" for row in subset),
                         "accepted_by_assistant": sum(row["eligible_for_training"] and row.get("reviewed_by") == "assistant" for row in subset),
                         "candidate_steps": sum(row["steps"] for row in subset if row["automatic_success"])}
    recorded = {row["layout_id"] for row in rows}
    output = directory / "dataset_index.json"
    write_json(output, {"schema": "banana-layout-dataset-v1", "source": manifest["source"],
                        "policy_model_used": bool(manifest.get("learner_states_used", False)), "counts": counts, "episodes": rows,
                        "unrecorded_layouts": [row for row in plan if row["layout_id"] not in recorded],
                        "training_ready": False,
                        "note": "Raw synchronized simulation records; review and LeRobot conversion are separate steps."})
    return output


def collect_dataset(config_path: Path, *, display: bool = False) -> Path:
    from .demonstrations import collect_demos
    expert_path, plan, metadata = make_layout_plan(config_path)
    return collect_demos(expert_path, display=display, layout_plan=plan, dataset_metadata=metadata)
