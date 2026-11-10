"""Columnar CPU norm statistics using official ALOHA transforms and RunningStats."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from banana_config import read_settings, require_openpi_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    settings = read_settings(args.config or root / "configs/banana_finetune.toml")
    checkout = require_openpi_commit(root)
    sys.path.insert(0, str(checkout / "src"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import pyarrow.parquet as pq
    from openpi.policies import aloha_policy
    from openpi.shared import normalize

    home = root / settings["data"]["directory"]
    dataset = home / settings["data"]["train_repo_id"]
    meta = json.loads((dataset / "meta/info.json").read_text())
    mapping = json.loads((dataset / "source_episodes.json").read_text())
    if mapping["source_split"] != "train" or any(row["split"] != "train" for row in mapping["episodes"]):
        raise ValueError("归一化只能使用训练划分。")
    if meta["fps"] != 50:
        raise ValueError("需要 50Hz 训练数据。")
    states, action_chunks = [], []
    frame_count = 0
    for row in mapping["episodes"]:
        index = row["lerobot_episode_index"]
        path = dataset / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if state.shape != (row["frames"], 14) or action.shape != state.shape:
            raise ValueError("训练状态/动作维度不匹配。")
        # Reuse exactly the state/action operations used by official AlohaInputs(adapt_to_pi=True).
        state = np.stack([aloha_policy._decode_state(s, adapt_to_pi=True) for s in state])
        action = aloha_policy._encode_actions_inv(action, adapt_to_pi=True)
        indices = np.minimum(np.arange(len(action))[:, None] + np.arange(50)[None, :], len(action) - 1)
        chunk = action[indices].reshape(-1, 14)
        if not np.isfinite(state).all() or not np.isfinite(chunk).all():
            raise ValueError("ALOHA 适配后出现非有限数值。")
        states.append(state)
        action_chunks.append(chunk)
        frame_count += len(action)
    # One update over the full population avoids batch-order histogram rebucketing effects.
    # Mean/std use the same samples and 50-step/end-padding weighting as the official loader.
    values = {"state": np.concatenate(states), "actions": np.concatenate(action_chunks)}
    stats = {}
    report = {"repo_id": settings["data"]["train_repo_id"], "split": "train",
        "episodes": len(mapping["episodes"]), "frames": frame_count,
        "action_horizon": 50, "action_rows_in_statistics": frame_count * 50,
        "adapt_to_pi": True, "use_delta_joint_actions": False, "use_quantile_norm": False,
        "method": "official ALOHA numeric transforms and RunningStats; complete-population CPU columnar input",
        "quantile_note": "histogram quantiles may differ from the official minibatch script's rebucketing; original pi0 uses mean/std",
        "statistics": {}, "source_hashes": {}}
    for key, array in values.items():
        running = normalize.RunningStats()
        running.update(array)
        stats[key] = running.get_statistics()
        empirical_std = np.std(array, axis=0)
        report["statistics"][key] = {"count": len(array), "minimum_std": float(stats[key].std.min()),
            "near_constant_dimensions": np.flatnonzero(stats[key].std < 1e-5).tolist(),
            "direct_mean_max_error": float(np.max(np.abs(stats[key].mean - np.mean(array, axis=0)))),
            "direct_std_max_error": float(np.max(np.abs(stats[key].std - empirical_std))),
            "mean": stats[key].mean.tolist(), "std": stats[key].std.tolist()}
    out = root / settings["paths"]["assets_base_dir"] / settings["model"]["config_name"] / settings["data"]["train_repo_id"]
    if (out / "norm_stats.json").exists():
        raise FileExistsError("统计已存在；请保留旧统计并明确决定是否重新计算，避免训练/部署不一致。")
    normalize.save(out, stats)
    for name in ("src/openpi/policies/aloha_policy.py", "src/openpi/shared/normalize.py"):
        report["source_hashes"][name] = hashlib.sha256((checkout / name).read_bytes()).hexdigest()
    report["norm_stats_sha256"] = hashlib.sha256((out / "norm_stats.json").read_bytes()).hexdigest()
    report["std_floor_applied"] = False
    (out / "normalization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(out), "frames": frame_count, "action_rows": frame_count * 50,
                      "statistics": report["statistics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
