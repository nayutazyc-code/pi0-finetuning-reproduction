"""Prepare a versioned, fixed-normalization pi0 continuation; never launch it."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "outputs/v6_recovery_20261026"
    corrections = json.loads((out / "corrections.json").read_text())
    raw = Path(corrections["merged_source"])
    dataset = ROOT / "data/lerobot" / raw.name
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    mapping = json.loads((dataset / "local/pi0_banana_train/source_episodes.json").read_text())
    old_dataset = ROOT / "data/lerobot/20261024-185715-merged-ebdbe3"
    old_mapping = json.loads((old_dataset / "local/pi0_banana_train/source_episodes.json").read_text())
    base_count = len(old_mapping["episodes"])
    assert len(mapping["episodes"]) == base_count + 4
    for before, after in zip(old_mapping["episodes"], mapping["episodes"][:base_count]):
        for key in ("layout_id", "frames", "source_trajectory_sha256", "split"):
            assert before[key] == after[key], key
    for split in ("validation", "test"):
        old = json.loads((old_dataset / f"local/pi0_banana_{split}/source_episodes.json").read_text())["episodes"]
        new = json.loads((dataset / f"local/pi0_banana_{split}/source_episodes.json").read_text())["episodes"]
        assert [(r["layout_id"], r["source_trajectory_sha256"]) for r in old] == [(r["layout_id"], r["source_trajectory_sha256"]) for r in new]
    old_weights = np.asarray(json.loads((ROOT / "assets_quality/banana_sampling.json").read_text())["weights"], np.float64)
    base_frames = sum(r["frames"] for r in old_mapping["episodes"])
    assert len(old_weights) == base_frames
    new_entries = mapping["episodes"][base_count:]
    total_frames = sum(r["frames"] for r in mapping["episodes"])
    weights = (old_weights / old_weights.sum() * 0.8 * total_frames).tolist()
    for row in new_entries:
        assert row["split"] == "train"
        episode = raw / row["source_episode"]
        meta = json.loads((episode / "episode.json").read_text())
        assert meta["recovery"]["source_split"] == "train" and not meta["recovery"]["independent_layout"]
        assert meta["eligible_for_training"] and meta["review_status"] == "accepted"
        with np.load(episode / "trajectory.npz") as arrays:
            phase = arrays["phase"]
        local = np.where(np.isin(phase, ["descend", "close_gripper", "lower_into_box", "release"]), 2.0, 1.0)
        weights.extend((local / local.sum() * 0.2 * total_frames / len(new_entries)).tolist())
    assert len(weights) == total_frames and np.isfinite(weights).all() and min(weights) > 0
    assets = ROOT / "assets_v7_recovery"
    config_path = ROOT / "configs/banana_finetune_v7_recovery.toml"
    if assets.exists() or config_path.exists():
        raise FileExistsError("Keep existing preparation; use a distinct candidate name")
    assets.mkdir()
    sampler = assets / "banana_sampling.json"
    sampler.write_text(json.dumps({"repo_id": "local/pi0_banana_train", "split": "train", "weights": weights,
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "groups": {"base": {"episodes": base_count, "target_share": 0.8},
                   "new_recovery": {"episodes": len(new_entries), "target_share": 0.2}},
        "method": "Preserve original base weights at 80% mass; equal new-recovery segment mass at 20%, precision-phase weight 2."}, indent=2) + "\n")
    config = (ROOT / "outputs/long_chunk_20261025/selected_training.toml").read_text()
    replacements = {
        'directory = "data/lerobot/20261024-185715-merged-ebdbe3"': f'directory = "data/lerobot/{raw.name}"',
        'num_train_steps = 8000': 'num_train_steps = 1000',
        'seed = 46': 'seed = 47', 'warmup_steps = 400': 'warmup_steps = 100',
        'peak_lr = 0.00002': 'peak_lr = 0.000005', 'decay_lr = 0.000002': 'decay_lr = 0.000001',
        'save_interval = 1000': 'save_interval = 500', 'keep_period = 2000': 'keep_period = 500',
        'initial_checkpoint = "checkpoints/pi0_banana_lora/banana_v5_quality_norm1000/999"':
            'initial_checkpoint = "checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999"',
        'assets_base_dir = "assets_quality_norm"': 'assets_base_dir = "assets_v7_recovery_norm"',
        'weights_file = "assets_quality/banana_sampling.json"': 'weights_file = "assets_v7_recovery/banana_sampling.json"',
    }
    for before, after in replacements.items():
        assert config.count(before) == 1, before
        config = config.replace(before, after)
    config = config[config.index("[data]"):]
    config_path.write_text("# Original pi0 H100 v6 continuation; fresh optimizer, exact initializer normalization.\n" + config)
    plan = {"status": "prepared_not_launched", "dataset": str(dataset.relative_to(ROOT)),
        "base_dataset": str(old_dataset.relative_to(ROOT)), "raw_source": str(raw.relative_to(ROOT)),
        "train_episodes": len(mapping["episodes"]), "base_train_frames": base_frames,
        "new_recovery_frames": total_frames - base_frames, "train_frames": total_frames,
        "new_independent_layouts": 0, "validation_test_trajectory_lineage_unchanged": True,
        "config": str(config_path.relative_to(ROOT)), "sampler": str(sampler.relative_to(ROOT)),
        "experiment": "banana_v7_recovery1000", "optimizer_updates": 1000, "micro_batch": 8, "accumulation": 2,
        "expected_initial_norm_sha256": "6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269",
        "normalization_source": "Copy exact committed v6 norm file on cloud; local older assets are not substituted.",
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (out / "training_preparation.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
