"""Run original-pi0 banana LoRA training using the existing server environment."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from banana_config import make_config, read_settings, require_openpi_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("describe", "train", "probe"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--exp-name", default="banana_pilot_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-updates", type=int, default=3)
    parser.add_argument("--probe-output", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    config_path = args.config or root / "configs/banana_finetune.toml"
    settings = read_settings(config_path)
    checkout = require_openpi_commit(root)
    dataset_home = root / settings["data"]["directory"]
    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)
    sys.path.insert(0, str(checkout / "src"))
    config = make_config(root, config_path, exp_name=args.exp_name, resume=args.resume)
    description = {"config_name": config.name, "model": asdict(config.model),
        "strict_lora_only": config.policy_metadata["strict_lora_only"], "dataset_home": str(dataset_home),
        "train_repo_id": settings["data"]["train_repo_id"], "assets_dir": str(config.assets_dirs),
        "micro_batch_size": settings["train"]["batch_size"],
        "gradient_accumulation_steps": settings["train"].get("gradient_accumulation_steps", 1),
        "effective_batch_size": config.batch_size,
        "training_step_unit": "optimizer updates",
        "checkpoint_dir": str(config.checkpoint_dir), "training": settings["train"], "model_options": settings["model"], "sampling": settings.get("sampling"), "resume": args.resume}
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)
    if args.command == "describe":
        return
    manifest = json.loads((dataset_home / "dataset_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("数据转换尚未完成。")
    repo = dataset_home / settings["data"]["train_repo_id"]
    mapping = json.loads((repo / "source_episodes.json").read_text())
    if mapping["source_split"] != "train" or any(r["split"] != "train" for r in mapping["episodes"]):
        raise ValueError("检测到非训练划分，拒绝训练。")
    norm_dir = config.assets_dirs / settings["data"]["train_repo_id"]
    if not (norm_dir / "norm_stats.json").is_file():
        raise FileNotFoundError("请先计算训练集归一化统计。")
    if not (root / settings["paths"]["initial_checkpoint"] / "params").exists():
        raise FileNotFoundError("缺少初始化检查点 params。")
    if settings["train"].get("require_initial_norm_match", False):
        initial_norm = root / settings["paths"]["initial_checkpoint"] / "assets" / settings["data"]["train_repo_id"] / "norm_stats.json"
        if json.loads(initial_norm.read_text()) != json.loads((norm_dir / "norm_stats.json").read_text()):
            raise ValueError("Initial checkpoint and training normalization values differ")
    run_record = root / "logs" / (args.exp_name + "-launch.json")
    run_record.parent.mkdir(exist_ok=True, parents=True)
    description["norm_stats_sha256"] = hashlib.sha256((norm_dir / "norm_stats.json").read_bytes()).hexdigest()
    description["dataset_manifest_sha256"] = hashlib.sha256((dataset_home / "dataset_manifest.json").read_bytes()).hexdigest()
    run_record.write_text(json.dumps(description, ensure_ascii=False, indent=2) + "\n")
    os.chdir(checkout)
    training = runpy.run_path(str(checkout / "scripts/train.py"), run_name="pi0_banana_train_entry")
    # The official logging helper expects an existing root logging handler.
    import logging
    logging.basicConfig(level=logging.INFO)
    if settings.get("sampling"):
        from banana_sampling import install_phase_sampler
        install_phase_sampler(root, settings)
    from banana_accumulation import install_accumulation
    step_fn = install_accumulation(training, settings["train"]["batch_size"],
                                    settings["train"].get("gradient_accumulation_steps", 1))
    if args.command == "probe":
        if args.probe_updates < 2 or args.probe_output is None:
            parser.error("probe requires --probe-output and at least two updates")
        from banana_probe import run_probe
        run_probe(training, config, step_fn, args.probe_updates, args.probe_output, description)
        return
    training["main"](config)


if __name__ == "__main__":
    main()
