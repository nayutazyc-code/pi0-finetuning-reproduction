"""Run on the Linux GPU server, inside the pinned official openpi environment."""
import argparse
from pathlib import Path
import logging
import subprocess
import hashlib


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
CONFIGS = {"aloha_sim": ("pi0_aloha_sim", 14), "libero": ("pi0_libero", 7), "banana": ("pi0_banana_lora", 14)}


def main() -> None:
    parser = argparse.ArgumentParser(description="只加载原版 pi0，不使用 --env 默认模型")
    parser.add_argument("--profile", choices=CONFIGS, default="aloha_sim")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint", help="训练配置对应的检查点根目录；LIBERO 必填")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--rtc", action="store_true", help="Enable original-pi0 JAX generation-time prefix guidance")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.denoise_steps < 1:
        parser.error("端口需要在 1–65535 之间，denoise-steps 需要为正整数。")
    if args.profile in {"libero", "banana"} and not args.checkpoint:
        parser.error("LIBERO/香蕉需要对应的原版 pi0 微调检查点。")

    from openpi.models.pi0_config import Pi0Config
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as training_config

    config_name, action_dim = CONFIGS[args.profile]
    if args.profile == "banana":
        from banana_config import make_config
        root = args.project_root.expanduser().resolve()
        config = make_config(root, args.training_config or root / "configs/banana_finetune.toml", exp_name="serve")
    else:
        config = training_config.get_config(config_name)
    if not isinstance(config.model, Pi0Config) or config.model.pi05:
        raise RuntimeError("模型配置不是原版 Pi0Config(pi05=False)，拒绝启动。")
    import openpi
    checkout = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENPI_COMMIT:
        raise RuntimeError(f"openpi 版本不匹配：请切换到 {OPENPI_COMMIT}，当前 {commit}。")
    checkpoint = args.checkpoint or "gs://openpi-assets/checkpoints/pi0_aloha_sim"
    policy = policy_config.create_trained_policy(
        config, checkpoint, sample_kwargs={"num_steps": args.denoise_steps},
    )
    if args.rtc:
        if args.profile != "banana":
            parser.error("RTC is currently supported only for the banana JAX profile")
        from rtc_policy import RTCPolicy
        policy = RTCPolicy(policy)
    metadata = dict(policy.metadata)
    metadata.update({
        "project": "pi0-mac-reproduction",
        "model_family": "pi0",
        "policy_config": config_name,
        "action_dim": action_dim,
        "action_horizon": config.model.action_horizon,
        "denoise_steps": args.denoise_steps,
        "checkpoint": str(checkpoint),
        "training_config_file": str(args.training_config) if args.training_config else None,
        "openpi_commit": commit,
    })
    if args.rtc:
        metadata["rtc"] = {"schema": "pi0-rtc-v1", "guidance": "clean_estimate_vjp", "action_space": "native_aloha_absolute", "time_direction": "1_noise_to_0_clean", "adapter_sha256": hashlib.sha256((Path(__file__).parent / "rtc_policy.py").read_bytes()).hexdigest()}
    if args.profile == "banana":
        stats_path = Path(checkpoint) / "assets/local/pi0_banana_train/norm_stats.json"
        metadata["norm_stats_sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    logging.info("Original pi0: %s; checkpoint=%s", metadata, checkpoint)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
