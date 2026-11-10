import argparse
import logging
import os
from pathlib import Path

from .config import load_config, PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="原版 pi0：Mac 仿真与远程 GPU 推理客户端")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("run", "运行传方块或香蕉入盒仿真，并向原版 pi0 请求动作"),
        ("connect", "只连接并读取服务器模型元数据，不发送观测"),
        ("infer-file", "发送一个真实 NPZ 观测并保存动作，不执行动作"),
    ):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/aloha_sim.toml")
        if command == "run":
            sub.add_argument("--display", action="store_true", help="通过配置的显示后端显示仿真相机画面")
            seeds = sub.add_mutually_exclusive_group()
            seeds.add_argument("--seed", type=int, help="设置并记录本次可复现的重置种子")
            seeds.add_argument("--fresh-seed", action="store_true", help="本次生成新种子；配合随机场景每次生成新位姿")
            sub.add_argument("--episodes", type=int, help="覆盖配置中的仿真次数")
            sub.add_argument("--port", type=int, help="覆盖本地推理转发端口")
        if command == "infer-file":
            sub.add_argument("--observation", type=Path, required=True)
    scene_parser = commands.add_parser("scene", help="本地预览香蕉与开口盒子，不连接 π0")
    scene_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/banana_box.toml")
    scene_parser.add_argument("--camera", choices=("top", "angle"), help="选择俯视或斜视相机")
    scene_parser.add_argument("--snapshot-only", action="store_true", help="只渲染并保存一张图片后退出")
    demo_parser = commands.add_parser("collect-demo", help="使用本地脚本专家采集香蕉演示，无需真机或模型")
    demo_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/banana_expert.toml")
    demo_parser.add_argument("--episodes", type=int, default=1)
    demo_parser.add_argument("--display", action="store_true")
    dataset_parser = commands.add_parser("collect-dataset", help="按固定种子采集不同布局并预先划分数据集")
    dataset_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/banana_dataset.toml")
    dataset_parser.add_argument("--display", action="store_true")
    recovery_parser = commands.add_parser("collect-recovery", help="Replay a train-only learner state and record expert corrections")
    recovery_parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/banana_quality_expert.toml")
    recovery_parser.add_argument("--rollout", type=Path, required=True)
    recovery_parser.add_argument("--source-episode", type=Path, required=True)
    recovery_parser.add_argument("--step", type=int, required=True)
    quality_parser = commands.add_parser("audit-demo", help="Record structural checks and advisory phase motion metrics")
    quality_parser.add_argument("--episode", type=Path, required=True)
    review_parser = commands.add_parser("review-demo", help="回看视频后，由用户明确接受或拒绝一条候选演示")
    review_parser.add_argument("--episode", type=Path, required=True)
    review_parser.add_argument("--reviewer", choices=("human", "assistant"), default="human")
    review_parser.add_argument("--reason")
    review_parser.add_argument("--evidence", type=Path)
    decision = review_parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", action="store_true")
    decision.add_argument("--reject", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Set this before importing gym_aloha/dm_control, and do not use Linux EGL on a Mac.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    try:
        if args.command == "collect-recovery":
            from .recovery import collect_recovery
            logging.info("Recovery candidates: %s", collect_recovery(args.config, args.rollout, args.source_episode, args.step))
            return
        if args.command == "audit-demo":
            import json
            from .quality import audit_episode
            print(json.dumps(audit_episode(args.episode), indent=2))
            return
        if args.command == "review-demo":
            from .demo_review import review_demo
            path = review_demo(args.episode, accept=args.accept, reviewer=args.reviewer,
                               reason=args.reason, evidence_path=args.evidence)
            logging.info("人工复核结果已保存：%s", path)
            return
        if args.command == "collect-dataset":
            from .layouts import collect_dataset
            directory = collect_dataset(args.config, display=args.display)
            logging.info("多布局数据集目录：%s", directory)
            return
        if args.command == "collect-demo":
            from .demonstrations import collect_demos
            directory = collect_demos(args.config, episodes=args.episodes, display=args.display)
            logging.info("演示采集目录：%s", directory)
            return
        if args.command == "scene":
            from .scene import preview_scene
            directory = preview_scene(args.config, camera=args.camera, snapshot_only=args.snapshot_only)
            logging.info("场景文件已保存：%s", directory)
            return
        config = load_config(args.config)
        if args.command == "run":
            if args.fresh_seed:
                import secrets
                config.simulation.seed = secrets.randbits(31)
            elif args.seed is not None:
                if not 0 <= args.seed < 2**32:
                    raise ValueError("seed must be in [0, 2**32)")
                config.simulation.seed = args.seed
            if args.episodes is not None:
                if not 1 <= args.episodes <= 100:
                    raise ValueError("episodes must be in [1,100]")
                config.simulation.episodes = args.episodes
            if args.port is not None:
                if not 1 <= args.port <= 65535:
                    raise ValueError("port must be in [1,65535]")
                config.connection.port = args.port
            from .simulation import run
            directory = run(config, display=args.display)
        elif args.command == "connect":
            import json
            from .remote import RemotePi0
            with RemotePi0(config) as client:
                print(json.dumps(client.metadata, ensure_ascii=False, indent=2))
            return
        else:
            import numpy as np
            from .observations import from_npz
            from .recording import new_run, write_json
            from .remote import RemotePi0
            observation = from_npz(args.observation, config.model.profile, config.simulation.prompt, resize=not config.model.send_full_resolution)
            directory = new_run(config, "infer-file")
            with RemotePi0(config) as client:
                actions, elapsed_ms, response = client.infer(observation)
                np.save(directory / "actions.npy", actions, allow_pickle=False)
                write_json(directory / "response.json", {
                    "server": client.metadata, "observation_file": str(args.observation.resolve()),
                    "action_shape": list(actions.shape), "round_trip_ms": elapsed_ms,
                    "server_timing": response.get("server_timing", {}),
                })
        logging.info("运行记录已保存：%s", directory)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        if args.command in {"scene", "collect-demo", "collect-dataset", "review-demo"}:
            logging.error("请查看 docs/仿真专家演示采集.md；此入口使用本地仿真，不连接模型。")
        else:
            logging.error("请先启动 server/serve_pi0.py，再打开 SSH 隧道；详见 README.md。")
        raise SystemExit(1) from exc
