"""Export explicitly reviewed demonstrations using openpi's pinned LeRobot API."""
import argparse
import hashlib
from importlib import metadata
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PROJECT_ROOT
from .recording import write_json

LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
JOINT_NAMES = [f"{side}_{name}" for side in ("left", "right") for name in (
    "waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
IMAGE_KEY = "observation.images.top"
SPLITS = ("train", "validation", "test")


def _selected_episodes(source: Path) -> tuple[list[dict], list[dict]]:
    selected, excluded, seen = [], [], set()
    plan = json.loads((source / "collection.json").read_text(encoding="utf-8")).get("layout_plan")
    if not plan:
        raise ValueError("需要含采集前布局划分的多布局数据。")
    planned = {row["layout_id"]: row["split"] for row in plan}
    for path in sorted(source.glob("episode_[0-9][0-9][0-9]")):
        data = json.loads((path / "episode.json").read_text(encoding="utf-8"))
        accepted = (data.get("eligible_for_training") and data.get("automatic_success")
                    and data.get("intentional_release") and not data.get("error")
                    and data.get("review_status", data.get("human_review")) == "accepted")
        if not accepted:
            excluded.append({"path": path.name, "reason": "not explicitly reviewed and accepted", "error": data.get("error")})
            continue
        layout = data.get("layout_id")
        if layout in seen or planned.get(layout) != data.get("split") or data.get("split") not in SPLITS:
            raise ValueError(f"{path.name}: 布局重复或划分与采集前计划不符。")
        seen.add(layout)
        reviewer = data.get("reviewed_by", "human")
        if reviewer == "assistant":
            if data.get("review_authorization") != "explicit_user_delegation":
                raise ValueError("助手审核缺少用户明确委托记录。")
            proof = source / data["assistant_review"]["evidence"]
            if not proof.resolve().is_relative_to(source):
                raise ValueError("审核证据路径超出原数据目录。")
            proof_data = json.loads(proof.read_text(encoding="utf-8"))
            if proof_data["trajectory_sha256"] != hashlib.sha256((path / "trajectory.npz").read_bytes()).hexdigest():
                raise ValueError("审核后轨迹内容发生变化。")
        data["source_path"] = path.name
        selected.append(data)
    if not selected:
        raise ValueError("没有可转换的已接受轨迹。")
    return selected, excluded


def _read_source(path: Path, expected_count: int) -> dict:
    with np.load(path / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for key in ("state", "action", "requested_action", "next_state"):
        if arrays[key].shape != (expected_count, 14) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{path.name}/{key}: 无效的维度或数值。")
    t = arrays["timestamp"]
    if (t.shape != (expected_count,) or not np.isfinite(t).all()
            or not np.allclose(t, np.arange(expected_count) / 50, atol=1e-7, rtol=0)):
        raise ValueError("时间戳不是从 0 开始的 50Hz 同步序列。")
    if len(list((path / "rgb").glob("*.png"))) != expected_count:
        raise ValueError("图像数与动作数不同。")
    return arrays


def _compare_saved_episode(root: Path, local_index: int, source: Path, arrays: dict) -> dict:
    import pyarrow.parquet as pq
    path = root / f"data/chunk-000/episode_{local_index:06d}.parquet"
    table = pq.read_table(path)
    count = len(arrays["state"])
    if len(table) != count:
        raise ValueError("输出帧数不同。")
    for feature, key in (("observation.state", "state"), ("action", "action")):
        values = np.asarray(table[feature].to_pylist(), dtype=np.float32)
        if not np.array_equal(values, arrays[key]):
            raise ValueError(f"{feature} 与来源不一致。")
    timestamp = np.asarray(table["timestamp"].to_pylist())
    if not np.allclose(timestamp, arrays["timestamp"], atol=2e-6, rtol=0):
        raise ValueError("输出时间戳与原记录不一致。")
    reencoded = 0
    image_hash = hashlib.sha256()
    for i, saved in enumerate(table[IMAGE_KEY].to_pylist()):
        raw = (source / "rgb" / f"{i:06d}.png").read_bytes()
        embedded = saved["bytes"]
        if embedded is None:
            raise ValueError("图片未嵌入 Parquet，数据不能独立移动。")
        image_hash.update(raw)
        if embedded != raw:
            reencoded += 1
            with Image.open(io.BytesIO(embedded)) as converted, Image.open(io.BytesIO(raw)) as original:
                if not np.array_equal(np.asarray(converted), np.asarray(original)):
                    raise ValueError(f"第 {i} 张图像存在像素变化。")
    return {"frames": count, "state_action_exact": True, "images_pixel_exact": True,
            "png_reencoded_count": reencoded, "source_png_stream_sha256": image_hash.hexdigest(),
            "timestamp_max_error_s": float(np.max(np.abs(timestamp - arrays["timestamp"])))}


def export_dataset(source: Path, output_root: Path | None = None) -> Path:
    from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    if CODEBASE_VERSION != "v2.1":
        raise ValueError("必须使用 openpi 固定提交对应的 LeRobot v2.1。")
    source = source.expanduser().resolve()
    if not source.is_relative_to((PROJECT_ROOT / "data/banana_demos").resolve()):
        raise ValueError("来源必须是当前项目 data/banana_demos 下的采集批次。")
    selected, excluded = _selected_episodes(source)
    output_root = (PROJECT_ROOT / "data/lerobot" / source.name if output_root is None else output_root).expanduser().resolve()
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError("输出或 partial 已存在；为保留既有结果，请使用新的输出目录。")
    partial.mkdir(parents=True)
    manifest = {"status": "converting", "source_session": source.name, "source_path": str(source),
                "lerobot_commit": LEROBOT_COMMIT, "codebase_version": CODEBASE_VERSION,
                "fps": 50, "image_storage": "lossless PNG embedded in Parquet", "action_dim": 14,
                "action_semantics": "absolute ALOHA joint-position targets; normalized gripper targets",
                "action_order": JOINT_NAMES, "state_order": JOINT_NAMES,
                "action_source": "executed action, not requested_action", "action_horizon_at_training": 50,
                "frame_alignment": "image and state before the same indexed action",
                "timestamps": "official writer uses frame_index / 50; readback compared to recorded timestamps",
                "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "review_policy": "explicit human or user-delegated assistant acceptance",
                "raw_data_modified": False, "pushed_to_hub": False, "excluded": excluded, "datasets": {},
                "openpi_data_settings": {"adapt_to_pi": True, "use_delta_joint_actions": False,
                    "default_prompt": selected[0]["prompt"], "action_sequence_keys": ["action"],
                    "repack": {"images.cam_high": IMAGE_KEY, "state": "observation.state", "actions": "action"}},
                "normalization_note": "LeRobot raw statistics are not openpi normalization assets; compute only from train after ALOHA transforms.",
                "packages": {name: metadata.version(name) for name in ("torch", "torchvision", "datasets", "pyarrow", "av", "numpy", "Pillow")}}
    write_json(partial / "dataset_manifest.json", manifest)
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        IMAGE_KEY: {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }
    for split in SPLITS:
        rows = [row for row in selected if row["split"] == split]
        if not rows:
            continue
        repo_id = f"local/pi0_banana_{split}"
        root = partial / repo_id
        dataset = LeRobotDataset.create(repo_id=repo_id, root=root, fps=50, robot_type="aloha_sim",
                    features=features.copy(), use_videos=False, image_writer_threads=4, video_backend="pyav")
        mappings, audits = [], []
        try:
            for local_index, row in enumerate(rows):
                original = source / row["source_path"]
                arrays = _read_source(original, row["steps"])
                for i in range(row["steps"]):
                    with Image.open(original / "rgb" / f"{i:06d}.png") as im:
                        image = np.asarray(im.convert("RGB")).copy()
                    if image.shape != (480, 640, 3):
                        raise ValueError("原始图像尺寸错误，转换不隐式缩放。")
                    dataset.add_frame({"observation.state": arrays["state"][i], "action": arrays["action"][i],
                                       IMAGE_KEY: image, "task": row["prompt"]})
                dataset.save_episode()
                audit = _compare_saved_episode(root, local_index, original, arrays)
                audits.append(audit)
                mappings.append({"lerobot_episode_index": local_index, "source_episode": row["source_path"],
                                 "layout_id": row["layout_id"], "split": split, "frames": row["steps"],
                                 "source_trajectory_sha256": hashlib.sha256((original / "trajectory.npz").read_bytes()).hexdigest(),
                                 "reviewed_by": row.get("reviewed_by", "human"),
                                 "review_evidence": row.get("assistant_review", {}).get("evidence"),
                                 "scene": row["scene"], "settled_initial_pose": row["settled_initial_pose"]})
                logging.info("已转换 %s %d/%d：%s，%d 帧", split, local_index + 1, len(rows), row["source_path"], row["steps"])
        finally:
            dataset.stop_image_writer()
        del dataset
        write_json(root / "source_episodes.json", {"source_session": source.name, "source_split": split, "episodes": mappings})
        # Open through the actual pinned reader and exercise the same 50-action query used by openpi.
        loaded = LeRobotDataset(repo_id=repo_id, root=root, delta_timestamps={"action": [i / 50 for i in range(50)]}, video_backend="pyav")
        first = loaded[0]
        edge = loaded[rows[0]["steps"] - 1]
        if tuple(first["action"].shape) != (50, 14) or tuple(edge["action"].shape) != (50, 14):
            raise ValueError("真实 LeRobot 读取器未生成 50×14 动作块。")
        first_source = _read_source(source / rows[0]["source_path"], rows[0]["steps"])
        if not np.array_equal(first["action"].numpy(), first_source["action"][:50]):
            raise ValueError("首个动作块与原始序列不一致。")
        if not np.array_equal(edge["action"].numpy(), np.repeat(first_source["action"][-1:], 50, axis=0)):
            raise ValueError("轨迹边界动作块越过 episode 或 padding 不符。")
        if tuple(first[IMAGE_KEY].shape) != (3, 480, 640):
            raise ValueError("读取器的图像形状不符。")
        item = {"repo_id": repo_id, "path": repo_id, "source_split": split,
                "episodes": len(rows), "frames": sum(row["steps"] for row in rows),
                "reader_action_shape": list(first["action"].shape), "reader_image_shape": list(first[IMAGE_KEY].shape),
                "reader_episode_boundary_padding": "confirmed; does not read next episode", "audits": audits}
        write_json(root / "conversion_audit.json", item)
        manifest["datasets"][split] = item
        write_json(partial / "dataset_manifest.json", manifest)
        del loaded
    manifest["status"] = "complete"
    manifest["total_episodes"] = sum(x["episodes"] for x in manifest["datasets"].values())
    manifest["total_frames"] = sum(x["frames"] for x in manifest["datasets"].values())
    write_json(partial / "dataset_manifest.json", manifest)
    partial.rename(output_root)
    logging.info("LeRobot 转换完成：%s；未上传，未开始训练。", output_root)
    return output_root


def main():
    parser = argparse.ArgumentParser(description="将已明确接受的香蕉演示转为固定版本 LeRobot v2.1")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Conversion and local readback are intentionally offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    export_dataset(args.source, args.output_root)


if __name__ == "__main__":
    main()
