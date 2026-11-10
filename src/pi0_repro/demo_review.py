"""Explicit human or user-delegated assistant review with provenance."""
from datetime import datetime
import json
import hashlib
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT
from .recording import write_json


def review_demo(episode_path: Path, *, accept: bool, reviewer: str = "human",
                reason: str | None = None, evidence_path: Path | None = None) -> Path:
    episode_path = episode_path.expanduser().resolve()
    if not episode_path.is_relative_to((PROJECT_ROOT / "data/banana_demos").resolve()):
        raise ValueError("请指定本项目 data/banana_demos 下的一个 episode 目录。")
    if reviewer not in {"human", "assistant"}:
        raise ValueError("reviewer 必须是 human 或 assistant。")
    if reviewer == "assistant" and (not reason or evidence_path is None):
        raise ValueError("助手审核必须提供原因和证据文件；需用户明确委托。")
    metadata_path = episode_path / "episode.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    proof = None
    if evidence_path is not None:
        evidence_path = evidence_path.expanduser().resolve()
        if not evidence_path.is_relative_to(episode_path.parent):
            raise ValueError("证据必须保存在本次采集目录内。")
        proof = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected = hashlib.sha256((episode_path / "trajectory.npz").read_bytes()).hexdigest()
        if (proof.get("episode") != data["episode"] or proof.get("layout_id") != data.get("layout_id")
                or proof.get("trajectory_sha256") != expected
                or proof.get("decision") != ("accepted" if accept else "rejected")):
            raise ValueError("审核证据与轨迹或决定不匹配。")
    if accept:
        from .quality import audit_episode
        if not audit_episode(episode_path)["structural_pass"]:
            raise ValueError("演示未通过结构与完整抓放检查，请查看 quality.json")
        if not data.get("automatic_success") or not data.get("intentional_release") or data.get("error"):
            raise ValueError("该轨迹未通过完整受控抓放判定，不能接受为成功演示。")
        with np.load(episode_path / "trajectory.npz", allow_pickle=False) as arrays:
            count = len(arrays["state"])
            for key in ("state", "action", "requested_action", "next_state"):
                if arrays[key].shape != (count, 14) or not np.isfinite(arrays[key]).all():
                    raise ValueError(f"{key} 的形状或数值无效。")
            timestamp = arrays["timestamp"]
            if count == 0 or timestamp.shape != (count,) or not np.isfinite(timestamp).all():
                raise ValueError("轨迹没有有效时间索引。")
            if not np.allclose(np.diff(timestamp), 0.02, atol=1e-7, rtol=0):
                raise ValueError("时间间隔与 50Hz 控制周期不符。")
            if not np.allclose(arrays["state"][1:], arrays["next_state"][:-1], atol=1e-6, rtol=0):
                raise ValueError("相邻步的状态连续性不符。")
            if len(list((episode_path / "rgb").glob("*.png"))) != count:
                raise ValueError("图像数量与动作行数不一致。")
            if "release" not in arrays["phase"].tolist():
                raise ValueError("没有计划松爪阶段。")
    decision = "accepted" if accept else "rejected"
    now = datetime.now().astimezone().isoformat()
    record = {"reviewer": reviewer, "decision": decision, "reviewed_at": now,
              "reason": reason, "evidence": None if evidence_path is None else str(evidence_path.relative_to(episode_path.parent))}
    data.setdefault("review_history", []).append(record)
    if reviewer == "human":
        data["human_review"] = decision
    else:
        data["assistant_review"] = record
        data["review_authorization"] = "explicit_user_delegation"
    data["review_status"] = decision
    data["reviewed_by"] = reviewer
    data["eligible_for_training"] = bool(accept)
    data["reviewed_at"] = now
    write_json(metadata_path, data)
    summary_path = episode_path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["episodes"] = [data if row.get("episode") == data.get("episode") else row
                               for row in summary["episodes"]]
        summary["accepted_demonstrations"] = sum(bool(row.get("eligible_for_training"))
                                                  for row in summary["episodes"])
        write_json(summary_path, summary)
    manifest_path = episode_path.parent / "collection.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("layout_plan"):
            from .layouts import write_dataset_index
            write_dataset_index(episode_path.parent)
    return metadata_path
