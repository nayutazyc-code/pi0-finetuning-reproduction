import time
import hashlib
import io
from PIL import Image
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

from .config import Config, OPENPI_COMMIT


class RemotePi0:
    """Use the official wire format and require original-pi0 server metadata."""

    def __init__(self, config: Config):
        self.config = config
        connection = config.connection
        host = connection.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        self.url = f"ws://{host}:{connection.port}"
        self._packer = msgpack_numpy.Packer()
        self._socket = connect(
            self.url,
            compression=None,
            max_size=16 * 1024 * 1024,
            open_timeout=connection.connect_timeout_seconds,
            close_timeout=5,
        )
        try:
            metadata = self._receive(connection.connect_timeout_seconds)
            if metadata.get("model_family") != "pi0":
                raise ValueError(
                    "服务器未声明原版 pi0。请使用本项目 server/serve_pi0.py 启动，"
                    "不要使用默认 --env LIBERO；未发送推理观测。"
                )
            if metadata.get("policy_config") != config.policy_config:
                raise ValueError(
                    f"服务器配置 {metadata.get('policy_config')!r} 与本地 "
                    f"{config.policy_config!r} 不匹配。"
                )
            if metadata.get("action_dim") != config.action_dim:
                raise ValueError("服务器声明的动作维度与本地配置不匹配。")
            if metadata.get("openpi_commit") != OPENPI_COMMIT:
                raise ValueError("服务器的 openpi 提交与本项目固定版本不一致。")
            if metadata.get("requires_full_resolution") and not config.model.send_full_resolution:
                raise ValueError("当前 ROI 策略需要原始分辨率；请使用对应 send_full_resolution=true 的配置。")
            if config.connection.image_transport == "png" and "png" not in metadata.get("image_transport", []):
                raise ValueError("当前服务端不支持 PNG，请连接无损图像传输端口 8001。")
            if config.simulation.rtc_enabled and metadata.get("rtc", {}).get("schema") != "pi0-rtc-v1":
                raise ValueError("当前服务端未启用生成阶段 RTC，请使用 --rtc 启动。")
            if config.simulation.rtc_enabled and metadata.get("action_horizon") != config.simulation.replan_steps:
                raise ValueError("服务端训练动作长度与本地 replan_steps 不匹配，未发送观测。")
            expected = config.model.expected_checkpoint_suffix
            if expected and not str(metadata.get("checkpoint", "")).rstrip("/").endswith("/" + expected):
                raise ValueError("服务端检查点与此配置指定的版本不匹配。")
            expected_digest = config.model.expected_norm_stats_sha256
            if expected_digest and metadata.get("norm_stats_sha256") != expected_digest:
                raise ValueError("服务端归一化资产与此配置指定的版本不匹配。")
            self.metadata = metadata
        except BaseException:
            self._socket.close()
            raise

    def _receive(self, timeout: float) -> dict[str, Any]:
        raw = self._socket.recv(timeout=timeout)
        if isinstance(raw, str):
            raise RuntimeError(f"远端错误：\n{raw}")
        result = msgpack_numpy.unpackb(raw)
        if not isinstance(result, dict):
            raise ValueError("服务器响应应为字典。")
        return result

    def infer(self, observation: dict[str, Any]) -> tuple[np.ndarray, float, dict]:
        start = time.perf_counter()
        encoding_ms = 0.0
        if self.config.connection.image_transport == "png":
            encode_start = time.perf_counter()
            observation = dict(observation)
            images = observation.pop("images")
            payloads, digests = {}, {}
            for name, image in images.items():
                image = np.ascontiguousarray(image)
                if image.dtype != np.uint8 or image.shape != (3, 480, 640):
                    raise ValueError("PNG 传输需要 uint8 CHW 640x480 RGB 图像。")
                buffer = io.BytesIO()
                Image.fromarray(image.transpose(1, 2, 0)).save(buffer, format="PNG", compress_level=1)
                payloads[name] = buffer.getvalue()
                digests[name] = hashlib.sha256(image.tobytes()).hexdigest()
            observation["images_png"], observation["image_sha256"] = payloads, digests
            encoding_ms = (time.perf_counter() - encode_start) * 1000
        packed = self._packer.pack(observation)
        self._socket.send(packed)
        result = self._receive(self.config.connection.inference_timeout_seconds)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        result["client_transport"] = {"payload_bytes": len(packed), "encoding_ms": encoding_ms}
        if "actions" not in result:
            raise ValueError("服务器响应中没有 actions。")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.config.action_dim or not len(actions):
            raise ValueError(f"动作形状应为 (H, {self.config.action_dim})，实际为 {actions.shape}。")
        if not np.isfinite(actions).all():
            raise ValueError("动作包含 NaN/Inf，停止执行。")
        return actions, elapsed_ms, result

    def close(self) -> None:
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
