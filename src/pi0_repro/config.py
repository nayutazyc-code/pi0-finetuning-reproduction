from dataclasses import dataclass, field
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_CONFIGS = {"aloha_sim": "pi0_aloha_sim", "libero": "pi0_libero"}
ACTION_DIMS = {"aloha_sim": 14, "libero": 7}
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


@dataclass
class ModelConfig:
    family: str = "pi0"
    profile: str = "aloha_sim"
    policy_config: str | None = None
    send_full_resolution: bool = False
    expected_checkpoint_suffix: str | None = None
    expected_norm_stats_sha256: str | None = None


@dataclass
class ConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    connect_timeout_seconds: float = 10.0
    inference_timeout_seconds: float = 180.0
    image_transport: str = "raw"


@dataclass
class SimulationConfig:
    task: str = "gym_aloha/AlohaTransferCube-v0"
    scene_config: str | None = None
    prompt: str = "Transfer cube"
    episodes: int = 1
    max_steps: int = 300
    replan_steps: int = 10
    seed: int = 7
    save_video: bool = True
    display: bool = False
    max_joint_velocity: float | None = None
    max_gripper_velocity: float | None = None
    max_joint_acceleration: float | None = None
    max_joint_jerk: float | None = None
    joint_target_filter_hz: float | None = None
    request_interval_steps: int | None = None
    render_on_request: bool = False
    video_fps: int = 50
    async_inference: bool = False
    prefetch_steps: int = 20
    real_time: bool = False
    control_pacing: str = "after_step"
    display_backend: str = "matplotlib"
    display_fps: float = 20.0
    display_layout: str = "policy"
    record_multiview: bool = False
    reuse_policy_camera: bool = False
    display_in_worker: bool = False
    chunk_blend_steps: int = 0
    chunk_blend_shape: str = "linear"
    temporal_ensemble: bool = False
    temporal_ensemble_max_chunks: int = 3
    temporal_ensemble_coeff: float = 0.01
    async_video: bool = False
    defer_video_render: bool = False
    recover_control_overruns: bool = False
    trace_control_timing: bool = False
    rtc_enabled: bool = False
    rtc_delay_steps: int = 15
    rtc_blend_steps: int = 20
    rtc_guidance_max: float = 2.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @property
    def policy_config(self) -> str:
        return self.model.policy_config or POLICY_CONFIGS[self.model.profile]

    @property
    def action_dim(self) -> int:
        return ACTION_DIMS[self.model.profile]


def load_config(path: Path) -> Config:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    unknown = set(data) - {"model", "connection", "simulation"}
    if unknown:
        raise ValueError(f"未知配置区：{sorted(unknown)}")
    config = Config(
        model=ModelConfig(**data.get("model", {})),
        connection=ConnectionConfig(**data.get("connection", {})),
        simulation=SimulationConfig(**data.get("simulation", {})),
    )
    if config.model.family != "pi0" or config.model.profile not in POLICY_CONFIGS:
        raise ValueError("本项目只接受原版 pi0：profile 必须为 aloha_sim 或 libero。")
    allowed = {"aloha_sim": {"pi0_aloha_sim", "pi0_banana_lora"}, "libero": {"pi0_libero"}}
    if config.policy_config not in allowed[config.model.profile]:
        raise ValueError("policy_config 与当前原版 π0 观测协议不匹配。")
    if type(config.model.send_full_resolution) is not bool:
        raise ValueError("send_full_resolution 必须是布尔值。")
    expected = config.model.expected_checkpoint_suffix
    if expected is not None and (not isinstance(expected, str) or not expected or expected.startswith("/") or ".." in expected.split("/")):
        raise ValueError("expected_checkpoint_suffix must name a relative checkpoint suffix")
    digest = config.model.expected_norm_stats_sha256
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("expected_norm_stats_sha256 must be a lowercase SHA-256 digest")
    connection = config.connection
    if not connection.host or any(c in connection.host for c in "/ \t\r\n"):
        raise ValueError("host 请填写主机名或 IP，不包含 ws:// 或路径。")
    if not 1 <= connection.port <= 65535:
        raise ValueError("端口必须在 1–65535 之间。")
    if connection.image_transport not in {"raw", "png"}:
        raise ValueError("image_transport 仅支持 raw 或 png。")
    if connection.image_transport == "png" and not config.model.send_full_resolution:
        raise ValueError("PNG 传输要求 send_full_resolution=true。")
    if connection.connect_timeout_seconds <= 0 or connection.inference_timeout_seconds <= 0:
        raise ValueError("连接和推理超时必须为正数。")
    simulation = config.simulation
    interval = simulation.request_interval_steps
    if interval is not None:
        if type(interval) is not int or not 0 < interval < simulation.replan_steps:
            raise ValueError("request_interval_steps must be an integer shorter than the action horizon")
        if not simulation.async_inference:
            raise ValueError("request_interval_steps requires async_inference=true")
        prefetch = simulation.replan_steps - interval
        if "prefetch_steps" in data.get("simulation", {}) and simulation.prefetch_steps != prefetch:
            raise ValueError("request_interval_steps conflicts with prefetch_steps")
        simulation.prefetch_steps = prefetch
    if type(simulation.video_fps) is not int or not 1 <= simulation.video_fps <= 50 or 50 % simulation.video_fps:
        raise ValueError("video_fps must be a positive divisor of the 50 Hz control rate")
    if simulation.render_on_request and simulation.task != "pi0_repro/BananaBox-v0":
        raise ValueError("render_on_request currently supports only BananaBox-v0")
    if min(simulation.episodes, simulation.max_steps, simulation.replan_steps) < 1:
        raise ValueError("episodes、max_steps、replan_steps 必须大于 0。")
    for name in ("max_joint_velocity", "max_gripper_velocity", "max_joint_acceleration", "max_joint_jerk", "joint_target_filter_hz", "display_fps"):
        value = getattr(simulation, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf")):
            raise ValueError(f"{name} 必须是有限正数。")
    if simulation.max_joint_jerk is not None and (simulation.max_joint_velocity is None or simulation.max_joint_acceleration is None):
        raise ValueError("max_joint_jerk requires finite joint velocity and acceleration limits")
    if simulation.joint_target_filter_hz is not None:
        if simulation.joint_target_filter_hz > 20:
            raise ValueError("joint_target_filter_hz must be at most 20 Hz")
        if simulation.max_joint_acceleration is None:
            raise ValueError("joint target filtering requires max_joint_acceleration")
    if simulation.display_backend not in {"matplotlib", "glfw"}:
        raise ValueError("display_backend 仅支持 matplotlib 或 glfw。")
    if simulation.display_fps is None:
        raise ValueError("display_fps 必须是有限正数。")
    if simulation.display_layout not in {"policy", "multiview"}:
        raise ValueError("display_layout must be policy or multiview")
    if simulation.display_layout == "multiview":
        if simulation.task != "pi0_repro/BananaBox-v0":
            raise ValueError("Multiview display currently requires BananaBox-v0")
        if not float(50 / simulation.display_fps).is_integer():
            raise ValueError("Multiview display_fps must divide the 50Hz control rate")
    if simulation.record_multiview and (simulation.display_layout != "multiview" or not simulation.save_video):
        raise ValueError("record_multiview requires multiview layout and save_video=true")
    if simulation.reuse_policy_camera and simulation.task != "pi0_repro/BananaBox-v0":
        raise ValueError("reuse_policy_camera currently requires BananaBox-v0")
    if simulation.display_in_worker and (simulation.display_layout != "multiview" or simulation.display_backend != "glfw"):
        raise ValueError("display_in_worker requires multiview layout and glfw backend")
    if simulation.defer_video_render:
        if simulation.task != "pi0_repro/BananaBox-v0" or not simulation.render_on_request or not simulation.save_video:
            raise ValueError("defer_video_render requires BananaBox-v0, render_on_request and save_video")
        if simulation.display and not simulation.display_in_worker:
            raise ValueError("defer_video_render requires headless operation or display_in_worker")
    if simulation.control_pacing not in {"after_step", "before_step"}:
        raise ValueError("control_pacing must be after_step or before_step")
    if simulation.control_pacing == "before_step" and not simulation.real_time:
        raise ValueError("before_step control pacing requires real_time=true")
    if simulation.recover_control_overruns and (not simulation.real_time or simulation.control_pacing != "before_step"):
        raise ValueError("recover_control_overruns requires real_time and before_step pacing")
    if simulation.trace_control_timing and simulation.task != "pi0_repro/BananaBox-v0":
        raise ValueError("trace_control_timing currently requires BananaBox-v0")
    for name in ("async_inference", "real_time", "async_video", "rtc_enabled", "render_on_request", "temporal_ensemble", "record_multiview", "reuse_policy_camera", "display_in_worker", "defer_video_render", "recover_control_overruns", "trace_control_timing"):
        if type(getattr(simulation, name)) is not bool:
            raise ValueError(f"{name} 必须是布尔值。")
    for name in ("prefetch_steps", "chunk_blend_steps"):
        value = getattr(simulation, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} 必须是非负整数。")
    if simulation.chunk_blend_steps >= simulation.replan_steps:
        raise ValueError("chunk_blend_steps 必须小于 replan_steps。")
    if simulation.chunk_blend_shape not in {"linear", "quintic"}:
        raise ValueError("chunk_blend_shape must be linear or quintic")
    if type(simulation.temporal_ensemble_max_chunks) is not int or not 2 <= simulation.temporal_ensemble_max_chunks <= 8:
        raise ValueError("temporal_ensemble_max_chunks must be an integer in [2, 8]")
    coefficient = simulation.temporal_ensemble_coeff
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not 0 <= coefficient <= 1:
        raise ValueError("temporal_ensemble_coeff must be finite and in [0, 1]")
    if simulation.temporal_ensemble and (not simulation.async_inference or simulation.chunk_blend_steps):
        raise ValueError("temporal ensemble requires async inference and chunk_blend_steps=0")
    if simulation.chunk_blend_shape == "quintic" and simulation.chunk_blend_steps:
        if not simulation.async_inference or simulation.chunk_blend_steps < 2:
            raise ValueError("quintic transitions require asynchronous overlap and at least two steps")
    if simulation.async_inference:
        if not simulation.real_time:
            raise ValueError("异步推理需要 real_time=true，避免无节拍仿真提前耗尽预测。")
        if not 0 < simulation.prefetch_steps < simulation.replan_steps:
            raise ValueError("异步 prefetch_steps 必须在 0 和 replan_steps 之间。")
    if simulation.rtc_enabled:
        if not simulation.async_inference:
            raise ValueError("RTC requires async inference")
        if simulation.chunk_blend_steps and simulation.chunk_blend_shape != "quintic":
            raise ValueError("RTC transitions require the forecast-aware quintic shape")
        if config.policy_config != "pi0_banana_lora":
            raise ValueError("RTC requires an original pi0 banana policy with a matching trained action horizon")
        if type(simulation.rtc_delay_steps) is not int or not 1 <= simulation.rtc_delay_steps < simulation.prefetch_steps:
            raise ValueError("RTC delay must be positive and shorter than the prefetch window")
        if type(simulation.rtc_blend_steps) is not int or not 1 <= simulation.rtc_blend_steps <= simulation.replan_steps:
            raise ValueError("RTC blend must be an integer within the configured action horizon")
        if not 0 < simulation.rtc_guidance_max <= 10:
            raise ValueError("RTC guidance_max must be in (0,10]")
    if simulation.scene_config is not None:
        if not isinstance(simulation.scene_config, str) or not simulation.scene_config:
            raise ValueError("scene_config 必须是非空文件路径。")
        scene_path = Path(simulation.scene_config).expanduser()
        if not scene_path.is_absolute():
            scene_path = path.resolve().parent / scene_path
        simulation.scene_config = str(scene_path.resolve())
    if simulation.task == "pi0_repro/BananaBox-v0":
        if config.model.profile != "aloha_sim" or simulation.scene_config is None:
            raise ValueError("香蕉任务需要 aloha_sim profile 和 scene_config。")
    return config
