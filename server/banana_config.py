"""Project-local original-pi0 configuration; the official checkout stays unchanged."""
from pathlib import Path
import subprocess
import tomllib
import math

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def read_settings(path: Path) -> dict:
    with path.open("rb") as stream:
        settings = tomllib.load(stream)
    if not {"data", "model", "train", "paths"} <= set(settings) or set(settings) - {"data", "model", "train", "paths", "sampling"}:
        raise ValueError("训练 TOML 需要 data/model/train/paths 四个区域。")
    if settings["model"]["family"] != "pi0" or settings["model"]["config_name"] != "pi0_banana_lora":
        raise ValueError("仅接受原版 π0 香蕉 LoRA 配置。")
    if settings["data"]["train_repo_id"] != "local/pi0_banana_train":
        raise ValueError("训练只能使用预先划分的 local/pi0_banana_train。")
    if type(settings["model"]["strict_lora_only"]) is not bool:
        raise ValueError("strict_lora_only 应为布尔值。")
    for key in ("batch_size", "num_train_steps", "save_interval", "log_interval"):
        if type(settings["train"][key]) is not int or settings["train"][key] < 1:
            raise ValueError(f"{key} 必须是正整数。")
    for section, key, default in (("model", "action_horizon", 50), ("train", "gradient_accumulation_steps", 1)):
        value = settings[section].get(key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")
    temporal = settings['model'].get('temporal_training', False)
    weight = settings['model'].get('temporal_loss_weight', 0.0)
    if type(temporal) is not bool or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid temporal loss configuration')
    if (temporal or weight) and (not settings['model'].get('weighted_flow_loss') or not settings.get('sampling')):
        raise ValueError('Temporal training requires weighted flow and the explicit training sampler')
    if weight and not temporal:
        raise ValueError('Temporal loss weight requires temporal_training=true')
    if temporal and settings['train'].get('gradient_accumulation_steps', 1) < 2:
        raise ValueError('Temporal metric logging requires the accumulation adapter')
    prefix_steps=settings['model'].get('near_term_steps',0)
    prefix_weight=settings['model'].get('near_term_weight',1.0)
    if type(prefix_steps) is not int or not 0<=prefix_steps<=settings['model'].get('action_horizon',50):
        raise ValueError('near_term_steps must be within the action horizon')
    if isinstance(prefix_weight,bool) or not isinstance(prefix_weight,(int,float)) or not math.isfinite(prefix_weight) or not 1<=prefix_weight<=10:
        raise ValueError('near_term_weight must be finite in [1,10]')
    if (prefix_steps and not temporal) or (not prefix_steps and prefix_weight!=1):
        raise ValueError('Near-term weighting needs explicit masked training and a nonzero prefix')
    return settings


def find_openpi(root: Path) -> Path:
    for candidate in (root / "openpi", root / "repos/openpi"):
        if (candidate / "src/openpi/training/config.py").is_file():
            return candidate
    raise FileNotFoundError("未找到固定版本 openpi。")


def require_openpi_commit(root: Path) -> Path:
    checkout = find_openpi(root)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if commit != OPENPI_COMMIT:
        raise ValueError(f"openpi 必须固定在 {OPENPI_COMMIT}；实际 {commit}")
    return checkout


def make_config(root: Path, config_path: Path, *, exp_name: str = "banana_pilot_v1", resume: bool = False):
    import flax.nnx as nnx
    from openpi.models import pi0_config
    from openpi.shared import nnx_utils
    from openpi.training import config as training_config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    settings = read_settings(config_path)
    train = settings["train"]
    model_class = pi0_config.Pi0Config
    model_extra = {}
    if settings["model"].get("weighted_flow_loss", False):
        from banana_model import BananaPi0Config
        model_class = BananaPi0Config
        if settings["model"].get("camera_crop"):
            from banana_roi import ROIPi0Config
            model_class = ROIPi0Config
        model_extra = {"dimension_weights": tuple([0.25]*7 + [1.0]*6 + [4.0] + [0.1]*18),
                       "image_augmentation": settings["model"].get("image_augmentation", True)}
        if settings['model'].get('temporal_training', False):
            import json
            from banana_temporal import ARM_JOINTS
            norm_path = root / settings['paths']['assets_base_dir'] / settings['model']['config_name'] / settings['data']['train_repo_id'] / 'norm_stats.json'
            stats = json.loads(norm_path.read_text())['norm_stats']['actions']
            scales = tuple(float(stats['std'][i]) + 1e-6 for i in ARM_JOINTS)
            if not all(math.isfinite(x) and x > 0 for x in scales):
                raise ValueError('Invalid native arm normalization scales')
            model_extra.update(temporal_training=True, temporal_loss_weight=settings['model'].get('temporal_loss_weight', 0.0),
                               temporal_arm_scales=scales,near_term_steps=settings['model'].get('near_term_steps',0),
                               near_term_weight=settings['model'].get('near_term_weight',1.0))
    model = model_class(pi05=False, action_dim=32, action_horizon=settings["model"].get("action_horizon", 50),
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora", **model_extra)
    freeze = (nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))
              if settings["model"]["strict_lora_only"] else model.get_freeze_filter())
    if settings["model"].get("train_action_projections", False):
        projections = nnx_utils.PathRegex("(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)/.*")
        freeze = nnx.All(freeze, nnx.Not(projections))
    data_class = training_config.LeRobotAlohaDataConfig
    data_extra = {}
    if settings["model"].get("camera_crop"):
        from banana_roi import ROIAlohaDataConfig
        data_class = ROIAlohaDataConfig
        data_extra = {"crop_bounds": tuple(settings["model"]["camera_crop"])}
    config = training_config.TrainConfig(
        name=settings["model"]["config_name"], exp_name=exp_name, model=model,
        data=data_class(
            repo_id=settings["data"]["train_repo_id"], adapt_to_pi=True,
            use_delta_joint_actions=False, default_prompt=settings["data"]["prompt"],
            base_config=training_config.DataConfig(prompt_from_task=True), **data_extra),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(root / settings["paths"]["initial_checkpoint"] / "params")),
        freeze_filter=freeze, ema_decay=None,
        batch_size=train["batch_size"] * train.get("gradient_accumulation_steps", 1),
        num_workers=train["num_workers"], seed=train["seed"],
        num_train_steps=train["num_train_steps"], save_interval=train["save_interval"],
        keep_period=train["keep_period"], log_interval=train["log_interval"],
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=train["warmup_steps"], peak_lr=train["peak_lr"],
            decay_steps=train["num_train_steps"], decay_lr=train["decay_lr"]),
        assets_base_dir=str(root / settings["paths"]["assets_base_dir"]),
        checkpoint_base_dir=str(root / settings["paths"]["checkpoint_base_dir"]),
        wandb_enabled=False, resume=resume, overwrite=False,
        policy_metadata={"task": "banana_box", "model_family": "pi0", "strict_lora_only": settings["model"]["strict_lora_only"] and not settings["model"].get("train_action_projections", False),
                         "train_action_projections": settings["model"].get("train_action_projections", False),
                         "weighted_flow_loss": settings["model"].get("weighted_flow_loss", False),
                         "temporal_training": settings['model'].get('temporal_training', False),
                         "temporal_loss_weight": settings['model'].get('temporal_loss_weight', 0.0),
                         "near_term_steps": settings['model'].get('near_term_steps',0),
                         "near_term_weight": settings['model'].get('near_term_weight',1.0),
                         "requires_full_resolution": bool(settings["model"].get("camera_crop")),
                         "camera_crop": settings["model"].get("camera_crop"),
                         "action_semantics": "absolute_aloha_joint_targets", "scene_controller_profile": "position_pd"})
    # Register only inside this process; never patch the official source or change its Git state.
    training_config._CONFIGS_DICT[config.name] = config
    return config
