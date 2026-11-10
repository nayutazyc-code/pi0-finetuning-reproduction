from pathlib import Path

import numpy as np
from openpi_client import image_tools


def image224(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"需要 H×W×3 的 RGB 图像，实际为 {image.shape}。")
    if image.dtype != np.uint8:
        raise ValueError("输入图像必须是 uint8 RGB；请明确转换后再调用。")
    return np.ascontiguousarray(image_tools.resize_with_pad(image, 224, 224))


def state_vector(state: np.ndarray, dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (dim,) or not np.isfinite(state).all():
        raise ValueError(f"状态必须是有限值的 ({dim},) 向量，实际为 {state.shape}。")
    return state


def aloha_observation(gym_obs: dict, prompt: str, *, resize: bool = True) -> dict:
    image = image224(gym_obs["pixels"]["top"]) if resize else np.asarray(gym_obs["pixels"]["top"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("需要 uint8 HWC RGB 观测。")
    return {
        "state": state_vector(gym_obs["agent_pos"], 14),
        "images": {"cam_high": np.ascontiguousarray(image.transpose(2, 0, 1))},
        "prompt": prompt,
    }


def from_npz(path: Path, profile: str, default_prompt: str, *, resize: bool = True) -> dict:
    # Object arrays are rejected; the observation file is data, not Python code.
    with np.load(path, allow_pickle=False) as data:
        prompt = str(data["prompt"].item()) if "prompt" in data else default_prompt
        if profile == "aloha_sim":
            return {
                "state": state_vector(data["state"], 14),
                "images": {"cam_high": np.ascontiguousarray((image224(data["image"]) if resize else data["image"]).transpose(2, 0, 1))},
                "prompt": prompt,
            }
        return {
            "observation/state": state_vector(data["state"], 8),
            "observation/image": image224(data["image"]),
            "observation/wrist_image": image224(data["wrist_image"]),
            "prompt": prompt,
        }
