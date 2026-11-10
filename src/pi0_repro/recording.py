from dataclasses import asdict
from datetime import datetime
from importlib import metadata
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from .config import Config, PROJECT_ROOT


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_run(config: Config, command: str) -> Path:
    directory = PROJECT_ROOT / "outputs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    directory.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("pi0-mac-reproduction", "openpi-client", "gym-aloha", "mujoco", "numpy", "dm-control", "websockets", "ruckig"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT / "repos/openpi"), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write_json(directory / "run.json", {
        "created_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "config": asdict(config),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": packages,
        "openpi_commit": commit,
        "source_code_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / "src/pi0_repro" / name).read_bytes()).hexdigest()
            for name in ("config.py", "execution.py", "jerk_reference.py", "simulation.py", "viewer.py", "remote.py", "banana_env.py", "scene.py", "diagnostics.py")
        },
        "optional_source_code_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / "src/pi0_repro" / name).read_bytes()).hexdigest()
            for name in ("multiview.py", "random_pose.py", "async_multiview.py", "deferred_video.py", "pacing.py", "control_timing.py")
            if (PROJECT_ROOT / "src/pi0_repro" / name).is_file()
        },
    })
    return directory
