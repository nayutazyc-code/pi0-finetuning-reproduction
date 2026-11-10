"""Detached, checked original-pi0 v7 probe and continuation job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "outputs/v7_recovery_20261026"
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    experiment = "banana_v7_recovery1000"
    checkpoint = root / "checkpoints/pi0_banana_lora" / experiment / "999"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "trained_pending_validation" or Path(f"/proc/{previous.get('pid', -1)}").exists():
            raise RuntimeError("Existing training job or completed candidate retained; no duplicate launch")
    if checkpoint.exists():
        raise FileExistsError("Do not overwrite an existing v7 checkpoint")
    status = {"pid": os.getpid(), "experiment": experiment, "status": "preflight", "started_at": time.time()}

    def update(**values):
        status.update(values)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    stopped = []
    try:
        update()
        config = root / "configs/banana_finetune_v7_recovery.toml"
        import tomllib
        settings = tomllib.loads(config.read_text())
        initial = root / settings["paths"]["initial_checkpoint"]
        norm = initial / "assets/local/pi0_banana_train/norm_stats.json"
        expected = "6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269"
        if hashlib.sha256(norm.read_bytes()).hexdigest() != expected or not (initial / "params/_METADATA").exists():
            raise ValueError("Initializer is not the expected committed v6 checkpoint")
        assets = root / settings["paths"]["assets_base_dir"] / "pi0_banana_lora/local/pi0_banana_train/norm_stats.json"
        assets.parent.mkdir(parents=True, exist_ok=True)
        if assets.exists() and assets.read_bytes() != norm.read_bytes():
            raise ValueError("Existing v7 normalization differs; preserve it for review")
        assets.write_bytes(norm.read_bytes())
        dataset = root / settings["data"]["directory"]
        manifest = json.loads((dataset / "dataset_manifest.json").read_text())
        mapping = json.loads((dataset / "local/pi0_banana_train/source_episodes.json").read_text())
        assert manifest["status"] == "complete" and len(mapping["episodes"]) == 26
        assert mapping["source_split"] == "train" and all(r["split"] == "train" for r in mapping["episodes"])
        assert sum(r["frames"] for r in mapping["episodes"]) == 29647
        sampler = json.loads((root / settings["sampling"]["weights_file"]).read_text())
        assert sampler["dataset_manifest_sha256"] == hashlib.sha256((dataset / "dataset_manifest.json").read_bytes()).hexdigest()
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if str(root / "server/serve_pi0.py").encode() in path.read_bytes():
                    pid = int(path.parent.name)
                    if pid != os.getpid():
                        os.kill(pid, signal.SIGTERM)
                        stopped.append(pid)
            except (OSError, ValueError):
                pass
        for _ in range(20):
            used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
            if used < 2048:
                break
            time.sleep(1)
        if used >= 2048:
            raise RuntimeError("Other GPU allocations remain; refusing to start training")
        python = root / "openpi/.venv/bin/python"
        entry = root / "server/banana_finetune.py"
        environment = (f"cd {shlex.quote(str(root))}; "
                       f"if [ -f env.sh ]; then source env.sh; fi; "
                       "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 JAX_PLATFORMS=cuda "
                       "XLA_PYTHON_CLIENT_PREALLOCATE=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled; "
                       f"export HF_HOME={shlex.quote(str(root / 'cache/huggingface'))} "
                       f"OPENPI_DATA_HOME={shlex.quote(str(root / 'cache/openpi'))} "
                       f"XDG_CACHE_HOME={shlex.quote(str(root / 'cache'))}; exec ")

        def run(command, logfile):
            with (output / logfile).open("wb") as log:
                return subprocess.run(["bash", "-lc", environment + shlex.join([str(x) for x in command])], stdout=log, stderr=subprocess.STDOUT).returncode

        common = ["--root", root, "--config", config]
        update(status="probing", stopped_policy_pids=stopped, idle_gpu_used_mib=used, norm_stats_sha256=expected)
        code = run([python, "-u", entry, "probe", *common, "--exp-name", experiment + "_probe", "--probe-updates", "3", "--probe-output", output / "probe.json"], "probe.log")
        if code:
            raise RuntimeError(f"Real-data probe failed with exit code {code}; training not launched")
        probe = json.loads((output / "probe.json").read_text())
        if (probe.get("status") != "passed" or probe.get("actual_action_shape") != [16, 100, 32]
                or probe.get("gpu_peak_used_mib") is None or probe.get("gpu_total_mib") is None
                or probe["gpu_total_mib"] - probe["gpu_peak_used_mib"] < 2048):
            raise RuntimeError("Probe shape/status or required 2048 MiB memory headroom check failed")
        update(status="training", probe=probe)
        code = run([python, "-u", entry, "train", *common, "--exp-name", experiment], "training.log")
        if code:
            raise RuntimeError(f"Training failed with exit code {code}")
        if not (checkpoint / "params/_METADATA").is_file() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            raise RuntimeError("Training exited without the expected committed final checkpoint")
        update(status="trained_pending_validation", checkpoint=str(checkpoint), finished_at=time.time())
    except BaseException as exc:
        update(status="failed", error=f"{type(exc).__name__}: {exc}", stopped_policy_pids=stopped, finished_at=time.time())
        if stopped:
            # Keep a failed optimization from leaving the previous model unavailable.
            restore = (f"export PI0_RTC=1; exec " + shlex.join([
                "bash", str(root / "server/run_banana_serve.sh"),
                str(root / "outputs/long_chunk_20261025/selected_training.toml"),
                str(root / "checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999"), "10"]))
            with (output / "restore_v6.log").open("ab") as log:
                process = subprocess.Popen(["bash", "-lc", restore], stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(restored_v6_pid=process.pid)
        raise


if __name__ == "__main__":
    main()
