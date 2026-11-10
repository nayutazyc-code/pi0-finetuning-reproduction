"""Read only aggregate status for the fixed v7 continuation experiment."""
import json
from pathlib import Path
import re
import time


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/v7_recovery_20261026"
    source = output / "status.json"
    if not source.exists():
        print(json.dumps({"status": "not_started"}))
        return
    status = json.loads(source.read_text())
    result = {"status": status["status"], "experiment": status["experiment"], "sampled_at": time.time(),
              "elapsed_seconds": time.time() - status["started_at"]}
    if status.get("error"):
        result["error"] = status["error"]
    probe = output / "probe.json"
    if probe.exists():
        data = json.loads(probe.read_text())
        result["probe"] = {key: data.get(key) for key in (
            "status", "completed_optimizer_updates", "actual_action_shape", "steady_update_seconds",
            "gpu_peak_used_mib", "gpu_total_mib")}
    log = output / "training.log"
    if log.exists():
        text = log.read_text(errors="replace")
        logged = [int(value) for value in re.findall(r"Step (\d+):", text)]
        progress = [int(value) for value in re.findall(r"(\d+)/1000", text)]
        result["last_logged_step"] = max(logged, default=None)
        result["completed_updates_lower_bound"] = max(progress + [step + 1 for step in logged], default=0)
        result["nonfinite_metric_lines"] = sum(
            bool(re.search(r"\b(?:nan|[+-]?inf)\b", line, re.I))
            for line in text.splitlines() if re.search(r"Step \d+:", line))
    final = root / "checkpoints/pi0_banana_lora/banana_v7_recovery1000/999"
    result["final_checkpoint_committed"] = (final / "_CHECKPOINT_METADATA").is_file()
    if result["status"] == "trained_pending_validation":
        result["completed_updates_lower_bound"] = 1000
    print(json.dumps(result))


if __name__ == "__main__":
    main()
