"""Structural rejection criteria and advisory motion-quality metrics; no loss filtering."""
import hashlib
import json
from pathlib import Path

import numpy as np
from .recording import write_json


def audit_episode(directory: Path) -> dict:
    meta = json.loads((directory / "episode.json").read_text())
    errors, metrics = [], {}
    with np.load(directory / "trajectory.npz", allow_pickle=False) as a:
        n = len(a["state"])
        for key in ("state", "next_state", "action", "requested_action"):
            if a[key].shape != (n, 14) or not np.isfinite(a[key]).all():
                errors.append(key + ": invalid shape or nonfinite values")
        if not n or a["timestamp"].shape != (n,) or not np.allclose(a["timestamp"], np.arange(n)*0.02, atol=1e-7, rtol=0):
            errors.append("invalid synchronized timestamps")
        if not np.allclose(a["state"][1:], a["next_state"][:-1], atol=1e-6, rtol=0):
            errors.append("discontinuous next_state")
        if len(list((directory / "rgb").glob("*.png"))) != n:
            errors.append("image count mismatch")
        phases = a["phase"]
        joints = [0,1,2,3,4,5,7,8,9,10,11,12]
        for phase in np.unique(phases):
            # Within-phase adjacent samples only: don't hide boundary jumps by
            # joining disjoint spans of the same phase.
            select = (phases[1:] == phase) & (phases[:-1] == phase)
            speed = np.abs(np.diff(a["action"][:, joints], axis=0)[select]) / 0.02
            metrics[str(phase)] = dict(frames=int(np.sum(phases == phase)),
                joint_speed_p95_rad_s=float(np.percentile(speed,95)) if speed.size else None,
                joint_speed_max_rad_s=float(speed.max()) if speed.size else None)
        delta = np.abs(np.diff(a["action"][:, joints], axis=0))
        max_delta = float(delta.max()) if delta.size else 0.0
    if not meta.get("automatic_success") or not meta.get("intentional_release") or meta.get("error"):
        errors.append("incomplete controlled grasp/place")
    report = dict(schema="banana-quality-v1", structural_pass=not errors, errors=errors,
        advisory={"max_joint_target_delta_rad": max_delta, "large_delta_over_0_05rad": max_delta > 0.05,
                  "phases": metrics},
        trajectory_sha256=hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest(),
        policy="structural failures block acceptance; motion thresholds nominate replay, never silently delete hard examples",
        review_required=True)
    write_json(directory / "quality.json", report)
    return report
