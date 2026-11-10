"""Explicit, versioned normalization adaptation for nearly constant action axes."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from .recording import write_json


def create_action_std_floor(source: Path, destination: Path, *, floor: float, dataset_manifest: Path):
    if not math.isfinite(floor) or not 0 < floor <= 1:
        raise ValueError("Action std floor must be finite in (0,1]")
    manifest=json.loads(dataset_manifest.read_text())
    if manifest.get("status") != "complete" or "train" not in manifest["datasets"]:
        raise ValueError("A complete dataset with an explicit train split is required")
    if destination.exists():
        raise FileExistsError("Normalization assets already exist; use a new version")
    data=json.loads(source.read_text())
    std=np.asarray(data["norm_stats"]["actions"]["std"],np.float64)
    updated=np.maximum(std,floor)
    data["norm_stats"]["actions"]["std"]=updated.tolist()
    destination.parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,data)
    report=dict(schema="banana-action-std-floor-v1",source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(dataset_manifest.read_bytes()).hexdigest(),
        split_used="train",floor_transformed_action_units=floor,
        changed_dimensions=np.flatnonzero(updated!=std).tolist(),old_action_std=std.tolist(),new_action_std=updated.tolist(),
        state_statistics_changed=False,action_means_changed=False,quantiles_changed=False,use_quantile_norm=False,
        target_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        meaning="Coordinate change for nearly constant action dimensions; initialization is not exactly function-preserving. Retrain/evaluate and serve with checkpoint-matched assets.")
    write_json(destination.parent/"normalization_report.json",report)
    return report
