"""Versioned merge of reviewed data; preserve splits, evidence and source lineage."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
import uuid

from .config import PROJECT_ROOT
from .recording import write_json
from .layouts import write_dataset_index


def merge_batches(sources):
    destination = PROJECT_ROOT / "data/banana_demos" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-merged-" + uuid.uuid4().hex[:6])
    destination.mkdir()
    (destination / "review_evidence").mkdir()
    plans, episodes, seen, lineage = [], [], set(), []
    for source in sources:
        source = Path(source).resolve()
        if not source.is_relative_to((PROJECT_ROOT / "data/banana_demos").resolve()):
            raise ValueError("Merge source outside project demonstrations")
        collection = json.loads((source / "collection.json").read_text())
        planned = {r["layout_id"]: r for r in collection.get("layout_plan", [])}
        for path in sorted(source.glob("episode_*/episode.json")):
            meta = json.loads(path.read_text())
            if not meta.get("eligible_for_training"):
                continue
            layout = meta["layout_id"]
            if layout in seen or planned[layout]["split"] != meta["split"]:
                raise ValueError("Repeated layout or changed split")
            seen.add(layout)
            index = len(episodes)
            target = destination / f"episode_{index:03d}"
            # Independent files, including JSON review metadata. Old batches stay immutable.
            shutil.copytree(path.parent, target)
            original = dict(batch=str(source), episode=meta["episode"],
                            episode_metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            meta["episode"], meta["merged_from"] = index, original
            if meta.get("reviewed_by") == "assistant":
                proof_path = source / meta["assistant_review"]["evidence"]
                proof = json.loads(proof_path.read_text())
                proof["merged_from"] = {"path": str(proof_path), "sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest()}
                proof["episode"] = index
                relative = f"review_evidence/episode_{index:03d}.json"
                write_json(destination / relative, proof)
                meta["assistant_review"] = {**meta["assistant_review"], "evidence": relative}
            write_json(target / "episode.json", meta)
            plans.append(planned[layout])
            episodes.append(meta)
        lineage.append(dict(source=str(source), collection_sha256=hashlib.sha256((source/"collection.json").read_bytes()).hexdigest()))
    write_json(destination / "collection.json", dict(source="reviewed_simulation_merge_v1", layout_plan=plans,
        policy_model_used=False, learner_states_used=any(r.get("recovery") for r in episodes),
        source_batches=lineage, expert_labels_only=True, split_policy="unchanged from each source plan",
        human_review_required=True, fps=50))
    write_json(destination / "summary.json", dict(source="reviewed_simulation_merge_v1", episodes=episodes,
        accepted_demonstrations=len(episodes)))
    write_dataset_index(destination)
    return destination
