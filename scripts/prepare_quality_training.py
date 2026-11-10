"""Create a train-only balanced sampler and conservative pi0 continuation config."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from pi0_repro.recording import write_json
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--action-std-floor",type=float,default=0.01)
    args=p.parse_args()
    home=args.dataset.resolve()
    manifest=home / "dataset_manifest.json"
    if json.loads(manifest.read_text()).get("status") != "complete":
        raise ValueError("LeRobot export is not complete")
    mapping=json.loads((home/"local/pi0_banana_train/source_episodes.json").read_text())
    source=Path(json.loads(manifest.read_text())["source_path"])
    groups={"original":[],"diversity":[],"recovery":[]}
    entries=[]
    for row in mapping["episodes"]:
        if row["split"] != "train": raise ValueError("Non-train source")
        episode=source/row["source_path"] if "source_path" in row else source/row["source_episode"]
        meta=json.loads((episode/"episode.json").read_text())
        with np.load(episode/"trajectory.npz",allow_pickle=False) as a:
            phases=a["phase"]
        group="recovery" if meta.get("recovery") else ("diversity" if meta["scene"].get("table_rgba") is not None else "original")
        weights=np.ones(len(phases),np.float64)
        for phase, factor in {"descend":1.5,"close_gripper":1.5,"lower_into_box":1.5,"release":1.5}.items():
            weights[phases==phase]=factor
        groups[group].append(len(entries))
        entries.append(dict(weights=weights,group=group,episode=row["lerobot_episode_index"],frames=len(weights)))
    shares={"original":0.55,"diversity":0.30,"recovery":0.15}
    total_share=sum(shares[g] for g in groups if groups[g])
    weights=[]
    count=sum(r["frames"] for r in entries)
    for row in entries:
        w=row["weights"]
        w=w/w.sum()*shares[row["group"]]/total_share/len(groups[row["group"]])*count
        weights.extend(w.tolist())
    output=ROOT/"assets_quality/banana_sampling.json"
    if output.exists() or (ROOT/"configs/banana_finetune_quality.toml").exists() or (ROOT/"assets_quality_norm/pi0_banana_lora/local/pi0_banana_train/norm_stats.json").exists():
        raise FileExistsError("Version existing quality settings before preparing another candidate")
    output.parent.mkdir(exist_ok=True, parents=True)
    write_json(output,dict(repo_id="local/pi0_banana_train",split="train",weights=weights,
        dataset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        groups={k:dict(episodes=len(v),target_share=shares[k]/total_share if v else 0) for k,v in groups.items()},
        method="group mass then equal episode mass; within-episode precision-phase weighting"))
    from pi0_repro.normalization_assets import create_action_std_floor
    create_action_std_floor(ROOT/"assets_v2/pi0_banana_lora/local/pi0_banana_train/norm_stats.json",
        ROOT/"assets_quality_norm/pi0_banana_lora/local/pi0_banana_train/norm_stats.json",
        floor=args.action_std_floor,dataset_manifest=manifest)
    config=(ROOT/"configs/banana_finetune_roi.toml").read_text()
    config=config.replace('data/lerobot/20261023-213415-07fb84',str(home.relative_to(ROOT)))
    config=config.replace('num_train_steps = 2000','num_train_steps = 1000').replace('seed = 44','seed = 45')
    config=config.replace('peak_lr = 0.00005','peak_lr = 0.00002').replace('decay_lr = 0.000005','decay_lr = 0.000002')
    config=config.replace('save_interval = 1000','save_interval = 500').replace('keep_period = 1000','keep_period = 500')
    config=config.replace('banana_v2_phase4000/2000','banana_v3_roi2000/1999').replace('assets_v2/banana_sampling.json','assets_quality/banana_sampling.json')
    config=config.replace('assets_base_dir = "assets_v2"','assets_base_dir = "assets_quality_norm"')
    config=config.replace('# v2 continues v1 weights, new optimizer and explicitly versioned state normalization.','# Action std floor is versioned; optimizer state starts fresh.')
    config='# Original pi0 continuation with versioned action std floor; checkpoint-matched serving required.\n'+config
    (ROOT/"configs/banana_finetune_quality.toml").write_text(config)
    print(json.dumps(dict(config="configs/banana_finetune_quality.toml",weights=str(output),frames=count,groups={k:len(v) for k,v in groups.items()}),indent=2))

if __name__=="__main__": main()
