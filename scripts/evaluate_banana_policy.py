"""Run pure-policy closed-loop episodes on predefined layouts, preserving every result."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"repos/openpi/packages/openpi-client/src")]
os.environ.setdefault("MUJOCO_GL","glfw")
from pi0_repro.config import load_config
from pi0_repro.simulation import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/banana_finetuned_eval.local.toml")
    parser.add_argument("--suite",choices=("smoke","validation","confirmation","test"),default="smoke")
    parser.add_argument("--experiment",required=True)
    parser.add_argument("--repeats",type=int,default=1,help="Repeat each fixed layout; policy sampling noise is not paired")
    parser.add_argument("--dataset",type=Path,default=ROOT/"data/lerobot/20261023-213415-07fb84")
    parser.add_argument("--layouts",nargs="+",help="Explicit layout IDs/source episode names; default-center is optional")
    args=parser.parse_args()
    if args.repeats < 1: parser.error("--repeats must be positive")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    dataset=args.dataset.resolve()
    mapping_root=dataset/"local"
    dataset_manifest=json.loads((dataset/"dataset_manifest.json").read_text())
    raw_root=Path(dataset_manifest["source_path"])
    candidates=[{"kind":"default","episode":None,"layout_id":"default-center","scene_path":ROOT/"configs/banana_demo_scene.toml","seed":7}]
    if args.suite=="confirmation":
        candidates=candidates*3
    else:
        split={"smoke":"train","validation":"validation","test":"test"}[args.suite]
        mapping=json.loads((mapping_root/f"pi0_banana_{split}"/"source_episodes.json").read_text())
        for row in (mapping["episodes"][:1] if args.suite=="smoke" else mapping["episodes"]):
            m=json.loads((raw_root/row["source_episode"]/"layout.json").read_text())
            candidates.append({"kind":split,"episode":row["source_episode"],"layout_id":row["layout_id"],"scene":m["scene"],"seed":m["reset_seed"]})
    if args.layouts:
        candidates=[r for r in candidates if r["layout_id"] in args.layouts or r["episode"] in args.layouts]
        if not candidates: raise ValueError("No requested layouts in this suite")
    out=ROOT/"outputs/banana_evaluations"/(datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    record={"experiment":args.experiment,"suite":args.suite,"dataset":str(dataset),"config_file":str(args.config.resolve()),"repeats_per_layout":args.repeats,"policy_noise_paired":False,"success_rule":"unchanged BananaBoxEnv physical grasp+containment+release+stability", "results":[]}
    candidates=[{**item,"repeat":repeat} for item in candidates for repeat in range(args.repeats)]
    for i,item in enumerate(candidates):
        config=load_config(args.config)
        config.simulation.episodes=1
        config.simulation.display=False
        config.simulation.seed=item["seed"]
        if "scene" in item:
            path=out/f"scene_{i:03d}.toml"
            path.write_text("[scene]\n"+"\n".join(f"{k} = {json.dumps(v)}" for k,v in item["scene"].items() if v is not None)+"\n")
        else: path=item["scene_path"]
        config.simulation.scene_config=str(path.resolve())
        directory=run(config)
        summary=json.loads((directory/"summary.json").read_text())
        record["results"].append({"layout_id":item["layout_id"],"repeat":item["repeat"],"kind":item["kind"],"source_episode":item["episode"],"run_directory":str(directory),"result":summary["episodes"][0]})
        record["successful_episodes"]=sum(int(x["result"]["success"]) for x in record["results"])
        record["completed_episodes"]=len(record["results"])
        (out/"evaluation.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        logging.info("SUITE_RESULT %s",record["results"][-1])
    logging.info("SUITE_COMPLETE %s",out)


if __name__=="__main__": main()
