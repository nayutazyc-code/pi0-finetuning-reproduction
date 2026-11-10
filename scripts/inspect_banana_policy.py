"""Compare deployed policy outputs against recorded train/validation demonstration states."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "repos/openpi/packages/openpi-client/src")]
from pi0_repro.config import load_config
from pi0_repro.observations import aloha_observation
from pi0_repro.remote import RemotePi0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/banana_finetuned_eval.local.toml")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep prior diagnostic results; choose a new output")
    rows = []
    config = load_config(args.config)
    source = ROOT/"data/banana_demos/20261023-213415-07fb84"
    with RemotePi0(config) as client:
        for split, name in (("train", "episode_000"), ("validation", "episode_004")):
            episode = source/name
            with np.load(episode/"trajectory.npz", allow_pickle=False) as data:
                states = data["state"].copy()
                actions = data["action"].copy()
                phase = data["phase"].copy()
            chosen = [0]
            for stage in ("approach", "descend", "close_gripper", "transport", "release"):
                ids = np.flatnonzero(phase==stage)
                chosen.append(int(ids[len(ids)//2]))
            for index in chosen:
                with Image.open(episode/"rgb"/f"{index:06d}.png") as im:
                    rgb = np.array(im.convert("RGB"))
                observation = aloha_observation({"pixels":{"top":rgb},"agent_pos":states[index]}, config.simulation.prompt, resize=not config.model.send_full_resolution)
                predicted, elapsed, _ = client.infer(observation)
                target = actions[np.minimum(np.arange(50)+index,len(actions)-1)]
                rows.append({"source_split":split,"episode":name,"frame":index,"phase":str(phase[index]),
                    "right_joint_mae_first10_radians":float(np.mean(np.abs(predicted[:10,7:13]-target[:10,7:13]))),
                    "right_gripper_mae_first10":float(np.mean(np.abs(predicted[:10,13]-target[:10,13]))),
                    "predicted_first_target_minus_state":(predicted[0]-states[index]).tolist(),
                    "expert_first_target_minus_state":(target[0]-states[index]).tolist(),
                    "round_trip_ms":elapsed,"prediction":predicted.tolist(),"target":target.tolist()})
                print(split,name,index,str(phase[index]),"right_joint_MAE",round(rows[-1]["right_joint_mae_first10_radians"],4),flush=True)
        result={"server":client.metadata,"rows":rows,
            "scope":"Teacher-forced diagnostics on 6 frames from one train and one validation episode; not rollout success rates, no test split used; stochastic predictions without RNG reset"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
