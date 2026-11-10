"""Compact actual training status; does not import JAX or load model parameters."""
import argparse
from pathlib import Path
import re,json,subprocess
p=argparse.ArgumentParser();p.add_argument("experiment");p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
r=a.root;log=r/"logs"/(a.experiment+".log");s=log.read_text(errors="replace")
metrics=re.findall(r"Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+), param_norm=([\d.eE+-]+)",s)
ck=r/"checkpoints/pi0_banana_lora"/a.experiment
code=r/"logs"/(a.experiment+".exit-code")
print(json.dumps({"experiment":a.experiment,"latest_metrics":metrics[-1:] or None,"checkpoint_steps":sorted(int(p.name) for p in ck.glob("*") if p.is_dir() and p.name.isdigit()),"exit_code":code.read_text().strip() if code.exists() else None,"last_progress":[x for x in s.splitlines() if "Progress on:" in x][-1:],"errors":[x for x in s.splitlines() if any(v in x for v in ("Traceback","RESOURCE_EXHAUSTED","out of memory"))][-2:]},indent=2))
