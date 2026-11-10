"""Freeze new independent training layouts before acquiring expert labels."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pi0_repro.layouts import make_layout_plan

out=ROOT/'outputs/motion_first_20261026';out.mkdir(exist_ok=True)
expert,plan,metadata=make_layout_plan(ROOT/'configs/banana_motion_dataset.toml')
assert len(plan)==12 and all(r['split']=='train' for r in plan)
old=json.loads((ROOT/'data/banana_demos/20261026-142111-merged-e54349/collection.json').read_text())
assert not ({r['layout_id'] for r in old['layout_plan']} & {r['layout_id'] for r in plan})
record={'expert_config':str(expert.relative_to(ROOT)),'expert_source_sha256':hashlib.sha256((ROOT/'src/pi0_repro/banana_expert.py').read_bytes()).hexdigest(),
        'layout_plan':plan,'metadata':metadata,'scope':'All12 new sources assigned train before collection; no success-based resampling. Source corrections remain dependent on these layouts; held-out validation/test states must not be labeled into training.'}
path=out/'new_layout_plan.json'
if path.exists():raise FileExistsError('Retain previous plan')
path.write_text(json.dumps(record,indent=2)+'\n')
print('Frozen12 independent train-only layouts; no old layout ID overlap.')
