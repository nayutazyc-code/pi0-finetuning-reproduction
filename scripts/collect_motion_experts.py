"""Collect the preassigned new training layouts after physical expert screening."""
from pathlib import Path
import hashlib
import json
import logging
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));os.environ.setdefault('MUJOCO_GL','glfw')
from pi0_repro.demonstrations import collect_demos

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
out=ROOT/'outputs/motion_first_20261026'
assert json.loads((out/'comparison.json').read_text())['complete']
screen=json.loads((out/'expert_screen/comparison.json').read_text());assert screen['complete']
assert all(r['success'] and not r['error'] for r in screen['results'] if r['profile']=='motion')
source_hash=hashlib.sha256((ROOT/'src/pi0_repro/banana_expert.py').read_bytes()).hexdigest()
assert source_hash==screen['expert_source_sha256']
assert hashlib.sha256((ROOT/'configs/banana_motion_expert.toml').read_bytes()).hexdigest()==screen['profile_sha256']['motion']
record_path=out/'data_collections.json'
if record_path.exists():raise FileExistsError('Do not repeat a recorded acquisition')
plan=json.loads((out/'new_layout_plan.json').read_text())
metadata={**plan['metadata'],'motion_label_revision':'quintic-tightIK-so3-softgrip-offline-entry-v1',
          'verified_expert_source_sha256':source_hash,'source_screen':'outputs/motion_first_20261026/expert_screen/comparison.json'}
directory=collect_demos(ROOT/'configs/banana_motion_expert.toml',layout_plan=plan['layout_plan'],dataset_metadata=metadata)
record={'stage':'full_experts_collected_pending_review','full_expert_collection':str(directory),'corrections':[],
        'scope':'Raw candidates only, no automatic review acceptance; fixed12 train-only layout assignments retained.'}
record_path.write_text(json.dumps(record,indent=2)+'\n')
print('FULL_EXPERT_COLLECTION',directory,flush=True)
