"""Select train-only v6 states and collect versioned expert candidates for review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
import numpy as np
from pi0_repro.recovery import collect_recovery


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');q.replace(p)


def select(row):
    source=Path(row['source_episode']);meta=read(source/'episode.json');assert meta['split']=='train'
    run=Path(row['run_directory']);tasks=[json.loads(s) for s in (run/'episode_000_task.jsonl').read_text().splitlines()]
    with np.load(run/'episode_000.npz',allow_pickle=False) as d:n=len(d['action']);grip=d['action'][:,13].copy()
    near=next((i for i,t in enumerate(tasks) if t.get('near_grasp_region') and not t.get('ever_grasped')),None)
    lift=next((i for i,t in enumerate(tasks) if t.get('ever_grasped')),n)
    inside=next((i for i,t in enumerate(tasks) if t.get('inside_box')),n)
    selected=[];availability=dict(near_available=near is not None,lift_available=lift<n,containment_available=inside<n)
    if near is not None:
        delayed=(lift-near)*.02>4.16
        upper=min(lift-1,n-2)
        target=min(near+(208 if delayed else 15),upper)
        choices=[i for i in range(near,upper+1) if tasks[i].get('near_grasp_region') and not tasks[i].get('ever_grasped')]
        if choices:
            i=min(choices,key=lambda i:abs(i-target));selected.append(('near_grasp_delay' if delayed else 'near_grasp_reference',i+1))
    box=np.asarray(meta['scene']['box_position'][:2])
    held=[i for i,t in enumerate(tasks[:-1]) if t.get('grasped') and grip[i]<.6 and
          np.linalg.norm(np.asarray(t['banana_position'][:2])-box)<.025]
    if held:
        delayed=lift<n and (inside-lift)*.02>12.58
        target=lift+629 if delayed else held[0]
        i=min(held,key=lambda i:abs(i-target));selected.append(('held_near_box_delay' if delayed else 'held_near_box_reference',i+1))
    availability['held_near_box_available']=bool(held)
    entries=[dict(role=role,step=step,rollout=str(run),source_episode=str(source),source_layout_id=meta['layout_id'],
                 source_split='train',selection='delay_trigger' if role.endswith('delay') else 'reference_coverage') for role,step in selected]
    return entries,availability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-new',type=int,default=100);p.add_argument('--select-only',action='store_true')
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    ledger=read(out/'comparison.json');assert ledger['complete']
    selection=out/'recovery_selection.json'
    if not selection.exists():
        plan=[];availability=[]
        for row in ledger['results']:
            if row['profile']!='headless':continue
            assert row['split']=='train'
            assert read(Path(row['run_directory'])/'server.json')['checkpoint'].endswith('/banana_v6_long100_accum8000/7999')
            entries,a=select(row)
            for e in entries:e['index']=len(plan);plan.append(e)
            availability.append(dict(parent=row['parent'],layout_id=row['layout_id'],**a))
        write(selection,dict(schema='v6-progress-recovery-selection-v1',source_ledger_sha256=sha(out/'comparison.json'),
            entries=plan,availability=availability,expert_config='configs/banana_motion_expert.toml',
            expert_config_sha256=sha(ROOT/'configs/banana_motion_expert.toml'),
            expert_source_sha256=sha(ROOT/'src/pi0_repro/banana_expert.py'),
            scope='Only headless train-parent v6 states. Delayed stages prioritized with training-expert thresholds; otherwise reference coverage explicitly labeled. Max one near-grasp and one held-near-box state per parent. No state replacement after outcomes. No acceptance/export/train.'))
    plan=read(selection);assert plan['source_ledger_sha256']==sha(out/'comparison.json')
    assert plan['expert_config_sha256']==sha(ROOT/plan['expert_config'])
    assert plan['expert_source_sha256']==sha(ROOT/'src/pi0_repro/banana_expert.py')
    if args.select_only:print(json.dumps(plan,indent=2));return
    path=out/'recovery_candidates.json'
    record=read(path) if path.exists() else dict(schema='v6-progress-recovery-candidates-v1',selection_sha256=sha(selection),complete=False,results=[])
    assert record['selection_sha256']==sha(selection)
    assert len({r['index'] for r in record['results']})==len(record['results'])
    done={r['index'] for r in record['results']};count=0
    for entry in plan['entries']:
        if entry['index'] in done:continue
        if count>=args.max_new:break
        row=dict(entry);print('START correction',entry['index'],entry['role'],flush=True)
        before=set((ROOT/'data/banana_demos').iterdir())
        try:
            collection=collect_recovery(ROOT/plan['expert_config'],Path(entry['rollout']),Path(entry['source_episode']),entry['step'])
            meta=read(collection/'episode_000/episode.json')
            row.update(collection=str(collection),automatic_success=meta['automatic_success'],error=meta.get('error'),steps=meta['steps'])
            print('END correction',entry['index'],row['automatic_success'],row['error'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}'
            row['partial_directories']=[str(p) for p in set((ROOT/'data/banana_demos').iterdir())-before]
            record['results'].append(row);write(path,record);raise
        record['results'].append(row);write(path,record);count+=1
    record['complete']=len(record['results'])==len(plan['entries']);write(path,record)
    print('CANDIDATES',len(record['results']),'/',len(plan['entries']),'pending explicit review',flush=True)


if __name__=='__main__':main()
