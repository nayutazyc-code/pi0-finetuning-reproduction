"""Label actual train-only learner states; retain every selected attempt and failure."""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MUJOCO_GL','glfw')
from pi0_repro.recovery import collect_recovery

out=ROOT/'outputs/motion_first_20261026';data=json.loads((out/'data_collections.json').read_text())
assert data['stage']=='learners_recorded'
learners=json.loads(Path(data['learner_ledger']).read_text());assert learners['complete']
path=out/'corrections.json'
if path.exists():raise FileExistsError('Preserve previous correction attempts')
plan=[];skipped=[]
for run in learners['results']:
    assert run['split']=='train'
    source=Path(run['source_episode']);meta=json.loads((source/'episode.json').read_text());assert meta['split']=='train'
    p=Path(run['run_directory']);tasks=[json.loads(s) for s in (p/'episode_000_task.jsonl').read_text().splitlines()]
    with np.load(p/'episode_000.npz',allow_pickle=False) as a:grip=a['action'][:,13];n=len(grip)
    near=next((i for i,r in enumerate(tasks) if r.get('near_grasp_region') and not r.get('ever_grasped')),None)
    first_lift=next((i for i,r in enumerate(tasks) if r.get('ever_grasped')),n)
    selected=[]
    if near is not None:
        for offset in (5,15):
            i=min(near+offset,first_lift-1,n-2)
            if i>=near and tasks[i].get('near_grasp_region') and not tasks[i].get('ever_grasped'):
                selected.append(('near_grasp_'+str(offset),i+1))
    box=np.asarray(meta['scene']['box_position'][:2])
    held=next((i for i,r in enumerate(tasks[:-1]) if r.get('grasped') and grip[i]<.6 and np.linalg.norm(np.asarray(r['banana_position'][:2])-box)<.025),None)
    if held is not None:selected.append(('held_near_box',held+1))
    supported=next((i for i,r in enumerate(tasks[:-1]) if r.get('ever_grasped') and r.get('inside_box') and r.get('supported_by_box') and r.get('stable') and not r.get('released') and grip[i]<.6),None)
    if supported is not None:selected.append(('supported_release',supported+1))
    seen=set()
    for role,step in selected:
        if step in seen:continue
        seen.add(step);plan.append({'index':len(plan),'role':role,'step':int(step),'rollout':str(p),'source_episode':str(source),'source_layout_id':meta['layout_id'],'source_split':'train'})
    skipped.append({'source_layout_id':meta['layout_id'],'held_entry_available':held is not None,'supported_entry_available':supported is not None,'selected_unique_states':len(seen)})
record={'scope':'Selected actual learner states only from12preassignednewtrainlayouts. Two near-grasp offsets5/15steps, heldXY<.025m, supportedinside+stable+robotcontact+grip<.6. State step is after the diagnostic row. No outcome-based replacement or holdout labels.',
        'selection_plan':plan,'state_availability':skipped,'complete':False,'results':[]}
path.write_text(json.dumps(record,indent=2)+'\n')
for entry in plan:
    print('START correction',entry['index']+1,'/',len(plan),entry['role'],entry['source_layout_id'],flush=True)
    row=dict(entry)
    before=set((ROOT/'data/banana_demos').iterdir())
    try:
        collection=collect_recovery(ROOT/'configs/banana_motion_expert.toml',Path(entry['rollout']),Path(entry['source_episode']),entry['step'])
        meta=json.loads((collection/'episode_000/episode.json').read_text())
        with np.load(collection/'episode_000/trajectory.npz',allow_pickle=False) as a:first_phase=str(a['phase'][0]) if len(a['phase']) else None
        row.update(collection=str(collection),automatic_success=meta['automatic_success'],error=meta.get('error'),steps=meta['steps'],first_phase=first_phase)
        print('END correction',entry['index']+1,'success',meta['automatic_success'],'first_phase',first_phase,'error',meta.get('error'),flush=True)
    except BaseException as exc:
        row['infrastructure_error']=f'{type(exc).__name__}: {exc}'
        row['partial_directories']=[str(p) for p in set((ROOT/'data/banana_demos').iterdir())-before]
        record['results'].append(row);path.write_text(json.dumps(record,indent=2)+'\n');raise
    record['results'].append(row);path.write_text(json.dumps(record,indent=2)+'\n')
record['complete']=True;path.write_text(json.dumps(record,indent=2)+'\n')
data['corrections']=[r['collection'] for r in record['results'] if r.get('collection')];data['stage']='corrections_collected_pending_review'
(out/'data_collections.json').write_text(json.dumps(data,indent=2)+'\n')
print('COMPLETE corrections',len(record['results']),flush=True)
