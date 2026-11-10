"""Frozen ACT command replay, release-label audit and isolated gripper branch."""
from pathlib import Path
import sys,json,hashlib,math
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
import numpy as np
from PIL import Image
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.scene import SceneConfig
import mujoco

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 out=ROOT/'outputs/act_release_diagnosis_20261029';parent=ROOT/'outputs/act_baseline_20261029';ledger=read(parent/'evaluation.json');plan=read(out/'plan.json');assert not (out/'diagnosis.json').exists();records=[];queries=[]
 for case in ledger['results']:
  path=Path(case['run_directory']);tasks=[json.loads(s) for s in (path/'task.jsonl').read_text().splitlines()];a=np.load(path/'trajectory.npz');states=a['state'];actions=a['requested_action'];n=len(states);indices=[]
  if case['index'] in plan['unique_lifted_cases']:
   first=next(i+1 for i,t in enumerate(tasks) if t['inside_box']);first=math.ceil(first/25)*25;indices=sorted(set([min(first,n-25),min(first+50,n-25),min(first+250,n-25),n-25]))
  env=BananaBoxEnv(SceneConfig(**case['scene']),max_episode_steps=n+200,render_observations=False,terminate_on_success=False);obs,_=env.reset(seed=case['seed']);error=0.;caseout=out/f"case_{case['index']:02d}";caseout.mkdir()
  try:
   for i in range(n):
    error=max(error,float(np.max(np.abs(obs['agent_pos']-states[i]))))
    if i in indices:
     p=caseout/f'query_{i:04d}.png';Image.fromarray(env.render()).save(p);q=dict(kind='learner',case=case['index'],step=i,image=str(p.relative_to(out)),state=obs['agent_pos'].tolist(),executed_gripper_targets=actions[i:i+25,13].tolist());queries.append(q)
    obs,_,_,_,info=env.step(actions[i])
   assert error==0.,f'Replay mismatch {case["index"]}:{error}'
   row=dict(index=case['index'],layout=case['layout'],split=case['split'],state_error=error,final_info=info,branches=[])
   if indices:
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,state,spec);counters={k:getattr(env,k) for k in ['steps','hold_steps','ever_grasped','ever_lifted','_needs_reset']}
    for mode in ['hold','open_only']:
     mujoco.mj_setState(env.physics.model.ptr,env.physics.data.ptr,state,spec)
     for k,v in counters.items():setattr(env,k,v)
     trace=[];qs=[];us=[]
     for j in range(100):
      u=actions[-1].copy()
      if mode=='open_only':
       x=min((j+1)/20,1.);alpha=x*x*x*(10+x*(-15+6*x));u[13]=actions[-1,13]+alpha*(1-actions[-1,13])
      us.append(u.copy());obs,_,_,_,info=env.step(u);trace.append(info);qs.append(obs['agent_pos'].copy())
     np.savez_compressed(caseout/(mode+'.npz'),state=qs,requested_action=us);write(caseout/(mode+'_tasks.json'),trace);row['branches'].append(dict(mode=mode,success_any=any(t['is_success'] for t in trace),released_any=any(t['released'] for t in trace),final={k:trace[-1][k] for k in ['is_success','inside_box','released','supported_by_box','stable']},first_release_step=next((j+1 for j,t in enumerate(trace) if t['released']),None)))
   records.append(row);write(out/'progress.json',dict(completed=len(records),records=records));print('REPLAY',case['index'],'error',error,'branches',row['branches'],flush=True)
  finally:env.close()
 # Audit the actually accepted training labels; stage-balanced samples are diagnostic only.
 dataset=ROOT/'data/lerobot/20261029-135803-ik-replaced-b8ed3a';raw=Path(read(dataset/'dataset_manifest.json')['source_path']);mapping=read(dataset/'local/pi0_banana_train/source_episodes.json');labels=[]
 for r in mapping['episodes']:
  ep=raw/r['source_episode'];a=np.load(ep/'trajectory.npz');ph=a['phase'];rel=np.flatnonzero(ph=='release');labels.append(dict(episode=r['source_episode'],frames=len(ph),release_frames=len(rel),release_action_range=None if not len(rel) else [float(a['action'][rel,13].min()),float(a['action'][rel,13].max())],open_action_frames=int(np.sum(a['action'][:,13]>=.8)),phase_frames={str(p):int(np.sum(ph==p)) for p in np.unique(ph)}))
  if len(rel):
   first=int(rel[0]);indices=sorted(set(max(0,min(first+d,len(ph)-100)) for d in [-50,-25,0,25,50]))
   for i in indices:
    target=out/f"teacher_{r['lerobot_episode_index']:02d}_{i:04d}.png";Image.open(ep/'rgb'/f'{i:06d}.png').save(target);queries.append(dict(kind='teacher',case=r['lerobot_episode_index'],step=i,image=target.name,state=a['state'][i].tolist(),target_actions=a['action'][i:i+100].tolist(),release_start=first,phase=str(ph[i])))
 write(out/'queries.json',queries);write(out/'diagnosis.json',dict(complete=True,records=records,training_labels=labels,scope='Exact replay of all16 recorded command trajectories. Six unique lifted cases get same-state hold vs gripper-only opening branches; original model/actions unchanged. Branch task successes are diagnostic counterfactuals, not policy success improvement. Teacher query selection phase-balanced, not a validation score.'))
if __name__=='__main__':main()
