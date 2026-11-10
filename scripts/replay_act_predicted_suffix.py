"""Compare recorded replanning to one frozen full100-step ACT prediction."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,mujoco
from PIL import Image
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.scene import SceneConfig

def main():
 out=ROOT/'outputs/act_release_diagnosis_20261029';p=json.loads((out/'suffix_plan.json').read_text());pred=json.loads((out/'full_predictions.json').read_text());cases=json.loads((ROOT/'outputs/act_baseline_20261029/evaluation.json').read_text())['results'];rows=[]
 for selection in p['cases']:
  case=cases[selection['case']];step=selection['step'];old=np.load(Path(case['run_directory'])/'trajectory.npz');row=next(r for r in pred['rows'] if r['case']==case['index'] and r['step']==step);chunk=np.array(row['action_chunk'],np.float32);assert np.array_equal(chunk[:25],old['requested_action'][step:step+25]);results=[];initial=None
  for mode in ['recorded_replanning','frozen_full_chunk']:
   env=BananaBoxEnv(SceneConfig(**case['scene']),max_episode_steps=2200,render_observations=False,terminate_on_success=False);obs,_=env.reset(seed=case['seed']);error=0.
   try:
    for i in range(step):
     error=max(error,float(np.max(abs(obs['agent_pos']-old['state'][i]))));obs,_,_,_,info=env.step(old['requested_action'][i])
    error=max(error,float(np.max(abs(obs['agent_pos']-old['state'][step]))));assert error==0
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;z=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,z,spec);digest=hashlib.sha256(z.tobytes()).hexdigest()
    if initial is None:initial=digest
    assert digest==initial;trace=[];states=[];us=old['requested_action'][step:step+100] if mode=='recorded_replanning' else chunk
    for j,u in enumerate(us):
     if mode=='recorded_replanning':assert np.array_equal(obs['agent_pos'],old['state'][step+j])
     obs,_,_,_,info=env.step(u);trace.append(info);states.append(obs['agent_pos'].copy())
    prefix=out/f"case_{case['index']:02d}";np.savez_compressed(prefix/(mode+'_suffix.npz'),state=states,requested_action=us);(prefix/(mode+'_suffix_tasks.json')).write_text(json.dumps(trace,indent=2)+'\n');Image.fromarray(env.physics.render(height=480,width=640,camera_id='angle')).save(prefix/(mode+'_suffix_angle.png'));results.append(dict(mode=mode,prefix_error=error,initial_integration_sha256=digest,success_any=any(t['is_success'] for t in trace),released_any=any(t['released'] for t in trace),final={k:info[k] for k in ['is_success','inside_box','released','supported_by_box','stable']},gripper_range=[float(us[:,13].min()),float(us[:,13].max())]))
   finally:env.close()
  rows.append(dict(case=case['index'],step=step,results=results));print('CASE',case['index'],[(r['mode'],r['success_any'],r['released_any']) for r in results],flush=True)
 dest=out/'suffix_results.json';assert not dest.exists();dest.write_text(json.dumps(dict(complete=True,rows=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=p['scope']),indent=2)+'\n')
if __name__=='__main__':main()
