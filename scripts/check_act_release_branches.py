"""Fresh-environment branches to avoid cached MuJoCo dynamics after state restore."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,mujoco
from PIL import Image
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.scene import SceneConfig

def main():
 out=ROOT/'outputs/act_release_diagnosis_20261029';plan=json.loads((out/'plan.json').read_text());cases=json.loads((ROOT/'outputs/act_baseline_20261029/evaluation.json').read_text())['results'];rows=[]
 def contacts(env):
  rs=[]
  for i in range(env.physics.data.ncon):
   c=env.physics.data.contact[i];g={int(c.geom1),int(c.geom2)}
   if not(g & set(env.banana_collision_geoms)):continue
   f=np.empty(6);mujoco.mj_contactForce(env.physics.model.ptr,env.physics.data.ptr,i,f);rs.append(dict(geoms=[env.physics.model.id2name(int(v),'geom') for v in (c.geom1,c.geom2)],distance=float(c.dist),normal_force=float(f[0]),robot_contact=bool(g & env.robot_geoms)))
  return rs
 for case in cases:
  if case['index'] not in plan['unique_lifted_cases']:continue
  original=np.load(Path(case['run_directory'])/'trajectory.npz');arms=original['requested_action'];before_hash=None;row=dict(case=case['index'],branches=[])
  for mode in ['hold','open_only']:
   env=BananaBoxEnv(SceneConfig(**case['scene']),max_episode_steps=2200,render_observations=False,terminate_on_success=False);obs,_=env.reset(seed=case['seed']);error=0.
   try:
    for i,u in enumerate(arms):
     error=max(error,float(np.max(abs(obs['agent_pos']-original['state'][i]))));obs,_,_,_,info=env.step(u)
    assert error==0
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;z=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,z,spec);digest=hashlib.sha256(z.tobytes()).hexdigest()
    if before_hash is None:before_hash=digest
    assert digest==before_hash
    before=contacts(env);qbefore=obs['agent_pos'].tolist();trace=[];qs=[];us=[]
    for j in range(100):
     u=arms[-1].copy()
     if mode=='open_only':
      x=min((j+1)/20,1.);alpha=x**3*(10+x*(-15+6*x));u[13]=arms[-1,13]+alpha*(1-arms[-1,13])
     us.append(u.copy());obs,_,_,_,info=env.step(u);trace.append(info);qs.append(obs['agent_pos'].copy())
    d=out/f"case_{case['index']:02d}";np.savez_compressed(d/(mode+'_fresh.npz'),state=qs,requested_action=us);(d/(mode+'_fresh_tasks.json')).write_text(json.dumps(trace,indent=2)+'\n');Image.fromarray(env.physics.render(height=480,width=640,camera_id='angle')).save(d/(mode+'_fresh_angle.png'))
    row['branches'].append(dict(mode=mode,prefix_state_error=error,initial_integration_sha256=digest,before_state=qbefore,after_state=obs['agent_pos'].tolist(),before_contacts=before,after_contacts=contacts(env),success_any=any(t['is_success'] for t in trace),released_any=any(t['released'] for t in trace),final=info))
   finally:env.close()
  rows.append(row);print('CASE',case['index'],[(b['mode'],b['success_any'],b['released_any']) for b in row['branches']],flush=True)
 p=out/'corrected_branches.json';assert not p.exists();p.write_text(json.dumps(dict(complete=True,rows=rows,scope='Same declared gripper-only intervention; fresh identical command-prefix replay for each branch. Exact recorded-state equality and initial full-integration hashes match. Supersedes unvalidated mj_setState branches, not a new parameter search or policy outcome.'),indent=2)+'\n')
if __name__=='__main__':main()
