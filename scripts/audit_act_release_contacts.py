"""Instrument the already declared gripper-opening branch, without new controls."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import mujoco
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
   names=[env.physics.model.id2name(int(v),'geom') for v in (c.geom1,c.geom2)];f=np.empty(6);mujoco.mj_contactForce(env.physics.model.ptr,env.physics.data.ptr,i,f);rs.append(dict(geoms=names,distance=float(c.dist),normal_force=float(f[0]),force_norm=float(np.linalg.norm(f[:3])),robot_contact=bool(g & env.robot_geoms)))
  return rs
 for case in cases:
  if case['index'] not in plan['unique_lifted_cases']:continue
  original=np.load(Path(case['run_directory'])/'trajectory.npz');branch=np.load(out/f"case_{case['index']:02d}/open_only.npz");env=BananaBoxEnv(SceneConfig(**case['scene']),max_episode_steps=2200,render_observations=False,terminate_on_success=False);obs,_=env.reset(seed=case['seed']);error=0.
  try:
   for i,u in enumerate(original['requested_action']):
    error=max(error,float(np.max(abs(obs['agent_pos']-original['state'][i]))));obs,_,_,_,info=env.step(u)
   before=contacts(env);qbefore=obs['agent_pos'].tolist()
   for i,u in enumerate(branch['requested_action']):
    obs,_,_,_,info=env.step(u);error=max(error,float(np.max(abs(obs['agent_pos']-branch['state'][i]))))
   assert error==0
   row=dict(case=case['index'],replay_error=error,before_contacts=before,after_contacts=contacts(env),before_state=qbefore,after_state=obs['agent_pos'].tolist(),final_target=branch['requested_action'][-1].tolist(),final_info=info);rows.append(row)
   Image.fromarray(env.render()).save(out/f"case_{case['index']:02d}/open_final_top.png");Image.fromarray(env.physics.render(height=480,width=640,camera_id='angle')).save(out/f"case_{case['index']:02d}/open_final_angle.png")
  finally:env.close()
 p=out/'contacts.json';assert not p.exists();p.write_text(json.dumps(dict(complete=True,rows=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Instrumentation replays exactly the six previously declared opening branches; not a new tuning value or policy result. Contacts include MuJoCo contacts used by original task flags, even when instantaneous normal force is zero.'),indent=2)+'\n');print('COMPLETE',len(rows))
if __name__=='__main__':main()
