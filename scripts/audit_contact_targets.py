"""Read-only expert/learner target tracking and post-step contact audit."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    import mujoco
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    out=ROOT/'outputs/contact_target_audit_20261029';out.mkdir(exist_ok=False)
    dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'
    raw=Path(read(dataset/'dataset_manifest.json')['source_path'])
    mapping=read(dataset/'local/pi0_banana_train/source_episodes.json')
    sources=[]
    for row in mapping['episodes']:
        source=raw/row['source_episode'];meta=read(source/'episode.json');layout=read(source/'layout.json')
        assert row['split']=='train' and meta['review_status']=='accepted'
        if layout.get('recovery') or (layout.get('layout') or {}).get('recovery'):continue
        assert sha(source/'trajectory.npz')==row['source_trajectory_sha256']
        sources.append(dict(kind='expert_train',name=row['source_episode'],source=str(source),
            trajectory=str(source/'trajectory.npz'),scene=meta['scene'],seed=meta['reset_seed'],final_position=meta['final_info']['banana_position']))
    for name,path in [('continued_release','20261028-210846-ed71d1'),('v6_release','20261028-213922-b9dac9')]:
        source=ROOT/'outputs'/path;meta=read(source/'run.json')
        sources.append(dict(kind='learner_diagnostic',name=name,source=str(source),trajectory=str(source/'episode_000.npz'),
            scene=read(source/'episode_000_scene.json')['scene'],seed=meta['config']['simulation']['seed']))
    plan=dict(schema='contact-target-audit-v1',script_sha256=sha(Path(__file__)),sources=sources,
        mapping_sha256=sha(dataset/'local/pi0_banana_train/source_episodes.json'),
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ('scene.py','banana_env.py')},
        scope='Read-only20Hz? sampling is50Hz (20ms) post-step force/target/state audit. Existing2ms force-aligned release diagnosis stays separate. Train experts and validation learner recordings never merged as training labels; no data or policy changes.')
    plan['scope']=plan['scope'].replace('Read-only20Hz? sampling is50Hz (20ms)','Read-only50Hz (20ms)')
    write(out/'plan.json',plan);report=dict(plan=plan,complete=False,records=[])
    def dist(values):
        a=np.asarray(values,float)
        return None if not a.size else dict(count=int(a.size),median=float(np.median(a)),p95=float(np.quantile(a,.95)),p99=float(np.quantile(a,.99)),max=float(np.max(a)))
    try:
        for item in sources:
            source=Path(item['trajectory'])
            with np.load(source,allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
            env=BananaBoxEnv(SceneConfig(**item['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
            try:
                obs,_=env.reset(seed=item['seed']);m=env.physics.model;data=env.physics.data
                joint=m.name2id('vx300s_right/wrist_angle','joint');q=int(m.jnt_qposadr[joint]);dof=int(m.jnt_dofadr[joint])
                actuator=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==joint)[0])
                geom_names=[m.id2name(i,'geom') or '' for i in range(m.ngeom)]
                rows=[];error=0.
                for i,action in enumerate(actions):
                    error=max(error,float(np.max(np.abs(obs['agent_pos']-reference[i]))))
                    obs,_,_,_,info=env.step(action)
                    contact_force=0.;table=False;box=False
                    for k in range(data.ncon):
                        c=data.contact[k]
                        if c.efc_address<0:continue
                        names=(geom_names[c.geom1],geom_names[c.geom2])
                        finger=any(n.startswith('vx300s_right/') and n.endswith(('10_left_gripper_finger','10_right_gripper_finger')) for n in names)
                        if finger and 'table' in names:
                            force=np.empty(6);mujoco.mj_contactForce(m.ptr,data.ptr,k,force);contact_force+=max(0.,float(force[0]));table=True
                        if any(n.startswith('vx300s_right/') for n in names) and any(n.startswith('box_') for n in names):box=True
                    target=float(data.ctrl[actuator]);actual=float(data.qpos[q]);tau=float(data.actuator_force[actuator])
                    rows.append([target,actual,target-actual,tau,float(data.qvel[dof]),float(data.qacc[dof]),contact_force,float(table),float(box),float(info['ever_grasped']),float(info['inside_box'])])
                assert error==0,(item['name'],error)
                if 'final_position' in item:np.testing.assert_allclose(info['banana_position'],item['final_position'],atol=1e-12,rtol=0.)
                a=np.asarray(rows);touch=a[:,7].astype(bool);moving=np.abs(a[:,4])>.02
                result=dict(name=item['name'],kind=item['kind'],trajectory_sha256=sha(source),frames=len(actions),state_error=error,success=info['is_success'],
                    finger_table_steps=int(touch.sum()),right_robot_box_steps=int(a[:,8].sum()),
                    peak_table_force_newton=float(a[:,6].max()),abs_tracking_error_rad=dist(abs(a[:,2])),abs_wrist_effort_nm=dist(abs(a[:,3])),
                    table_contact_effort_nm=dist(abs(a[touch,3])),noncontact_moving_effort_nm=dist(abs(a[~touch&moving,3])),
                    fractions_exceed_effort={str(limit):float(np.mean(abs(a[:,3])>limit)) for limit in (1.,1.5,2.,2.5)},
                    default_force_limited=bool(m.actuator_forcelimited[actuator]),default_force_range=np.asarray(m.actuator_forcerange[actuator]).tolist())
                np.savez_compressed(out/(item['name']+'.npz'),samples=a,columns=np.asarray(['target','post_q','error','actuator_effort_nm','post_qvel','post_qacc','finger_table_normal_force_n','finger_table_active','right_robot_box_active','ever_grasped','inside_box']))
                report['records'].append(result);write(out/'summary.json',report)
                print(item['kind'],item['name'],'table_steps',result['finger_table_steps'],'effort',result['abs_wrist_effort_nm'],'noncontact',result['noncontact_moving_effort_nm'],flush=True)
            finally:env.close()
        report['complete']=True
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write(out/'summary.json',report)


if __name__=='__main__':main()
