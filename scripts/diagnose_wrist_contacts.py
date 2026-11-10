"""Exact frozen-control replay with aligned2ms contact/constraint diagnostics."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write

OUT=ROOT/'outputs/wrist_contact_diagnosis_20261028'
SOURCES=dict(control=(ROOT/'outputs/20261028-210846-ed71d1',419),v6=(ROOT/'outputs/20261028-213922-b9dac9',397))


def replay(source,peak,output,instrument):
    import mujoco
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    meta=read(source/'run.json');scene=read(source/'episode_000_scene.json')['scene']
    with np.load(source/'episode_000.npz',allow_pickle=False) as data:
        actions=data['action'].copy();reference=data['state'].copy()
    env=BananaBoxEnv(SceneConfig(**scene),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
    states=[];samples=[];tasks=[]
    try:
        assert (source/'scene.xml').read_text()==env.xml+'\n','Scene XML differs beyond the recording newline'
        obs,_=env.reset(seed=meta['config']['simulation']['seed'])
        assert np.max(np.abs(obs['agent_pos']-reference[0]))<=1e-6
        physics=env.physics;model=physics.model;data=physics.data
        joint=model.name2id('vx300s_right/wrist_angle','joint');dof=int(model.jnt_dofadr[joint]);qpos=int(model.jnt_qposadr[joint])
        actuator=int(np.flatnonzero(np.asarray(model.actuator_trnid)[:,0]==joint)[0])
        assert physics.legacy_step and int(model.opt.integrator)==int(mujoco.mjtIntegrator.mjINT_EULER)
        original=physics.step
        start,finish=peak-60,peak+60
        groups={'contact':np.arange(5,8),'joint_limit':np.array([3]),'joint_friction':np.array([1]),'equality':np.array([0])}
        def instrumented_step(nstep=1):
            if not start<=env.steps<=finish:return original(nstep=nstep)
            # Match the legacy Euler sequence exactly: step2, then step1. Forces
            # and Jacobians belong to the saved pre-integration state; qpos after
            # step2 belongs to the next2ms state. Reading adds no forward solve.
            with physics.check_invalid_state():
                for substep in range(nstep):
                    before_q=float(data.qpos[qpos]);before_v=float(data.qvel[dof]);before_time=float(data.time)
                    mujoco.mj_step2(model.ptr,data.ptr)
                    torques={}
                    for label,kinds in groups.items():
                        weights=np.asarray(data.efc_force).copy()
                        weights[~np.isin(np.asarray(data.efc_type),kinds)]=0
                        torque=np.zeros(model.nv);mujoco.mj_mulJacTVec(model.ptr,data.ptr,torque,weights)
                        torques[label]=float(torque[dof])
                    contacts=[]
                    for i in range(data.ncon):
                        contact=data.contact[i];force=np.zeros(6)
                        mujoco.mj_contactForce(model.ptr,data.ptr,i,force)
                        names=[model.id2name(int(g),'geom') or '' for g in (contact.geom1,contact.geom2)]
                        contacts.append(dict(geoms=names,distance_m=float(contact.dist),dim=int(contact.dim),
                            active=bool(contact.efc_address>=0),force_contact_frame=force.tolist()))
                    samples.append(dict(control_step=env.steps,substep=substep,pre_physics_time=before_time,
                        pre_episode_time=env.steps*.02+substep*.002,pre_q=before_q,pre_qvel=before_v,
                        post_q=float(data.qpos[qpos]),post_qvel=float(data.qvel[dof]),target=float(data.ctrl[actuator]),
                        qacc=float(data.qacc[dof]),actuator_force=float(data.actuator_force[actuator]),
                        qfrc_actuator=float(data.qfrc_actuator[dof]),qfrc_passive=float(data.qfrc_passive[dof]),
                        qfrc_bias=float(data.qfrc_bias[dof]),qfrc_constraint=float(data.qfrc_constraint[dof]),
                        constraint_torque=torques,contacts=contacts))
                    mujoco.mj_step1(model.ptr,data.ptr)
        if instrument:physics.step=instrumented_step
        for action in actions:
            states.append(obs['agent_pos'].copy());obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
        states=np.asarray(states);error=float(np.max(np.abs(states-reference)))
        summary=dict(source=str(source),source_npz_sha256=sha(source/'episode_000.npz'),source_scene_sha256=sha(source/'episode_000_scene.json'),
            instrumented=instrument,source_state_max_abs_error=error,reference_match=error<=1e-6,steps=len(actions),
            final_success=tasks[-1]['is_success'],samples=len(samples),focus_joint='vx300s_right/wrist_angle',dof=dof,qpos_address=qpos,
            focus_peak_state_index=peak,physics_dt=.002,force_alignment='Solved forces/Jacobians use saved pre-integration q/qvel at t; post_q/post_qvel are t+2ms. No added forward solves. Task/control samples at20ms.',
            scope='Open-loop replay only; no policy inference/training/candidate claim. Read-only force instrumentation must pass full recorded state replay check.')
        np.savez_compressed(output/'states.npz',state=states)
        if instrument:(output/'substeps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
        write(output/'summary.json',summary)
        assert summary['reference_match'],f'Replay drift: {error}'
        return summary
    finally:env.close()


def main():
    OUT.mkdir(exist_ok=False)
    plan=dict(schema='wrist-contact-force-replay-v1',sources={k:dict(path=str(v[0]),peak_state_index=v[1],npz_sha256=sha(v[0]/'episode_000.npz')) for k,v in SOURCES.items()},
        script_sha256=sha(Path(__file__)),source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ['banana_env.py','scene.py']},
        method='Full frozen-action plain replay then full replay with only peak+-60 control steps instrumented at2ms. Each must reproduce all recorded pre-step14D states within1e-6.')
    write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes());results=[]
    for model,(source,peak) in SOURCES.items():
        for instrument in (False,True):
            sub=OUT/(model+('_instrumented' if instrument else '_plain'));sub.mkdir()
            result=replay(source,peak,sub,instrument);results.append(result);write(OUT/'summary.json',dict(complete=False,results=results))
            print(model,instrument,result['source_state_max_abs_error'],result['samples'],flush=True)
    write(OUT/'summary.json',dict(complete=True,results=results))


if __name__=='__main__':main()
