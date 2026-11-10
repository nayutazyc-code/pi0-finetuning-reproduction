"""Sustained tracking-effort guard: exact expert invariance then frozen learners."""
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import mujoco
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.motion_quality import motion_quality
    source_audit=ROOT/'outputs/contact_target_audit_20261029'
    audit=read(source_audit/'summary.json');assert audit['complete']
    from pi0_repro.wrist_effort_guard import WristEffortGuard
    out=ROOT/'outputs/sustained_wrist_guard_screen_20261029';out.mkdir(exist_ok=False)
    experts=[s for s in audit['plan']['sources'] if s['kind']=='expert_train']
    peak=max(r['abs_wrist_effort_nm']['max'] for r in audit['records'] if r['kind']=='expert_train')
    limit=math.ceil(1.5*peak/.25)*.25
    assert limit==1.
    plan=dict(schema='sustained-wrist-guard-frozen-screen-v1',audit_sha256=sha(source_audit/'summary.json'),
        previous_failed_screen_sha256=sha(ROOT/'outputs/wrist_effort_screen_20261029/summary.json'),script_sha256=sha(Path(__file__)),
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ('scene.py','banana_env.py','wrist_effort_guard.py')},
        calibration='One rule: limit=ceil(1.5*max trainexpert POSTSTEP demand/0.25)*0.25=1Nm; activate after3high samples>limit; release after3samples<limit/2. Samples precede new command, using prior command and current measured joint state.',
        treatment='Only right wrist actuator effort is capped while sustained guard is active. Fixed kp/damping/contacts/actions; no task, banana position, contact-oracle or phase input. Normal intra-step expert pulses remain permitted.',
        gate=dict(success='both successes retained',active_jerk_p99_ratio_max=1.05,global_jerk_peak_ratio_max=.90,wrist_jerk_peak_ratio_max=.80,physical_increment_ratio_max=1.05,
                  experts='all20 exact state replay; zero activation'),
        scope='New duration-based intervention, not relabeling failed6Nm static cap. Frozen commands only. Do not run live trial if this gate fails.')
    write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    report=dict(plan=plan,complete=False,calibration=[],results=[])
    experts=[s for s in audit['plan']['sources'] if s['kind']=='expert_train']
    learners=[s for s in audit['plan']['sources'] if s['kind']=='learner_diagnostic']
    try:
        for item in experts:
            with np.load(item['trajectory'],allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
            env=BananaBoxEnv(SceneConfig(**item['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
            try:
                obs,_=env.reset(seed=item['seed']);p=env.physics;m=p.model;data=p.data
                j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
                guard=WristEffortGuard(p,limit_nm=limit);error=0.;activations=0
                for i,action in enumerate(actions):
                    error=max(error,float(np.max(abs(obs['agent_pos']-reference[i]))))
                    receipt=guard.before_step();activations+=int(receipt['active']);obs,*_=env.step(action)
                assert error==0 and activations==0,(item['name'],error,activations)
                row=dict(name=item['name'],source_sha256=sha(Path(item['trajectory'])),state_error=error,guard_active_steps=activations)
                report['calibration'].append(row);write(out/'summary.json',report)
                print('CALIBRATE',item['name'],row['guard_active_steps'],flush=True)
            finally:env.close()
        report.update(expert_poststep_peak_nm=peak,candidate_limit_nm=limit)
        write(out/'summary.json',report)
        for item in learners:
            with np.load(item['trajectory'],allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
            for capped in (False,True):
                env=BananaBoxEnv(SceneConfig(**item['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
                try:
                    m=env.physics.model;j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
                    gain=m.actuator_gainprm.copy();bias=m.actuator_biasprm.copy();damping=m.dof_damping.copy()
                    guard=WristEffortGuard(env.physics,limit_nm=limit) if capped else None
                    np.testing.assert_array_equal(gain,m.actuator_gainprm);np.testing.assert_array_equal(bias,m.actuator_biasprm);np.testing.assert_array_equal(damping,m.dof_damping)
                    obs,_=env.reset(seed=item['seed']);states=[];tasks=[];post_efforts=[];guard_rows=[]
                    for action in actions:
                        guard_rows.append(guard.before_step() if guard else {'active':False})
                        states.append(obs['agent_pos'].copy());obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                        post_efforts.append(float(env.physics.data.actuator_force[a]))
                    states=np.asarray(states);quality=motion_quality(states,actions,tasks)
                    arms=[0,1,2,3,4,5,7,8,9,10,11,12]
                    moving=np.max(np.abs(np.diff(states[:,arms],axis=0)/.02),axis=1)>.02
                    wristjerk=np.abs(np.diff(states[:,11].astype(float),n=3)/.02**3)
                    active=moving[2:]
                    row=dict(name=item['name'],capped=capped,limit_nm=limit if capped else None,success=tasks[-1]['is_success'],
                        state_error=float(np.max(np.abs(states-reference))),quality=quality,guard_active_steps=sum(r['active'] for r in guard_rows),
                        active_wrist_jerk_peak=float(wristjerk[active].max()),max_post_effort_nm=float(np.max(np.abs(post_efforts))))
                    if not capped:assert row['state_error']==0
                    if capped:assert all(abs(f)<=limit+1e-9 for f,r in zip(post_efforts,guard_rows) if r['active'])
                    report['results'].append(row)
                    np.savez_compressed(out/(item['name']+('_capped' if capped else '_baseline')+'.npz'),state=states,action=actions,guard_active=np.asarray([r['active'] for r in guard_rows]))
                    write(out/'summary.json',report)
                    print('SCREEN',item['name'],capped,row['success'],quality['active_frame_max_jerk_rad_s3'],row['active_wrist_jerk_peak'],flush=True)
                finally:env.close()
        gates=[]
        for item in learners:
            b,c=[r for r in report['results'] if r['name']==item['name']];g=plan['gate']
            a,z=b['quality']['active_frame_max_jerk_rad_s3'],c['quality']['active_frame_max_jerk_rad_s3']
            checks=dict(success=bool(c['success']),active_p99=z['p99']<=a['p99']*g['active_jerk_p99_ratio_max'],
                global_peak=z['max']<=a['max']*g['global_jerk_peak_ratio_max'],wrist_peak=c['active_wrist_jerk_peak']<=b['active_wrist_jerk_peak']*g['wrist_jerk_peak_ratio_max'],
                physical_increment=c['quality']['physical_joint_increment_rad']['max']<=b['quality']['physical_joint_increment_rad']['max']*g['physical_increment_ratio_max'])
            gates.append(dict(name=item['name'],checks=checks,passed=all(checks.values()),p99_ratio=z['p99']/a['p99'],global_peak_ratio=z['max']/a['max'],wrist_peak_ratio=c['active_wrist_jerk_peak']/b['active_wrist_jerk_peak']))
        report.update(complete=True,gates=gates,passed=all(r['passed'] for r in gates))
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write(out/'summary.json',report)
    print('COMPLETE','limit_nm',limit,'passed',report['passed'],'gates',gates,flush=True)


if __name__=='__main__':main()
