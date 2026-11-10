"""Exact candidate replay against bound baseline/tight-only expert references."""
from pathlib import Path
import sys
import hashlib
import json
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from experiment_ik_continuous import OUT,FIRST
from evaluate_v6_progress import read,sha,write
from precision_quality_gates import score
from select_replication_takeovers import box_metrics


def main():
    import numpy as np
    import mujoco
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.recovery import restore_learner_state
    from pi0_repro.motion_quality import motion_quality
    plan=read(OUT/'plan.json');ledger=read(OUT/'candidates.json');first=read(FIRST/'aggregate.json');oldplan=read(FIRST/'plan.json')
    assert ledger['complete'] and len(ledger['results'])==8 and not (OUT/'audit.json').exists()
    checks=[]
    def check(n,v):checks.append(dict(name=n,passed=bool(v)))
    check('plan_binding',ledger['plan_sha256']==sha(OUT/'plan.json'))
    check('first_audit_bound',plan['first_audit_sha256']==sha(FIRST/'audit.json') and read(FIRST/'audit.json')['passed'])
    check('first_result_bound',plan['first_aggregate_sha256']==sha(FIRST/'aggregate.json') and plan['references']['baseline']==first['groups']['baseline']['records'] and plan['references']['tight_ik']==first['groups']['tight_ik']['records'])
    check('same_gates',plan['gates']==oldplan['gates'])
    check('evaluator_reproduces_first_failed_gates',score(first['groups']['baseline']['records'],first['groups']['tight_ik']['records'],oldplan['gates'])==first['gate_checks'])
    check('off_mode_compatibility',len(ledger['compatibility'])==2 and all(r['exact_arrays'] for r in ledger['compatibility']))
    for c in ledger['compatibility']:
        reference=plan['references']['baseline' if c['profile']=='baseline' else 'tight_ik'][0]
        with np.load(Path(c['collection'])/'episode_000/trajectory.npz') as a,np.load(Path(reference['collection'])/'episode_000/trajectory.npz') as b:
            check('compatibility_arrays:'+c['profile'],all(np.array_equal(a[k],b[k]) for k in ('state','action','next_state','phase')))
    for n,d in plan['source_sha256'].items():check('source:'+n,sha(ROOT/'src/pi0_repro'/n)==sha(OUT/'frozen_source'/n)==d)
    for p,c in plan['configs'].items():check('config:'+p,sha(Path(c['path']))==c['sha256'])
    for profile,rows in plan['references'].items():
        for r in rows:check(f'reference:{profile}:{r["parent"]}',sha(Path(r['collection'])/'episode_000/trajectory.npz')==r['trajectory_sha256'])
    proofs=OUT/'replays';proofs.mkdir(exist_ok=False);records=[]
    for row in ledger['results']:
        i=row['index'];entry=plan['entries'][row['entry_index']];parent=plan['source_parents'][row['parent']]
        path=Path(row['collection'])/'episode_000';meta=read(path/'episode.json');quality=read(path/'quality.json');collection=read(path.parent/'collection.json')
        source=Path(entry['source_episode'])
        check(f'{i}:source_parent',sha(source/'episode.json')==parent['source_metadata_sha256'] and sha(source/'trajectory.npz')==parent['source_trajectory_sha256'])
        check(f'{i}:train_lineage',meta['split']=='train' and meta['recovery']['source_split']=='train' and meta['recovery']['source_layout_id']==entry['source_layout_id'])
        check(f'{i}:pending',not meta['eligible_for_training'] and meta['human_review']=='pending' and not meta.get('assistant_review'))
        check(f'{i}:config',collection['expert_config']==plan['configs']['continuous']['expert_config'])
        check(f'{i}:code',all(plan['source_sha256'].get(n)==d for n,d in collection['code_sha256'].items()))
        check(f'{i}:outcome',meta['automatic_success']==row['automatic_success'] and meta['error']==row['error'])
        with np.load(path/'trajectory.npz',allow_pickle=False) as data:
            states=data['state'].copy();following=data['next_state'].copy();actions=data['action'].copy();phase=data['phase'].copy();times=data['timestamp'].copy()
        n=len(actions);check(f'{i}:arrays',states.shape==following.shape==actions.shape==(n,14) and np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(following).all())
        check(f'{i}:continuity',np.array_equal(states[1:],following[:-1]))
        check(f'{i}:timestamps',np.allclose(np.diff(times),.02,atol=1e-7,rtol=0))
        check(f'{i}:frames',len(list((path/'rgb').glob('*.png')))==n)
        env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=4000,terminate_on_success=False,render_observations=False)
        try:
            env.reset(seed=meta['reset_seed']);proof=proofs/f'case_{i:02d}';proof.mkdir()
            request={k:entry[k] for k in ('rollout','source_episode','step','source_split')};obs,_,provenance=restore_learner_state(env,request,proof)
            check(f'{i}:takeover',provenance['replay_state_max_error']==0)
            spec=mujoco.mjtState.mjSTATE_INTEGRATION;start=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,start,spec)
            start_sha=hashlib.sha256(start.tobytes()).hexdigest();ref=plan['references']['baseline'][row['parent']]
            check(f'{i}:same_physical_start',start_sha==ref['start_integration_sha256'])
            m=env.physics.model;d=env.physics.data;j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
            err=posterr=0.;tasks=[];forces=[]
            for k,action in enumerate(actions):
                err=max(err,float(np.max(abs(obs['agent_pos']-states[k]))));obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                posterr=max(posterr,float(np.max(abs(obs['agent_pos']-following[k]))));forces.append(float(d.actuator_force[a]))
            check(f'{i}:exact_pre_post',err==posterr==0)
            if n:check(f'{i}:environment_outcome',tasks[-1]['is_success']==meta['environment_success'])
            np.savez_compressed(proof/'samples.npz',integration_start=start,poststep_wrist_effort_nm=forces)
        finally:env.close()
        jerk=lambda q:None if len(q)<4 else float(np.max(abs(np.diff(q[:,11].astype(float),n=3)/.02**3)))
        lift=phase=='lift';valid=np.array([all(lift[k:k+4]) for k in range(max(0,n-3))],bool);j=abs(np.diff(states[:,11].astype(float),n=3)/.02**3) if n>=4 else np.empty(0)
        mask=lift[:-1]&lift[1:];delta=np.diff(actions[:,11].astype(float))
        records.append(dict(index=i,parent=row['parent'],profile='continuous',role=entry['role'],collection=row['collection'],source_layout_id=entry['source_layout_id'],
            automatic_success=meta['automatic_success'],error=meta['error'],structural_pass=quality['structural_pass'],frames=n,
            trajectory_sha256=sha(path/'trajectory.npz'),start_integration_sha256=start_sha,pre_post_state_error=max(err,posterr),
            first_second_wrist_peak=jerk(states[:51]),first_three_second_wrist_peak=jerk(states[:151]),full_lift_wrist_peak=None if not valid.any() else float(j[valid].max()),
            lift_wrist_target_repeat_fraction=None if not mask.any() else float(np.mean(abs(delta[mask])<1e-8)),
            motion_quality=motion_quality(states,actions,tasks) if n>=4 else None,box=box_metrics(following,actions,tasks,meta['scene']) if n else dict(available=False,first_above_to_containment_seconds=None),
            poststep_effort_peak_nm=None if not forces else float(np.max(abs(np.asarray(forces)))),video=str(path/'preview.mp4'),video_sha256=sha(path/'preview.mp4')))
        print('AUDIT',row['parent'],row['automatic_success'],flush=True)
    base=plan['references']['baseline'];candidate=sorted(records,key=lambda r:r['parent']);tests=score(base,candidate,plan['gates'])
    failed=[c for c in checks if not c['passed']];passed=not failed and all(t['passed'] for t in tests)
    write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,plan_sha256=sha(OUT/'plan.json'),ledger_sha256=sha(OUT/'candidates.json'),gate_evaluator_sha256=sha(ROOT/'scripts/precision_quality_gates.py')))
    result=dict(schema='precision-continuous-result-v1',audit_passed=not failed,quality_gate_passed=passed,gate_checks=tests,
        groups={p:dict(attempts=len(v),successes=sum(r['automatic_success'] for r in v),records=v) for p,v in [('baseline',base),('tight_only',plan['references']['tight_ik']),('continuous',candidate)]},
        decision='quality_pass_pending_explicit_data_review' if passed else 'do_not_accept_export_or_train',scope=plan['scope'])
    write(OUT/'aggregate.json',result);(OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes());(OUT/'gate_evaluator_source.py').write_bytes((ROOT/'scripts/precision_quality_gates.py').read_bytes())
    write(OUT/'training_decision.json',dict(quality_gate_passed=passed,accepted=False,exported=False,training_started=False,aggregate_sha256=sha(OUT/'aggregate.json'),decision=result['decision']))
    print(json.dumps(dict(audit_checks=len(checks),failed_checks=failed,gate_checks=tests,decision=result['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
