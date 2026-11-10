"""Exact-replay and frozen full-phase quality gates for the IK tolerance pair."""
import hashlib
import json
from pathlib import Path
import sys
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from experiment_ik_precision import OUT,PRIOR
from evaluate_v6_progress import read,sha,write
from select_replication_takeovers import box_metrics


def main():
    import numpy as np
    import mujoco
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.recovery import restore_learner_state
    from pi0_repro.motion_quality import motion_quality
    from pi0_repro.banana_expert import load_expert_config
    from dataclasses import asdict
    plan=read(OUT/'plan.json');ledger=read(OUT/'candidates.json');previous=read(PRIOR/'aggregate.json')
    assert ledger['complete'] and len(ledger['results'])==len(plan['cases'])==16
    assert not (OUT/'audit.json').exists()
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('plan_binding',ledger['plan_sha256']==sha(OUT/'plan.json'))
    check('selection_binding',plan['source_selection_sha256']==sha(PRIOR/'takeovers.json'))
    check('prior_acquisition_binding',plan['source_acquisition_sha256']==sha(PRIOR/'acquisition.json'))
    check('runner_frozen',plan['runner_sha256']==sha(OUT/'runner_source.py')==sha(ROOT/'scripts/experiment_ik_precision.py'))
    for n,d in plan['source_sha256'].items():check('source:'+n,sha(ROOT/'src/pi0_repro'/n)==sha(OUT/'frozen_source'/n)==d)
    configs={p:asdict(load_expert_config(Path(c['path']))) for p,c in plan['configs'].items()}
    a,b=[dict(configs[p]) for p in ('baseline','tight_ik')];check('tolerance_values',a['ik_position_tolerance']==.0002 and b['ik_position_tolerance']==.00002)
    b['ik_position_tolerance']=a['ik_position_tolerance'];check('single_variable',a==b)
    for p,c in plan['configs'].items():check('config:'+p,sha(Path(c['path']))==c['sha256'] and configs[p]==c['expert_config'])
    records=[];starts={};proofs=OUT/'replays';proofs.mkdir(exist_ok=False)
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        i=row['index'];profile=row['profile'];entry=plan['entries'][row['entry_index']];parent=plan['source_parents'][row['parent']]
        source=Path(entry['source_episode']);path=Path(row['collection'])/'episode_000';meta=read(path/'episode.json');collection=read(path.parent/'collection.json');quality=read(path/'quality.json')
        check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        check(f'{i}:original_source',sha(source/'episode.json')==parent['source_metadata_sha256'] and sha(source/'trajectory.npz')==parent['source_trajectory_sha256'])
        check(f'{i}:expert_config',collection['expert_config']==configs[profile])
        check(f'{i}:source_code',all(plan['source_sha256'].get(n)==d for n,d in collection['code_sha256'].items()))
        check(f'{i}:train_lineage',meta['split']=='train' and meta['recovery']['source_split']=='train' and meta['recovery']['source_layout_id']==entry['source_layout_id'])
        check(f'{i}:pending',meta['human_review']=='pending' and not meta['eligible_for_training'] and not meta.get('assistant_review'))
        check(f'{i}:outcome',row['automatic_success']==meta['automatic_success'] and row['error']==meta['error'])
        with np.load(path/'trajectory.npz',allow_pickle=False) as data:
            states=data['state'].copy();following=data['next_state'].copy();actions=data['action'].copy();times=data['timestamp'].copy();phases=data['phase'].copy()
        n=len(actions);check(f'{i}:arrays',states.shape==following.shape==actions.shape==(n,14) and np.isfinite(states).all() and np.isfinite(following).all() and np.isfinite(actions).all())
        check(f'{i}:continuity',np.array_equal(states[1:],following[:-1]))
        check(f'{i}:timestamps',np.allclose(np.diff(times),.02,atol=1e-7,rtol=0))
        check(f'{i}:frames',len(list((path/'rgb').glob('*.png')))==n)
        prior=next(r for r in previous['groups']['unload']['records'] if r['parent']==row['parent'])
        if profile=='baseline':
            with np.load(Path(prior['collection'])/'episode_000/trajectory.npz',allow_pickle=False) as old:
                check(f'{i}:fresh_baseline_reproduces_previous',np.array_equal(states,old['state']) and np.array_equal(actions,old['action']) and np.array_equal(following,old['next_state']) and np.array_equal(phases,old['phase']))
        env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=4000,terminate_on_success=False,render_observations=False)
        try:
            env.reset(seed=meta['reset_seed']);proof=proofs/f'case_{i:02d}';proof.mkdir()
            request={k:entry[k] for k in ('rollout','source_episode','step','source_split')}
            obs,_,provenance=restore_learner_state(env,request,proof);check(f'{i}:takeover',provenance['replay_state_max_error']==0)
            spec=mujoco.mjtState.mjSTATE_INTEGRATION;start=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,start,spec)
            start_sha=hashlib.sha256(start.tobytes()).hexdigest();starts.setdefault(row['parent'],[]).append(start_sha);check(f'{i}:prior_start',start_sha==prior['start_integration_sha256'])
            m=env.physics.model;data=env.physics.data;j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
            err=next_err=0.;tasks=[];forces=[]
            for k,action in enumerate(actions):
                err=max(err,float(np.max(abs(obs['agent_pos']-states[k]))));obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                next_err=max(next_err,float(np.max(abs(obs['agent_pos']-following[k]))));forces.append(float(data.actuator_force[a]))
            check(f'{i}:exact_pre_post',err==next_err==0)
            if n:check(f'{i}:environment_outcome',tasks[-1]['is_success']==meta['environment_success'])
            np.savez_compressed(proof/'samples.npz',integration_start=start,poststep_wrist_effort_nm=forces)
        finally:env.close()
        jerk=lambda q:None if len(q)<4 else float(np.max(abs(np.diff(q[:,11].astype(float),n=3)/.02**3)))
        lift=phases=='lift';valid=np.array([all(lift[k:k+4]) for k in range(max(0,n-3))],bool)
        wristjerk=abs(np.diff(states[:,11].astype(float),n=3)/.02**3) if n>=4 else np.empty(0)
        qdiff=np.diff(actions[:,11].astype(float));liftpairs=lift[:-1]&lift[1:]
        mq=motion_quality(states,actions,tasks) if n>=4 else None
        r=dict(index=i,parent=row['parent'],profile=profile,role=entry['role'],collection=row['collection'],source_layout_id=entry['source_layout_id'],
            automatic_success=meta['automatic_success'],error=meta['error'],structural_pass=quality['structural_pass'],frames=n,
            trajectory_sha256=sha(path/'trajectory.npz'),start_integration_sha256=start_sha,pre_post_state_error=max(err,next_err),
            first_second_wrist_peak=jerk(states[:51]),first_three_second_wrist_peak=jerk(states[:151]),
            full_lift_wrist_peak=None if not valid.any() else float(wristjerk[valid].max()),
            lift_wrist_target_repeat_fraction=None if not liftpairs.any() else float(np.mean(abs(qdiff[liftpairs])<1e-8)),
            motion_quality=mq,box=box_metrics(following,actions,tasks,meta['scene']) if n else dict(available=False,first_above_to_containment_seconds=None),
            poststep_effort_peak_nm=None if not forces else float(np.max(abs(np.asarray(forces)))),video=str(path/'preview.mp4'),video_sha256=sha(path/'preview.mp4'))
        records.append(r);print('AUDIT',row['parent'],profile,'success',r['automatic_success'],'liftpeak',r['full_lift_wrist_peak'],flush=True)
    for parent,values in starts.items():check('same_integration_start:'+str(parent),len(values)==2 and values[0]==values[1])
    groups={p:sorted([r for r in records if r['profile']==p],key=lambda r:r['parent']) for p in ('baseline','tight_ik')};base,candidate=groups.values();g=plan['gates'];tests=[]
    def compare(name,values_a,values_b,stat,limit,additive=False):
        available=bool(values_a and values_b and all(v is not None for v in values_a+values_b))
        a=float(stat(values_a)) if available else None;b=float(stat(values_b)) if available else None
        tests.append(dict(metric=name,baseline=a,candidate=b,ratio=None if not available or not a else b/a,difference=None if not available else b-a,passed=available and b<=(a+limit if additive else a*limit)))
    for name,key,stat,threshold in [('first_three_second_wrist_peak_median','first_three_second_wrist_peak',np.median,g['first_three_second_wrist_peak_median_ratio_max']),
        ('full_lift_wrist_peak_median','full_lift_wrist_peak',np.median,g['full_lift_wrist_peak_median_ratio_max']),
        ('first_second_wrist_peak_worst','first_second_wrist_peak',max,g['first_second_wrist_peak_worst_ratio_max'])]:compare(name,[r[key] for r in base],[r[key] for r in candidate],stat,threshold)
    get=lambda r,k:None if r['motion_quality'] is None or r['motion_quality']['active_frame_max_jerk_rad_s3'] is None else r['motion_quality']['active_frame_max_jerk_rad_s3'][k]
    compare('whole_active_arm_jerk_p99_median',[get(r,'p99') for r in base],[get(r,'p99') for r in candidate],np.median,g['whole_active_arm_jerk_p99_median_ratio_max'])
    compare('whole_arm_jerk_worst',[get(r,'max') for r in base],[get(r,'max') for r in candidate],max,g['whole_arm_jerk_worst_ratio_max'])
    compare('duration_median',[r['frames']*.02 for r in base],[r['frames']*.02 for r in candidate],np.median,g['duration_median_ratio_max'])
    for name,stat,limit in [('low_motion_pause_total_seconds',np.median,g['idle_total_median_added_seconds_max']),('low_motion_pause_max_seconds',max,g['idle_max_worst_added_seconds_max'])]:compare(name,[None if r['motion_quality'] is None else r['motion_quality'][name] for r in base],[None if r['motion_quality'] is None else r['motion_quality'][name] for r in candidate],stat,limit,True)
    compare('box_to_containment_median',[r['box']['first_above_to_containment_seconds'] for r in base],[r['box']['first_above_to_containment_seconds'] for r in candidate],np.median,g['box_to_containment_median_ratio_max'])
    tests.append(dict(metric='all16_success_structural_transport_coverage',passed=len(records)==16 and all(r['automatic_success'] and r['structural_pass'] and r['motion_quality'] is not None and r['motion_quality']['coverage']['transport_motion_comparable'] for r in records)))
    ratios=[None if get(a,'p99') is None or get(b,'p99') is None or get(a,'p99')==0 else get(b,'p99')/get(a,'p99') for a,b in zip(base,candidate,strict=True)]
    tests.append(dict(metric='individual_active_arm_jerk_p99',ratios=ratios,passed=all(r is not None and r<=g['individual_active_arm_jerk_p99_ratio_max'] for r in ratios)))
    improved=sum(a['full_lift_wrist_peak'] is not None and b['full_lift_wrist_peak'] is not None and b['full_lift_wrist_peak']<=.8*a['full_lift_wrist_peak'] for a,b in zip(base,candidate,strict=True))
    tests.append(dict(metric='pairs_with_20pct_full_lift_improvement',count=improved,passed=improved>=g['minimum_pairs_with_20pct_full_lift_improvement']))
    failed=[x for x in checks if not x['passed']];passed=not failed and all(x['passed'] for x in tests)
    write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,plan_sha256=sha(OUT/'plan.json'),ledger_sha256=sha(OUT/'candidates.json')))
    result=dict(schema='ik-position-precision-result-v1',audit_passed=not failed,quality_gate_passed=passed,gate_checks=tests,groups={p:dict(attempts=len(v),successes=sum(r['automatic_success'] for r in v),records=v) for p,v in groups.items()},
        decision='quality_pass_pending_explicit_data_review' if passed else 'do_not_accept_export_or_train',scope=plan['boundary'])
    write(OUT/'aggregate.json',result);(OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'training_decision.json',dict(quality_gate_passed=passed,accepted=False,exported=False,training_started=False,aggregate_sha256=sha(OUT/'aggregate.json'),decision=result['decision']))
    print(json.dumps(dict(audit_checks=len(checks),failed_checks=failed,gates=tests,decision=result['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
