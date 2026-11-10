"""Frozen-gate audit for eight-parent same-state teacher replication."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from replicate_contact_train import OUT,SUFFIX
from evaluate_v6_progress import read,sha,write
from select_replication_takeovers import box_metrics


def main():
    import numpy as np
    import mujoco
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.recovery import restore_learner_state
    from pi0_repro.motion_quality import motion_quality
    from profile_v6_runtime import inspect_run
    from analyze_rtc_pair import equal
    from pi0_repro.banana_expert import load_expert_config
    plan=read(OUT/'plan.json');acq=read(OUT/'acquisition.json');sel=read(OUT/'takeovers.json');teachers=read(OUT/'teacher_candidates.json')
    assert acq['complete'] and sel['complete'] and teachers['complete']
    assert not (OUT/'audit.json').exists()
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('acquisition_bound',acq['plan_sha256']==sha(OUT/'plan.json'))
    check('selector_bound',sel['plan_sha256']==sha(OUT/'plan.json') and sel['acquisition_sha256']==sha(OUT/'acquisition.json'))
    check('teachers_bound',teachers['plan_sha256']==sha(OUT/'plan.json') and teachers['takeovers_sha256']==sha(OUT/'takeovers.json'))
    check('eight_distinct_disjoint_train',len(plan['parents'])==8 and len({p['layout_id'] for p in plan['parents']})==8 and not set(p['layout_id'] for p in plan['parents']).intersection(plan['excluded_prior_six_layout_ids']))
    check('service_unchanged',acq['service_before']==acq['service_after'] and acq['metadata_before']==acq['metadata_after'])
    check('route',acq['route_before']==acq['route_after'] and not acq['route_before'].startswith(('utun','tun','tap')))
    check('v6_norm',acq['metadata_after']['checkpoint'].endswith('/'+SUFFIX) and acq['metadata_after']['norm_stats_sha256']==plan['config_by_parent'][0]['model']['expected_norm_stats_sha256'])
    for n,d in plan['source_sha256'].items():check('frozen:'+n,sha(OUT/'frozen_source'/n)==d)
    source_hashes=dict(plan['source_sha256'],**teachers['auxiliary_source_sha256'])
    for n,d in teachers['auxiliary_source_sha256'].items():check('auxiliary_frozen:'+n,sha(OUT/'frozen_source'/n)==d)
    for p,row in zip(plan['parents'],acq['results'],strict=True):
        run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json')
        check(f'learner{p["parent"]}:configuration',equal(meta['config'],plan['config_by_parent'][p['parent']]))
        check(f'learner{p["parent"]}:server',server['checkpoint'].endswith('/'+SUFFIX) and server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'learner{p["parent"]}:metadata_source',sha(Path(p['source_episode'])/'episode.json')==p['source_metadata_sha256'])
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for n,d in meta[field].items():check(f'learner{p["parent"]}:code:{n}',plan['source_sha256'][n]==d)
        check(f'learner{p["parent"]}:recomputed',equal(inspect_run(run),row['diagnostics']))
    records=[];starts={};proofs=OUT/'teacher_replays';proofs.mkdir(exist_ok=False)
    for case,row in zip(teachers['cases'],teachers['results'],strict=True):
        i=row['index'];profile=row['profile'];entry=sel['entries'][row['takeover_index']]
        check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        path=Path(row['collection'])/'episode_000';meta=read(path/'episode.json');collection=read(path.parent/'collection.json');quality=read(path/'quality.json')
        expected=asdict(load_expert_config(Path(plan['expert_configs'][profile]['path'])))
        check(f'{i}:expert_config',collection['expert_config']==expected)
        check(f'{i}:expert_code',all(source_hashes.get(n)==d for n,d in collection['code_sha256'].items()))
        check(f'{i}:split_lineage',meta['split']=='train' and meta['recovery']['source_split']=='train' and meta['recovery']['source_layout_id']==entry['source_layout_id'])
        check(f'{i}:pending',not meta['eligible_for_training'] and meta['human_review']=='pending' and not meta.get('assistant_review'))
        check(f'{i}:outcome',meta['automatic_success']==row['automatic_success'] and meta['error']==row['error'])
        with np.load(path/'trajectory.npz',allow_pickle=False) as d:
            actions=d['action'].copy();states=d['state'].copy();following=d['next_state'].copy();times=d['timestamp'].copy()
        n=len(actions)
        check(f'{i}:array_shape',states.shape==following.shape==actions.shape==(n,14) and np.isfinite(states).all() and np.isfinite(actions).all())
        check(f'{i}:continuity',np.allclose(states[1:],following[:-1],atol=1e-6,rtol=0))
        check(f'{i}:timestamps',np.allclose(np.diff(times),.02,atol=1e-7,rtol=0))
        check(f'{i}:frames',len(list((path/'rgb').glob('*.png')))==n)
        env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=4000,terminate_on_success=False,render_observations=False)
        try:
            env.reset(seed=meta['reset_seed']);proof=proofs/f'case_{i:02d}';proof.mkdir()
            request=dict(rollout=entry['rollout'],source_episode=entry['source_episode'],step=entry['step'],source_split='train')
            obs,_,provenance=restore_learner_state(env,request,proof)
            check(f'{i}:takeover_match',provenance['replay_state_max_error']==0)
            spec=mujoco.mjtState.mjSTATE_INTEGRATION
            start=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,start,spec)
            start_sha=hashlib.sha256(start.tobytes()).hexdigest();starts.setdefault(row['parent'],[]).append(start_sha)
            m=env.physics.model;data=env.physics.data;j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
            error=next_error=0.;tasks=[];forces=[]
            for k,action in enumerate(actions):
                error=max(error,float(np.max(abs(obs['agent_pos']-states[k]))));obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                next_error=max(next_error,float(np.max(abs(obs['agent_pos']-following[k]))));forces.append(float(data.actuator_force[a]))
            check(f'{i}:pre_post_exact',error==next_error==0)
            if n:check(f'{i}:environment_outcome',tasks[-1]['is_success']==meta['environment_success'])
            np.savez_compressed(proof/'samples.npz',integration_start=start,poststep_wrist_effort_nm=forces)
            jerk=lambda q:None if len(q)<4 else float(np.max(abs(np.diff(q[:,11].astype(float),n=3)/.02**3)))
            mq=motion_quality(states,actions,tasks) if n>=4 else None
            # Supplemental boundary diagnostic includes actual preceding learner states.
            with np.load(Path(entry['rollout'])/'episode_000.npz',allow_pickle=False) as d:
                preceding=d['state'][max(0,entry['step']-2):entry['step']].copy();prior_target=d['action'][entry['step']-1].copy()
            boundary=jerk(np.concatenate([preceding,states[:51]])) if n else None
            record=dict(index=i,parent=row['parent'],profile=profile,role=entry['role'],collection=row['collection'],
                source_layout_id=entry['source_layout_id'],source_split='train',independent_new_layout=False,takeover_step=entry['step'],
                automatic_success=meta['automatic_success'],structural_pass=quality['structural_pass'],error=meta['error'],frames=n,
                trajectory_sha256=sha(path/'trajectory.npz'),start_integration_sha256=start_sha,recorded_pre_post_state_max_error=max(error,next_error) if n else None,
                first_second_wrist_jerk_peak=jerk(states[:51]),first_two_second_wrist_jerk_peak=jerk(states[:101]),
                boundary_first_second_wrist_jerk_peak=boundary,first_target_arm_delta_rad=None if not n else float(np.max(abs(actions[0,[0,1,2,3,4,5,7,8,9,10,11,12]]-prior_target[[0,1,2,3,4,5,7,8,9,10,11,12]]))),
                motion_quality=mq,poststep_effort_peak_nm=None if not forces else float(np.max(abs(np.asarray(forces)))),
                box=box_metrics(following,actions,tasks,meta['scene']) if n else dict(available=False,reason='No recorded correctionsteps'),
                video=str(path/'preview.mp4'),video_sha256=sha(path/'preview.mp4'),eligible_for_training=False)
            records.append(record);print('AUDIT',row['parent'],profile,'success',record['automatic_success'],'jerk1s',record['first_second_wrist_jerk_peak'],flush=True)
        finally:env.close()
    for parent,values in starts.items():check('paired_integration_start:'+str(parent),len(values)==2 and values[0]==values[1])
    groups={p:sorted([r for r in records if r['profile']==p],key=lambda r:r['parent']) for p in ('clearance','unload')}
    base,candidate=groups['clearance'],groups['unload'];g=plan['gates'];gate_checks=[]
    def compare(metric,values_a,values_b,stat,limit):
        available=bool(values_a and values_b and all(v is not None for v in values_a+values_b))
        a=float(stat(values_a)) if available else None;b=float(stat(values_b)) if available else None
        gate_checks.append(dict(metric=metric,baseline=a,candidate=b,ratio=None if not available or not a else b/a,passed=available and b<=a*limit))
    compare('first_second_median',[r['first_second_wrist_jerk_peak'] for r in base],[r['first_second_wrist_jerk_peak'] for r in candidate],np.median,g['median_first_second_wrist_jerk_ratio_max'])
    compare('first_second_worst',[r['first_second_wrist_jerk_peak'] for r in base],[r['first_second_wrist_jerk_peak'] for r in candidate],max,g['worst_first_second_wrist_jerk_ratio_max'])
    compare('first_two_second_worst',[r['first_two_second_wrist_jerk_peak'] for r in base],[r['first_two_second_wrist_jerk_peak'] for r in candidate],max,g['worst_first_two_second_wrist_jerk_ratio_max'])
    compare('first_two_second_median',[r['first_two_second_wrist_jerk_peak'] for r in base],[r['first_two_second_wrist_jerk_peak'] for r in candidate],np.median,g['median_first_two_second_wrist_jerk_ratio_max'])
    get=lambda r,k:None if r['motion_quality'] is None else r['motion_quality']['active_frame_max_jerk_rad_s3'][k]
    compare('whole_episode_active_p99_median',[get(r,'p99') for r in base],[get(r,'p99') for r in candidate],np.median,g['covered_active_p99_ratio_max'])
    compare('whole_episode_arm_worst',[get(r,'max') for r in base],[get(r,'max') for r in candidate],max,g['worst_whole_episode_arm_jerk_ratio_max'])
    compare('duration_median',[r['frames']*.02 for r in base],[r['frames']*.02 for r in candidate],np.median,g['median_duration_ratio_max'])
    exposure=sel['selected_parents']==8 and sel['sustained_effort_states']>=plan['exposure_gate']['minimum_sustained_effort_states']
    gate_checks.append(dict(metric='predeclared_exposure',selected=sel['selected_parents'],sustained=sel['sustained_effort_states'],passed=exposure))
    gate_checks.append(dict(metric='all8_unload_success_and_structural',passed=len(candidate)==8 and all(r['automatic_success'] and r['structural_pass'] for r in candidate)))
    failed=[c for c in checks if not c['passed']]
    passed=not failed and all(t['passed'] for t in gate_checks)
    subgroup={}
    for role in ('sustained_effort','reference_without_sustained_effort'):
        subgroup[role]={}
        for profile,rows in groups.items():
            selected=[r for r in rows if r['role']==role]
            values=[r['first_second_wrist_jerk_peak'] for r in selected]
            subgroup[role][profile]=dict(count=len(selected),successes=sum(r['automatic_success'] for r in selected),
                first_second_median=None if not values or None in values else float(np.median(values)),
                first_second_worst=None if not values or None in values else max(values),
                transport_coverage=sum(r['motion_quality'] is not None and r['motion_quality']['coverage']['transport_motion_comparable'] for r in selected))
    summary=dict(schema='eight-train-contact-replication-summary-v1',complete=True,audit_passed=not failed,quality_gate_passed=passed,
        groups={k:dict(attempts=len(v),successes=sum(r['automatic_success'] for r in v),records=v) for k,v in groups.items()},gate_checks=gate_checks,
        subgroup_diagnostics=subgroup,
        learner_box_records=[dict(parent=r['parent'],box=r['learner_box']) for r in sel['records']],
        decision='quality_pass_pending_explicit_visual_review' if passed else 'do_not_accept_export_or_train',
        scope='Eight disjoint existing trainingparents. Same reconstructed physical states paired for teachers; no policy-noise pairing. Allfailures/missing/reference states retained. Originalfour-state results notpooled. Boundary-lookback andbox metrics supplemental, not alternative gates.')
    (OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,acquisition_sha256=sha(OUT/'acquisition.json'),teachers_sha256=sha(OUT/'teacher_candidates.json')))
    write(OUT/'aggregate.json',summary)
    write(OUT/'training_decision.json',dict(quality_gate_passed=passed,training_started=False,accepted=False,exported=False,decision=summary['decision'],aggregate_sha256=sha(OUT/'aggregate.json')))
    print(json.dumps(dict(audit_checks=len(checks),failed_checks=failed,gate_checks=gate_checks,decision=summary['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
