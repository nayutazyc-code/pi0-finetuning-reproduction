"""Train-only full-demonstration replay and predeclared regeneration gates."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from experiment_ik_continuous import OUT
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    import mujoco
    from dataclasses import asdict
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.motion_quality import motion_quality
    plan=read(OUT/'relabel_plan.json');ledger=read(OUT/'relabel_candidates.json')
    assert ledger['complete'] and len(ledger['results'])==20 and not (OUT/'relabel_audit.json').exists()
    checks=[]
    def check(n,v):checks.append(dict(name=n,passed=bool(v)))
    check('plan',ledger['plan_sha256']==sha(OUT/'relabel_plan.json'))
    check('combined_round_pass',read(OUT/'aggregate.json')['quality_gate_passed'])
    check('legacy_controls',len(ledger['legacy_compatibility'])==len(plan['config_groups']) and all(c['exact_arrays'] for c in ledger['legacy_compatibility']))
    check('base_manifest',sha(Path(plan['base_dataset'])/'dataset_manifest.json')==plan['base_manifest_sha256'])
    check('base_mapping',sha(Path(plan['base_dataset'])/'local/pi0_banana_train/source_episodes.json')==plan['base_train_mapping_sha256'])
    for n,d in plan['source_sha256'].items():check('source:'+n,sha(ROOT/'src/pi0_repro'/n)==d)
    canonical=lambda s:json.loads(json.dumps(asdict(SceneConfig(**s))))
    records=[];proofs=OUT/'fulltrain_replays';proofs.mkdir(exist_ok=True)
    assert not list(proofs.iterdir()),'Retain nonempty replay attempt; do not overwrite'
    for entry,row in zip(plan['entries'],ledger['results'],strict=True):
        i=row['index'];source=Path(entry['source_episode']);original=read(source/'episode.json')
        check(f'{i}:original_immutable',sha(source/'episode.json')==entry['source_metadata_sha256'] and sha(source/'trajectory.npz')==entry['trajectory_sha256'])
        check(f'{i}:train_layout',row['layout_id']==entry['layout_id'] and original['split']==entry['layout']['split']=='train')
        starts=[]
        for profile,path in [('original',source),('continuous',Path(row['collection'])/'episode_000')]:
            meta=read(path/'episode.json')
            quality=read(path/'quality.json') if (path/'quality.json').exists() else dict(structural_pass=None,scope='Legacy reference has no stored quality report; original files remain untouched and read-only structural checks are recorded here.')
            if profile=='continuous':
                collection=read(path.parent/'collection.json');config=plan['config_groups'][entry['group']]['profiles']['precision']
                check(f'{i}:config',collection['expert_config']==config['config'] and sha(Path(config['path']))==config['sha256'])
                check(f'{i}:code',all(plan['source_sha256'].get(n)==d for n,d in collection['code_sha256'].items()))
                check(f'{i}:pending',not meta['eligible_for_training'] and meta['human_review']=='pending' and not meta.get('assistant_review'))
                check(f'{i}:lineage',meta['split']=='train' and meta['layout_id']==entry['layout_id'] and collection['dataset']['source_episode']==entry['source_episode'] and not meta.get('recovery'))
            check(f'{i}:{profile}:same_scene',canonical(meta['scene'])==canonical(original['scene']) and meta['reset_seed']==original['reset_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as d:
                states=d['state'].copy();actions=d['action'].copy();following=d['next_state'].copy();times=d['timestamp'].copy()
            n=len(actions);check(f'{i}:{profile}:arrays',states.shape==following.shape==actions.shape==(n,14) and np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(following).all())
            check(f'{i}:{profile}:continuity',np.array_equal(states[1:],following[:-1]))
            check(f'{i}:{profile}:timestamps',np.allclose(np.diff(times),.02,atol=1e-7,rtol=0))
            check(f'{i}:{profile}:frames',len(list((path/'rgb').glob('*.png')))==n)
            env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=4000,terminate_on_success=False,render_observations=False)
            try:
                obs,_=env.reset(seed=meta['reset_seed']);spec=mujoco.mjtState.mjSTATE_INTEGRATION;start=np.empty(mujoco.mj_stateSize(env.physics.model.ptr,spec));mujoco.mj_getState(env.physics.model.ptr,env.physics.data.ptr,start,spec)
                h=hashlib.sha256(start.tobytes()).hexdigest();starts.append(h);error=posterror=0.;tasks=[]
                for k,action in enumerate(actions):
                    error=max(error,float(np.max(abs(obs['agent_pos']-states[k]))));obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                    posterror=max(posterror,float(np.max(abs(obs['agent_pos']-following[k]))))
                check(f'{i}:{profile}:exact_pre_post',error==posterror==0)
                if n:check(f'{i}:{profile}:environment_outcome',tasks[-1]['is_success']==meta.get('environment_success',meta['automatic_success']))
                np.savez_compressed(proofs/f'{i:02d}_{profile}_start.npz',integration_start=start)
            finally:env.close()
            records.append(dict(index=i,profile=profile,layout_id=entry['layout_id'],collection=str(path.parent),episode_path=str(path),source_episode=entry['source_episode'],
                automatic_success=meta['automatic_success'],error=meta['error'],structural_pass=quality['structural_pass'],frames=n,
                trajectory_sha256=sha(path/'trajectory.npz'),pre_post_state_error=max(error,posterror),start_integration_sha256=h,
                motion_quality=motion_quality(states,actions,tasks) if n>=4 else None,video=str(path/'preview.mp4'),video_sha256=sha(path/'preview.mp4')))
            print('FULL AUDIT',i,profile,meta['automatic_success'],flush=True)
        check(f'{i}:same_start',starts[0]==starts[1])
    groups={p:[r for r in records if r['profile']==p] for p in ('original','continuous')};base,candidate=groups.values();g=plan['gates'];tests=[]
    def compare(name,a,b,stat,limit,additive=False):
        available=bool(a and b and all(v is not None for v in a+b));a=float(stat(a)) if available else None;b=float(stat(b)) if available else None
        tests.append(dict(metric=name,baseline=a,candidate=b,ratio=None if not available or not a else b/a,difference=None if not available else b-a,passed=available and b<=(a+limit if additive else a*limit)))
    get=lambda r,k:None if r['motion_quality'] is None or r['motion_quality']['active_frame_max_jerk_rad_s3'] is None else r['motion_quality']['active_frame_max_jerk_rad_s3'][k]
    compare('median_whole_active_jerk_p99',[get(r,'p99') for r in base],[get(r,'p99') for r in candidate],np.median,g['median_whole_active_jerk_p99_ratio_max'])
    compare('worst_whole_arm_jerk',[get(r,'max') for r in base],[get(r,'max') for r in candidate],max,g['worst_whole_arm_jerk_ratio_max'])
    compare('median_duration',[r['frames']*.02 for r in base],[r['frames']*.02 for r in candidate],np.median,g['median_duration_ratio_max'])
    for name,stat,limit in [('low_motion_pause_total_seconds',np.median,g['median_idle_total_added_seconds_max']),('low_motion_pause_max_seconds',max,g['worst_idle_max_added_seconds_max'])]:compare(name,[None if r['motion_quality'] is None else r['motion_quality'][name] for r in base],[None if r['motion_quality'] is None else r['motion_quality'][name] for r in candidate],stat,limit,True)
    ratios=[None if get(a,'p99') is None or get(b,'p99') is None or not get(a,'p99') else get(b,'p99')/get(a,'p99') for a,b in zip(base,candidate,strict=True)]
    tests.append(dict(metric='individual_active_jerk_p99',ratios=ratios,passed=all(r is not None and r<=g['individual_active_jerk_p99_ratio_max'] for r in ratios)))
    tests.append(dict(metric='all20_success_structural_transport_coverage',passed=len(candidate)==20 and all(r['automatic_success'] and r['structural_pass'] and r['motion_quality'] is not None and r['motion_quality']['coverage']['transport_motion_comparable'] for r in candidate)))
    failed=[c for c in checks if not c['passed']];passed=not failed and all(t['passed'] for t in tests)
    write(OUT/'relabel_audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,plan_sha256=sha(OUT/'relabel_plan.json'),ledger_sha256=sha(OUT/'relabel_candidates.json')))
    write(OUT/'relabel_aggregate.json',dict(audit_passed=not failed,quality_gate_passed=passed,gate_checks=tests,groups=groups,
        decision='quality_pass_pending_explicit_data_review' if passed else 'do_not_accept_export_or_train',scope='Original20fulltrain layouts, paired recorded originalreferences versusnew combinedexpert. No heldout labels. Originalsource metadata/trajectories unchanged.'))
    (OUT/'relabel_analyzer_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(checks=len(checks),failed_checks=failed,gates=tests,quality_gate_passed=passed),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
