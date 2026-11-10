"""Exact candidate replay, lineage, physical tracking and first-second comparison."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.recovery import restore_learner_state
    from pi0_repro.motion_quality import motion_quality
    out=ROOT/'outputs/contact_recovery_20261029';first=read(out/'clearance_candidates.json');new=read(out/'unload_candidates.json');selection=read(out/'selection.json')
    assert first['complete'] and new['complete'] and len(new['results'])==4
    assert not (out/'unload_audit.json').exists()
    checks=[];records=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('selection_bound',first['selection_sha256']==new['selection_sha256']==sha(out/'selection.json'))
    check('first_candidates_bound',new['previous_candidates_sha256']==sha(out/'clearance_candidates.json'))
    for batch,ledger in [('clearance_reference',first),('unload',new)]:
        for entry,row in zip(selection['plan']['entries'],ledger['results'],strict=True):
            index=row['index'];prefix=f'{batch}:{index}'
            check(prefix+':same_source',all(entry[k]==row[k] for k in entry))
            path=Path(row['collection'])/'episode_000';meta=read(path/'episode.json');quality=read(path/'quality.json')
            source=read(Path(entry['source_episode'])/'episode.json')
            collection=read(path.parent/'collection.json')
            check(prefix+':expert_options',collection['expert_config']['recovery_grasp_min_height']==.008 and collection['expert_config'].get('recovery_target_bridge_steps',0)==0 and collection['expert_config'].get('recovery_unload_steps',0)==(10 if batch=='unload' else 0))
            check(prefix+':train',meta['split']=='train' and source['split']=='train' and entry['source_split']=='train')
            check(prefix+':no_accept',not meta['eligible_for_training'] and meta['human_review']=='pending' and not meta.get('assistant_review'))
            check(prefix+':outcome',meta['automatic_success']==row['automatic_success'] and meta['error']==row['error'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as d:
                actions=d['action'].copy();reference=d['state'].copy();following=d['next_state'].copy();times=d['timestamp'].copy()
            check(prefix+':continuity',np.allclose(reference[1:],following[:-1],atol=1e-6,rtol=0))
            check(prefix+':time',np.allclose(np.diff(times),.02,atol=1e-7,rtol=0))
            check(prefix+':images',len(list((path/'rgb').glob('*.png')))==len(actions))
            env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=4000,terminate_on_success=False,render_observations=False)
            try:
                env.reset(seed=meta['reset_seed']);proof=out/f'unload_replay_{batch}_{index}';proof.mkdir(exist_ok=False)
                request=dict(rollout=entry['rollout'],source_episode=entry['source_episode'],step=entry['step'],source_split='train')
                obs,_,provenance=restore_learner_state(env,request,proof)
                check(prefix+':takeover',provenance['replay_state_max_error']==0)
                m=env.physics.model;data=env.physics.data;j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
                forces=[];states=[];tasks=[];error=0.
                for i,action in enumerate(actions):
                    states.append(obs['agent_pos'].copy());error=max(error,float(np.max(np.abs(obs['agent_pos']-reference[i]))))
                    obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info);forces.append(float(data.actuator_force[a]))
                check(prefix+':exact_replay',error==0)
                check(prefix+':physical_outcome',bool(tasks[-1]['is_success'])==bool(meta['environment_success']))
                states=np.asarray(states);forces=np.asarray(forces)
                np.savez_compressed(proof/'samples.npz',state=states,action=actions,poststep_wrist_effort_nm=forces)
                learner=Path(entry['rollout'])
                with np.load(learner/'episode_000.npz',allow_pickle=False) as d:learnerstates=d['state'][entry['step']:entry['step']+51].copy()
                def jerk(q):return float(np.max(np.abs(np.diff(q[:,11].astype(float),n=3)/.02**3)))
                record=dict(batch=batch,index=index,collection=row['collection'],trajectory_sha256=sha(path/'trajectory.npz'),
                    source_parent_layout_id=source['layout_id'],independent_layout=False,source_split='train',takeover_replay_error=provenance['replay_state_max_error'],correction_replay_error=error,
                    automatic_success=meta['automatic_success'],structural_pass=quality['structural_pass'],review_status='pending',eligible_for_training=False,
                    frames=len(actions),poststep_wrist_effort_peak_nm=float(np.max(abs(forces))),first_second_poststep_wrist_effort_peak_nm=float(np.max(abs(forces[:50]))),
                    first_second_wrist_jerk_peak=float(jerk(states[:51])),first_two_second_wrist_jerk_peak=float(jerk(states[:101])),original_learner_next_second_wrist_jerk_peak=float(jerk(learnerstates)),
                    poststep_high_effort_steps=int(np.sum(abs(forces)>1)),motion_quality=motion_quality(states,actions,tasks))
                records.append(record);print('AUDIT',batch,index,'success',record['automatic_success'],'state_error',error,'effort',record['poststep_wrist_effort_peak_nm'],flush=True)
            finally:env.close()
    failed=[c for c in checks if not c['passed']]
    write(out/'unload_audit.json',dict(schema='contact-recovery-unload-audit-v1',passed=not failed,checks=checks,failed_checks=failed,records=records,
        selection_sha256=sha(out/'selection.json'),first_candidates_sha256=sha(out/'clearance_candidates.json'),unload_candidates_sha256=sha(out/'unload_candidates.json'),script_sha256=sha(Path(__file__)),
        scope='Expert corrections from four fixed training-parent states; offline exact replays, not new policy inference or model improvement. First-second comparisons are expert takeover versus continued frozen learner commands, not matched learned policies. Candidates remain unaccepted/unexported.'))
    if failed:raise SystemExit(1)
    plan=read(out/'unload_plan.json')
    assert plan['clearance_candidates_sha256']==sha(out/'clearance_candidates.json') and plan['selection_sha256']==sha(out/'selection.json')
    base=[r for r in records if r['batch']=='clearance_reference'];candidate=[r for r in records if r['batch']=='unload']
    tests=[]
    for key,stat,limit in [('first_second_wrist_jerk_peak',np.median,plan['gates']['median_first_second_wrist_jerk_ratio_max']),('first_second_wrist_jerk_peak',max,plan['gates']['worst_first_second_wrist_jerk_ratio_max'])]:
        a=float(stat([r[key] for r in base]));b=float(stat([r[key] for r in candidate]));tests.append(dict(metric=key,statistic=stat.__name__,baseline=a,candidate=b,ratio=b/a,passed=b<=a*limit))
    a=float(np.median([r['motion_quality']['active_frame_max_jerk_rad_s3']['p99'] for r in base]));b=float(np.median([r['motion_quality']['active_frame_max_jerk_rad_s3']['p99'] for r in candidate]))
    tests.append(dict(metric='whole_episode_active_jerk_p99_median',baseline=a,candidate=b,ratio=b/a,passed=b<=a*plan['gates']['covered_active_p99_ratio_max']))
    a=float(np.median([r['frames']*.02 for r in base]));b=float(np.median([r['frames']*.02 for r in candidate]))
    tests.append(dict(metric='duration_median',baseline=a,candidate=b,ratio=b/a,passed=b<=a*plan['gates']['median_duration_ratio_max']))
    tests.append(dict(metric='all4success_and_structural',passed=all(r['automatic_success'] and r['structural_pass'] for r in candidate)))
    for stat,key,limit in [(max,'first_two_second_wrist_jerk_peak',plan['gates']['worst_first_two_second_wrist_jerk_ratio_max']),(np.median,'first_two_second_wrist_jerk_peak',plan['gates']['median_first_two_second_wrist_jerk_ratio_max'])]:
        a=float(stat([r[key] for r in base]));b=float(stat([r[key] for r in candidate]));tests.append(dict(metric=key,statistic=stat.__name__,baseline=a,candidate=b,ratio=b/a,passed=b<=a*limit))
    a=max(r['motion_quality']['active_frame_max_jerk_rad_s3']['max'] for r in base);b=max(r['motion_quality']['active_frame_max_jerk_rad_s3']['max'] for r in candidate)
    tests.append(dict(metric='whole_episode_worst_arm_jerk',baseline=a,candidate=b,ratio=b/a,passed=b<=a*plan['gates']['worst_whole_episode_arm_jerk_ratio_max']))
    write(out/'unload_comparison.json' ,dict(schema='expert-contact-unload-comparison-v1',passed=all(t['passed'] for t in tests),gate_checks=tests,
        audit_sha256=sha(out/'unload_audit.json'),plan_sha256=sha(out/'unload_plan.json'),
        scope='Offline expert-generated targets on four dependent training-parent states, not pi0 policy improvement. No acceptance/export/training. All8pair outcomes retained.'))
    print('UNLOAD_COMPARISON',tests,flush=True)


if __name__=='__main__':main()
