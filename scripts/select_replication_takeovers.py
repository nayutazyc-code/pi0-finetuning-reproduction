"""Exact train-only learner replay, frozen takeover selection and box diagnosis."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from replicate_contact_train import OUT
from evaluate_v6_progress import read,sha,write


def box_metrics(states,actions,tasks,scene):
    """Post-step state/task alignment; missing coverage stays null."""
    import numpy as np
    from pi0_repro.motion_quality import _spans,ARM_JOINTS
    states=np.asarray(states,float);actions=np.asarray(actions,float)
    assert len(states)==len(tasks)==len(actions)
    box=np.asarray(scene['box_position'][:2],float)
    mask=np.array([bool(t['grasped'] and not t['inside_box'] and np.linalg.norm(np.asarray(t['banana_position'][:2])-box)<=.04) for t in tasks])
    inds=np.flatnonzero(mask)
    if not len(inds):return dict(available=False,held_above_box_seconds=None,first_above_to_containment_seconds=None,low_motion_dwell_max_seconds=None,low_motion_dwell_total_seconds=None)
    speed=np.r_[np.inf,np.max(abs(np.diff(states[:,ARM_JOINTS],axis=0)),axis=1)/.02]
    grip=np.r_[np.inf,np.max(abs(np.diff(actions[:,[6,13]],axis=0)),axis=1)/.02]
    spans=_spans(mask&(speed<.015)&(grip<.05),.02,.5)
    entry=next((i for i in range(int(inds[0]),len(tasks)) if tasks[i]['inside_box']),None)
    return dict(available=True,held_above_box_seconds=float(mask.sum()*.02),first_above_to_containment_seconds=None if entry is None else float((entry-inds[0])*.02),
        first_above_box_seconds=float((inds[0]+1)*.02),low_motion_dwell_max_seconds=max([s['duration_seconds'] for s in spans],default=0.),
        low_motion_dwell_total_seconds=sum(s['duration_seconds'] for s in spans),spans=spans,
        scope='Diagnostic geometry/speed thresholds only, not policy input or intent. Held bananaXY within4cm of box, outsidecontainment; no missingphase scored aszero.')


def main():
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    plan=read(OUT/'plan.json');ledger=read(OUT/'acquisition.json')
    assert ledger['complete'] and len(ledger['results'])==8
    assert not (OUT/'takeovers.json').exists()
    report=dict(schema='eight-train-contact-takeovers-v1',complete=False,plan_sha256=sha(OUT/'plan.json'),acquisition_sha256=sha(OUT/'acquisition.json'),
        script_sha256=sha(Path(__file__)),records=[],entries=[],scope='Only8fixed trainparents; exact50Hz post-step replay. Primary sustained-effort states and no-effort references remain distinct; unavailable states retained, no replacements.')
    samples=OUT/'learner_replays';samples.mkdir(exist_ok=False)
    for parent,row in zip(plan['parents'],ledger['results'],strict=True):
        source=Path(parent['source_episode']);meta=read(source/'episode.json');run=Path(row['run_directory'])
        assert meta['split']=='train' and row['split']=='train' and meta['layout_id']==row['layout_id']
        assert sha(source/'episode.json')==parent['source_metadata_sha256'] and sha(source/'trajectory.npz')==parent['source_trajectory_sha256']
        canonical=lambda s:json.loads(json.dumps(asdict(SceneConfig(**s))))
        assert canonical(read(run/'scene.json')['scene'])==canonical(meta['scene'])
        with np.load(run/'episode_000.npz',allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
        env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
        try:
            obs,_=env.reset(seed=parent['seed']);m=env.physics.model;data=env.physics.data
            j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
            names=[m.id2name(g,'geom') or '' for g in range(m.ngeom)]
            high=0;error=0.;picked=None;fallback=None;tasks=[];post=[];efforts=[]
            for i,action in enumerate(actions):
                error=max(error,float(np.max(abs(obs['agent_pos']-reference[i]))));obs,_,_,_,info=env.step(action);info.pop('executed_action')
                post.append(obs['agent_pos'].copy());tasks.append(info)
                effort=float(data.actuator_force[a]);efforts.append(effort);high=high+1 if abs(effort)>1. else 0
                table=any(c.efc_address>=0 and 'table' in (names[c.geom1],names[c.geom2]) and any(n.startswith('vx300s_right/') and n.endswith(('10_left_gripper_finger','10_right_gripper_finger')) for n in (names[c.geom1],names[c.geom2])) for c in data.contact[:data.ncon])
                eligible=info['touching_table'] and not info['grasped'] and 1<=i+1<len(actions)
                if fallback is None and eligible and i+1>=15 and info.get('near_grasp_region'):
                    fallback=dict(step=i+1,role='reference_without_sustained_effort',demand_nm=effort)
                if picked is None and eligible and high>=3 and table:
                    picked=dict(step=i+1,role='sustained_effort',demand_nm=effort,high_steps=high)
            assert error==0,(parent['parent'],error)
            entry=picked or fallback
            if entry:
                entry=dict(index=len(report['entries']),parent=parent['parent'],source_layout_id=meta['layout_id'],rollout=str(run),source_episode=str(source),source_split='train',independent_layout=False,**entry)
                report['entries'].append(entry)
            record=dict(parent=parent['parent'],layout_id=meta['layout_id'],selected=entry,missing=entry is None,
                replay14d_error=error,source_trajectory_sha256=sha(run/'episode_000.npz'),max_abs_poststep_effort_nm=float(np.max(abs(np.asarray(efforts)))),
                learner_box=box_metrics(post,actions,tasks,meta['scene']))
            report['records'].append(record)
            np.savez_compressed(samples/f'parent_{parent["parent"]:02d}.npz',post_state=post,poststep_effort_nm=efforts)
            write(OUT/'takeovers.json',report)
            print('SELECT',parent['parent'],None if entry is None else (entry['role'],entry['step']),'box',record['learner_box'],flush=True)
        finally:env.close()
    report.update(complete=True,selected_parents=len(report['entries']),sustained_effort_states=sum(e['role']=='sustained_effort' for e in report['entries']))
    write(OUT/'takeovers.json',report);(OUT/'selector_source.py').write_bytes(Path(__file__).read_bytes())
    print('COMPLETE selected',report['selected_parents'],'sustained',report['sustained_effort_states'])


if __name__=='__main__':main()
