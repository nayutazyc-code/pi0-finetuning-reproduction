"""Train-only persistent-contact-state selection; no held-out learner labels."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    import json
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from dataclasses import asdict
    old=ROOT/'outputs/v6_progress_20261028';ledger=read(old/'comparison.json');assert ledger['complete']
    out=ROOT/'outputs/contact_recovery_20261029';out.mkdir(exist_ok=False)
    plan=dict(schema='contact-recovery-train-selection-v1',source_ledger_sha256=sha(old/'comparison.json'),script_sha256=sha(Path(__file__)),
        expert_config=str(ROOT/'configs/banana_motion_expert.toml'),expert_config_sha256=sha(ROOT/'configs/banana_motion_expert.toml'),
        rule='At most one earliest eligible takeover per train parent:3 consecutive post-step wrist demand samples>1Nm, right finger/table contact at takeover, banana table-resting or physically grasped. Step=i+1 preserves observed post-step state.',
        budget='Six existing headless preassigned training-parent v6 records audited. Retain parents with no eligible state. No replacement after recovery outcomes.',
        scope='New expert candidate labels only after exact replay, never learner actions as labels. Train parents only; no validation/test replay converted to training and no training launched.',sources=[],entries=[])
    report=dict(plan=plan,complete=False,records=[]);write(out/'selection.json',report)
    for row in ledger['results']:
        if row['profile']!='headless':continue
        assert row['split']=='train'
        source=Path(row['source_episode']);meta=read(source/'episode.json');assert meta['split']=='train'
        run=Path(row['run_directory']);cfg=read(run/'run.json');server=read(run/'server.json')
        assert server['checkpoint'].endswith('/banana_v6_long100_accum8000/7999')
        canonical=lambda x:json.loads(json.dumps(asdict(SceneConfig(**x))))
        assert canonical(read(run/'scene.json')['scene'])==canonical(meta['scene'])
        plan['sources'].append(dict(source_episode=str(source),source_episode_sha256=sha(source/'episode.json'),run=str(run),trajectory_sha256=sha(run/'episode_000.npz'),source_layout_id=meta['layout_id']))
        with np.load(run/'episode_000.npz',allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
        env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
        try:
            obs,_=env.reset(seed=cfg['config']['simulation']['seed']);m=env.physics.model;data=env.physics.data
            j=m.name2id('vx300s_right/wrist_angle','joint');a=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==j)[0])
            names=[m.id2name(g,'geom') or '' for g in range(m.ngeom)]
            high=0;error=0.;picked=None;pressure=[]
            for i,action in enumerate(actions):
                error=max(error,float(np.max(abs(obs['agent_pos']-reference[i]))));obs,_,_,_,info=env.step(action)
                effort=float(data.actuator_force[a]);high=high+1 if abs(effort)>1. else 0
                finger_table=any(c.efc_address>=0 and 'table' in (names[c.geom1],names[c.geom2]) and
                    any(n.startswith('vx300s_right/') and n.endswith(('10_left_gripper_finger','10_right_gripper_finger')) for n in (names[c.geom1],names[c.geom2])) for c in data.contact[:data.ncon])
                pressure.append(effort)
                if picked is None and high>=3 and finger_table and (info['touching_table'] or info['grasped']) and i+1<len(actions):
                    picked=dict(index=len(plan['entries']),rollout=str(run),source_episode=str(source),step=i+1,source_split='train',source_layout_id=meta['layout_id'],
                        role='persistent_wrist_tracking_effort',demand_nm=effort,high_steps=high,contact_evidence='right finger/table at takeover',independent_layout=False)
            assert error==0,(run,error)
            if picked:plan['entries'].append(picked)
            report['records'].append(dict(parent=row['parent'],replay_state_error=error,selected=picked is not None,max_abs_poststep_demand_nm=float(np.max(abs(np.asarray(pressure)))),frames=len(actions)))
            write(out/'selection.json',report);print('PARENT',row['parent'],'selected',picked,flush=True)
        finally:env.close()
    report['complete']=True;write(out/'selection.json',report);(out/'selector_source.py').write_bytes(Path(__file__).read_bytes())
    print('COMPLETE',len(plan['entries']),'train-only candidates')


if __name__=='__main__':main()
