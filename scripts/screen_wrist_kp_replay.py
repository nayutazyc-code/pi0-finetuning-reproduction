"""One predeclared right-wrist stiffness candidate on two frozen action traces."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write
from diagnose_wrist_contacts import SOURCES

OUT=ROOT/'outputs/wrist_kp_replay_screen_20261028'


def main():
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.motion_quality import motion_quality
    OUT.mkdir(exist_ok=False)
    plan=dict(schema='single-right-wrist-kp-replay-screen-v1',candidate_scale=.75,baseline_scale=1.,
        changed='Only vx300s_right/wrist_angle position-servo kp200to150, compiled actuator gain[0] and linear bias[1] changed together; damping/contact/other gains unchanged.',
        gate='Both retained source traces must keep final success, active jerk p99<=1.1baseline and worst peak<=.9baseline. Frozen action replay is not closed-loop validation.',script_sha256=sha(Path(__file__)))
    write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes());results=[]
    for model,(source,_) in SOURCES.items():
        scene=SceneConfig(**read(source/'episode_000_scene.json')['scene']);meta=read(source/'run.json')
        with np.load(source/'episode_000.npz',allow_pickle=False) as x:actions=x['action'].copy();reference=x['state'].copy()
        for scale in (1.,.75):
            env=BananaBoxEnv(scene,max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
            try:
                m=env.physics.model;joint=m.name2id('vx300s_right/wrist_angle','joint');actuator=int(np.flatnonzero(np.asarray(m.actuator_trnid)[:,0]==joint)[0])
                gain=m.actuator_gainprm.copy();bias=m.actuator_biasprm.copy()
                m.actuator_gainprm[actuator,0]*=scale;m.actuator_biasprm[actuator,1]*=scale
                assert m.actuator_gainprm[actuator,0]==-m.actuator_biasprm[actuator,1]==200*scale
                obs,_=env.reset(seed=meta['config']['simulation']['seed']);states=[];tasks=[]
                for action in actions:
                    states.append(obs['agent_pos'].copy());obs,_,_,_,info=env.step(action);info.pop('executed_action');tasks.append(info)
                states=np.asarray(states);quality=motion_quality(states,actions,tasks)
                row=dict(model=model,scale=scale,kp=200*scale,source=str(source),source_sha256=sha(source/'episode_000.npz'),
                         quality=quality,success=tasks[-1]['is_success'],max_state_difference_from_record=float(np.max(np.abs(states-reference))))
                if scale==1.:assert row['max_state_difference_from_record']==0
                results.append(row);np.savez_compressed(OUT/f'{model}_kp{int(200*scale)}.npz',state=states,action=actions)
                write(OUT/'summary.json',dict(complete=False,results=results));print(model,scale,row['success'],quality['active_frame_max_jerk_rad_s3'],flush=True)
            finally:env.close()
    gates=[]
    for model in SOURCES:
        base,candidate=[r for r in results if r['model']==model]
        a,b=base['quality']['active_frame_max_jerk_rad_s3'],candidate['quality']['active_frame_max_jerk_rad_s3']
        gates.append(dict(model=model,passed=candidate['success'] and b['p99']<=a['p99']*1.1 and b['max']<=a['max']*.9,
                          p99_ratio=b['p99']/a['p99'],peak_ratio=b['max']/a['max']))
    write(OUT/'summary.json',dict(complete=True,results=results,gates=gates,passed=all(g['passed'] for g in gates)))


if __name__=='__main__':main()
