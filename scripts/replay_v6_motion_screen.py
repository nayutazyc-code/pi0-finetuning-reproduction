"""Frozen v6 references: limiter equivalence and scene response, never closed-loop policy evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL', 'glfw')
import numpy as np
from pi0_repro.config import load_config
from pi0_repro.execution import JointTargetLimiter, BackgroundVideoWriter
from pi0_repro.motion_quality import motion_quality
from pi0_repro.scene import SceneConfig


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs')
    out.mkdir(parents=True,exist_ok=False)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    from pi0_repro.banana_env import BananaBoxEnv
    source=ROOT/'outputs/20261027-204014-c0490a'
    scene=json.loads((source/'scene.json').read_text())['scene']
    seed=json.loads((source/'run.json').read_text())['config']['simulation']['seed']
    with np.load(source/'episode_000.npz',allow_pickle=False) as data:
        target=data['blended_action'].copy(); reference=data['state'].copy(); recorded_action=data['action'].copy()
    summary={'scope':'Open-loop frozen v6 blended references; no new inference, no training, no claim of closed-loop improvement.',
             'source':str(source),'source_npz_sha256':sha(source/'episode_000.npz'),
             'source_scene_sha256':sha(source/'scene.json'),'script_sha256':sha(Path(__file__)),
             'complete':False,'limiter_equivalence':[],'replays':[]}
    write(out/'summary.json',summary)
    # Check all 18 existing v6 trajectories, no physics steps and no network.
    old=json.loads((ROOT/'outputs/motion_first_20261026/comparison.json').read_text())
    env=BananaBoxEnv(SceneConfig(**scene),max_episode_steps=len(target),terminate_on_success=False,render_observations=False)
    try:
        env.reset(seed=seed)
        for row in [r for r in old['results'] if r['profile']=='v6']:
            directory=Path(row['run_directory'])
            with np.load(directory/'episode_000.npz',allow_pickle=False) as data:
                references=data['blended_action'].copy(); initial=data['state'][0].copy(); original=data['action'].copy()
            result={'layout':row['layout'],'repeat':row['repeat'],'source':str(directory),
                    'source_npz_sha256':sha(directory/'episode_000.npz'),'frames':len(original),'profiles':{}}
            for label,acc in [('baseline',40.),('acc20',20.),('acc2',2.)]:
                settings=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml').simulation
                settings.max_joint_acceleration=acc
                limiter=JointTargetLimiter(initial,settings,env.action_space.low,env.action_space.high)
                commands=[]
                for reference_target in references:
                    command=env.project_action(limiter.limit(reference_target))
                    commands.append(command); limiter.update_executed(command)
                commands=np.asarray(commands)
                error=np.max(abs(commands-original),axis=1)
                result['profiles'][label]={'max_command_error_rad':float(error.max()),'changed_frames_gt_1e_7':int((error>1e-7).sum())}
                if label=='baseline': assert np.array_equal(commands,original),'Baseline command reconstruction differs'
            summary['limiter_equivalence'].append(result)
        write(out/'summary.json',summary)
    finally: env.close()
    # Physical diagnostic on one fixed source scene, using identical references.
    for label,acc,scale in [('baseline',40.,1.),('acc20',20.,1.),('acc2',2.,1.),('wrist125',40.,1.25)]:
        directory=out/label; directory.mkdir()
        replay_scene=SceneConfig(**dict(scene,wrist_damping_scale=scale))
        settings=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml').simulation
        settings.max_joint_acceleration=acc
        env=BananaBoxEnv(replay_scene,max_episode_steps=len(target),terminate_on_success=False,render_observations=False)
        states=[]; actions=[]; tasks=[]; writer=None
        try:
            obs,_=env.reset(seed=seed)
            initial_error=float(abs(obs['agent_pos']-reference[0]).max())
            # Changed joint damping also changes the tiny settling residual.
            np.testing.assert_allclose(obs['agent_pos'],reference[0],atol=1e-5 if scale!=1 else 1e-7,rtol=0)
            limiter=JointTargetLimiter(obs['agent_pos'],settings,env.action_space.low,env.action_space.high)
            writer=BackgroundVideoWriter(directory/'episode_000.mp4',fps=50)
            for reference_target in target:
                states.append(obs['agent_pos'].copy()); writer.append_data(env.render())
                obs,_,_,_,info=env.step(limiter.limit(reference_target))
                action=np.asarray(info.pop('executed_action'),np.float32)
                limiter.update_executed(action); actions.append(action); tasks.append(info)
            writer.append_data(env.render())
            states=np.asarray(states); actions=np.asarray(actions)
            np.savez_compressed(directory/'trajectory.npz',state=states,action=actions,blended_action=target)
            (directory/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
            (directory/'scene.xml').write_text(env.xml)
            write(directory/'scene.json',asdict(replay_scene))
            row={'profile':label,'max_joint_acceleration':acc,'wrist_damping_scale':scale,
                 'initial_state_difference_from_source':initial_error,
                 'max_command_difference_from_source':float(abs(actions-recorded_action).max()),
                 'max_state_difference_from_source':float(abs(states-reference).max()),
                 'motion_quality':motion_quality(states,actions,tasks),
                 'short_dwell':motion_quality(states,actions,tasks,pause_min_seconds=.16),
                 'video':str(directory/'episode_000.mp4'),
                 'state_time_semantics':'state[i] is pre-step, task[i] post-step; video frame i=state[i], event at task index i maps to frame i+1.'}
            if label in ('baseline','acc20'):
                assert np.array_equal(actions,recorded_action),'Unchanged limiter changed commands'
                np.testing.assert_allclose(states,reference,atol=1e-6,rtol=0)
            summary['replays'].append(row)
            write(out/'summary.json',summary)
            print(label,row['max_command_difference_from_source'],row['max_state_difference_from_source'],row['motion_quality']['active_frame_max_jerk_rad_s3'],flush=True)
        finally:
            if writer:writer.close()
            env.close()
    summary['complete']=True
    write(out/'summary.json',summary)


if __name__=='__main__': main()
