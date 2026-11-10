"""Full contact-rich recorded task: live RGB versus deferred RGB at every frame."""
import hashlib
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    import imageio.v2 as imageio
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.deferred_video import DeferredTopVideo
    source=ROOT/'outputs/20261028-221218-7add88'
    out=ROOT/'outputs/deferred_video_full_replay_20261029';out.mkdir(exist_ok=False)
    meta=read(source/'run.json');scene=read(source/'episode_000_scene.json')['scene']
    with np.load(source/'episode_000.npz',allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
    plan=dict(source=str(source),trajectory_sha256=sha(source/'episode_000.npz'),script_sha256=sha(Path(__file__)),
        scope='Complete frozen-action replay, all25Hz video frames plus terminal; all raw RGB hashes must equal deferred rendering. No cloud or closed-loop timing inference.')
    write(out/'plan.json',plan)
    env=BananaBoxEnv(SceneConfig(**scene),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
    result=dict(plan=plan,complete=False)
    try:
        obs,_=env.reset(seed=meta['config']['simulation']['seed'])
        video=DeferredTopVideo(env.physics,env.xml,max_steps=len(actions),stride=2)
        pixels=[];max_error=0.
        for step,action in enumerate(actions):
            max_error=max(max_error,float(np.max(np.abs(obs['agent_pos']-reference[step]))))
            if step%2==0:
                before=env.physics.get_state().copy();video.capture(env.physics,step)
                pixels.append(hashlib.sha256(env.render().tobytes()).hexdigest())
                np.testing.assert_array_equal(before,env.physics.get_state())
            obs,*_=env.step(action)
        video.capture(env.physics,len(actions));pixels.append(hashlib.sha256(env.render().tobytes()).hexdigest())
        final=env.physics.get_state().copy()
        receipt=video.finish(out/'episode.mp4',fps=25,writer_factory=imageio.get_writer)
        assert max_error==0,max_error
        assert pixels==receipt['pixel_sha256']
        np.testing.assert_array_equal(final,env.physics.get_state())
        result.update(complete=True,frames=len(pixels),all_raw_rgb_equal=True,replay14d_state_max_error=max_error,live_final_state_unchanged_by_deferred_render=True,video_receipt=receipt)
        print('PASS full task',len(pixels),'frames; live/deferred raw RGB equal;14D error',max_error)
    except BaseException as e:result['error']=f'{type(e).__name__}: {e}';raise
    finally:
        env.close();write(out/'summary.json',result)


if __name__=='__main__':main()
