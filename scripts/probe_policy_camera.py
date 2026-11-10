"""Native paired camera-lifecycle probe, not cloud policy evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from dm_control.mujoco import engine
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from profile_v6_runtime import distribution
    run=args.rollout.resolve();meta=json.loads((run/'run.json').read_text())
    scene=json.loads((run/'episode_000_scene.json').read_text())['scene']
    with np.load(run/'episode_000.npz',allow_pickle=False) as data:
        actions=data['action'].copy();reference=data['state'].copy()
    env=BananaBoxEnv(SceneConfig(**scene),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
    camera=None;rows=[];state_error=0.
    try:
        obs,_=env.reset(seed=meta['config']['simulation']['seed'])
        camera=engine.Camera(env.physics,height=480,width=640,camera_id='top')
        # Startup GPU/context work is excluded from steady comparisons.
        env.physics.render(height=480,width=640,camera_id='top');camera.render()
        for step,action in enumerate(actions):
            state_error=max(state_error,float(np.max(np.abs(obs['agent_pos']-reference[step]))))
            if step%10==0:
                before=env.physics.get_state().copy();control=env.physics.data.ctrl.copy();stamp=float(env.physics.data.time)
                frames={};timings={}
                for kind in (('legacy','persistent') if (step//10)%2==0 else ('persistent','legacy')):
                    started=time.perf_counter()
                    frame=(env.physics.render(height=480,width=640,camera_id='top') if kind=='legacy' else camera.render().copy())
                    timings[kind]=(time.perf_counter()-started)*1000
                    frames[kind]=frame.copy()
                np.testing.assert_array_equal(frames['legacy'],frames['persistent'])
                np.testing.assert_array_equal(before,env.physics.get_state())
                np.testing.assert_array_equal(control,env.physics.data.ctrl)
                assert float(env.physics.data.time)==stamp
                rows.append(dict(step=step,**timings,pixel_equal=True,physics_state_unchanged=True))
            obs,*_=env.step(action)
        assert state_error<1e-5,state_error
        result=dict(schema='policy-camera-lifecycle-probe-v1',scope='Paired frozen-action native replay across one existing task. No cloud, no policy outcome improvement claim.',
            source_rollout=str(run),source_trajectory_sha256=hashlib.sha256((run/'episode_000.npz').read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            render_pairs=len(rows),all_pixels_equal=True,physics_state_unchanged=True,
            replay_state_max_error=state_error,legacy_ms=distribution([r['legacy'] for r in rows]),
            persistent_ms=distribution([r['persistent'] for r in rows]),
            median_paired_saved_ms=float(np.median([r['legacy']-r['persistent'] for r in rows])),rows=rows)
        (out/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    finally:
        if camera is not None:camera._scene.free()
        env.close()


if __name__=='__main__':main()
