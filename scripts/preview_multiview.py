"""Render display-only cameras and verify that the policy top pixels stay untouched."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
import numpy as np
from PIL import Image
from pi0_repro.scene import load_scene_config
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.multiview import MultiViewComposer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scene',type=Path,default=ROOT/'configs/banana_random_scene.toml')
    p.add_argument('--seed',type=int,default=2026102801)
    p.add_argument('--display-seconds',type=float,default=0)
    args=p.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs') and 0<=args.display_seconds<=60
    out.mkdir(parents=True,exist_ok=False)
    scene=load_scene_config(args.scene)
    env=BananaBoxEnv(scene,max_episode_steps=100,render_observations=False,display_cameras=True)
    viewer=None
    try:
        obs,info=env.reset(seed=args.seed)
        top=env.render();composer=MultiViewComposer();timings=[]
        for i in range(12):
            start=time.perf_counter()
            front=env.render_camera('display_front',height=480,width=640)
            side=env.render_camera('display_side',height=240,width=320)
            wrist=env.render_camera('display_wrist_right',height=240,width=320)
            render_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();frame=composer.compose(front,side,wrist,step=0,info=info)
            timings.append(dict(render_ms=render_ms,compose_ms=(time.perf_counter()-start)*1000))
        np.testing.assert_array_equal(top,env.render())
        for name,data in [('policy_top',top),('front',front),('side',side),('wrist',wrist),('multiview',frame)]:
            Image.fromarray(data).save(out/(name+'.png'))
        (out/'scene.xml').write_text(env.xml)
        record={'scope':'Display-only native camera preview; no policy inference and no control steps.',
                'seed':args.seed,'base_scene':asdict(scene),'reset':env.last_reset,'composer':composer.describe(),
                'policy_top_equal_before_after_display_render':True,'policy_top_sha256':hashlib.sha256(top.tobytes()).hexdigest(),
                'steady_camera_median_ms':float(np.median([t['render_ms'] for t in timings[2:]])),
                'steady_compose_median_ms':float(np.median([t['compose_ms'] for t in timings[2:]])),
                'timings':timings}
        (out/'preview.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k in ['policy_top_equal_before_after_display_render','steady_camera_median_ms','steady_compose_median_ms','reset']},indent=2),flush=True)
        if args.display_seconds:
            from pi0_repro.viewer import GlfwViewer
            from pi0_repro.execution import WindowClosed
            viewer=GlfwViewer(composer.width,composer.height)
            end=time.monotonic()+args.display_seconds
            try:
                while time.monotonic()<end:
                    viewer.draw(frame,'Original pi0 v6 | front + side + right wrist | preview')
                    time.sleep(.04)
            except WindowClosed:pass
    finally:
        if viewer:viewer.close()
        env.close()


if __name__=='__main__':main()
