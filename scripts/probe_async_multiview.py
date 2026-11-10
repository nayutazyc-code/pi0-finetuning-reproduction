"""Local neutral-hold timing probe; not a policy-success experiment."""
import argparse
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
from pi0_repro.async_multiview import AsyncMultiViewRenderer
from pi0_repro.banana_env import BananaBoxEnv
from pi0_repro.scene import load_scene_config


def dist(values):
    return {'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':float(np.max(values))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs');out.mkdir(exist_ok=False,parents=True)
    env=BananaBoxEnv(load_scene_config(ROOT/'configs/banana_random_scene.toml'),max_episode_steps=150,
                     render_observations=False,display_cameras=True)
    renderer=None
    try:
        obs,info=env.reset(seed=2026102801);top_before=env.render().copy()
        renderer=AsyncMultiViewRenderer(env.xml,env.physics.get_state(),info,fps=25)
        frame,first=renderer.wait_for_step(0)
        np.testing.assert_array_equal(top_before,env.render())
        Image.fromarray(frame).save(out/'first.png')
        hold=obs['agent_pos'].copy();rows=[];apply_times=[]
        deadline=time.perf_counter()+.02
        for step in range(120):
            started=time.perf_counter();top_ms=0.
            if step%2==0:
                t=time.perf_counter();env.render();top_ms=(time.perf_counter()-t)*1000
                renderer.submit(env.physics.get_state(),step=step,info=info)
            t=time.perf_counter();frame,receipt=renderer.poll(current_step=step);exchange_ms=(time.perf_counter()-t)*1000
            compute_ms=(time.perf_counter()-started)*1000
            time.sleep(max(0,deadline-time.perf_counter()))
            apply=time.perf_counter();apply_times.append(apply)
            obs,_,_,_,info=env.step(hold)
            late=max(0.,apply-deadline)
            deadline=(apply+.02) if late>.002 else (deadline+.02)
            rows.append({'step':step,'top_render_ms':top_ms,'exchange_ms':exchange_ms,'compute_ms':compute_ms,'frame':receipt})
        renderer.submit(env.physics.get_state(),step=120,info=info)
        frame,last=renderer.wait_for_step(120,timeout=3)
        Image.fromarray(frame).save(out/'last.png')
        unique={r['frame']['worker_frame_id']:r['frame'] for r in rows}
        result={'scope':'120-step neutral-hold presentation timing only, not closed-loop policy evaluation.',
                'policy_top_equal_before_after_renderer_startup':True,'control_apply_interval_ms':dist(np.diff(apply_times)*1000),
                'control_compute_ms':dist([r['compute_ms'] for r in rows]),
                'snapshot_exchange_ms':dist([r['exchange_ms'] for r in rows]),
                'top_render_ms':dist([r['top_render_ms'] for r in rows if r['top_render_ms']]),
                'worker_render_ms':dist([r['worker_render_ms'] for r in unique.values()]),
                'worker_compose_ms':dist([r['worker_compose_ms'] for r in unique.values()]),
                'frame_age_steps':dist([r['frame']['frame_age_control_steps'] for r in rows]),
                'renderer':renderer.diagnostics(),'first':first,'last':last,'rows':rows}
        (out/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','first','last']},indent=2),flush=True)
    finally:
        if renderer:renderer.close()
        env.close()


if __name__=='__main__':main()
