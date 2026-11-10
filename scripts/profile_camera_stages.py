"""Frozen-command camera stage timing; no policy inference or package edits."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL', 'glfw')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rollout', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs')
    out.mkdir(exist_ok=False, parents=True)
    from dm_control.mujoco import engine
    import mujoco
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from profile_v6_runtime import distribution, write, sha

    run = args.rollout.resolve()
    meta = json.loads((run/'run.json').read_text())
    scene = json.loads((run/'episode_000_scene.json').read_text())['scene']
    latency = [json.loads(s) for s in (run/'episode_000_latency.jsonl').read_text().splitlines()]
    requests = {r['step'] for r in latency}
    with np.load(run/'episode_000.npz', allow_pickle=False) as d:
        actions, reference = d['action'].copy(), d['state'].copy()
    plan = dict(schema='camera-stage-frozen-replay-v1', source_rollout=str(run),
        source_trajectory_sha256=sha(run/'episode_000.npz'), script_sha256=sha(Path(__file__)),
        engine_sha256=sha(Path(engine.__file__)),
        passes=['legacy', 'persistent', 'persistent', 'legacy'],
        render_schedule='Every second control state (25fps) union recorded inference request states; 50Hz pacing.',
        scope='Frozen commands and recorded request schedule, not closed-loop policy outcomes. Native wall/thread CPU stage envelopes; GL submission is not total GPU time.',
        assertions='Exact recorded14D state replay; sampled instrumented/native RGB equality; render leaves physics state/control/time unchanged.')
    write(out/'plan.json', plan)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())

    class Camera(engine.Camera):
        def __init__(self, *a, **kw):
            self.timing = {}
            super().__init__(*a, **kw)

        def update(self, scene_option=None):
            wall, cpu = time.perf_counter(), time.thread_time()
            super().update(scene_option=scene_option)
            self.timing.update(scene_update_ms=(time.perf_counter()-wall)*1000,
                               scene_update_cpu_ms=(time.thread_time()-cpu)*1000)

        def _render_on_gl_thread(self, depth, overlays):
            assert not depth and not overlays
            wall, cpu = time.perf_counter(), time.thread_time()
            mujoco.mjr_render(self._rect, self._scene.ptr, self._physics.contexts.mujoco.ptr)
            rendered = time.perf_counter()
            mujoco.mjr_readPixels(self._rgb_buffer, None, self._rect, self._physics.contexts.mujoco.ptr)
            self.timing.update(gl_submit_ms=(rendered-wall)*1000,
                pixel_read_ms=(time.perf_counter()-rendered)*1000,
                gl_body_ms=(time.perf_counter()-wall)*1000,
                gl_body_cpu_ms=(time.thread_time()-cpu)*1000)

    ledger = dict(plan=plan, complete=False, passes=[])
    write(out/'summary.json', ledger)
    try:
        for repeat, kind in enumerate(plan['passes']):
            env = BananaBoxEnv(SceneConfig(**scene), max_episode_steps=len(actions),
                              terminate_on_success=False, render_observations=False)
            camera = None
            rows, checks, collections = [], [], []
            gc_start = {}
            def on_gc(phase, info):
                if phase == 'start': gc_start[info['generation']] = time.perf_counter()
                elif info['generation'] in gc_start:
                    begin = gc_start.pop(info['generation'])
                    collections.append(dict(start=begin, end=time.perf_counter(), generation=info['generation']))
            try:
                obs, _ = env.reset(seed=meta['config']['simulation']['seed'])
                # Context/shader startup is separate from running-control timings.
                env.render()
                if kind == 'persistent':
                    camera = Camera(env.physics, height=480, width=640, camera_id='top')
                    camera.render()
                deadline = time.perf_counter()
                gc.callbacks.append(on_gc)
                max_error = 0.
                for step, action in enumerate(actions):
                    max_error = max(max_error, float(np.max(np.abs(obs['agent_pos']-reference[step]))))
                    if step % 2 == 0 or step in requests:
                        state = env.physics.get_state().copy()
                        ctrl = env.physics.data.ctrl.copy()
                        stamp = float(env.physics.data.time)
                        wall, cpu = time.perf_counter(), time.thread_time()
                        if kind == 'legacy': camera = Camera(env.physics, height=480, width=640, camera_id='top')
                        created = time.perf_counter()
                        frame = camera.render()
                        rendered = time.perf_counter()
                        if kind == 'persistent': frame = frame.copy()
                        copied = time.perf_counter()
                        timing = camera.timing.copy()
                        if kind == 'legacy': camera._scene.free(); camera = None
                        ended = time.perf_counter()
                        rows.append(dict(step=step, start=wall, end=ended,
                            total_ms=(ended-wall)*1000, caller_cpu_ms=(time.thread_time()-cpu)*1000,
                            constructor_ms=(created-wall)*1000, render_ms=(rendered-created)*1000,
                            copy_ms=(copied-rendered)*1000, free_ms=(ended-copied)*1000, **timing))
                        np.testing.assert_array_equal(state, env.physics.get_state())
                        np.testing.assert_array_equal(ctrl, env.physics.data.ctrl)
                        assert stamp == float(env.physics.data.time)
                        if step % 50 == 0:
                            np.testing.assert_array_equal(frame, env.render())
                            checks.append(dict(step=step, rgb_equal=True))
                    # Late samples reset the clock, retaining their actual timings.
                    deadline += .02
                    delay = deadline-time.perf_counter()
                    if delay > 0: time.sleep(delay)
                    else: deadline = time.perf_counter()
                    obs, *_ = env.step(action)
                assert max_error == 0, max_error
                gc.callbacks.remove(on_gc)
                for row in rows:
                    row['gc_overlap_ms'] = sum(max(0., min(row['end'],g['end'])-max(row['start'],g['start']))*1000 for g in collections)
                    row.pop('start'); row.pop('end')
                result = dict(index=repeat, profile=kind, render_calls=len(rows),
                    replay_state_max_error=max_error, sampled_pixel_checks=checks,
                    timings={k:distribution([r[k] for r in rows]) for k in rows[0] if k.endswith('_ms')},
                    worst_calls=sorted(rows,key=lambda r:r['total_ms'],reverse=True)[:10],
                    gc_collections=len(collections))
                (out/f'pass_{repeat}_{kind}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                ledger['passes'].append(result)
                write(out/'summary.json', ledger)
                print('PASS',repeat,kind,'total',result['timings']['total_ms'],'worst',result['worst_calls'][0],flush=True)
            finally:
                if on_gc in gc.callbacks: gc.callbacks.remove(on_gc)
                if camera is not None: camera._scene.free()
                env.close()
        ledger['complete'] = True
    except BaseException as e:
        ledger['error'] = f'{type(e).__name__}: {e}'
        raise
    finally: write(out/'summary.json', ledger)


if __name__ == '__main__': main()
