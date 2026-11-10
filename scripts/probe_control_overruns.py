"""Native exact-state replay and declared injected waits; no cloud policy."""
import os
from pathlib import Path
import sys
import time
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    import numpy as np
    import mujoco
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.pacing import next_apply_deadline
    from pi0_repro.control_timing import ControlTimingTrace
    from profile_v6_runtime import distribution
    source=ROOT/'outputs/20261028-221218-7add88'
    out=ROOT/'outputs/control_overrun_probe_20261029';out.mkdir(exist_ok=False)
    meta=read(source/'run.json');scene=SceneConfig(**read(source/'episode_000_scene.json')['scene'])
    with np.load(source/'episode_000.npz',allow_pickle=False) as d:actions=d['action'].copy();reference=d['state'].copy()
    waits={221:.035,601:.050,901:.080}
    plan=dict(schema='control-overrun-native-probe-v1',source=str(source),source_sha256=sha(source/'episode_000.npz'),
        script_sha256=sha(Path(__file__)),source_hashes={n:sha(ROOT/'src/pi0_repro'/n) for n in ('pacing.py','control_timing.py','banana_env.py')},
        passes=['baseline','recover','recover','baseline'],injected_post_step_wait_seconds=waits,
        scope='Same frozen actions,50Hz physicaldt,25Hz native top rendering, no cloud/GUI/encoder. Injected sleeps test deadline response, not naturally occurring OS stall rates.',
        gate=dict(median_injected_gap_saved_seconds_min=.015,min_recovered_injected_gap_seconds=.02,exact_recorded_14d=True,exact_diagnostic_integration_state=True))
    write(out/'plan.json',plan);ledger=dict(plan=plan,complete=False,passes=[]);write(out/'summary.json',ledger)
    def env(trace):return BananaBoxEnv(scene,max_episode_steps=len(actions),terminate_on_success=False,render_observations=False,collect_step_timings=trace)
    a,b=env(False),env(True)
    try:
        a.reset(seed=meta['config']['simulation']['seed']);b.reset(seed=meta['config']['simulation']['seed'])
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        sa=np.empty(mujoco.mj_stateSize(a.physics.model.ptr,spec));sb=np.empty_like(sa)
        pixel_checks=0
        for i,action in enumerate(actions):
            a.step(action);b.step(action)
            mujoco.mj_getState(a.physics.model.ptr,a.physics.data.ptr,sa,spec)
            mujoco.mj_getState(b.physics.model.ptr,b.physics.data.ptr,sb,spec)
            np.testing.assert_array_equal(sa,sb)
            if i in (0,221,450,601,901,len(actions)-1):
                np.testing.assert_array_equal(a.render(),b.render());pixel_checks+=1
        ledger['diagnostic_fidelity']=dict(integration_states_compared=len(actions),all_integration_states_equal=True,sampled_rgb_equal=pixel_checks)
    finally:a.close();b.close()
    write(out/'summary.json',ledger)
    try:
        for index,kind in enumerate(plan['passes']):
            e=env(True);trace=None;rows=[];max_error=0.
            try:
                obs,_=e.reset(seed=meta['config']['simulation']['seed']);e.render()
                origin=time.perf_counter();deadline=origin+.02;trace=ControlTimingTrace(origin)
                for step,action in enumerate(actions):
                    trace.step=step
                    max_error=max(max_error,float(np.max(np.abs(obs['agent_pos']-reference[step]))))
                    if step%2==0:e.render()
                    start=time.perf_counter();requested=max(0.,deadline-start)
                    cpu,process=time.thread_time(),time.process_time()
                    if requested:time.sleep(requested)
                    apply=time.perf_counter();waited=apply-start
                    wait_cpu=time.thread_time()-cpu;wait_process=time.process_time()-process
                    env_start,env_cpu,env_process=time.perf_counter(),time.thread_time(),time.process_time()
                    obs,*_=e.step(action)
                    injected_start=time.perf_counter()
                    if step in waits:time.sleep(waits[step])
                    finished=time.perf_counter();late=max(0.,apply-deadline)
                    rows.append(dict(step=step,apply_seconds=apply-origin,scheduled_seconds=deadline-origin,
                        requested_wait_seconds=requested,pacing_wait_seconds=waited,pacing_thread_cpu_seconds=wait_cpu,
                        pacing_process_cpu_seconds=wait_process,oversleep_seconds=max(0.,waited-requested) if requested else 0.,
                        environment_wall_seconds=finished-env_start,environment_thread_cpu_seconds=time.thread_time()-env_cpu,
                        environment_process_cpu_seconds=time.process_time()-env_process,
                        injected_wait_seconds=waits.get(step,0.),actual_injected_wall_seconds=finished-injected_start if step in waits else 0.,
                        environment_phases=e.last_step_timings))
                    deadline=next_apply_deadline(deadline,apply,finished,late,recover_overrun=kind=='recover')
                assert max_error==0,max_error
                gaps=np.diff([r['apply_seconds'] for r in rows]);events=[]
                for step in waits:
                    events.append(dict(step=step,injected_seconds=waits[step],actual_environment_wall_seconds=rows[step]['environment_wall_seconds'],
                        next_apply_gap_seconds=float(gaps[step]),gap_minus_environment_seconds=float(gaps[step]-rows[step]['environment_wall_seconds'])))
                normal=[g for i,g in enumerate(gaps) if i not in waits]
                result=dict(index=index,profile=kind,replay14d_error=max_error,apply_gap_ms=distribution(gaps*1000),normal_gap_ms=distribution(np.asarray(normal)*1000),injected_events=events,
                    trace=trace.close())
                trace=None
                (out/f'pass_{index}_{kind}.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows))
                ledger['passes'].append(result);write(out/'summary.json',ledger)
                print('PASS',index,kind,'normal',result['normal_gap_ms'],'injected gaps',[round(x['next_apply_gap_seconds']*1000,2) for x in events],flush=True)
            finally:
                if trace:trace.close()
                e.close()
        base=[p for p in ledger['passes'] if p['profile']=='baseline'];new=[p for p in ledger['passes'] if p['profile']=='recover']
        saved=[b['next_apply_gap_seconds']-n['next_apply_gap_seconds'] for bp,np_ in zip(base,new) for b,n in zip(bp['injected_events'],np_['injected_events'])]
        minimum=min(e['next_apply_gap_seconds'] for p in new for e in p['injected_events'])
        passed=float(np.median(saved))>=.015 and minimum>=.02
        ledger.update(complete=True,passed=passed,median_injected_gap_saved_seconds=float(np.median(saved)),min_recovered_injected_gap_seconds=minimum)
    except BaseException as exc:ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:write(out/'summary.json',ledger)
    if not ledger['passed']:raise SystemExit(1)


if __name__=='__main__':main()
