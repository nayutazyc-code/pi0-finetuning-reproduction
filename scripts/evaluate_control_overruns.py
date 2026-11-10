"""Fresh alternating continued-model default versus overrun deadline recovery."""
import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from evaluate_v6_progress import read,sha,write
from evaluate_random_multiview import SOURCE_NAMES
from pi0_repro.config import load_config
from profile_v6_runtime import inspect_run

OUT=ROOT/'outputs/control_overrun_ab_20261029'
SUFFIX='banana_v6_progress_control1000/999'
SEED=20261027
PORT=18077


def configured(out,profile='baseline',model='control'):
    cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml')
    cfg.connection.port=PORT
    cfg.model.expected_checkpoint_suffix=SUFFIX if model=='control' else 'banana_v6_long100_accum8000/7999'
    s=cfg.simulation;s.scene_config=str(out/(profile+'.toml'));s.seed=SEED;s.episodes=1
    s.max_steps=2000;s.display=False;s.save_video=True;s.video_fps=25
    s.defer_video_render=False;s.trace_control_timing=True;s.recover_control_overruns=profile=='recover'
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    previous=ROOT/'outputs/continued_wrist_kp_ab_20261028'
    old=read(previous/'comparison.json');assert old['complete'] and old['restoration']['verified']
    scene=previous/'baseline.toml'
    source_names=(*SOURCE_NAMES, 'deferred_video.py', 'pacing.py', 'control_timing.py')
    native=read(ROOT/'outputs/control_overrun_probe_20261029/summary.json')
    assert native['complete'] and native['diagnostic_fidelity']['all_integration_states_equal'] and native['median_injected_gap_saved_seconds']>0
    amendment=ROOT/'outputs/control_wakeup_diagnosis_20261029/native_screen_amendment.json'
    assert read(amendment)['native_summary_sha256']==sha(ROOT/'outputs/control_overrun_probe_20261029/summary.json')
    plan=dict(schema='continued-control-overrun-ab-v1',attempts=12,layout='episode_004',reset_seed=SEED,
        cases=[dict(index=2*pair+i,model='control',layout='episode_004',repeat=pair+1,profile=profile) for pair in range(6) for i,profile in enumerate(('baseline','recover') if pair%2==0 else ('recover','baseline'))],
        checkpoint_suffix=SUFFIX,prior_comparison_sha256=sha(previous/'comparison.json'),
        scene_sha256={profile:sha(scene) for profile in ('baseline','recover')},script_sha256=sha(Path(__file__)),
        config_by_profile={profile:asdict(configured(OUT,profile)) for profile in ('baseline','recover')},
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in source_names},config=asdict(configured(OUT)),
        native_probe_sha256=sha(ROOT/'outputs/control_overrun_probe_20261029/summary.json'),
        native_original_gate_passed=native['passed'],native_scope_amendment_sha256=sha(amendment),
        treatment='Only recover_control_overruns false->true. Normal clock unchanged; after post-apply overrun next deadline=max(finish,last_apply+20ms), without extra idle. Same common trace instrumentation, original live top25fps, kp200, continued weights, RTC20.',
        gate=dict(median_trial_apply_p99_ratio_max=1.05,worst_apply_interval_ratio_max=.9,gap_over50_rate_ratio_max=1.,running_underruns_max=0,
                  coverage='candidate>=baseline>0',covered_jerk_p99_ratio_max=1.1,worst_jerk_peak_ratio_max=1.1,covered_duration_ratio_max=1.1,
                  correctness='Source/prefix/timestamp/step-phase/clock audit; native exact-state diagnostic replay. Original native>=15ms gate failed and remains failed; conditional live diagnostic is explicitly amended before outcomes.',
                  interpretation='Normal50Hz p99 near20ms floor is a non-regression check; target is worst interval. Even all passed supports independent replication only, no automatic default promotion.'),
        selection='One deliberately selected validation layout, not population or generalization evidence.',
        sampling='Six counterbalanced AB/BA pairs; fixed reset, unpaired policy noise. No deterministic trajectory pairing.',
        budget='All12 fresh attempts, including failures/waits; no success-only selection, early stop or retries.',
        preflight='Non-tunnel route;6 no-step warmups, last4 median<800ms/max<1500ms. Record all warmups.',
        scope='Headless top-video timing pilot, normal workload and no injected waits in live trials; GC/step wall/thread/process clocks common to both arms. No GUI/OS-cause guarantee; simulation-time video does not reproduce wall-clock stalls.',
        restoration='Select v6 and verify checkpoint/norm metadata before owned tunnel cleanup.')
    assert {k:v for k,v in plan['config_by_profile']['baseline']['simulation'].items() if k not in ('scene_config','recover_control_overruns')} == {k:v for k,v in plan['config_by_profile']['recover']['simulation'].items() if k not in ('scene_config','recover_control_overruns')}
    if OUT.exists():assert read(OUT/'plan.json')==plan,'Existing plan differs; preserve evidence'
    else:
        OUT.mkdir();write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes())
        for profile in ('baseline','recover'): (OUT/(profile+'.toml')).write_bytes(scene.read_bytes())
        frozen=OUT/'frozen_source';frozen.mkdir()
        for n in source_names:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    if args.prepare_only:print('PREPARED',OUT);return
    assert not (OUT/'comparison.json').exists(),'Existing execution retained; never rerun in place'
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    def route():
        raw=subprocess.check_output(['route','-n','get',target],text=True,timeout=15)
        match=re.search(r'interface:\s*(\S+)',raw);assert match
        interface=match.group(1);assert not interface.startswith(('utun','tun','tap')),'Virtual route active'
        return interface
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
         '-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def select(model):
        cmd=['/root/pi0-repro/openpi/.venv/bin/python','-u','/root/pi0-repro/server/progress_policy_select.py',
             '--root','/root/pi0-repro','--model',model]
        result=subprocess.run(ssh+[target,shlex.join(cmd)],capture_output=True,text=True,timeout=60)
        if result.returncode:raise RuntimeError(result.stderr[-2000:])
        return json.loads(result.stdout)
    os.environ.setdefault('MUJOCO_GL','glfw')
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    import numpy as np
    def endpoint(model):
        deadline=time.monotonic()+100
        while True:
            try:
                with RemotePi0(configured(OUT,model=model)) as client:return client.metadata
            except (OSError,TimeoutError,EOFError):
                if time.monotonic()>deadline:raise
            except RuntimeError as exc:
                if 'ConnectionRefusedError' not in str(exc) or time.monotonic()>deadline:raise
            time.sleep(1)
    ledger=dict(plan=plan,complete=False,results=[],restoration=None)
    write(OUT/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'experiment.log')])
    tunnel=None;awake=None;selected=False;log=(OUT/'tunnel.log').open('xb')
    try:
        ledger['route_before']=route();write(OUT/'comparison.json',ledger)
        with socket.socket() as sock:sock.bind(('127.0.0.1',PORT))
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{PORT}:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        # Set before dispatch so a partial remote selection also invokes restoration.
        selected=True;ledger['selection']=dict(dispatch=select('control'),metadata=endpoint('control'))
        write(OUT/'comparison.json',ledger)
        warm=OUT/'warmup';warm.mkdir();cfg=configured(OUT);env=_make_env(cfg,warm)
        try:
            obs,_=env.reset(seed=SEED)
            observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            observation['rtc']=dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)
            requests=[]
            with RemotePi0(cfg) as client:
                for i in range(6):
                    actions,ms,response=client.infer(observation)
                    assert actions.shape==(100,14) and np.isfinite(actions).all()
                    requests.append(dict(index=i,round_trip_ms=ms,server_timing=response.get('server_timing')))
                    write(warm/'requests.json',dict(scope='No environment steps; excluded from outcomes',requests=requests))
            tail=[r['round_trip_ms'] for r in requests[2:]]
            assert max(tail)<1500 and np.median(tail)<800,'Warm latency gate failed'
        finally:env.close()
        for case in plan['cases']:
            assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
            assert sha(OUT/(case['profile']+'.toml'))==plan['scene_sha256'][case['profile']]
            row=dict(case,route_interface=route());ledger['results'].append(row);write(OUT/'comparison.json',ledger)
            print('START',case,flush=True)
            existing={p.name for p in (ROOT/'outputs').iterdir() if p.is_dir()}
            try:
                path=run(configured(OUT,case['profile']));row['run_directory']=str(path);d=inspect_run(path);row['diagnostics']=d
                assert d['prefix_reconstruction']['passed'] and d['summary']['raw_prediction_max_error']==0 and d['summary']['max_action_timestamp_error_steps']==0
                assert not d['summary']['result']['error'] and not d['summary']['result']['aborted_by_window_close']
            except BaseException as exc:
                row['operational_error']=f'{type(exc).__name__}: {exc}'
                row['retained_new_run_directories']=[str(p) for p in (ROOT/'outputs').iterdir() if p.is_dir() and p.name not in existing]
                raise
            finally:write(OUT/'comparison.json',ledger)
            print('END',case['profile'],case['repeat'],'success',d['summary']['result']['success'],'apply_ms',d['action_apply_ms'],flush=True)
        ledger['route_after']=route();ledger['complete']=True
    except BaseException as exc:ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if selected:
            try:ledger['restoration']=dict(dispatch=select('v6'),metadata=endpoint('v6'),verified=True)
            except BaseException as exc:ledger['restoration']=dict(verified=False,error=f'{type(exc).__name__}: {exc}')
        write(OUT/'comparison.json',ledger)
        if tunnel is not None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        if awake is not None:awake.terminate()
        log.close()
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
