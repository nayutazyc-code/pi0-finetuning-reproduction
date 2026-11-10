"""Predeclared v6 RTC20 runtime-cost diagnostic, retaining every task outcome.

Diagnostic seeds are deliberately chosen from earlier recordings (ordinary,
slow grasp, failed grasp). This is neither a fresh reliability estimate nor
paired policy-noise evaluation. Startup, teardown and analysis are untimed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from compare_v6_rtc import SERVICE_CODE
from evaluate_random_multiview import SOURCE_NAMES

PROFILES = ('control', 'video', 'multiview')
SEEDS = (2026102801, 2026102802, 2026102805)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def configured(profile, seed, port):
    worker = profile == 'worker_multiview'
    reuse = profile in ('reuse_multiview','worker_multiview')
    if profile in ('legacy_multiview', 'reuse_multiview','worker_multiview'):
        profile = 'multiview'
    cfg = load_config(ROOT/'configs/banana_v6_random_multiview_pi0.toml')
    s = cfg.simulation
    cfg.connection.port = port
    s.episodes = 1
    s.seed = seed
    s.display = profile == 'multiview'
    s.save_video = profile != 'control'
    s.record_multiview = profile == 'multiview'
    s.display_layout = 'multiview' if profile == 'multiview' else 'policy'
    s.reuse_policy_camera = reuse
    s.display_in_worker = worker
    return cfg


def distribution(values):
    import numpy as np
    a = np.asarray(values, float)
    return None if not a.size else dict(median=float(np.median(a)), p95=float(np.quantile(a,.95)),
                                       p99=float(np.quantile(a,.99)), max=float(a.max()))


def inspect_run(directory):
    import numpy as np
    from summarize_banana_execution import summarize
    from pi0_repro.motion_quality import motion_quality
    from analyze_rtc_pair import reconstructed_prefix
    summary = summarize(directory)['episodes'][0]
    controls = [json.loads(s) for s in (directory/'episode_000_control.jsonl').read_text().splitlines()]
    tasks = [json.loads(s) for s in (directory/'episode_000_task.jsonl').read_text().splitlines()]
    latency = [json.loads(s) for s in (directory/'episode_000_latency.jsonl').read_text().splitlines()]
    cfg = json.loads((directory/'run.json').read_text())['config']
    with np.load(directory/'episode_000.npz', allow_pickle=False) as data:
        quality = motion_quality(data['state'], data['action'], tasks)
        prefix = reconstructed_prefix(data, latency, cfg['simulation']['chunk_blend_steps'])
    # Entry i is action i. Its gap includes previous step i-1 physics and
    # bookkeeping, followed by entry i's pre-action preparation and pacing.
    events = []
    for i in range(1,len(controls)):
        gap = controls[i]['action_apply_wall_seconds']-controls[i-1]['action_apply_wall_seconds']
        if gap > .05:
            events.append(dict(step=i, gap_ms=gap*1000, previous_control=controls[i-1],
                               current_control=controls[i], task=tasks[i-1]))
    return dict(summary=summary, motion_quality=quality, prefix_reconstruction=prefix,
                action_apply_ms=distribution(np.diff([r['action_apply_wall_seconds'] for r in controls])*1000),
                gaps_over_50ms=len(events), gaps_over_100ms=sum(e['gap_ms']>100 for e in events),
                longest_gaps=sorted(events,key=lambda e:e['gap_ms'],reverse=True)[:12],
                evidence_scope='Wall-clock gaps and simulation-time movement are separate; nested stage timers are not additive causal slices.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--ssh-target')
    p.add_argument('--port', type=int, default=18051)
    p.add_argument('--check-only', action='store_true')
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--camera-ablation', action='store_true', help='Six matched-reset multiview trials: legacy versus persistent top camera')
    modes.add_argument('--display-ablation', action='store_true', help='Six matched-reset trials: persistent camera, parent versus worker window')
    modes.add_argument('--optimized-ablation', action='store_true', help='Old camera/parent window versus reused camera/worker window bundle')
    p.add_argument('--repeats',type=int,choices=(1,2,3),default=1)
    args = p.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs') and 1024 < args.port < 65536
    profiles = (('legacy_multiview','worker_multiview') if args.optimized_ablation else
                ('reuse_multiview','worker_multiview') if args.display_ablation else
                ('legacy_multiview','reuse_multiview') if args.camera_ablation else PROFILES)
    cases = [dict(index=r*len(SEEDS)*len(profiles)+i*len(profiles)+j, repeat=r+1,
                  seed=seed, profile=profiles[(r+i+j)%len(profiles)])
             for r in range(args.repeats) for i,seed in enumerate(SEEDS) for j in range(len(profiles))]
    source_hashes = lambda: {n:sha(ROOT/'src/pi0_repro'/n) for n in SOURCE_NAMES}
    plan = dict(schema='v6-runtime-cost-v1', cases=cases, attempts=len(cases), noise_paired=False,
        source_sha256=source_hashes(), script_sha256=sha(Path(__file__)),
        source_profile_sha256=sha(ROOT/'configs/banana_v6_random_multiview_pi0.toml'),
        profiles={k:asdict(configured(k,SEEDS[0],args.port)) for k in profiles},
        camera_ablation=args.camera_ablation,
        display_ablation=args.display_ablation,
        optimized_ablation=args.optimized_ablation,
        repeats=args.repeats,
        runtime_adoption_gate={'median_trial_apply_p99_ratio_max':.90,
            'worst_apply_interval_ratio_max':1.,'gap_over50_rate_per1000_steps_ratio_max':1.,
            'running_underruns_max':0,'transport_coverage_min':'baseline',
            'covered_median_simulated_duration_ratio_max':1.10,
            'covered_median_active_jerk_p99_ratio_max':1.10,
            'scope':'All checks and preserved visual review required before changing the helper default. No automatic promotion and no population/physical-robot claim.'},
        seed_selection='Previously observed ordinary, slow-grasp and no-lift cases; diagnostic selection, no population inference.',
        comparison='Control includes fresh policy-camera rendering and cloud inference. Video adds top25fps recording. Multiview adds native GUI, read-only renderer, and a second25fps recording: a bundle, not separate component effects.',
        gates='Retain all failures/underruns; abort only on source, timestamp, prefix or metadata corruption, service change, or operational exception. No reruns or automatic policy promotion.',
        timing='Running action-application intervals exclude initial wait and post-loop teardown. Logs retain inference waits. Analysis occurs between attempts, not during timed control.',
        constraints='Original pi0 v6/7999, H100, RTC20, client bridge20, same scene/controller/filter, no training or cloud mutations.')
    if args.check_only:
        print(json.dumps(plan,indent=2)); return
    import re
    target = args.ssh_target or (ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    frozen = out/'frozen_source'; frozen.mkdir()
    for n in SOURCE_NAMES: (frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')])
    os.environ.setdefault('MUJOCO_GL','glfw')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import numpy as np
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run, _make_env
    from pi0_repro.observations import aloha_observation
    ssh = ['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes',
           '-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10',
           '-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def identity():
        cmd = shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',SERVICE_CODE])
        r = subprocess.run(ssh+[target,cmd],capture_output=True,text=True,timeout=25)
        if r.returncode: raise RuntimeError('Read-only service check failed: '+r.stderr[-1500:])
        return json.loads(r.stdout)
    ledger = dict(plan=plan,complete=False,results=[],endpoint_before=None,endpoint_after=None)
    write(out/'comparison.json',ledger)
    log = (out/'tunnel.log').open('xb')
    tunnel = None
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        with socket.socket() as s: s.bind(('127.0.0.1',args.port))
        tunnel = subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',
            f'127.0.0.1:{args.port}:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        deadline = time.monotonic()+20
        while True:
            if tunnel.poll() is not None: raise RuntimeError('Owned tunnel exited')
            try:
                with RemotePi0(configured('control',SEEDS[0],args.port)) as c: metadata=c.metadata
                break
            except OSError:
                if time.monotonic()>deadline: raise
                time.sleep(.25)
        assert metadata['transport_proxy']['kind']=='lossless-png'
        ledger['endpoint_before']=dict(metadata=metadata,service_identity=identity())
        write(out/'comparison.json',ledger)
        cfg = configured('control',SEEDS[0],args.port)
        warmdir = out/'preflight'; warmdir.mkdir()
        env = _make_env(cfg,warmdir)
        preflight=[]
        try:
            obs,_=env.reset(seed=SEEDS[0])
            adapted=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(np.asarray(obs['agent_pos'],np.float32)[None],100,axis=0)
            prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            with RemotePi0(cfg) as c:
                for i in range(6):
                    action,ms,response=c.infer({**adapted,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,
                        valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)})
                    assert action.shape==(100,14) and np.isfinite(action).all()
                    preflight.append(dict(round_trip_ms=ms,server_timing=response.get('server_timing')))
        finally: env.close()
        write(out/'preflight.json',dict(scope='Six no-step requests, excluded from outcomes',requests=preflight))
        assert max(r['round_trip_ms'] for r in preflight[2:])<1200,'Network preflight failed'
        for case in cases:
            assert source_hashes()==plan['source_sha256'],'Source changed'
            row=dict(case)
            ledger['results'].append(row); write(out/'comparison.json',ledger)
            print('START',case,flush=True)
            try:
                directory=run(configured(case['profile'],case['seed'],args.port))
                row['run_directory']=str(directory)
                row['diagnostics']=inspect_run(directory)
                d=row['diagnostics']; s=d['summary']
                assert s['result']['error'] is None,s['result']['error']
                assert s['raw_prediction_max_error']==0 and s['max_action_timestamp_error_steps']==0
                assert d['prefix_reconstruction']['passed'],d['prefix_reconstruction']
                assert source_hashes()==plan['source_sha256']
            except BaseException as e:
                row['operational_error']=f'{type(e).__name__}: {e}'; raise
            finally: write(out/'comparison.json',ledger)
            print('END',case['index'],s['result']['success'],d['action_apply_ms'],flush=True)
        ledger['endpoint_after']=dict(service_identity=identity())
        assert ledger['endpoint_after']['service_identity']==ledger['endpoint_before']['service_identity']
        ledger['complete']=True
    except BaseException as e:
        ledger['error']=f'{type(e).__name__}: {e}'; raise
    finally:
        write(out/'comparison.json',ledger)
        if tunnel is not None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        log.close();inhibitor.terminate()
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
