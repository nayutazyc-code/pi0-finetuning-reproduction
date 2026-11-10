"""Predeclared eight disjoint train-parent v6 acquisitions for expert replication."""
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
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from evaluate_v6_progress import read,sha,write
from evaluate_random_multiview import SOURCE_NAMES
from pi0_repro.config import load_config

OUT=ROOT/'outputs/contact_recovery_replication_20261029'
PORT=18079
SUFFIX='banana_v6_long100_accum8000/7999'


def choose_parents():
    import numpy as np
    dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'
    mapping=read(dataset/'local/pi0_banana_train/source_episodes.json')
    raw=Path(read(dataset/'dataset_manifest.json')['source_path'])
    old=read(ROOT/'outputs/contact_recovery_20261029/selection.json')
    excluded={r['source_layout_id'] for r in old['plan']['sources']}
    candidates=[]
    for row in mapping['episodes']:
        source=raw/row['source_episode'];meta=read(source/'episode.json');layout=read(source/'layout.json')
        assert row['split']==meta['split']=='train' and meta['review_status']=='accepted' and meta['eligible_for_training']
        if layout.get('recovery') or (layout.get('layout') or {}).get('recovery'):continue
        assert sha(source/'trajectory.npz')==row['source_trajectory_sha256']
        s=meta['scene']
        candidates.append(dict(source_episode=str(source),layout_id=meta['layout_id'],scene=s,seed=meta['reset_seed'],
            source_trajectory_sha256=row['source_trajectory_sha256'],source_metadata_sha256=sha(source/'episode.json'),
            features=[s['banana_position'][0],s['banana_position'][1],s['banana_yaw_degrees'],s['box_position'][0],s['box_position'][1]]))
    candidates.sort(key=lambda r:r['layout_id'])
    anchors=[i for i,p in enumerate(candidates) if p['layout_id'] in excluded]
    assert len(candidates)==20 and len(anchors)==6
    x=np.asarray([p['features'] for p in candidates]);x=(x-x.min(0))/np.maximum(np.ptp(x,axis=0),1e-9)
    chosen=[]
    while len(chosen)<8:
        remaining=[i for i in range(len(candidates)) if i not in anchors+chosen]
        distance=lambda i:float(np.min(np.sum((x[i]-x[anchors+chosen])**2,axis=1)))
        i=max(remaining,key=distance);chosen.append(i)
    return [candidates[i] for i in chosen],sorted(excluded),sha(dataset/'local/pi0_banana_train/source_episodes.json')


def configured(parent):
    cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml');cfg.connection.port=PORT
    s=cfg.simulation;s.scene_config=str(OUT/f'parent_{parent["parent"]:02d}.toml')
    s.seed=parent['seed'];s.episodes=1;s.max_steps=2000;s.display=False;s.save_video=True;s.video_fps=25
    assert not s.recover_control_overruns and not s.trace_control_timing and not s.defer_video_render and not s.reuse_policy_camera
    return cfg


def prepare():
    parents,excluded,mapping_sha=choose_parents()
    for i,p in enumerate(parents):p['parent']=i
    prior=ROOT/'outputs/contact_recovery_20261029'
    names=(*SOURCE_NAMES,'deferred_video.py','pacing.py','control_timing.py','banana_expert.py','recovery.py','demonstrations.py','quality.py')
    gate=read(prior/'unload_plan.json')['gates']
    plan=dict(schema='eight-train-parent-contact-replication-v1',parents=parents,excluded_prior_six_layout_ids=excluded,
        selection='Deterministic farthest-first normalized bananaXY/yaw+boxXY coverage, anchored to prior6; ties sorted layout_id. No learner outcomes consulted.',
        mapping_sha256=mapping_sha,script_sha256=sha(Path(__file__)),source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in names},
        config_by_parent=[asdict(configured(p)) for p in parents],attempts=8,checkpoint_suffix=SUFFIX,
        expert_configs={name:dict(path=str(prior/(name+'_expert.toml')),sha256=sha(prior/(name+'_expert.toml'))) for name in ('clearance','unload')},
        inherited_gate_sha256=sha(prior/'unload_plan.json'),gates=gate,
        takeover_rule='Earliest post-step3consecutive wrist effort>1Nm with right-finger/table contact and table-resting/notgrasped banana. If absent: first near-grasp table-resting/notgrasped state after >=15controlsteps. Label fallback explicitly as reference_without_sustained_effort; if neither exists retain missing/no replacement.',
        exposure_gate=dict(required_selected_parents=8,minimum_sustained_effort_states=4),
        teacher_pairing='Same physical learner-state replay for clearance and unload. Order alternates perparent; all16expert outcomes retained, no retries/replacements.',
        box_diagnosis='For learner+teachers separately: held banana XY within4cm of box and notinside; time-to-firstcontainment and low-motion dwells. Missing boxphase is null/unavailable, notzero.',
        training_condition='Only all inherited numerical gates, exposure, lineage/physical audits and explicit data review permit acceptance/export. Then data-only matched continuation with unchanged batch/steps and separate validation. Failed gates remainfailed; no tuning/default promotion.',
        runtime='Original v6H100/denoise10 RTC20/client20 kp200, original camera/controller/filter, headless top25fps. No newruntimeoptions, injection, contact fallback, or expert during learner runs.',
        service='Use existing v6 process without restart; metadata+serviceidentity before/after. No training/checkpoint/norm mutation.',
        source_scope='Eight other existing trainparents, not newly independent layouts or population estimate. No validation/test data used to label, select or tune. Policy noise unpaired.',
        preflight='Public route;6 no-step warmups, last4median<800ms/max<1500ms. Startup not taskoutcome. Preserve all errors/waits.')
    if OUT.exists():assert read(OUT/'plan.json')==plan,'Preserve existing differing plan'
    else:
        OUT.mkdir();write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes())
        frozen=OUT/'frozen_source';frozen.mkdir()
        for n in names:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
        for p in parents:
            (OUT/f'parent_{p["parent"]:02d}.toml').write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in p['scene'].items() if v is not None)+'\n')
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    plan=prepare()
    if args.prepare_only:print('PREPARED',[Path(p['source_episode']).name for p in plan['parents']]);return
    assert not (OUT/'acquisition.json').exists(),'No in-place retry'
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def route():
        raw=subprocess.check_output(['route','-n','get',target],text=True,timeout=15);m=re.search(r'interface:\s*(\S+)',raw);assert m
        assert not m.group(1).startswith(('utun','tun','tap')),'Virtual routeactive'
        return m.group(1)
    from compare_v6_rtc import SERVICE_CODE
    def identity():
        r=subprocess.run(ssh+[target,shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',SERVICE_CODE])],capture_output=True,text=True,timeout=25)
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        return json.loads(r.stdout)
    os.environ.setdefault('MUJOCO_GL','glfw')
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    from profile_v6_runtime import inspect_run
    import numpy as np
    ledger=dict(plan_sha256=sha(OUT/'plan.json'),complete=False,results=[])
    write(OUT/'acquisition.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(OUT/'experiment.log')])
    tunnel=awake=None;log=(OUT/'tunnel.log').open('xb')
    try:
        ledger['route_before']=route();ledger['service_before']=identity();write(OUT/'acquisition.json',ledger)
        with socket.socket() as s:s.bind(('127.0.0.1',PORT))
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{PORT}:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        end=time.monotonic()+25
        while True:
            if tunnel.poll() is not None:raise RuntimeError('Owned tunnel exited')
            try:
                with RemotePi0(configured(plan['parents'][0])) as c:ledger['metadata_before']=c.metadata
                break
            except (OSError,TimeoutError,EOFError):
                if time.monotonic()>end:raise
                time.sleep(.3)
        warm=OUT/'warmup';warm.mkdir();cfg=configured(plan['parents'][0]);env=_make_env(cfg,warm);requests=[]
        try:
            obs,_=env.reset(seed=cfg.simulation.seed)
            observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            observation['rtc']=dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)
            with RemotePi0(cfg) as c:
                for i in range(6):
                    a,ms,response=c.infer(observation);assert a.shape==(100,14) and np.isfinite(a).all()
                    requests.append(dict(index=i,round_trip_ms=ms,server_timing=response.get('server_timing')))
                    write(warm/'requests.json',dict(requests=requests,scope='No policy-control env.step; reset/settling only; excludedoutcomes'))
            tail=[r['round_trip_ms'] for r in requests[2:]];assert np.median(tail)<800 and max(tail)<1500,'Readiness gatefailed'
        finally:env.close()
        for p in plan['parents']:
            assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
            assert sha(Path(p['source_episode'])/'episode.json')==p['source_metadata_sha256']
            row=dict(parent=p['parent'],layout_id=p['layout_id'],source_episode=p['source_episode'],split='train',route=route())
            ledger['results'].append(row);write(OUT/'acquisition.json',ledger);print('START',p['parent'],Path(p['source_episode']).name,flush=True)
            existing={d.name for d in (ROOT/'outputs').iterdir() if d.is_dir()}
            try:
                path=run(configured(p));d=inspect_run(path)
                row.update(run_directory=str(path),diagnostics=d)
                assert d['prefix_reconstruction']['passed'] and d['summary']['raw_prediction_max_error']==0 and d['summary']['max_action_timestamp_error_steps']==0
                assert not d['summary']['result']['error'] and not d['summary']['result']['aborted_by_window_close']
            except BaseException as e:
                row['operational_error']=f'{type(e).__name__}: {e}'
                row['retained_partial_run_directories']=[str(d) for d in (ROOT/'outputs').iterdir() if d.is_dir() and d.name not in existing]
                raise
            finally:write(OUT/'acquisition.json',ledger)
            print('END',p['parent'],'success',d['summary']['result']['success'],'underruns',d['summary']['result']['execution']['inference_underruns'],flush=True)
        ledger['service_after']=identity();assert ledger['service_before']==ledger['service_after']
        with RemotePi0(configured(plan['parents'][0])) as c:ledger['metadata_after']=c.metadata
        assert ledger['metadata_before']==ledger['metadata_after'];ledger['route_after']=route();ledger['complete']=True
    except BaseException as e:ledger['error']=f'{type(e).__name__}: {e}';raise
    finally:
        write(OUT/'acquisition.json',ledger)
        if tunnel is not None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        if awake is not None:awake.terminate()
        log.close()
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__':main()
