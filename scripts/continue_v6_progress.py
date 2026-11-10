"""Versioned train-only v6 acquisition with wall/thread CPU tail diagnostics."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from compare_v6_rtc import SERVICE_CODE
from evaluate_random_multiview import SOURCE_NAMES
from profile_v6_runtime import inspect_run


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def choose_parents(rows,raw):
    import numpy as np
    candidates=[]
    for r in rows:
        p=raw/r['source_episode'];m=read(p/'episode.json');layout=read(p/'layout.json')
        assert r['split']=='train' and m['review_status']=='accepted' and m['eligible_for_training']
        if layout.get('recovery') or layout.get('layout',{}).get('recovery'):continue
        assert sha(p/'trajectory.npz')==r['source_trajectory_sha256']
        s=m['scene'];xyz=s['banana_position'];box=s['box_position']
        candidates.append(dict(source_episode=str(p),layout_id=r['layout_id'],scene=s,seed=m['reset_seed'],
            source_trajectory_sha256=r['source_trajectory_sha256'],features=[xyz[0],xyz[1],s['banana_yaw_degrees'],box[0],box[1]]))
    # Farthest-point geometric coverage only; no learner outcome is consulted.
    x=np.array([r['features'] for r in candidates]);x=(x-x.min(0))/np.maximum(np.ptp(x,axis=0),1e-9)
    selected=[min(range(len(candidates)),key=lambda i:candidates[i]['layout_id'])]
    while len(selected)<6:
        distances=np.min(np.sum((x[:,None]-x[selected][None])**2,axis=-1),axis=1)
        distances[selected]=-1;selected.append(int(np.argmax(distances)))
    return [candidates[i] for i in selected]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--port',type=int,default=18061)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    import numpy as np
    dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3';mapping=read(dataset/'local/pi0_banana_train/source_episodes.json')
    raw=Path(read(dataset/'dataset_manifest.json')['source_path']);parents=choose_parents(mapping['episodes'],raw)
    heldout={r['layout_id'] for split in ('validation','test') for r in read(dataset/f'local/pi0_banana_{split}/source_episodes.json')['episodes']}
    assert not heldout.intersection(p['layout_id'] for p in parents)
    cases=[dict(index=i*2+j,parent=i,profile=('headless','worker')[(i+j)%2]) for i in range(6) for j in range(2)]
    source_hashes=lambda:{n:sha(ROOT/'src/pi0_repro'/n) for n in SOURCE_NAMES}
    plan=dict(schema='v6-train-state-tail-v1',parents=parents,cases=cases,attempts=12,
        source_sha256=source_hashes(),script_sha256=sha(Path(__file__)),dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),
        profile_sha256=sha(ROOT/'configs/banana_v6_bridge20_pi0.toml'),noise_paired=False,
        selection='Six geometrically spaced full training parents selected without learner outcomes; alternating headless/worker order.',
        constraints='Original v6 RTC20/client20, same plant, no policy fallback, no holdout labels. Headless and worker are workload bundles, not isolated causal treatments.',
        correction_sources='Only headless runs become possible v6 recovery sources; worker runs are workload/tail diagnostics. Review precedes labels/export/train.',
        timing='Fresh wall and thread CPU envelopes; difference includes waits/scheduling/other-thread work, not one uniquely identified cause. Preflight/startup excluded from running intervals.')
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    frozen=out/'frozen_source';frozen.mkdir()
    for n in SOURCE_NAMES:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    scenes=[]
    for i,parent in enumerate(parents):
        q=out/f'parent_{i:02d}.toml';q.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in parent['scene'].items() if v is not None)+'\n');scenes.append(q)
    def configured(case):
        cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml');s=cfg.simulation
        cfg.connection.port=args.port;s.scene_config=str(scenes[case['parent']]);s.seed=parents[case['parent']]['seed'];s.episodes=1;s.max_steps=2000
        s.display=case['profile']=='worker';s.save_video=s.display;s.record_multiview=s.display
        s.display_layout='multiview' if s.display else 'policy';s.display_in_worker=s.display;s.reuse_policy_camera=s.display
        s.display_fps=25.;s.video_fps=25
        return cfg
    ledger=dict(plan=plan,complete=False,results=[],endpoint_before=None,endpoint_after=None);write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')]);os.environ.setdefault('MUJOCO_GL','glfw')
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    import re
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def identity():
        command=shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',SERVICE_CODE])
        r=subprocess.run(ssh+[target,command],capture_output=True,text=True,timeout=25)
        if r.returncode:raise RuntimeError(r.stderr[-1000:])
        return json.loads(r.stdout)
    log=(out/'tunnel.log').open('xb');tunnel=None;awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        with socket.socket() as s:s.bind(('127.0.0.1',args.port))
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{args.port}:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+20
        while True:
            if tunnel.poll() is not None:raise RuntimeError('Owned tunnel exited')
            try:
                with RemotePi0(configured(cases[0])) as c:metadata=c.metadata
                break
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(.25)
        ledger['endpoint_before']=dict(metadata=metadata,service_identity=identity());write(out/'comparison.json',ledger)
        warmdir=out/'preflight';warmdir.mkdir();cfg=configured(cases[0]);env=_make_env(cfg,warmdir);requests=[]
        try:
            obs,_=env.reset(seed=cfg.simulation.seed);adapted=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            with RemotePi0(cfg) as c:
                for i in range(6):
                    action,ms,response=c.infer({**adapted,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)})
                    assert action.shape==(100,14) and np.isfinite(action).all();requests.append(dict(round_trip_ms=ms,server_timing=response.get('server_timing'),transport_timing=response.get('transport_timing'),client_transport=response.get('client_transport')))
        finally:env.close()
        write(out/'preflight.json',dict(scope='Six no-step readiness requests, never task outcomes',requests=requests))
        assert max(r['round_trip_ms'] for r in requests[2:])<1500 and np.median([r['round_trip_ms'] for r in requests[2:]])<800,'Readiness gate failed; no tasks started'
        for case in cases:
            assert source_hashes()==plan['source_sha256'],'Source changed during collection'
            row=dict(case);row['source_episode']=parents[case['parent']]['source_episode'];row['layout_id']=parents[case['parent']]['layout_id'];row['split']='train'
            ledger['results'].append(row);write(out/'comparison.json',ledger);print('START',case,flush=True)
            try:
                directory=run(configured(case));row['run_directory']=str(directory);d=inspect_run(directory);row['diagnostics']=d
                assert not d['summary']['result']['error'] and not d['summary']['result']['aborted_by_window_close']
                assert d['prefix_reconstruction']['passed'] and d['summary']['raw_prediction_max_error']==0 and d['summary']['max_action_timestamp_error_steps']==0
                assert source_hashes()==plan['source_sha256']
            except BaseException as exc:row['operational_error']=f'{type(exc).__name__}: {exc}';raise
            finally:write(out/'comparison.json',ledger)
            print('END',case['index'],d['summary']['result']['success'],d['action_apply_ms'],flush=True)
        ledger['endpoint_after']=dict(service_identity=identity());assert ledger['endpoint_after']['service_identity']==ledger['endpoint_before']['service_identity']
        ledger['complete']=True
    except BaseException as exc:ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        write(out/'comparison.json',ledger)
        if tunnel is not None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        log.close();awake.terminate()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
