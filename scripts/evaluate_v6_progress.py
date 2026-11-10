"""Separate six-layout validation pilot: v6, matched control, recovery candidate."""
import argparse
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
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from evaluate_random_multiview import SOURCE_NAMES
from compare_v6_rtc import LAYOUTS
from profile_v6_runtime import inspect_run


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--network-preflight',type=Path)
    args=parser.parse_args()
    parent=ROOT/'outputs/v6_progress_20261028';training=read(parent/'training_plan_train_only.json')
    out=args.out.resolve() if args.out else parent/'evaluation'
    assert out.is_relative_to(parent) and out!=parent,'Evaluation must remain under its versioned parent'
    dataset=ROOT/training['arms'][0]['dataset'];mapping=read(dataset/'local/pi0_banana_validation/source_episodes.json')
    allowed={r['source_episode'] for r in mapping['episodes'] if r['split']=='validation'};assert set(LAYOUTS)<=allowed
    raw=Path(read(dataset/'dataset_manifest.json')['source_path']);layouts={n:read(raw/n/'layout.json') for n in LAYOUTS}
    registry={'v6':{'config':'outputs/long_chunk_20261025/selected_training.toml','suffix':'banana_v6_long100_accum8000/7999'}}
    for arm in training['arms']:registry[arm['arm']]={'config':arm['config'],'suffix':arm['experiment']+'/999'}
    cases=[dict(index=i*6+j,model=model,layout=LAYOUTS[(i+j)%6],repeat=1) for i,model in enumerate(registry) for j in range(6)]
    plan=dict(schema='v6-progress-validation-pilot-v1',models=registry,cases=cases,attempts=18,noise_paired=False,
        training_plan_sha256=sha(parent/'training_plan_train_only.json'),script_sha256=sha(Path(__file__)),
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in SOURCE_NAMES},validation_mapping_sha256=sha(dataset/'local/pi0_banana_validation/source_episodes.json'),
        objective_priority=['physical jumps and phase/low-motion stalls','retain transport coverage','success secondary'],
        comparison='New same-instance pilot, six fixed validation layouts once per checkpoint. Sequential model blocks, rotated layout order; latency and noise unpaired, drift not eliminated. No pooling with old cohorts or test-layout use.',
        gate={'median_active_jerk_p99_ratio_max':.9,'covered_median_active_jerk_p99_ratio_max':.9,'worst_peak_ratio_max':1.1,
              'worst_increment_ratio_max':1.1,'worst_low_motion_pause_add_seconds_max':.1,'coverage_min':'each reference',
              'covered_duration_ratio_max':1.1,'references':['v6','control'],'decision':'Even a passed small pilot only supports replication; no automatic default/checkpoint promotion.'},
        runtime='All models: RTC20/client20, original top640x480, H100/denoise10, physical controller unchanged, headless topvideo25fps. Preserve all failures and inference waits; source/prefix/timestamp errors abort.',
        restoration='Restore v6 and verify checkpoint/norm metadata before closing owned tunnel.')
    if args.network_preflight:
        preflight=args.network_preflight.resolve();proof=read(preflight)
        assert preflight.is_relative_to(parent) and proof['passed'] and proof['probe_complete']
        assert time.time()-proof['verified_at']<1800,'Network preflight older than30minutes'
        interruption=parent/'evaluation_transport_interruption.json'
        assert read(interruption)['restoration_verified'] and not read(parent/'evaluation/comparison.json')['complete']
        plan['network_retest']=dict(preflight=str(preflight.relative_to(parent)),preflight_sha256=sha(preflight),
            interruption_sha256=sha(interruption),original_incomplete_ledger_sha256=sha(parent/'evaluation/comparison.json'),
            scope='Separate complete pilot after a network operational interruption. Same18cases and gates; no pooling or outcome-filtered retries.')
    if out.exists():assert read(out/'plan.json')==plan,'Existing evaluation plan differs'
    else:
        out.mkdir();write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
        frozen=out/'frozen_source';frozen.mkdir()
        for n in SOURCE_NAMES:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    if args.prepare_only:print('PREPARED',out);return
    assert not (out/'comparison.json').exists(),'Preserve previous pilot; no reruns'
    ledger=dict(plan=plan,complete=False,results=[],selected_models=[],restoration=None);write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')]);os.environ.setdefault('MUJOCO_GL','glfw')
    import numpy as np
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    scenes={}
    for name,d in layouts.items():
        p=out/(name+'.toml');p.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in d['scene'].items() if v is not None)+'\n');scenes[name]=p
    def configured(model,layout=LAYOUTS[0]):
        cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml');cfg.connection.port=18071;cfg.model.expected_checkpoint_suffix=registry[model]['suffix']
        s=cfg.simulation;s.episodes=1;s.scene_config=str(scenes[layout]);s.seed=layouts[layout]['reset_seed'];s.max_steps=2000;s.display=False;s.save_video=True;s.video_fps=25
        return cfg
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    import re
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def remote(code):
        r=subprocess.run(ssh+[target,shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-u','-c',code])],capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    def select_model(model):
        command=['/root/pi0-repro/openpi/.venv/bin/python','-u','/root/pi0-repro/server/progress_policy_select.py','--root','/root/pi0-repro','--model',model]
        r=subprocess.run(ssh+[target,shlex.join(command)],capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    def endpoint(model):
        deadline=time.monotonic()+100
        while True:
            try:
                with RemotePi0(configured(model)) as c:return c.metadata
            except (OSError,TimeoutError):
                if time.monotonic()>deadline:raise
                time.sleep(1)
            except RuntimeError as exc:
                if 'ConnectionRefusedError' not in str(exc) or time.monotonic()>deadline:raise
                time.sleep(1)
    tunnel=None;awake=None;started=False;log=(out/'tunnel.log').open('xb')
    try:
        status=remote("import json;from pathlib import Path;print((Path('/root/pi0-repro/outputs/v6_progress_20261028/cloud_training/status.json')).read_text())")
        assert status['status']=='both_trained_pending_validation'
        with socket.socket() as s:s.bind(('127.0.0.1',18071))
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L','127.0.0.1:18071:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);started=True
        for model in registry:
            selected=select_model(model);metadata=endpoint(model);ledger['selected_models'].append(dict(model=model,dispatch=selected,metadata=metadata));write(out/'comparison.json',ledger)
            warm=out/('warmup_'+model);warm.mkdir();cfg=configured(model);env=_make_env(cfg,warm);requests=[]
            try:
                obs,_=env.reset(seed=cfg.simulation.seed);observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
                prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
                with RemotePi0(cfg) as c:
                    for i in range(6):
                        a,ms,response=c.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)})
                        assert a.shape==(100,14) and np.isfinite(a).all();requests.append(dict(round_trip_ms=ms,server_timing=response.get('server_timing')))
            finally:env.close()
            write(warm/'requests.json',dict(scope='No physical environment steps; excluded from outcomes',requests=requests))
            for case in [c for c in cases if c['model']==model]:
                assert all(sha(ROOT/'src/pi0_repro'/n)==digest for n,digest in plan['source_sha256'].items())
                row=dict(case);ledger['results'].append(row);write(out/'comparison.json',ledger);print('START',case,flush=True)
                try:
                    runpath=run(configured(model,case['layout']));row['run_directory']=str(runpath);row['diagnostics']=inspect_run(runpath)
                    d=row['diagnostics'];assert d['prefix_reconstruction']['passed'] and d['summary']['raw_prediction_max_error']==0 and d['summary']['max_action_timestamp_error_steps']==0
                    assert not d['summary']['result']['error'] and not d['summary']['result']['aborted_by_window_close']
                except BaseException as exc:row['operational_error']=f'{type(exc).__name__}: {exc}';raise
                finally:write(out/'comparison.json',ledger)
                print('END',case['index'],d['summary']['result']['success'],d['motion_quality']['active_frame_max_jerk_rad_s3'],flush=True)
        ledger['complete']=True
    except BaseException as exc:ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if started:
            try:ledger['restoration']=dict(dispatch=select_model('v6'),metadata=endpoint('v6'),verified=True)
            except BaseException as exc:ledger['restoration']=dict(verified=False,error=f'{type(exc).__name__}: {exc}')
        write(out/'comparison.json',ledger)
        if tunnel is not None:
            tunnel.terminate()
            try:tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        if awake is not None:awake.terminate()
        log.close()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
