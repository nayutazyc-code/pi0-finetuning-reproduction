"""Frozen50-case OOD suite for the latest original-pi0 candidate."""
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
ALL_SOURCE_NAMES=(*SOURCE_NAMES,'deferred_video.py','pacing.py','control_timing.py','banana_expert.py','expert_path.py','recovery.py','demonstrations.py','quality.py','layouts.py')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    import numpy as np
    parent=ROOT/'outputs/release_retreat_20261029';out=ROOT/'outputs/ood50_20261029'
    mapping_path=ROOT/'data/lerobot/20261029-135803-ik-replaced-b8ed3a/local/pi0_banana_train/source_episodes.json'
    train=read(mapping_path)['episodes'];rng=np.random.default_rng(2026102950)
    cases=[];layouts={}
    for j in range(50):
        group='position' if j<20 else 'yaw' if j<40 else 'appearance'
        scene=dict(banana_position=[float(rng.uniform(.16,.20)),float(rng.uniform(.46,.50)),.045],banana_yaw_degrees=float(rng.uniform(0,20)),banana_mass_kg=.12,banana_friction=[.8,.005,.0001],box_position=[.13,.72,0.],camera='top',settle_seconds=.6,controller_profile='position_pd')
        if group=='position':
            side=j%4
            if side<2:scene['banana_position'][0]=float(rng.uniform(.12,.15) if side==0 else rng.uniform(.21,.24))
            else:scene['banana_position'][1]=float(rng.uniform(.42,.45) if side==2 else rng.uniform(.51,.54))
        elif group=='yaw':scene['banana_yaw_degrees']=float(rng.uniform(-60,-20) if j%2==0 else rng.uniform(40,80))
        else:
            scene['table_rgba']=[[.18,.25,.50,1.],[.55,.22,.25,1.]][j%2]
            scene['light_scale']=.65 if (j//2)%2==0 else 1.4
        name=f'ood_{j:02d}';layouts[name]=dict(scene=scene,reset_seed=2026102900+j)
        cases.append(dict(index=j,block=0,model='candidate_damping2',layout=name,repeat=1,group=group))
    # Freeze an interleaved, outcome-independent order across shift types.
    order=rng.permutation(50).tolist();cases=[dict(cases[k],index=i) for i,k in enumerate(order)]
    registry={'candidate_damping2':dict(checkpoint_model='candidate',damping_scale=2.,suffix='banana_v6_ik_candidate1000/999'),'v6':dict(checkpoint_model='v6',damping_scale=1.,suffix='banana_v6_long100_accum8000/7999')};blocks=['candidate_damping2']
    plan=dict(schema='finetuning-ood50-v1',attempts=50,cases=cases,layouts=layouts,models=registry,seed=2026102950,
        groups=dict(position=20,yaw=20,appearance=10),
        reference_training_envelope=dict(banana_x=[.16,.20],banana_y=[.46,.50],yaw_degrees=[0,20],light_scale=[.85,1.15]),
        scope='OOD relative to local task fine-tuning data, not a claim of novelty to pi0 pretraining. Bounded tabletop XY/yaw and appearance only, not arbitrary3D pose, new objects or camera changes. Appearance jointly shifts palette and illumination; cannot isolate their effects.',
        success_rule='Unchanged BananaBoxEnv physical grasp, containment, release and stability;2000-step40s cap; no expert fallback.',
        sampling='50 fixed seeded independent resets, one policy attempt each, no outcome-based replacement/retry. All failures/waits retained. No training or threshold tuning. Prior validation/test layouts not reused.',
        client_profile_sha256=sha(ROOT/'configs/banana_v6_bridge20_pi0.toml'),script_sha256=sha(Path(__file__)),
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ALL_SOURCE_NAMES},train_mapping_sha256=sha(mapping_path),
        runtime='candidate999, RTC20/client20,H100,denoise10,wristdamping2,original top640x480 andROI unchanged;headless25fps video; no added policy cameras.',restoration='Restore and verify original v6 metadata before closing owned tunnel.')
    if out.exists():assert read(out/'plan.json')==plan,'Existing plan differs'
    else:
        out.mkdir();write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes());frozen=out/'frozen_source';frozen.mkdir()
        for n in ALL_SOURCE_NAMES:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    if args.prepare_only:print('PREPARED',out);return
    assert not (out/'comparison.json').exists(),'Preserve existing attempts; do not rerun'
    ledger=dict(plan=plan,complete=False,results=[],selected_models=[],restoration=None);write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')]);os.environ.setdefault('MUJOCO_GL','glfw')
    import numpy as np
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    scenes={}
    for name,d in layouts.items():
        for scale in (1.,2.):
            scene=dict(d['scene'],wrist_damping_scale=scale)
            p=out/(name+'_damping'+str(int(scale))+'.toml');p.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in scene.items() if v is not None)+'\n');scenes[(name,scale)]=p
    def configured(model,layout=cases[0]['layout']):
        cfg=load_config(ROOT/'configs/banana_v6_bridge20_pi0.toml');cfg.connection.port=18082;cfg.model.expected_checkpoint_suffix=registry[model]['suffix']
        s=cfg.simulation;s.episodes=1;s.scene_config=str(scenes[(layout,registry[model]['damping_scale'])]);s.seed=layouts[layout]['reset_seed'];s.max_steps=2000;s.display=False;s.save_video=True;s.video_fps=25
        return cfg
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    import re
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def route():
        import re
        raw=subprocess.check_output(['route','-n','get',target],text=True,timeout=15)
        match=re.search(r'interface:\s*(\S+)',raw);assert match and not match.group(1).startswith(('utun','tun','tap')),'Virtualroute active'
        return match.group(1)
    def remote(code):
        r=subprocess.run(ssh+[target,shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-u','-c',code])],capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    def select_model(model):
        command=['/root/pi0-repro/openpi/.venv/bin/python','-u','/root/pi0-repro/outputs/ik_continuous_20261029/cloud_policy_select.py','--root','/root/pi0-repro','--model',registry[model]['checkpoint_model']]
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
        status=remote("import json;from pathlib import Path;print((Path('/root/pi0-repro/outputs/ik_continuous_20261029/cloud_training/status.json')).read_text())")
        assert status['status']=='both_trained_pending_validation'
        assert read(parent/'closed_loop_replication_direct/audit.json')['passed']
        assert read(parent/'damping_screen/results.json')['complete'] and read(parent/'damping_screen/results.json')['passed'],'Frozen damping screen did not pass'
        assert sha(ROOT/'configs/banana_v6_bridge20_pi0.toml')==plan['client_profile_sha256']
        with socket.socket() as s:s.bind(('127.0.0.1',18082))
        tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L','127.0.0.1:18082:127.0.0.1:8001',target],stdout=log,stderr=subprocess.STDOUT)
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]);started=True
        for block,model in enumerate(blocks):
            ledger.setdefault('routes',[]).append(dict(block=block,interface=route()))
            selected=select_model(model);metadata=endpoint(model);ledger['selected_models'].append(dict(block=block,model=model,dispatch=selected,metadata=metadata));write(out/'comparison.json',ledger)
            warm=out/(f'warmup_{block:02d}_'+model);warm.mkdir();cfg=configured(model);env=_make_env(cfg,warm);requests=[]
            try:
                obs,_=env.reset(seed=cfg.simulation.seed);observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
                prefix=np.repeat(obs['agent_pos'][None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
                with RemotePi0(cfg) as c:
                    for i in range(6):
                        a,ms,response=c.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=10,blend_steps=20,guidance_max=.5)})
                        assert a.shape==(100,14) and np.isfinite(a).all();requests.append(dict(round_trip_ms=ms,server_timing=response.get('server_timing')))
            finally:env.close()
            write(warm/'requests.json',dict(scope='No physical environment steps; excluded from outcomes',requests=requests))
            assert max(r['round_trip_ms'] for r in requests[2:])<1500 and np.median([r['round_trip_ms'] for r in requests[2:]])<800,'Readiness gatefailed'
            for case in [c for c in cases if c['block']==block]:
                assert all(sha(ROOT/'src/pi0_repro'/n)==digest for n,digest in plan['source_sha256'].items())
                row=dict(case,route=route());ledger['results'].append(row);write(out/'comparison.json',ledger);print('START',case,flush=True)
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
