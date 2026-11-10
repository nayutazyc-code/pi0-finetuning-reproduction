"""Conditional48-case factorial: v6/new weights x original/doubled wrist damping."""
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    training_parent=ROOT/'outputs/ik_continuous_20261029';parent=ROOT/'outputs/release_retreat_20261029';training=read(training_parent/'training_plan_train_only.json')
    out=args.out.resolve() if args.out else parent/'closed_loop'
    assert out.is_relative_to(parent) and out!=parent,'Evaluation must remain under its versioned parent'
    dataset=ROOT/training['arms'][0]['dataset'];mapping=read(dataset/'local/pi0_banana_validation/source_episodes.json')
    allowed={r['source_episode'] for r in mapping['episodes'] if r['split']=='validation'};assert set(LAYOUTS)<=allowed
    raw=Path(read(dataset/'dataset_manifest.json')['source_path']);layouts={n:read(raw/n/'layout.json') for n in LAYOUTS}
    candidate=next(a for a in training['arms'] if a['arm']=='candidate')
    registry={}
    for name,checkpoint_model,scale in [('v6','v6',1.),('candidate','candidate',1.),('v6_damping2','v6',2.),('candidate_damping2','candidate',2.)]:
        registry[name]=dict(checkpoint_model=checkpoint_model,damping_scale=scale,
            config='outputs/long_chunk_20261025/selected_training.toml' if checkpoint_model=='v6' else candidate['config'],
            suffix='banana_v6_long100_accum8000/7999' if checkpoint_model=='v6' else candidate['experiment']+'/999')
    blocks=['v6','candidate','v6_damping2','candidate_damping2','candidate_damping2','v6_damping2','candidate','v6']
    cases=[dict(index=i*6+j,block=i,model=model,layout=LAYOUTS[(i+j)%6],repeat=1 if i<4 else 2) for i,model in enumerate(blocks) for j in range(6)]
    plan=dict(schema='release-damping-factorial-validation-v1',models=registry,cases=cases,attempts=48,noise_paired=False,
        training_plan_sha256=sha(training_parent/'training_plan_train_only.json'),damping_screen_plan_sha256=sha(parent/'damping_screen/plan.json'),conditional_on_screen_pass=True,client_profile_sha256=sha(ROOT/'configs/banana_v6_bridge20_pi0.toml'),script_sha256=sha(Path(__file__)),
        source_sha256={n:sha(ROOT/'src/pi0_repro'/n) for n in ALL_SOURCE_NAMES},validation_mapping_sha256=sha(dataset/'local/pi0_banana_validation/source_episodes.json'),
        objective_priority=['physical jumps and phase/low-motion stalls','retain transport coverage','success secondary'],
        comparison='Four profiles(weight checkpoint x wrist damping1or2), two reverse-order blocks each, six fixed validation layouts twice.48attempts. Noise unpaired; no pooling with prior cohorts or testlabels.',
        gate={'median_active_jerk_p99_ratio_max':.8,'covered_median_active_jerk_p99_ratio_max':.8,'worst_peak_ratio_max':1.1,
              'worst_increment_ratio_max':1.05,'worst_low_motion_pause_add_seconds_max':.1,'coverage_min':'each reference',
              'covered_duration_ratio_max':1.1,'covered_wall_duration_ratio_max':1.1,'median_total_pause_add_seconds_max':.1,'worst_all_frame_peak_ratio_max':1.1,'success_shortfall_vs_weaker_reference_max':1,'minimum_successes':6,'references':['v6','candidate'],'treatment':'candidate_damping2','median_apply_p99_ratio_max':1.1,'worst_apply_ratio_max':1.1,'inference_underruns_not_greater_than_each_reference':True,'decision':'Even a passed small pilot only supports replication; no automatic default/checkpoint promotion.'},
        runtime='All models: RTC20/client20, original top640x480, H100/denoise10, only declared wrist damping differs; all gains/contacts/controller semantics fixed, headless topvideo25fps. Preserve all failures and inference waits; source/prefix/timestamp errors abort.',
        restoration='Restore v6 and verify checkpoint/norm metadata before closing owned tunnel.')
    if out.exists():assert read(out/'plan.json')==plan,'Existing evaluation plan differs'
    else:
        out.mkdir();write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
        frozen=out/'frozen_source';frozen.mkdir()
        for n in ALL_SOURCE_NAMES:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
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
        for scale in (1.,2.):
            scene=dict(d['scene'],wrist_damping_scale=scale)
            p=out/(name+'_damping'+str(int(scale))+'.toml');p.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in scene.items() if v is not None)+'\n');scenes[(name,scale)]=p
    def configured(model,layout=LAYOUTS[0]):
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
