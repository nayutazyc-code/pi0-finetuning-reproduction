"""Same-session matched-reset RTC20/RTC10 experiment; all 36 outcomes retained."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from pi0_repro.remote import RemotePi0
from screen_v6_motion import extra_diagnostics, sha, source_hashes, write

PROFILES = {'rtc20': 'banana_v6_bridge20_pi0.toml', 'rtc10': 'banana_v6_rtc10_pi0.toml'}
LAYOUTS = ['episode_004','episode_012','episode_019','episode_020','episode_026','episode_029']
SERVICE_CODE = r'''
from pathlib import Path
import json
expected='/root/pi0-repro/checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999'
found=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=[p.decode() for p in path.read_bytes().split(b'\0') if p]
  assert not any(p.endswith(('/banana_finetune.py','/motion_train_job.py')) for p in args),'Training active'
  if any(p.endswith('/server/serve_pi0.py') for p in args):
   assert args[args.index('--checkpoint')+1]==expected,'Wrong checkpoint'
   assert '--rtc' in args and args[args.index('--denoise-steps')+1]=='10'
   stat=(path.parent/'stat').read_text().rsplit(')',1)[1].split()
   found.append({'pid':int(path.parent.name),'starttime_ticks':int(stat[19]),'checkpoint':expected})
 except OSError: pass
assert len(found)==1,'Expected exactly one existing v6 service'
print(json.dumps(found[0]))
'''


def profile_check():
    configs={k:asdict(load_config(ROOT/'configs'/v)) for k,v in PROFILES.items()}
    assert configs['rtc10']['simulation']['rtc_blend_steps']==10
    assert configs['rtc20']['simulation']['rtc_blend_steps']==20
    configs['rtc10']['simulation']['rtc_blend_steps']=20
    assert configs['rtc10']==configs['rtc20'],'Only RTC taper may differ'


def make_cases(layouts):
    cases=[]
    for repeat in range(1,4):
        for index,layout in enumerate(LAYOUTS):
            order=['rtc20','rtc10'] if (repeat-1+index)%2==0 else ['rtc10','rtc20']
            for profile in order:
                cases.append(dict(index=len(cases),profile=profile,layout=layout,repeat=repeat,
                                  reset_seed=layouts[layout]['reset_seed']))
    return cases


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-target',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18031)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',args.ssh_target)
    out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs') and 1024<args.port<65536
    profile_check()
    dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'
    mapping=dataset/'local/pi0_banana_validation/source_episodes.json'
    allowed={r['source_episode'] for r in json.loads(mapping.read_text())['episodes'] if r['split']=='validation'}
    assert set(LAYOUTS)<=allowed
    raw=Path(json.loads((dataset/'dataset_manifest.json').read_text())['source_path'])
    layouts={k:json.loads((raw/k/'layout.json').read_text()) for k in LAYOUTS}
    cases=make_cases(layouts)
    plan={'schema':'v6-rtc-same-session-v1','profiles':PROFILES,'layouts':LAYOUTS,'repeats':[1,2,3],
          'cases':cases,'attempts':36,'repeat_warmup_counts':[2,3,4],'noise_paired':False,
          'objective_priority':['motion jumps and low-motion dwell','phase/motion coverage','task success secondary'],
          'source_sha256':source_hashes(),'metric_sha256':sha(ROOT/'src/pi0_repro/motion_quality.py'),
          'profile_sha256':{k:sha(ROOT/'configs'/v) for k,v in PROFILES.items()},
          'script_sha256':sha(Path(__file__)),'helper_sha256':sha(ROOT/'scripts/screen_v6_motion.py'),
          'validation_mapping_sha256':sha(mapping),
          'selection_gate':{'all_and_covered_median_p99_ratio_max':.90,'worst_max_jerk_ratio_max':1.10,
              'worst_physical_increment_ratio_max':1.10,'worst_pause_ge_0_5s_add_seconds_max':.10,
              'median_pause_ge_0_16s_total_add_seconds_max':.10,'covered_count_min':'baseline',
              'covered_median_simulated_duration_ratio_max':1.10,'humanvideo_review_required':True},
          'selection':'All numerical gates plus video review; no automatic configuration promotion. Missing coverage is insufficient, never perfect smoothness. Sampling remains unpaired.',
          'preflight':'Six no-step requests; ignore initial two graph warmups, require last4 RTT max<1200ms and median<800ms.',
          'runtime_error_rule':'Preserve failed/infrastructure attempts; abort on inference underrun or broken source/timestamp/prefix integrity, never silently rerun.',
          'video':'50fps simulation time without retiming; wall-clock waits reported separately.',
          'service':'Use existing v6 process without restart, verify identical process identity at end. No training, plant, checkpoint or norm change.'}
    if args.check_only:
        print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (out/'screen_helper_source.py').write_bytes((ROOT/'scripts/screen_v6_motion.py').read_bytes())
    (out/'service_check_source.py').write_text(SERVICE_CODE)
    ledger={'plan':plan,'complete':False,'results':[],'endpoint_before':None,'endpoint_after':None}
    write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')])
    os.environ.setdefault('MUJOCO_GL','glfw')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import numpy as np
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    from pi0_repro.motion_quality import motion_quality
    spec=importlib.util.spec_from_file_location('rtc_summary',ROOT/'scripts/summarize_banana_execution.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    scenes={}
    for layout,meta in layouts.items():
        path=out/(layout+'.toml')
        path.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in meta['scene'].items() if v is not None)+'\n')
        scenes[layout]=path
    def configured(profile,layout=LAYOUTS[0]):
        cfg=load_config(ROOT/'configs'/PROFILES[profile])
        cfg.connection.port=args.port;cfg.simulation.display=False
        cfg.simulation.scene_config=str(scenes[layout]);cfg.simulation.seed=layouts[layout]['reset_seed']
        return cfg
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
         '-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def service_identity():
        command=shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',SERVICE_CODE])
        result=subprocess.run(ssh+[args.ssh_target,command],capture_output=True,text=True,timeout=20)
        if result.returncode:raise RuntimeError('Service verification failed: '+result.stderr[-2000:])
        return json.loads(result.stdout)
    with socket.socket() as probe:probe.bind(('127.0.0.1',args.port))
    tunnel_log=(out/'tunnel.log').open('xb')
    tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{args.port}:127.0.0.1:8001',args.ssh_target],
                            stdout=tunnel_log,stderr=subprocess.STDOUT,start_new_session=True)
    inhibitor=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    def endpoint():
        deadline=time.monotonic()+30
        while True:
            if tunnel.poll() is not None:raise RuntimeError('Own evaluation SSH tunnel exited')
            try:
                with RemotePi0(configured('rtc20')) as client:
                    metadata=client.metadata
                    assert metadata['transport_proxy']['kind']=='lossless-png'
                    return dict(metadata=metadata,metadata_verified=True,service_identity=service_identity())
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise
                time.sleep(.5)
    def warm(profile,count,directory):
        directory.mkdir()
        cfg=configured(profile);env=_make_env(cfg,directory)
        rows=[]
        try:
            obs,_=env.reset(seed=cfg.simulation.seed)
            observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(np.asarray(obs['agent_pos'],np.float32)[None],100,axis=0)
            prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            with RemotePi0(cfg) as client:
                for i in range(count):
                    actions,ms,response=client.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,
                        delay_steps=0 if i==0 else 10,blend_steps=0 if i==0 else cfg.simulation.rtc_blend_steps,guidance_max=.5)})
                    assert actions.shape==(100,14) and np.isfinite(actions).all()
                    rows.append(dict(round_trip_ms=ms,rtc=response.get('rtc'),server_timing=response.get('server_timing'),policy_timing=response.get('policy_timing')))
            write(directory/'requests.json',{'scope':'No environment control steps; excluded from outcomes.','profile':profile,'requests':rows})
            return rows
        finally:env.close()
    try:
        ledger['endpoint_before']=endpoint();write(out/'comparison.json',ledger)
        preflight=warm('rtc20',6,out/'preflight')
        measured=[r['round_trip_ms'] for r in preflight[2:]]
        if max(measured)>=1200 or np.median(measured)>=800:raise RuntimeError('Network preflight failed; no task attempts started')
        for repeat in range(1,4):
            for profile in (['rtc20','rtc10'] if repeat%2 else ['rtc10','rtc20']):
                warm(profile,repeat+1,out/f'warmup_r{repeat}_{profile}')
            for case in [c for c in cases if c['repeat']==repeat]:
                assert source_hashes()==plan['source_sha256'],'Source changed during comparison'
                assert sha(ROOT/'scripts/screen_v6_motion.py')==plan['helper_sha256'],'Helper changed'
                row=dict(case);row['scene_sha256']=sha(scenes[case['layout']])
                print(f"START {case['index']+1}/36 {case['profile']} {case['layout']} r{repeat}",flush=True)
                try:
                    path=run(configured(case['profile'],case['layout']));row['run_directory']=str(path)
                    details=summary.summarize(path)
                    tasks=[json.loads(line) for line in (path/'episode_000_task.jsonl').read_text().splitlines()]
                    with np.load(path/'episode_000.npz',allow_pickle=False) as data:
                        quality=motion_quality(data['state'],data['action'],tasks);extra=extra_diagnostics(data,tasks)
                    latency=[json.loads(line) for line in (path/'episode_000_latency.jsonl').read_text().splitlines()]
                    prefix_error=max((r.get('rtc_request') or {}).get('planned_vs_executed_prefix_max_error',0) for r in latency)
                    episode=details['episodes'][0]
                    row.update(diagnostics=details,motion_quality=quality,extra=extra,prefix_error=prefix_error,
                               video=str(path/'episode_000.mp4'),video_sha256=sha(path/'episode_000.mp4'))
                    assert source_hashes()==plan['source_sha256'],'Source changed during recording'
                    assert episode['raw_prediction_max_error']==0 and episode['max_action_timestamp_error_steps']==0 and prefix_error==0
                    assert episode['result']['execution']['inference_underruns']==0,'Inference underrun contaminates comparison; retained, aborting'
                except BaseException as exc:
                    row['infrastructure_error']=f'{type(exc).__name__}: {exc}'
                    ledger['results'].append(row);write(out/'comparison.json',ledger);raise
                ledger['results'].append(row);write(out/'comparison.json',ledger)
                print('END',case['index']+1,'success',episode['result']['success'],'p99',quality['active_frame_max_jerk_rad_s3']['p99'],
                      'dwell',quality['low_motion_pause_max_seconds'],flush=True)
        ledger['endpoint_after']=endpoint()
        assert ledger['endpoint_after']['service_identity']==ledger['endpoint_before']['service_identity'],'Service process changed'
        ledger['complete']=True
    except BaseException as exc:
        ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if ledger['endpoint_after'] is None:
            try:ledger['endpoint_after']=endpoint()
            except BaseException as exc:ledger['endpoint_after']={'metadata_verified':False,'error':f'{type(exc).__name__}: {exc}'}
        write(out/'comparison.json',ledger)
        tunnel.terminate()
        try:tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        tunnel_log.close();inhibitor.terminate()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
