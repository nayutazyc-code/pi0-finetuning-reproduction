"""Twelve predeclared random planar poses, original v6 RTC20, all outcomes retained."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from pi0_repro.random_pose import sample_banana_scene
from pi0_repro.scene import load_scene_config
from pi0_repro.remote import RemotePi0
from compare_v6_rtc import SERVICE_CODE

PROFILE=ROOT/'configs/banana_v6_random_multiview_pi0.toml'
SCENE=ROOT/'configs/banana_random_scene.toml'
SOURCE_NAMES=('config.py','execution.py','jerk_reference.py','simulation.py','viewer.py','remote.py',
              'banana_env.py','scene.py','diagnostics.py','motion_quality.py','recording.py','cli.py',
              '__main__.py','random_pose.py','multiview.py','async_multiview.py')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():return {name:sha(ROOT/'src/pi0_repro'/name) for name in SOURCE_NAMES}


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-target',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--port',type=int,default=18041);p.add_argument('--check-only',action='store_true')
    p.add_argument('--continue-from',type=Path)
    p.add_argument('--retain-transport-warnings',action='store_true')
    args=p.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs') and 1024<args.port<65536
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',args.ssh_target)
    import numpy as np
    scene=load_scene_config(SCENE);profile=load_config(PROFILE)
    assert scene.randomize_banana_pose and scene.banana_x_range==(.16,.20) and scene.banana_y_range==(.46,.50)
    assert scene.banana_yaw_range_degrees==(0.,20.) and scene.box_position==(.13,.72,0.)
    assert profile.simulation.rtc_blend_steps==20 and profile.simulation.chunk_blend_steps==20
    assert profile.simulation.joint_target_filter_hz==3 and profile.simulation.max_joint_acceleration==40
    cases=[]
    for index,seed in enumerate(range(2026102801,2026102813)):
        sampled=sample_banana_scene(scene,np.random.default_rng(seed))
        cases.append({'index':index,'seed':seed,'planned_scene':json.loads(json.dumps(asdict(sampled))),
                      'display':index==0})
    assert len({json.dumps(c['planned_scene'],sort_keys=True) for c in cases})==12
    training_path=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3/local/pi0_banana_train/source_episodes.json'
    training=json.loads(training_path.read_text())['episodes']
    existing={(tuple(r['scene']['banana_position']),r['scene']['banana_yaw_degrees']) for r in training}
    assert all((tuple(c['planned_scene']['banana_position']),c['planned_scene']['banana_yaw_degrees']) not in existing for c in cases)
    continuation=None
    if args.continue_from:
        previous=args.continue_from.resolve()
        assert previous.is_relative_to(ROOT/'outputs')
        prior=json.loads((previous/'comparison.json').read_text())
        assert not prior['complete'] and 0<len(prior['results'])<12
        assert prior['plan']['source_sha256']==sources() and prior['plan']['profile_sha256']==sha(PROFILE)
        assert prior['plan']['cases']==cases
        assert [r['index'] for r in prior['results']]==list(range(len(prior['results'])))
        assert prior.get('error')=='AssertionError: Inference underrun; retain and abort'
        assert all(not r.get('infrastructure_error') or r['infrastructure_error']=='AssertionError: Inference underrun; retain and abort' for r in prior['results'])
        continuation={'path':str(previous),'comparison_sha256':sha(previous/'comparison.json'),
                      'plan_sha256':sha(previous/'plan.json'),'preserved_attempts':len(prior['results']),
                      'scope':'No seed is repeated. Original transport-aborted ledger and its successful outcomes remain untouched.'}
        cases=cases[len(prior['results']):]
    plan={'schema':'v6-random-planar-multiview-v1','attempts':len(cases),'whole_suite_attempts':12,'cases':cases,'noise_paired':False,
          'sampling':'One uniformly sampled x/y/yaw for each predeclared independent seed; no outcome-based retries or resampling.',
          'scope':'Fresh combinations within v6 training-acquisition envelope, fixed box. No arbitrary 3D pose or whole-table generalization claim.',
          'training_mapping_sha256':sha(training_path),'profile_sha256':sha(PROFILE),'scene_config_sha256':sha(SCENE),
          'source_sha256':sources(),'script_sha256':sha(Path(__file__)),
          'policy_camera':'Original top RGB640x480 -> cam_high and unchanged cloud ROI; no display pixels sent to policy.',
          'display_camera':'Front master, side and real rigidly attached right-wrist camera; all from one read-only clone snapshot.',
          'display_pipeline':'Spawned renderer, bounded input and latest shared RGB buffer; can lag, per-video-frame source-step receipts retained.',
          'first_attempt_visible':True,'other_attempts_visible':False,'all_attempts_record_multiview':True,
          'video_fps':25,'control_hz_target':50,'preflight':'6 no-step requests, final4 max<1200ms median<800ms.',
          'continuation_of':continuation,'retain_transport_warnings':args.retain_transport_warnings,
          'service':'Existing v6 process must match before/after; no training, checkpoint, norm or physical-controller change.',
          'success_rule':'Unchanged full-task containment, release, box support and stable hold; every failure kept.'}
    if args.check_only:print(json.dumps(plan,indent=2));return
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan);(out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    ledger={'plan':plan,'complete':False,'results':[],'endpoint_before':None,'endpoint_after':None}
    write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')])
    os.environ.setdefault('MUJOCO_GL','glfw');os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    from pi0_repro.simulation import run,_make_env
    from pi0_repro.observations import aloha_observation
    from pi0_repro.motion_quality import motion_quality
    spec=importlib.util.spec_from_file_location('random_summary',ROOT/'scripts/summarize_banana_execution.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    def configured(seed,display=False):
        cfg=load_config(PROFILE);cfg.connection.port=args.port
        cfg.simulation.episodes=1;cfg.simulation.seed=seed;cfg.simulation.display=display
        return cfg
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
         '-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def identity():
        command=shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',SERVICE_CODE])
        r=subprocess.run(ssh+[args.ssh_target,command],capture_output=True,text=True,timeout=20)
        if r.returncode:raise RuntimeError('Existing v6 service check failed: '+r.stderr[-1500:])
        return json.loads(r.stdout)
    with socket.socket() as probe:probe.bind(('127.0.0.1',args.port))
    log=(out/'tunnel.log').open('xb')
    tunnel=subprocess.Popen(ssh+['-o','ExitOnForwardFailure=yes','-N','-L',f'127.0.0.1:{args.port}:127.0.0.1:8001',args.ssh_target],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    inhibitor=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    def endpoint():
        deadline=time.monotonic()+30
        while True:
            if tunnel.poll() is not None:raise RuntimeError('Own SSH tunnel exited')
            try:
                with RemotePi0(configured(2026102800)) as client:
                    assert client.metadata['transport_proxy']['kind']=='lossless-png'
                    return {'metadata':client.metadata,'metadata_verified':True,'service_identity':identity()}
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise
                time.sleep(.5)
    try:
        ledger['endpoint_before']=endpoint();write(out/'comparison.json',ledger)
        warm=out/'preflight';warm.mkdir();cfg=configured(2026102800)
        env=_make_env(cfg,warm)
        try:
            obs,_=env.reset(seed=cfg.simulation.seed)
            observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},cfg.simulation.prompt,resize=False)
            prefix=np.repeat(np.asarray(obs['agent_pos'],np.float32)[None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
            rows=[]
            with RemotePi0(cfg) as client:
                for i in range(6):
                    actions,ms,response=client.infer({**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=0 if i==0 else 10,blend_steps=0 if i==0 else 20,guidance_max=.5)})
                    assert actions.shape==(100,14) and np.isfinite(actions).all()
                    rows.append({'round_trip_ms':ms,'rtc':response.get('rtc'),'server_timing':response.get('server_timing')})
            write(warm/'requests.json',{'scope':'No task/control steps, excluded from attempt counts.','requests':rows})
            measured=[r['round_trip_ms'] for r in rows[2:]]
            if max(measured)>=1200 or np.median(measured)>=800:raise RuntimeError('Transport preflight failed')
        finally:env.close()
        for case in cases:
            assert sources()==plan['source_sha256'],'Source changed during evaluation'
            row=dict(case)
            print('START',case['index']+1,'/12 seed',case['seed'],case['planned_scene']['banana_position'],case['planned_scene']['banana_yaw_degrees'],flush=True)
            try:
                directory=run(configured(case['seed'],case['display']));row['run_directory']=str(directory)
                details=summary.summarize(directory);ep=details['episodes'][0]
                actual=json.loads((directory/'episode_000_scene.json').read_text())
                assert actual['scene']==case['planned_scene'],'Realized pose does not match predeclared seed'
                assert actual['reset']['seed']==case['seed'] and actual['reset']['randomized']
                tasks=[json.loads(s) for s in (directory/'episode_000_task.jsonl').read_text().splitlines()]
                with np.load(directory/'episode_000.npz',allow_pickle=False) as data:quality=motion_quality(data['state'],data['action'],tasks)
                latency=[json.loads(s) for s in (directory/'episode_000_latency.jsonl').read_text().splitlines()]
                prefix_error=max((r.get('rtc_request') or {}).get('planned_vs_executed_prefix_max_error',0) for r in latency)
                receipts=[json.loads(s) for s in (directory/'episode_000_multiview.jsonl').read_text().splitlines()]
                assert receipts and all(0<=r['source_control_step']<=r['target_control_step'] for r in receipts)
                assert all(r['frame_age_control_steps']==r['target_control_step']-r['source_control_step'] for r in receipts)
                if ep['result']['success']:
                    assert all(tasks[-1][k] for k in ['inside_box','released','supported_by_box','stable','is_success'])
                    assert tasks[-1]['success_hold_steps']>=tasks[-1]['required_hold_steps']
                row.update(diagnostics=details,motion_quality=quality,reset=actual['reset'],prefix_error=prefix_error,
                           success=ep['result']['success'],video=str(directory/'episode_000_multiview.mp4'),
                           video_sha256=sha(directory/'episode_000_multiview.mp4'),policy_video_sha256=sha(directory/'episode_000.mp4'),
                           display_frame_age_steps={'median':float(np.median([r['frame_age_control_steps'] for r in receipts])),
                                                    'max':max(r['frame_age_control_steps'] for r in receipts)},
                           display_unique_frames=len({r['worker_frame_id'] for r in receipts}))
                assert ep['raw_prediction_max_error']==0 and ep['max_action_timestamp_error_steps']==0 and prefix_error==0
                if ep['result']['execution']['inference_underruns']:
                    row['transport_warning']={'underruns':ep['result']['execution']['inference_underruns'],
                        'wait_seconds':ep['result']['execution']['underrun_wait_seconds'],
                        'scope':'Task outcome retained; timing includes network/inference wait, not renderer-only overhead.'}
                    assert args.retain_transport_warnings,'Inference underrun; retain and abort'
                assert not ep['result'].get('aborted_by_window_close'),'Visible attempt was aborted'
                assert sources()==plan['source_sha256'],'Source changed during episode'
            except BaseException as exc:
                row['infrastructure_error']=f'{type(exc).__name__}: {exc}';ledger['results'].append(row);write(out/'comparison.json',ledger);raise
            ledger['results'].append(row);write(out/'comparison.json',ledger)
            print('END',case['index']+1,'success',row['success'],'age',row['display_frame_age_steps'],flush=True)
        ledger['endpoint_after']=endpoint()
        assert ledger['endpoint_before']['service_identity']==ledger['endpoint_after']['service_identity']
        ledger['complete']=True
    except BaseException as exc:ledger['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if ledger['endpoint_after'] is None:
            try:ledger['endpoint_after']=endpoint()
            except BaseException as exc:ledger['endpoint_after']={'metadata_verified':False,'error':f'{type(exc).__name__}: {exc}'}
        write(out/'comparison.json',ledger)
        tunnel.terminate()
        try:tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        log.close();inhibitor.terminate()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
