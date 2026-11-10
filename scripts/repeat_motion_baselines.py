"""Frozen 36-attempt motion-first reference; headless, no test-layout tuning."""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from pi0_repro.config import load_config
from pi0_repro.motion_quality import motion_quality
from pi0_repro.observations import aloha_observation
from pi0_repro.remote import RemotePi0
from pi0_repro.simulation import _make_env,run


def write(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ssh-target',required=True);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+',args.ssh_target):raise ValueError('Invalid SSH target')
    out=ROOT/'outputs/motion_first_20261026';out.mkdir(exist_ok=True)
    plan_path=out/'baseline_plan.json'
    if plan_path.exists():raise FileExistsError('Keep every attempt; do not overwrite plan')
    blocks=[('v6',0),('control',0),('control',1),('v6',1),('v6',2),('control',2)]
    layouts=['episode_004','episode_012','episode_019','episode_020','episode_026','episode_029']
    configs={'v6':'banana_v6_bridge20_pi0.toml','control':'banana_v8_control32_pi0.toml'}
    plan={'objective_priority':['active-motion jumps and low-motion dwell','phase/motion coverage so stillness is not rewarded','task success secondary'],
          'blocks':blocks,'layouts':layouts,'attempts':36,'configurations':configs,
          'config_sha256':{k:hashlib.sha256((ROOT/'configs'/v).read_bytes()).hexdigest() for k,v in configs.items()},
          'metric_sha256':hashlib.sha256((ROOT/'src/pi0_repro/motion_quality.py').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'noise_paired':False,'repeat_warmup_counts':[2,3,4],
          'selection':'Compare per-stage motion coverage, active frame-max jerk p99/peak physical increment and dwell durations first; success is a secondary outcome. No single scalar or victory from stationary failures. Preserve all repeats, use no test layouts or validation states for training.'}
    write(plan_path,plan)
    ledger={'plan':plan,'complete':False,'results':[]};write(out/'comparison.json',ledger)
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(out/'experiment.log')])
    spec=importlib.util.spec_from_file_location('motion_summary',ROOT/'scripts/summarize_banana_execution.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    dataset=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'
    manifest=json.loads((dataset/'dataset_manifest.json').read_text());raw=Path(manifest['source_path'])
    allowed={r['source_episode'] for r in json.loads((dataset/'local/pi0_banana_validation/source_episodes.json').read_text())['episodes']}
    socket=(ROOT/'outputs/v6_recovery_20261026/ssh_control_socket.txt').read_text().strip()
    remote_root='/root/pi0-repro'
    def select(model):
        command=[remote_root+'/openpi/.venv/bin/python',remote_root+'/server/select_v8_policy.py','--root',remote_root,'--model',model]
        subprocess.run(['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=8',args.ssh_target,shlex.join(command)],check=True,capture_output=True,text=True,timeout=60)
        config=load_config(ROOT/'configs'/configs[model]);config.connection.connect_timeout_seconds=1
        deadline=time.monotonic()+90
        while True:
            try:
                with RemotePi0(config) as client:return client.metadata
            except (OSError,TimeoutError):
                if time.monotonic()>=deadline:raise
                time.sleep(2)
            except RuntimeError as exc:
                # The PNG sidecar can be ready before its policy upstream. This
                # is a bounded startup condition, never a swallowed trial error.
                if 'Image transport failed' not in str(exc) or 'ConnectionRefusedError' not in str(exc) or time.monotonic()>=deadline:
                    raise
                time.sleep(2)
    try:
        for block,(model,repeat) in enumerate(blocks):
            print(f'BLOCK {block+1}/6 model={model} repeat={repeat+1}',flush=True)
            metadata=select(model);warm=out/f'warmup_{block:02d}';warm.mkdir()
            config=load_config(ROOT/'configs'/configs[model]);env=_make_env(config,warm)
            try:
                obs,_=env.reset(seed=config.simulation.seed)
                observation=aloha_observation({'pixels':{'top':env.render()},'agent_pos':obs['agent_pos']},config.simulation.prompt,resize=False)
                prefix=np.repeat(np.asarray(obs['agent_pos'],np.float32)[None],100,axis=0);prefix[:,[6,13]]=np.clip(prefix[:,[6,13]],0,1)
                warm_rows=[]
                with RemotePi0(config) as client:
                    for i in range(2+repeat):
                        request={**observation,'rtc':dict(schema='pi0-rtc-v1',prefix=prefix,valid_steps=100,delay_steps=0 if i==0 else 10,blend_steps=0 if i==0 else 20,guidance_max=.5)}
                        actions,ms,response=client.infer(request);assert actions.shape==(100,14) and np.isfinite(actions).all()
                        warm_rows.append({'round_trip_ms':ms,'rtc':response['rtc']})
                write(warm/'requests.json',{'scope':'No environment steps; extra predeclared calls advance policy noise keys between repeats. Excluded from task outcomes.','requests':warm_rows,'metadata':metadata})
            finally:env.close()
            for layout in layouts:
                index=len(ledger['results']);assert layout in allowed
                meta=json.loads((raw/layout/'layout.json').read_text());scene=out/f'scene_{index:02d}.toml'
                scene.write_text('[scene]\n'+'\n'.join(f'{k} = {json.dumps(v)}' for k,v in meta['scene'].items() if v is not None)+'\n')
                config=load_config(ROOT/'configs'/configs[model]);config.simulation.display=False
                config.simulation.scene_config=str(scene);config.simulation.seed=meta['reset_seed']
                row={'index':index,'phase':'validation','profile':model,'repeat':repeat+1,'block':block,'layout':layout}
                print(f'START {index+1}/36 {model} repeat={repeat+1} {layout}',flush=True)
                try:
                    p=run(config);details=summary.summarize(p)
                    with np.load(p/'episode_000.npz',allow_pickle=False) as a:states,actions=a['state'],a['action']
                    tasks=[json.loads(s) for s in (p/'episode_000_task.jsonl').read_text().splitlines()]
                    row.update(run_directory=str(p),diagnostics=details,motion_quality=motion_quality(states,actions,tasks))
                    e=details['episodes'][0];m=row['motion_quality'];j=m['active_frame_max_jerk_rad_s3']
                    print(f"END {index+1} success={e['result']['success']} active_jerk_p99={j['p99'] if j else None} dwell_max={m['low_motion_pause_max_seconds']}",flush=True)
                except BaseException as exc:
                    row['infrastructure_error']=f'{type(exc).__name__}: {exc}';ledger['results'].append(row);write(out/'comparison.json',ledger);raise
                ledger['results'].append(row);write(out/'comparison.json',ledger)
        ledger['complete']=True;write(out/'comparison.json',ledger);print('COMPLETE 36/36',flush=True)
    finally:
        try:
            metadata=select('v6');write(out/'restored_v6_metadata.json',metadata)
        except BaseException as exc:
            write(out/'restore_failure.json',{'error':f'{type(exc).__name__}: {exc}','scope':'Restore attempt is not proof of availability.'})


if __name__=='__main__':main()
