"""Fresh cloud-pi0 successful examples on one fixed scene; retain every attempt."""
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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'repos/openpi/packages/openpi-client/src')]
from pi0_repro.config import load_config
from pi0_repro.remote import RemotePi0

REMOTE_ROOT = '/root/pi0-repro'
PROFILES = {'v6': 'banana_v6_bridge20_pi0.toml', 'v8_control': 'banana_v8_control32_pi0.toml',
            'v9': 'banana_v9_motion_pi0.toml'}

SELECT_CODE = r'''
from pathlib import Path
import json,sys,hashlib,fcntl,subprocess,os,signal,time
r=Path('/root/pi0-repro'); model=sys.argv[1]; out=r/sys.argv[2]
out.mkdir(parents=True,exist_ok=True)
models={
 'v6':(r/'outputs/long_chunk_20261025/selected_training.toml',r/'checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999'),
 'v8_control':(r/'configs/banana_finetune_v8_control.toml',r/'checkpoints/pi0_banana_lora/banana_v8_control32_3000/2999'),
 'v9':(r/'configs/banana_finetune_v9_motion.toml',Path('/root/pi0-motion-checkpoints/banana_v9_motion1500/1499'))}
config,checkpoint=models[model]
assert config.is_file() and (checkpoint/'_CHECKPOINT_METADATA').is_file() and (checkpoint/'params/_METADATA').is_file()
digest=hashlib.sha256((checkpoint/'assets/local/pi0_banana_train/norm_stats.json').read_bytes()).hexdigest()
assert digest=='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
with (out/'selection.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 policies=[]
 for f in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   a=f.read_bytes().split(b'\0')
   assert str(r/'server/banana_finetune.py').encode() not in a,'Training active; do not switch'
   assert str(r/'server/motion_train_job.py').encode() not in a,'Supervisor active; do not switch'
   if str(r/'server/serve_pi0.py').encode() in a:policies.append(int(f.parent.name))
  except OSError:pass
 assert len(policies)<=1,'Ambiguous policy processes'
 for pid in policies:os.kill(pid,signal.SIGTERM)
 for _ in range(40):
  used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if used<2048:break
  time.sleep(1)
 assert used<2048,'GPU not released'
 env=dict(os.environ,PI0_RTC='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 with (out/(model+'_serve.log')).open('ab') as log:
  p=subprocess.Popen(['bash',str(r/'server/run_banana_serve.sh'),str(config),str(checkpoint),'10'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record={'model':model,'pid':p.pid,'checkpoint':str(checkpoint),'norm_stats_sha256':digest,'dispatch_only':True}
 (out/(model+'_dispatch.json')).write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record))
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-target', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18001)
    parser.add_argument('--max-attempts', type=int, default=6)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.ssh_target):
        raise ValueError('Invalid SSH alias')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'outputs') or not 1 <= args.max_attempts <= 12:
        raise ValueError('Invalid output or attempt limit')
    configs = {model: load_config(ROOT / 'configs' / filename) for model, filename in PROFILES.items()}
    comparable = []
    for config in configs.values():
        item = asdict(config)
        item['model'].pop('expected_checkpoint_suffix')
        comparable.append(item)
    assert all(item == comparable[0] for item in comparable), 'Execution settings differ'
    dataset = ROOT / 'data/lerobot/20261024-185715-merged-ebdbe3'
    mapping = json.loads((dataset / 'local/pi0_banana_validation/source_episodes.json').read_text())
    assert any(row['source_episode'] == 'episode_019' and row['split'] == 'validation' for row in mapping['episodes'])
    raw = Path(json.loads((dataset / 'dataset_manifest.json').read_text())['source_path'])
    layout = json.loads((raw / 'episode_019/layout.json').read_text())
    plan = {'order': list(PROFILES), 'layout': 'episode_019', 'reset_seed': layout['reset_seed'],
            'max_attempts_per_model': args.max_attempts, 'selection': 'First successful full episode, all failures retained.',
            'scope': 'Selected successful examples only, not a success-rate experiment or aggregate ranking.',
            'noise_paired': False, 'warmup_requests_per_model': 2, 'video_fps': 50,
            'playback': 'Simulation time, no retiming. Wall-clock inference waits are reported separately.',
            'config_sha256': {model: sha(ROOT / 'configs' / name) for model, name in PROFILES.items()},
            'script_sha256': sha(Path(__file__)), 'selector_sha256': hashlib.sha256(SELECT_CODE.encode()).hexdigest()}
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    scene = out / 'scene.toml'
    scene.write_text('[scene]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k, v in layout['scene'].items() if v is not None) + '\n')
    plan['scene_sha256'] = sha(scene)
    write(out / 'plan.json', plan)
    ledger = {'plan': plan, 'complete': False, 'attempts': [], 'selected': {}, 'selections': [], 'restoration': None}
    write(out / 'comparison.json', ledger)
    (out / 'selector_source.py').write_text(SELECT_CODE)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / 'experiment.log')])
    key = str(Path.home() / '.ssh/id_ed25519')
    ssh = ['ssh', '-i', key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
           '-o', 'ControlPath=none', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import numpy as np
    from pi0_repro.simulation import run, _make_env
    from pi0_repro.observations import aloha_observation
    from pi0_repro.motion_quality import motion_quality
    spec = importlib.util.spec_from_file_location('demo_summary', ROOT / 'scripts/summarize_banana_execution.py')
    summarizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarizer)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    tunnel_log = (out / 'tunnel.log').open('xb')
    tunnel = subprocess.Popen(ssh + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                             f'127.0.0.1:{args.port}:127.0.0.1:8001', args.ssh_target],
                             stdout=tunnel_log, stderr=subprocess.STDOUT, start_new_session=True)
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])

    def configured(model):
        config = load_config(ROOT / 'configs' / PROFILES[model])
        config.connection.port = args.port
        config.simulation.display = False
        config.simulation.scene_config = str(scene)
        config.simulation.seed = layout['reset_seed']
        return config

    def ready(model):
        config = configured(model)
        config.connection.connect_timeout_seconds = 1
        deadline = time.monotonic() + 120
        while True:
            if tunnel.poll() is not None:
                raise RuntimeError('SSH video tunnel exited; inspect tunnel.log')
            try:
                with RemotePi0(config) as client:
                    assert client.metadata['transport_proxy']['kind'] == 'lossless-png'
                    return client.metadata
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
            except RuntimeError as exc:
                if 'Image transport failed: ConnectionRefusedError' not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def select(model):
        command = shlex.join([REMOTE_ROOT + '/openpi/.venv/bin/python', '-B', '-c', SELECT_CODE,
                              model, str(out.relative_to(ROOT))])
        result = subprocess.run(ssh + [args.ssh_target, command], capture_output=True, text=True, timeout=90)
        result.check_returncode()
        record = json.loads(result.stdout)
        metadata = ready(model)
        record.update(metadata=metadata, metadata_verified=True)
        ledger['selections'].append(record)
        write(out / 'comparison.json', ledger)
        return metadata

    try:
        for model in PROFILES:
            print('SELECT', model, flush=True)
            select(model)
            warm = out / (model + '_warmup')
            warm.mkdir()
            config = configured(model)
            env = _make_env(config, warm)
            try:
                obs, _ = env.reset(seed=layout['reset_seed'])
                observation = aloha_observation({'pixels': {'top': env.render()}, 'agent_pos': obs['agent_pos']},
                                                config.simulation.prompt, resize=False)
                prefix = np.repeat(np.asarray(obs['agent_pos'], np.float32)[None], 100, axis=0)
                prefix[:, [6, 13]] = np.clip(prefix[:, [6, 13]], 0, 1)
                rows = []
                with RemotePi0(config) as client:
                    for index in range(2):
                        actions, ms, response = client.infer({**observation, 'rtc': dict(schema='pi0-rtc-v1',
                            prefix=prefix, valid_steps=100, delay_steps=0 if index == 0 else 10,
                            blend_steps=0 if index == 0 else 20, guidance_max=.5)})
                        assert actions.shape == (100, 14) and np.isfinite(actions).all()
                        rows.append({'round_trip_ms': ms, 'rtc': response.get('rtc')})
                write(warm / 'requests.json', {'scope': 'No environment steps, excluded from attempt counts.', 'requests': rows})
            finally:
                env.close()
            for attempt in range(1, args.max_attempts + 1):
                row = {'model': model, 'attempt': attempt}
                print('START', model, attempt, flush=True)
                try:
                    directory = run(configured(model))
                    details = summarizer.summarize(directory)
                    result = details['episodes'][0]['result']
                    tasks = [json.loads(line) for line in (directory / 'episode_000_task.jsonl').read_text().splitlines()]
                    with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
                        quality = motion_quality(data['state'], data['action'], tasks)
                    row.update(run_directory=str(directory), success=bool(result['success']),
                               diagnostics=details, motion_quality=quality, video=str(directory / 'episode_000.mp4'))
                    if result['success']:
                        last = tasks[-1]
                        assert all(last[k] for k in ('inside_box', 'released', 'supported_by_box', 'stable', 'is_success'))
                        assert last['success_hold_steps'] >= last['required_hold_steps']
                        row.update(video_sha256=sha(directory / 'episode_000.mp4'), success_seconds=last['simulation_time_seconds'])
                        ledger['selected'][model] = row
                except BaseException as exc:
                    row['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
                    ledger['attempts'].append(row)
                    write(out / 'comparison.json', ledger)
                    raise
                ledger['attempts'].append(row)
                write(out / 'comparison.json', ledger)
                print('END', model, attempt, 'success', row['success'], flush=True)
                if row['success']:
                    break
            if model not in ledger['selected']:
                raise RuntimeError('No success within predeclared attempt limit: ' + model)
        ledger['complete'] = True
    except BaseException as exc:
        ledger['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            select('v8_control')
            ledger['restoration'] = {'model': 'v8_control', 'metadata_verified': True}
        except BaseException as exc:
            ledger['restoration'] = {'error': f'{type(exc).__name__}: {exc}', 'metadata_verified': False}
        write(out / 'comparison.json', ledger)
        tunnel.terminate()
        try:
            tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()
        tunnel_log.close()
        inhibitor.terminate()
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
