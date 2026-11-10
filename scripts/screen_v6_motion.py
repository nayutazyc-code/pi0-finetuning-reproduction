"""Predeclared, single-factor original-pi0 v6 runtime screen; retain all attempts."""
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
from record_three_success_videos import SELECT_CODE, REMOTE_ROOT

PROFILES = {'baseline': 'banana_v6_bridge20_pi0.toml', 'acc20': 'banana_v6_acc20_pi0.toml',
            'rtc30': 'banana_v6_rtc30_pi0.toml', 'wrist125': 'banana_v6_wrist125_pi0.toml'}
BLOCKS = [('episode_019', ['baseline', 'acc20', 'rtc30', 'wrist125']),
          ('episode_020', ['acc20', 'wrist125', 'baseline', 'rtc30']),
          ('episode_026', ['rtc30', 'baseline', 'wrist125', 'acc20'])]
SOURCES = ('config.py', 'execution.py', 'jerk_reference.py', 'simulation.py', 'viewer.py',
           'remote.py', 'banana_env.py', 'scene.py', 'diagnostics.py', 'motion_quality.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def source_hashes():
    return {name: sha(ROOT / 'src/pi0_repro' / name) for name in SOURCES}


def check_profiles():
    configs = {name: load_config(ROOT / 'configs' / path) for name, path in PROFILES.items()}
    base = asdict(configs['baseline'])
    for name, config in configs.items():
        candidate = asdict(config)
        if name == 'acc20':
            assert candidate['simulation']['max_joint_acceleration'] == 20.
            candidate['simulation']['max_joint_acceleration'] = 40.
        elif name == 'rtc30':
            assert candidate['simulation']['rtc_blend_steps'] == 30
            candidate['simulation']['rtc_blend_steps'] = 20
        elif name == 'wrist125':
            assert Path(candidate['simulation']['scene_config']).name == 'banana_demo_scene_wrist125.toml'
            candidate['simulation']['scene_config'] = base['simulation']['scene_config']
        assert candidate == base, f'Unexpected additional config difference: {name}'
    from pi0_repro.scene import load_scene_config
    normal = asdict(load_scene_config(ROOT / 'configs/banana_demo_scene.toml'))
    wrist = asdict(load_scene_config(ROOT / 'configs/banana_demo_scene_wrist125.toml'))
    assert wrist['wrist_damping_scale'] == 1.25
    wrist['wrist_damping_scale'] = 1.
    assert wrist == normal


def extra_diagnostics(data, tasks):
    """Prespecified supplementary measures; frozen motion-first-v1 stays unchanged."""
    import numpy as np
    from pi0_repro.motion_quality import motion_quality
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    state, action = data['state'].astype(float), data['action'].astype(float)
    def dist(x):
        x = np.asarray(x)
        return {'p95': float(np.quantile(x, .95)), 'p99': float(np.quantile(x, .99)), 'max': float(np.max(x))}
    short = motion_quality(state, action, tasks, pause_min_seconds=.16)
    return {'scope': 'Supplementary; original >=0.5s dwell and motion-first-v1 retained separately. State is pre-step; task rows are post-step.',
            'short_low_motion_min_seconds': .16,
            'short_low_motion_total_seconds': short['low_motion_pause_total_seconds'],
            'short_low_motion_max_seconds': short['low_motion_pause_max_seconds'],
            'short_low_motion_spans': short['low_motion_pause_spans'],
            'submitted_frame_max_jerk_rad_s3': dist(np.max(abs(np.diff(action[:, joints], n=3, axis=0)), axis=1)/.02**3),
            'post_step_command_tracking_residual_rad': dist(abs(state[1:, joints] - action[:-1, joints]).ravel()),
            'right_wrist_axis11_increment_rad': dist(abs(np.diff(state[:, 11]))),
            'right_wrist_axis11_jerk_rad_s3': dist(abs(np.diff(state[:, 11], n=3))/.02**3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-target', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18021)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', args.ssh_target)
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / 'outputs') and 1024 < args.port < 65536
    check_profiles()
    dataset = ROOT / 'data/lerobot/20261024-185715-merged-ebdbe3'
    mapping_path = dataset / 'local/pi0_banana_validation/source_episodes.json'
    allowed = {r['source_episode'] for r in json.loads(mapping_path.read_text())['episodes'] if r['split'] == 'validation'}
    raw = Path(json.loads((dataset / 'dataset_manifest.json').read_text())['source_path'])
    layouts = {name: json.loads((raw / name / 'layout.json').read_text()) for name, _ in BLOCKS}
    assert set(layouts) <= allowed
    plan = {'schema': 'v6-runtime-motion-screen-v1', 'blocks': BLOCKS, 'attempts': 12,
            'selection': 'All attempts; no retry, no first-success selection. Small unpaired screen, not general superiority.',
            'objective': ['motion jumps and low-motion dwell', 'motion coverage and phase duration', 'success secondary'],
            'comparison': 'One factor per candidate. Wrist damping changes scene dynamics; other scene fields identical.',
            'promotion_gate': 'Candidate is only a provisional recommendation if median active jerk p99 improves >=10%, worst physical increment <=110% control, median short dwell increases <=0.2s, transport-covered count >=control, covered median duration <=110% control. Inspect all videos; otherwise retain baseline.',
            'source_sha256': source_hashes(), 'script_sha256': sha(Path(__file__)),
            'selector_sha256': hashlib.sha256(SELECT_CODE.encode()).hexdigest(),
            'profile_sha256': {k: sha(ROOT / 'configs' / v) for k,v in PROFILES.items()},
            'validation_mapping_sha256': sha(mapping_path),
            'scene_seeds': {k: v['reset_seed'] for k,v in layouts.items()},
            'noise_paired': False, 'display': False, 'video_fps': 50,
            'warmup_scope': 'Six requests, no environment steps, excluded from all attempts. First two warm graphs; final four require RTT max<1200ms and median<800ms before any episode.',
            'playback': 'Simulation time; no speed changes; wall-clock waits separately recorded.',
            'end_service': 'Verified v6, including on failure; no training or checkpoint changes.'}
    if args.check_only:
        print(json.dumps(plan, indent=2)); return
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', plan)
    (out / 'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'selector_source.py').write_text(SELECT_CODE)
    ledger = {'plan': plan, 'complete': False, 'attempts': [], 'selection': None, 'final_endpoint': None}
    write(out / 'comparison.json', ledger)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(out / 'experiment.log')])
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import numpy as np
    from pi0_repro.simulation import run, _make_env
    from pi0_repro.observations import aloha_observation
    from pi0_repro.motion_quality import motion_quality
    spec = importlib.util.spec_from_file_location('motion_summary', ROOT / 'scripts/summarize_banana_execution.py')
    summary = importlib.util.module_from_spec(spec); spec.loader.exec_module(summary)
    scenes = {}
    for name, layout in layouts.items():
        for variant, scale in [('normal', 1.), ('wrist125', 1.25)]:
            scene = dict(layout['scene'], wrist_damping_scale=scale)
            path = out / f'{name}_{variant}.toml'
            path.write_text('[scene]\n' + '\n'.join(f'{k} = {json.dumps(v)}' for k,v in scene.items() if v is not None) + '\n')
            scenes[name, variant] = path
    def configured(profile, layout_name='episode_019'):
        config = load_config(ROOT / 'configs' / PROFILES[profile])
        config.connection.port = args.port
        config.simulation.display = False
        config.simulation.scene_config = str(scenes[layout_name, 'wrist125' if profile == 'wrist125' else 'normal'])
        config.simulation.seed = layouts[layout_name]['reset_seed']
        return config
    ssh = ['ssh', '-i', str(Path.home() / '.ssh/id_ed25519'), '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=10',
           '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', '-o', 'IPQoS=none']
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    tunnel_log = (out / 'tunnel.log').open('xb')
    tunnel = subprocess.Popen(ssh + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
                              f'127.0.0.1:{args.port}:127.0.0.1:8001', args.ssh_target],
                              stdout=tunnel_log, stderr=subprocess.STDOUT, start_new_session=True)
    inhibitor = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())])
    def ready():
        config = configured('baseline'); config.connection.connect_timeout_seconds = 1
        deadline = time.monotonic()+120
        while True:
            if tunnel.poll() is not None: raise RuntimeError('Own SSH tunnel exited')
            try:
                with RemotePi0(config) as client:
                    assert client.metadata['transport_proxy']['kind'] == 'lossless-png'
                    return client.metadata
            except (OSError, TimeoutError):
                if time.monotonic() >= deadline: raise
                time.sleep(1)
            except RuntimeError as exc:
                if 'Image transport failed: ConnectionRefusedError' not in str(exc) or time.monotonic() >= deadline: raise
                time.sleep(1)
    try:
        command = shlex.join([REMOTE_ROOT+'/openpi/.venv/bin/python', '-B', '-c', SELECT_CODE,
                              'v6', str(out.relative_to(ROOT))])
        selected = subprocess.run(ssh + [args.ssh_target, command], capture_output=True, text=True, timeout=90)
        if selected.returncode:
            raise RuntimeError('V6 selection failed: '+selected.stderr[-3000:])
        ledger['selection'] = dict(dispatch=json.loads(selected.stdout), metadata=ready(), metadata_verified=True)
        write(out / 'comparison.json', ledger)
        warm = out / 'warmup'; warm.mkdir()
        env = _make_env(configured('baseline'), warm)
        try:
            obs, _ = env.reset(seed=layouts['episode_019']['reset_seed'])
            observation = aloha_observation({'pixels': {'top': env.render()}, 'agent_pos': obs['agent_pos']},
                                            configured('baseline').simulation.prompt, resize=False)
            prefix = np.repeat(np.asarray(obs['agent_pos'], np.float32)[None], 100, axis=0)
            prefix[:, [6,13]] = np.clip(prefix[:, [6,13]], 0, 1)
            rows = []
            with RemotePi0(configured('baseline')) as client:
                for index in range(6):
                    actions, ms, response = client.infer({**observation, 'rtc': dict(schema='pi0-rtc-v1',
                        prefix=prefix, valid_steps=100, delay_steps=0 if index == 0 else 10,
                        blend_steps=0 if index == 0 else 30, guidance_max=.5)})
                    assert actions.shape == (100,14) and np.isfinite(actions).all()
                    rows.append({'round_trip_ms': ms, 'rtc': response.get('rtc'),
                                 'server_timing': response.get('server_timing'), 'policy_timing': response.get('policy_timing')})
            write(warm / 'requests.json', {'requests': rows})
            measured = [row['round_trip_ms'] for row in rows[2:]]
            if max(measured) >= 1200 or float(np.median(measured)) >= 800:
                raise RuntimeError('Transport preflight too slow for fair motion screen; requests retained, no episodes started')
        finally: env.close()
        index = 0
        for layout, profiles in BLOCKS:
            for profile in profiles:
                assert source_hashes() == plan['source_sha256'], 'Source changed during experiment'
                index += 1
                row = {'index': index, 'profile': profile, 'layout': layout,
                       'scene_sha256': sha(scenes[layout, 'wrist125' if profile == 'wrist125' else 'normal'])}
                print(f'START {index}/12 {layout} {profile}', flush=True)
                try:
                    directory = run(configured(profile, layout))
                    row['run_directory'] = str(directory)
                    details = summary.summarize(directory)
                    tasks = [json.loads(line) for line in (directory/'episode_000_task.jsonl').read_text().splitlines()]
                    with np.load(directory/'episode_000.npz', allow_pickle=False) as data:
                        quality = motion_quality(data['state'], data['action'], tasks)
                        extra = extra_diagnostics(data, tasks)
                    latency = [json.loads(line) for line in (directory/'episode_000_latency.jsonl').read_text().splitlines()]
                    prefix_error = max((r.get('rtc_request') or {}).get('planned_vs_executed_prefix_max_error',0) for r in latency)
                    row.update(diagnostics=details, motion_quality=quality, extra=extra,
                               success=details['episodes'][0]['result']['success'], max_prefix_forecast_error=prefix_error,
                               video=str(directory/'episode_000.mp4'), video_sha256=sha(directory/'episode_000.mp4'))
                    assert source_hashes() == plan['source_sha256'], 'Source changed during recording'
                    assert details['episodes'][0]['raw_prediction_max_error'] == 0
                    assert details['episodes'][0]['max_action_timestamp_error_steps'] == 0
                    assert prefix_error == 0
                except BaseException as exc:
                    row['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
                    ledger['attempts'].append(row); write(out/'comparison.json',ledger); raise
                ledger['attempts'].append(row); write(out/'comparison.json',ledger)
                print('END', index, profile, 'success',row['success'], 'p99',quality['active_frame_max_jerk_rad_s3']['p99'],
                      'pause',extra['short_low_motion_total_seconds'], flush=True)
        ledger['complete'] = True
    except BaseException as exc:
        ledger['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try: ledger['final_endpoint'] = {'model':'v6', 'metadata':ready(), 'metadata_verified':True}
        except BaseException as exc: ledger['final_endpoint'] = {'error':f'{type(exc).__name__}: {exc}', 'metadata_verified':False}
        write(out/'comparison.json',ledger)
        tunnel.terminate()
        try: tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired: tunnel.kill(); tunnel.wait()
        tunnel_log.close(); inhibitor.terminate()
    print('COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
