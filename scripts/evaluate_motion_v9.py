"""Fixed, headless v9 validation; no training, test layouts, or automatic promotion."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'repos/openpi/packages/openpi-client/src')]
LAYOUTS = ('episode_004', 'episode_012', 'episode_019', 'episode_020', 'episode_026', 'episode_029')
BASELINE = Path('outputs/motion_first_20261026')
OUTPUT = Path('outputs/motion_v9_validation_20261027')
VALIDATION_MEMBERSHIP = Path('data/lerobot/20261024-185715-merged-ebdbe3/local/pi0_banana_validation/source_episodes.json')
CONFIGS = {'v9': 'banana_v9_motion_pi0.toml', 'control': 'banana_v8_control32_pi0.toml'}
CONTRACT_KEYS = ('task', 'model_family', 'policy_config', 'action_dim', 'action_horizon',
                 'action_semantics', 'scene_controller_profile', 'denoise_steps', 'camera_crop',
                 'requires_full_resolution', 'openpi_commit', 'norm_stats_sha256', 'rtc')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparable_config(config):
    """Only version pin and predeclared connection/scene/display overrides differ."""
    config = json.loads(json.dumps(config))
    config['model'].pop('expected_checkpoint_suffix')
    config['connection'].pop('port')
    for key in ('scene_config', 'seed', 'display'):
        config['simulation'].pop(key)
    return config


def baseline_binding(root=ROOT):
    """Recover source bindings from old run receipts, not an invented old plan field."""
    from pi0_repro.config import load_config
    base = root / BASELINE
    plan = json.loads((base / 'baseline_plan.json').read_text())
    ledger = json.loads((base / 'comparison.json').read_text())
    if not ledger['complete'] or ledger['plan'] != plan or tuple(plan['layouts']) != LAYOUTS:
        raise ValueError('Baseline must be complete and preserve the frozen six layouts')
    if plan['repeat_warmup_counts'] != [2, 3, 4]:
        raise ValueError('Unexpected baseline warmup schedule')
    membership = json.loads((root / VALIDATION_MEMBERSHIP).read_text())['episodes']
    allowed = {row['source_episode'] for row in membership if row['split'] == 'validation'}
    if not set(LAYOUTS) <= allowed:
        raise ValueError('Every frozen layout must belong to the existing validation split')
    for relative, digest in [('scripts/repeat_motion_baselines.py', plan['script_sha256']),
                             ('src/pi0_repro/motion_quality.py', plan['metric_sha256']),
                             ('configs/' + CONFIGS['control'], plan['config_sha256']['control'])]:
        if sha(root / relative) != digest:
            raise ValueError(f'Frozen baseline binding changed: {relative}')
    rows = [r for r in ledger['results'] if r['profile'] == 'control']
    expected = {(layout, repeat) for layout in LAYOUTS for repeat in (1, 2, 3)}
    if len(rows) != 18 or {(r['layout'], r['repeat']) for r in rows} != expected:
        raise ValueError('Expected exactly 18 unique control baseline attempts')
    reference = asdict(load_config(root / 'configs' / CONFIGS['control']))
    sources, layouts, receipts, contract = None, {}, [], None
    for row in rows:
        if row.get('phase') != 'validation':
            raise ValueError('Control baseline row is not a validation attempt')
        if row.get('infrastructure_error'):
            raise ValueError('Control baseline has an infrastructure failure')
        directory = Path(row['run_directory'])
        receipt_path = directory / 'run.json'
        receipt = json.loads(receipt_path.read_text())
        metadata = json.loads((directory / 'server.json').read_text())
        observed_contract = {key: metadata[key] for key in CONTRACT_KEYS}
        if contract is None:
            contract = observed_contract
        if observed_contract != contract:
            raise ValueError('Baseline serving contract differs across repeats')
        if sources is None:
            sources = receipt['source_code_sha256']
        if receipt['source_code_sha256'] != sources:
            raise ValueError('Baseline execution source differs across repeats')
        if comparable_config(receipt['config']) != comparable_config(reference):
            raise ValueError('Baseline recorded execution settings differ from control config')
        scene = Path(receipt['config']['simulation']['scene_config'])
        binding = {'scene_file': str(scene), 'scene_sha256': sha(scene),
                   'reset_seed': receipt['config']['simulation']['seed']}
        if layouts.setdefault(row['layout'], binding) != binding:
            # Original scene_NN filenames vary; content and seed must not.
            prior = layouts[row['layout']]
            if (prior['scene_sha256'], prior['reset_seed']) != (binding['scene_sha256'], binding['reset_seed']):
                raise ValueError('Baseline layout or reset seed differs across repeats')
        receipts.append({'layout': row['layout'], 'repeat': row['repeat'],
                         'run_directory': str(directory), 'run_json_sha256': sha(receipt_path),
                         'server_json_sha256': sha(directory / 'server.json')})
    if not sources:
        raise ValueError('No baseline source receipts')
    for filename, digest in sources.items():
        if sha(root / 'src/pi0_repro' / filename) != digest:
            raise ValueError(f'Execution source differs from baseline: {filename}')
    return {'baseline_plan_sha256': sha(base / 'baseline_plan.json'),
            'baseline_comparison_sha256': sha(base / 'comparison.json'),
            'baseline_runner_sha256': plan['script_sha256'],
            'metric_sha256': plan['metric_sha256'], 'execution_source_sha256': sources,
            'metadata_contract': contract,
            'validation_membership_file': str(VALIDATION_MEMBERSHIP),
            'validation_membership_sha256': sha(root / VALIDATION_MEMBERSHIP),
            'source_binding_origin': 'All 18 baseline control run.json source_code_sha256 receipts',
            'layouts': layouts, 'control_receipts': receipts}


def build_plan(port, root=ROOT):
    from pi0_repro.config import load_config
    binding = baseline_binding(root)
    control = asdict(load_config(root / 'configs' / CONFIGS['control']))
    candidate = asdict(load_config(root / 'configs' / CONFIGS['v9']))
    if comparable_config(candidate) != comparable_config(control):
        raise ValueError('v9 execution differs from frozen control beyond checkpoint suffix')
    if any(candidate['simulation'][key] != control['simulation'][key] for key in ('scene_config', 'seed')):
        raise ValueError('Warmup scene and seed must match control')
    if candidate['connection']['image_transport'] != 'png' or not candidate['model']['expected_checkpoint_suffix']:
        raise ValueError('v9 must pin its checkpoint and use the PNG sidecar')
    return {'schema': 'motion-v9-validation-v1', 'binding': binding,
            'attempts': 18, 'layouts': list(LAYOUTS), 'repeats': [1, 2, 3],
            'repeat_warmup_counts': [2, 3, 4], 'noise_paired': False,
            'rng_protocol': 'Restart v9 once per repeat; advance with 2/3/4 warmup requests, then six layouts in baseline order.',
            'configurations': CONFIGS,
            'config_sha256': {k: sha(root / 'configs' / v) for k, v in CONFIGS.items()},
            'scripts_sha256': {name: sha(root / 'scripts' / name) for name in
                               ('evaluate_motion_v9.py', 'summarize_motion_v9.py', 'summarize_banana_execution.py')},
            'selector_sha256': sha(root / 'server/select_motion_policy.py'),
            'overrides': {'connection.port': port, 'simulation.display': False,
                          'simulation.scene_config': 'Frozen baseline scene per layout',
                          'simulation.seed': 'Frozen baseline reset_seed per layout'},
            'objective_priority': ['active-frame max jerk p99 and worst physical joint increment',
                                   'threshold dwell, near-to-lift and inside-to-success-or-timeout',
                                   'task-stage coverage: stationary failures have no smoothness advantage',
                                   'success secondary'],
            'selection': 'Report all repeats and failures; no scalar winner, automatic promotion, or success-rate gate.',
            'restore_model': 'control'}


def backend_startup_error(exc):
    # Exactly the sidecar's refused upstream at startup; never retry inference.
    return isinstance(exc, RuntimeError) and re.fullmatch(
        r'远端错误：\nImage transport failed: ConnectionRefusedError: \[Errno (?:111|61)\] (?:Connect call failed|Connection refused)[^\n]*',
        str(exc)) is not None


def select_policy(model, args, receipt, config, remote_class):
    command = [args.remote_root + '/openpi/.venv/bin/python',
               args.remote_root + '/server/select_motion_policy.py',
               '--root', args.remote_root, '--model', model]
    receipt.update(model=model, command=command, startup_errors=[])
    ssh = ['ssh', '-S', str(args.ssh_socket), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
           args.ssh_target, shlex.join(command)]
    result = subprocess.run(ssh, capture_output=True, text=True, timeout=90)
    receipt.update(selector_returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    result.check_returncode()
    deadline = time.monotonic() + 90
    config.connection.connect_timeout_seconds = 1
    try:
        while True:
            try:
                with remote_class(config) as client:
                    metadata = client.metadata
                    if metadata.get('transport_proxy', {}).get('kind') != 'lossless-png':
                        raise ValueError('Expected live PNG sidecar and backend metadata')
                    if {key: metadata.get(key) for key in CONTRACT_KEYS} != args.metadata_contract:
                        raise ValueError('Serving contract differs from frozen baseline')
                    receipt['metadata'] = metadata
                    return metadata
            except (OSError, TimeoutError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and not backend_startup_error(exc):
                    raise
                receipt['startup_errors'].append({'error': f'{type(exc).__name__}: {exc}'})
                if time.monotonic() >= deadline:
                    raise
                time.sleep(min(2, max(0, deadline - time.monotonic())))
    finally:
        config.connection.connect_timeout_seconds = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-target', required=True)
    parser.add_argument('--ssh-socket', type=Path, default=ROOT / 'outputs/v6_recovery_20261026/ssh_control_socket.txt',
                        help='Socket path, or existing text file containing the socket path')
    parser.add_argument('--remote-root', default='/root/pi0-repro')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--check-only', action='store_true', help='Print validated plan without writing evidence, SSH, inference or physics')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.ssh_target) or args.ssh_target.startswith('-'):
        raise ValueError('Invalid SSH target')
    if not 1 <= args.port <= 65535 or not args.remote_root.startswith('/'):
        raise ValueError('Invalid port or remote root')
    plan = build_plan(args.port)
    args.metadata_contract = plan['binding']['metadata_contract']
    plan['ssh_target'] = args.ssh_target
    plan['remote_root'] = args.remote_root
    if args.check_only:
        print(json.dumps(plan, indent=2))
        return
    out = ROOT / OUTPUT
    out.mkdir(exist_ok=False)  # Existing evidence is never overwritten or resumed.
    write(out / 'plan.json', plan)
    if args.ssh_socket.is_file():
        args.ssh_socket = Path(args.ssh_socket.read_text().strip())
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    import numpy as np
    from pi0_repro.config import load_config
    from pi0_repro.motion_quality import motion_quality
    from pi0_repro.observations import aloha_observation
    from pi0_repro.remote import RemotePi0
    from pi0_repro.simulation import _make_env, run
    summary = load_module('execution_summary_v9', ROOT / 'scripts/summarize_banana_execution.py')
    ledger = {'plan': plan, 'complete': False, 'results': [], 'selections': [], 'restore': None}
    write(out / 'comparison.json', ledger)
    def config_for(model):
        config = load_config(ROOT / 'configs' / CONFIGS[model])
        config.connection.port = args.port
        config.simulation.display = False
        return config
    def select(model):
        receipt = {}
        ledger['selections'].append(receipt)
        try:
            return select_policy(model, args, receipt, config_for(model), RemotePi0)
        finally:
            write(out / 'comparison.json', ledger)
    try:
        for repeat in (1, 2, 3):
            metadata = select('v9')
            warm = out / f'warmup_{repeat:02d}'
            warm.mkdir()
            config = config_for('v9')
            env = _make_env(config, warm)
            requests = []
            try:
                obs, _ = env.reset(seed=config.simulation.seed)
                observation = aloha_observation({'pixels': {'top': env.render()}, 'agent_pos': obs['agent_pos']},
                                                config.simulation.prompt, resize=False)
                prefix = np.repeat(np.asarray(obs['agent_pos'], np.float32)[None], 100, axis=0)
                prefix[:, [6, 13]] = np.clip(prefix[:, [6, 13]], 0, 1)
                with RemotePi0(config) as client:
                    for i in range(repeat + 1):
                        request = {**observation, 'rtc': dict(schema='pi0-rtc-v1', prefix=prefix, valid_steps=100,
                                   delay_steps=0 if i == 0 else 10, blend_steps=0 if i == 0 else 20, guidance_max=.5)}
                        actions, ms, response = client.infer(request)
                        if actions.shape != (100, 14) or not np.isfinite(actions).all():
                            raise ValueError('Invalid warmup actions')
                        requests.append({'round_trip_ms': ms, 'rtc': response['rtc']})
            finally:
                env.close()
                write(warm / 'requests.json', {'metadata': metadata, 'requests': requests,
                      'scope': 'Predeclared RNG advance only, no environment steps; excluded from task outcomes.'})
            for layout in LAYOUTS:
                index = len(ledger['results'])
                row = {'index': index, 'profile': 'v9', 'phase': 'validation', 'repeat': repeat, 'layout': layout}
                ledger['results'].append(row)
                write(out / 'comparison.json', ledger)
                config = config_for('v9')
                source = plan['binding']['layouts'][layout]
                scene = out / f'scene_{index:02d}.toml'
                scene.write_bytes(Path(source['scene_file']).read_bytes())
                if sha(scene) != source['scene_sha256']:
                    raise ValueError('Frozen scene changed during evaluation')
                config.simulation.scene_config = str(scene)
                config.simulation.seed = source['reset_seed']
                print(f'START {index + 1}/18 repeat={repeat} {layout}', flush=True)
                before = {p for p in (ROOT / 'outputs').iterdir() if p.is_dir()}
                try:
                    directory = run(config)
                    row['run_directory'] = str(directory)
                    details = summary.summarize(directory)
                    with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
                        states, actions = data['state'], data['action']
                    tasks = [json.loads(s) for s in (directory / 'episode_000_task.jsonl').read_text().splitlines()]
                    row.update(diagnostics=details, motion_quality=motion_quality(states, actions, tasks))
                except BaseException as exc:
                    row['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
                    row['new_output_directories_at_failure'] = sorted(str(p) for p in
                        {p for p in (ROOT / 'outputs').iterdir() if p.is_dir()} - before)
                    raise  # No task retries or selective exclusion.
                finally:
                    write(out / 'comparison.json', ledger)
                print(f'END {index + 1}/18 success={details["episodes"][0]["result"]["success"]}', flush=True)
        ledger['complete'] = True
    except BaseException as exc:
        ledger['fatal_error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            metadata = select('control')  # Restore only at the end, including failure paths.
            ledger['restore'] = {'verified': True, 'metadata': metadata}
        except BaseException as exc:
            ledger['restore'] = {'verified': False, 'error': f'{type(exc).__name__}: {exc}'}
        write(out / 'comparison.json', ledger)
        print(json.dumps({'complete': ledger['complete'], 'restore': ledger['restore']}), flush=True)
    if not ledger['restore']['verified']:
        raise RuntimeError('Control restoration failed; see retained comparison.json')


if __name__ == '__main__':
    main()
