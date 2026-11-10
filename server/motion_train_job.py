"""Guarded single original-pi0 motion pilot; never alter earlier checkpoints."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import tomllib

RELATIVE_OUTPUT = Path('outputs/motion_first_20261026')
CHECKPOINT_BASE = Path('/root/pi0-motion-checkpoints')
NORM_SHA = '6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
HEADROOM_BYTES = 2048 * 1024**2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a project-relative path')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Project path escapes root')
    return resolved


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lineage(dataset, split):
    record = json.loads((dataset / f'local/pi0_banana_{split}/source_episodes.json').read_text())
    require(record['source_split'] == split, 'Split mapping mismatch')
    require(all(row['split'] == split for row in record['episodes']), 'Mixed split mapping')
    return [(row['layout_id'], row['source_trajectory_sha256'], row['frames']) for row in record['episodes']]


def validate(root, plan):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', plan['experiment'])), 'Invalid experiment name')
    require(plan.get('restore_model') in ('v6', 'control'), 'Invalid baseline restoration model')
    require(plan.get('norm_stats_sha256') == NORM_SHA, 'Unexpected normalization pin')
    required_sources = {'server/banana_model.py', 'server/banana_config.py', 'server/banana_accumulation.py',
                        'server/banana_sampling.py', 'server/banana_temporal.py', 'server/banana_temporal_data.py',
                        'server/banana_finetune.py', 'server/banana_probe.py', 'server/banana_roi.py',
                        'server/temporal_preflight.py', 'server/motion_train_job.py', 'server/motion_train_status.py',
                        'server/select_v8_policy.py', 'tests/test_near_term_loss.py', 'tests/test_temporal_loss.py'}
    required_sources.add('server/motion_norm_audit.py')
    require(required_sources <= set(plan['source_sha256']), 'Training sources are not fully bound')
    for name, expected in plan['source_sha256'].items():
        require(digest(relative_path(root, name)) == expected, 'Source hash mismatch: ' + name)
    config = relative_path(root, plan['config'])
    require(digest(config) == plan['config_sha256'], 'Training configuration changed')
    settings = tomllib.loads(config.read_text())
    model, train, paths = (settings[key] for key in ('model', 'train', 'paths'))
    require(model['family'] == 'pi0' and model['config_name'] == 'pi0_banana_lora', 'Only original banana pi0 accepted')
    require(model['action_horizon'] == 100 and model.get('weighted_flow_loss'), 'Expected weighted H100 flow matching')
    require(model.get('strict_lora_only') is True and model.get('train_action_projections') is True,
            'Expected LoRA plus the configured action/state/time projections')
    require(model.get('camera_crop') == [275, 110, 540, 345], 'Expected unchanged camera ROI')
    require(model.get('temporal_training') is True and model.get('temporal_loss_weight') == 0, 'Expected masked training without temporal regularization')
    near_steps = plan.get('near_term_steps', 30)
    near_weight = plan.get('near_term_weight', 2)
    require(model.get('near_term_steps', 0) == near_steps and model.get('near_term_weight', 1.0) == near_weight,
            'Near-term objective changed')
    require(train['batch_size'] == 8 and train.get('gradient_accumulation_steps') == 4, 'Expected effective batch32')
    require(train['num_train_steps'] == plan.get('num_train_steps', 1500)
            and plan['final_step'] == train['num_train_steps'] - 1,
            'Pilot update budget changed')
    require(train['save_interval'] == 500 and train['keep_period'] == 1000000, 'Expected rolling checkpoint retention')
    require(train.get('require_initial_norm_match') is True, 'Initializer normalization guard required')
    require(Path(paths['checkpoint_base_dir']) == CHECKPOINT_BASE, 'Expected separate checkpoint volume')
    require(paths['initial_checkpoint'] == plan['initial_checkpoint'], 'Initializer differs from plan')
    require(settings['data']['directory'] == plan['dataset'] and settings['data']['train_repo_id'] == 'local/pi0_banana_train', 'Dataset differs from plan')
    require(settings.get('sampling', {}).get('weights_file') == plan['sampler'], 'Sampler differs from plan')
    dataset = relative_path(root, plan['dataset'])
    manifest = dataset / 'dataset_manifest.json'
    require(digest(manifest) == plan['dataset_manifest_sha256'], 'Dataset manifest changed')
    require(json.loads(manifest.read_text())['status'] == 'complete', 'Dataset export incomplete')
    mapping = json.loads((dataset / 'local/pi0_banana_train/source_episodes.json').read_text())
    rows = mapping['episodes']
    require(type(plan['train_episodes']) is int and plan['train_episodes'] > 0 and
            type(plan['train_frames']) is int and plan['train_frames'] > 0, 'Training counts must be positive integers')
    require(mapping['source_split'] == 'train' and len(rows) == plan['train_episodes'], 'Training mapping count mismatch')
    require(all(row['split'] == 'train' and row['lerobot_episode_index'] == index for index, row in enumerate(rows)), 'Training episodes not contiguous/train-only')
    require(all(type(row['frames']) is int and row['frames'] > 0 for row in rows), 'Invalid episode frame count')
    require(sum(row['frames'] for row in rows) == plan['train_frames'], 'Training frame count mismatch')
    base = relative_path(root, plan['base_dataset'])
    for split in ('validation', 'test'):
        require(lineage(base, split) == lineage(dataset, split), 'Held-out lineage changed: ' + split)
    sampler = relative_path(root, plan['sampler'])
    require(digest(sampler) == plan['sampler_sha256'], 'Sampler file changed')
    weights = json.loads(sampler.read_text())
    require(weights['split'] == 'train' and weights['repo_id'] == 'local/pi0_banana_train', 'Sampler split changed')
    require(weights['dataset_manifest_sha256'] == plan['dataset_manifest_sha256'], 'Sampler manifest binding mismatch')
    require(len(weights['weights']) == plan['train_frames'] and
            all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
                for value in weights['weights']), 'Invalid sampler weights')
    initial = relative_path(root, plan['initial_checkpoint'])
    require((initial / '_CHECKPOINT_METADATA').is_file() and (initial / 'params/_METADATA').is_file(), 'Initializer not committed')
    norm = initial / 'assets/local/pi0_banana_train/norm_stats.json'
    require(digest(norm) == NORM_SHA, 'Initializer normalization mismatch')
    destination = relative_path(root, paths['assets_base_dir']) / 'pi0_banana_lora/local/pi0_banana_train/norm_stats.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(destination.read_bytes() == norm.read_bytes(), 'Existing motion normalization differs')
    else:
        shutil.copy2(norm, destination)
    initial_bytes = sum(path.stat().st_size for path in initial.rglob('*') if path.is_file())
    probe_volume = CHECKPOINT_BASE
    while not probe_volume.exists():
        probe_volume = probe_volume.parent
    free = shutil.disk_usage(probe_volume).free
    required = 2 * initial_bytes + HEADROOM_BYTES
    require(free >= required, 'Checkpoint volume lacks room for two snapshots plus 2048MiB')
    final = CHECKPOINT_BASE / model['config_name'] / plan['experiment'] / str(plan['final_step'])
    require(not final.parent.exists(), 'Candidate checkpoint directory already exists')
    return settings, final, {'checkpoint_volume_free_bytes': free, 'initializer_checkpoint_bytes': initial_bytes,
                            'required_peak_bytes': required, 'norm_stats_sha256': NORM_SHA,
                            'train_episodes': len(rows), 'train_frames': plan['train_frames'],
                            'heldout_lineage_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    parent = root / RELATIVE_OUTPUT
    plan = json.loads((parent / 'training_plan.json').read_text())
    out = parent / 'cloud_training'
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'job.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = out / 'status.json'
    if status_path.exists():
        raise FileExistsError('Existing motion job retained; inspect status or explicitly version another experiment')
    status = {'pid': os.getpid(), 'experiment': plan['experiment'], 'status': 'preflight', 'started_at': time.time()}

    def update(**values):
        status.update(values)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)

    def interrupted(signum, frame):
        raise InterruptedError('Supervisor received signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    python = root / 'openpi/.venv/bin/python'
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled',
        HF_HOME=str(root / 'cache/huggingface'), OPENPI_DATA_HOME=str(root / 'cache/openpi'),
        XDG_CACHE_HOME=str(root / 'cache'), PYTHONPATH=str(root / 'server') + ':' + str(root / 'openpi/src'))

    def run(command, logfile, *, cpu=False):
        env = dict(environment)
        if cpu:
            env['JAX_PLATFORMS'] = 'cpu'
        keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'WANDB_MODE', 'HF_HOME', 'OPENPI_DATA_HOME', 'XDG_CACHE_HOME', 'PYTHONPATH')
        shell = 'if [ -f env.sh ]; then source env.sh; fi; export ' + ' '.join(key + '=' + shlex.quote(env[key]) for key in keys)
        shell += '; exec ' + shlex.join([str(value) for value in command])
        with (out / logfile).open('xb') as log:
            child = subprocess.Popen(['bash', '-lc', shell], cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                return child.wait()
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=20)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=20)
                raise

    stopped = []
    gpu_phase_entered = False

    def wait_for_idle_gpu():
        for _ in range(30):
            used = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                              text=True, timeout=5).splitlines()[0])
            if used < 2048:
                return used
            time.sleep(1)
        raise RuntimeError('GPU allocations remain; refusing training')

    try:
        update()
        settings, final, preflight = validate(root, plan)
        update(preflight=preflight)
        for name in ('test_near_term_loss.py', 'test_temporal_loss.py'):
            if run([python, root / 'tests' / name], name + '.log', cpu=True):
                raise RuntimeError('CPU numerical/gradient test failed: ' + name)
        if run([python, root / 'server/temporal_preflight.py', '--root', root, '--config', root / plan['config'],
                '--output', out / 'reader_audit.json'], 'reader_audit.log', cpu=True):
            raise RuntimeError('Real-reader episode-mask audit failed')
        if run([python, root / 'server/motion_norm_audit.py', '--root', root, '--config', root / plan['config'],
                '--output', out / 'normalization_audit.json'], 'normalization_audit.log', cpu=True):
            raise RuntimeError('Train-only normalization compatibility audit failed')
        policy_pids = []
        for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                parts = cmdline.read_bytes().split(b'\0')
                if str(root / 'server/banana_finetune.py').encode() in parts:
                    raise RuntimeError('Another project training process is active')
                if str(root / 'server/serve_pi0.py').encode() in parts:
                    policy_pids.append(int(cmdline.parent.name))
            except OSError:
                pass
        if len(policy_pids) > 1:
            raise RuntimeError('Multiple project policy processes; refusing ambiguous stop')
        update(status='pausing_policy', stopped_policy_pids=policy_pids)
        for pid in policy_pids:
            stopped.append(pid)
            os.kill(pid, signal.SIGTERM)
        gpu_phase_entered = True
        used = wait_for_idle_gpu()
        update(status='probing', idle_gpu_used_mib=used)
        common = ['--root', root, '--config', root / plan['config']]
        if run([python, '-u', root / 'server/banana_finetune.py', 'probe', *common, '--exp-name', plan['experiment'] + '_probe',
                '--probe-updates', '3', '--probe-output', out / 'probe.json'], 'probe.log'):
            raise RuntimeError('Real GPU probe failed; full training not started')
        probe = json.loads((out / 'probe.json').read_text())
        require(probe.get('status') == 'passed' and probe.get('actual_action_shape') == [32, 100, 32]
                and probe.get('completed_optimizer_updates') == 3, 'Invalid probe status/shape/update count')
        require(isinstance(probe.get('gpu_peak_used_mib'), int) and isinstance(probe.get('gpu_total_mib'), int)
                and probe['gpu_total_mib'] - probe['gpu_peak_used_mib'] >= 2048, 'GPU probe lacks 2048MiB headroom')
        require(all(math.isfinite(value) for step in probe['timings'] for value in step['metrics'].values()), 'Nonfinite probe metric')
        # A probe is not permission for later source/config changes or exhausted
        # storage; recheck the same bound plan before the saved continuation.
        validate(root, plan)
        wait_for_idle_gpu()
        update(status='training', probe={key: probe[key] for key in ('steady_update_seconds', 'gpu_peak_used_mib', 'gpu_total_mib', 'actual_action_shape')})
        if run([python, '-u', root / 'server/banana_finetune.py', 'train', *common, '--exp-name', plan['experiment']], 'training.log'):
            raise RuntimeError('Motion continuation failed')
        require((final / '_CHECKPOINT_METADATA').is_file() and (final / 'params/_METADATA').is_file(), 'Final checkpoint not committed')
        final_norm = final / 'assets/local/pi0_banana_train/norm_stats.json'
        require(digest(final_norm) == NORM_SHA, 'Final checkpoint normalization digest mismatch')
        update(status='trained_pending_validation', checkpoint=str(final), completed_optimizer_updates=1500,
               final_checkpoint_committed=True, norm_stats_sha256=NORM_SHA, finished_at=time.time())
    except BaseException as exc:
        update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=time.time())
        raise
    finally:
        if stopped or gpu_phase_entered:
            try:
                command = [python, root / 'server/select_v8_policy.py', '--root', root, '--model', plan['restore_model']]
                result = subprocess.run([str(value) for value in command], cwd=root, env=environment,
                                        capture_output=True, text=True, timeout=90)
                (out / 'restore_policy.log').write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError('Baseline selector failed with exit code ' + str(result.returncode))
                dispatch = json.loads(result.stdout)
                update(restore_model=plan['restore_model'], restored_policy_pid=dispatch['pid'],
                       restoration_dispatch_succeeded=True, restore_verified=False)
            except BaseException as exc:
                update(restore_model=plan['restore_model'], restoration_dispatch_succeeded=False,
                       restore_verified=False, restore_error=f'{type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
