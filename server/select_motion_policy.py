"""Dispatch a completed motion pilot for validation, or restore its v8 control."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


OUTPUT = Path('outputs/motion_first_20261026')
FINAL_CHECKPOINT = Path('/root/pi0-motion-checkpoints/pi0_banana_lora/banana_v9_motion1500/1499')
NORM_SHA = '6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(root, value):
    path = Path(value)
    require(not path.is_absolute() and '..' not in path.parts, 'Expected a project-relative path')
    result = (root / path).resolve()
    require(result.is_relative_to(root), 'Path escapes project root')
    return result


def validate_candidate(root, checkpoint=FINAL_CHECKPOINT):
    parent = root / OUTPUT
    plan = json.loads((parent / 'training_plan.json').read_text())
    status = json.loads((parent / 'cloud_training/status.json').read_text())
    require(plan['experiment'] == 'banana_v9_motion1500' and plan['final_step'] == 1499,
            'Unexpected motion experiment or final step')
    require(status.get('status') == 'trained_pending_validation', 'Motion training has not completed')
    require(status.get('experiment') == plan['experiment'], 'Training status belongs to another experiment')
    require(status.get('completed_optimizer_updates') == 1500 and status.get('final_checkpoint_committed') is True,
            'Training completion is not recorded')
    require(status.get('checkpoint') == str(checkpoint), 'Status checkpoint differs from selected checkpoint')
    require(status.get('norm_stats_sha256') == plan['norm_stats_sha256'] == NORM_SHA,
            'Unexpected training normalization digest')
    config = relative_path(root, plan['config'])
    require(digest(config) == plan['config_sha256'], 'Training configuration changed since plan review')
    require(bool(plan.get('source_sha256')), 'Plan has no source binding')
    for name, expected in plan['source_sha256'].items():
        require(digest(relative_path(root, name)) == expected, 'Bound source changed: ' + name)
    for path in (checkpoint / '_CHECKPOINT_METADATA', checkpoint / 'params/_METADATA'):
        require(path.is_file() and path.stat().st_size > 0, 'Final checkpoint metadata is missing or empty')
        json.loads(path.read_text())
    params = checkpoint / 'params'
    # The pinned Orbax writer uses OCDBT. Commit markers alone are insufficient
    # if migration omitted the binary manifests or parameter data directory.
    require(any(path.stat().st_size > 0 for path in params.rglob('manifest.ocdbt') if path.is_file()),
            'Final parameter OCDBT manifest is missing')
    require(any(path.is_file() and path.stat().st_size > 0
                for directory in params.rglob('d') if directory.is_dir() for path in directory.rglob('*')),
            'Final parameter data is missing')
    require(digest(checkpoint / 'assets/local/pi0_banana_train/norm_stats.json') == NORM_SHA,
            'Final checkpoint normalization differs from training plan')
    return config


def process_inventory(root, proc=Path('/proc')):
    policies = []
    training = str(root / 'server/banana_finetune.py').encode()
    serving = str(root / 'server/serve_pi0.py').encode()
    require(proc.is_dir(), 'Linux process inventory is unavailable')
    for cmdline in proc.glob('[0-9]*/cmdline'):
        try:
            parts = cmdline.read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise RuntimeError('Cannot inspect complete process inventory') from exc
        require(training not in parts, 'Project training process remains active')
        if serving in parts:
            policies.append(int(cmdline.parent.name))
    require(len(policies) <= 1, 'Multiple project policy processes; refusing ambiguous stop')
    return policies


def stop_policy_and_wait(root, policies):
    for pid in policies:
        try:
            parts = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
            require(str(root / 'server/serve_pi0.py').encode() in parts, 'Policy process identity changed')
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
    for _ in range(30):
        remaining = process_inventory(root)
        values = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True, timeout=5).splitlines()
        require(bool(values), 'GPU memory inventory is empty')
        used = [int(value.strip()) for value in values]
        if not remaining and all(value < 2048 for value in used):
            return used
        time.sleep(1)
    raise RuntimeError('Policy exit or GPU idle check failed; no replacement dispatched')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=('v9', 'control'), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / OUTPUT / 'cloud_training'
    require(out.is_dir(), 'Motion training output directory is missing')
    # The supervisor holds this lock through its baseline-restoration finally
    # block. A completed status alone must not race that restoration dispatch.
    with (out / 'job.lock').open('r+') as training_lock, (out / 'selection.lock').open('a') as selection_lock:
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(selection_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process_inventory(root)
        record_path = out / 'motion_selected_policy_process.json'
        record = {'model': args.model, 'status': 'checking', 'verified': False, 'started_at_unix': time.time()}
        try:
            if args.model == 'control':
                result = subprocess.run(
                    [str(root / 'openpi/.venv/bin/python'), str(root / 'server/select_v8_policy.py'),
                     '--root', str(root), '--model', 'control'], cwd=root,
                    capture_output=True, text=True, timeout=90)
                (out / 'motion_control_selector.log').write_text(result.stdout + result.stderr)
                require(result.returncode == 0, 'Control selector failed; inspect motion_control_selector.log')
                delegated = json.loads(result.stdout)
                record.update(pid=delegated['pid'], checkpoint=delegated['checkpoint'],
                              status='dispatched', scope='Dispatch only; verify metadata separately.')
            else:
                config = validate_candidate(root)
                policies = process_inventory(root)
                atomic_json(record_path, dict(record, status='stopping_policy', stopped_policy_pids=policies))
                used = stop_policy_and_wait(root, policies)
                # Nothing may alter the reviewed serving sources or completed
                # candidate while the old policy is relinquishing the GPU.
                validate_candidate(root)
                require(not process_inventory(root), 'A policy appeared before dispatch; refusing duplicate launch')
                env = dict(os.environ)
                env.update(PI0_RTC='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                with (out / 'motion_v9_serve.log').open('ab') as log:
                    process = subprocess.Popen(
                        ['bash', str(root / 'server/run_banana_serve.sh'), str(config), str(FINAL_CHECKPOINT), '10'],
                        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                record.update(status='dispatched', pid=process.pid, checkpoint=str(FINAL_CHECKPOINT),
                              idle_gpu_used_mib=used, norm_stats_sha256=NORM_SHA,
                              scope='Dispatch only; verify checkpoint, RTC and PNG metadata separately.')
            atomic_json(record_path, record)
            print(json.dumps(record))
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}', finished_at_unix=time.time())
            atomic_json(record_path, record)
            raise


if __name__ == '__main__':
    main()
