"""Read-only motion-pilot aggregates; no model/norm tensors or launch descriptions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def summarize(root):
    parent = root / 'outputs/motion_first_20261026'
    out = parent / 'cloud_training'
    path = out / 'status.json'
    if not path.exists():
        return {'status': 'not_started'}
    status = json.loads(path.read_text())
    plan = json.loads((parent / 'training_plan.json').read_text())
    result = {key: status.get(key) for key in ('status', 'experiment', 'error', 'restore_model',
              'restored_policy_pid', 'restoration_dispatch_succeeded', 'restore_verified', 'restore_error')}
    result['sampled_at'] = time.time()
    result['elapsed_seconds'] = status.get('finished_at', result['sampled_at']) - status['started_at']
    result['completed_optimizer_updates'] = status.get('completed_optimizer_updates')
    result['preflight'] = status.get('preflight')
    probe_path = out / 'probe.json'
    if probe_path.exists():
        try:
            probe = json.loads(probe_path.read_text())
        except (OSError, json.JSONDecodeError):
            # Probe sampling writes its small JSON repeatedly. A concurrent read
            # must not turn a transient write into a failed job or lost snapshot.
            probe = {'status': 'snapshot_not_ready'}
        result['probe'] = {key: probe.get(key) for key in ('status', 'completed_optimizer_updates', 'actual_action_shape',
            'steady_update_seconds', 'gpu_peak_used_mib', 'gpu_total_mib', 'error')}
        allowed = {'loss', 'flow_loss', 'temporal_loss', 'weighted_temporal_loss', 'grad_norm', 'param_norm'}
        result['probe']['timings'] = [{'optimizer_update': row['optimizer_update'], 'seconds': row['seconds'],
            'metrics': {key: value for key, value in row['metrics'].items() if key in allowed}}
            for row in probe.get('timings', [])]
    log = out / 'training.log'
    if log.exists():
        text = log.read_text(errors='replace')
        lines = [line for line in text.splitlines() if re.search(r'Step \d+:', line)]
        steps = [int(value) for value in re.findall(r'Step (\d+):', text)]
        result['last_logged_step'] = max(steps, default=None)
        result['completed_updates_lower_bound'] = max([value + 1 for value in steps] +
            [int(value) for value in re.findall(r'(\d+)/1500', text)] + [status.get('completed_optimizer_updates', 0)])
        result['last_metric_line'] = lines[-1] if lines else None
        result['nonfinite_metric_lines'] = sum(bool(re.search(r'\b(?:nan|[+-]?inf)\b', line, re.I)) for line in lines)
    final = Path('/root/pi0-motion-checkpoints/pi0_banana_lora') / plan['experiment'] / str(plan['final_step'])
    result['final_checkpoint_committed'] = (final / '_CHECKPOINT_METADATA').is_file() and (final / 'params/_METADATA').is_file()
    norm = final / 'assets/local/pi0_banana_train/norm_stats.json'
    if result['final_checkpoint_committed'] and norm.exists():
        result['norm_stats_sha256'] = hashlib.sha256(norm.read_bytes()).hexdigest()
        result['normalization_digest_matches_plan'] = result['norm_stats_sha256'] == plan['norm_stats_sha256']
    reader = out / 'reader_audit.json'
    if reader.exists():
        try:
            result['reader_audit'] = {key: value for key, value in json.loads(reader.read_text()).items() if key != 'samples'}
        except (OSError, json.JSONDecodeError):
            result['reader_audit'] = {'status': 'snapshot_not_ready'}
    normalization=out/'normalization_audit.json'
    if normalization.exists():
        try:
            record=json.loads(normalization.read_text())
            result['normalization_audit']={key:record.get(key) for key in ('status','training_frames','norm_stats_sha256','actions','states')}
        except (OSError,json.JSONDecodeError):
            result['normalization_audit']={'status':'snapshot_not_ready'}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(summarize(args.root.resolve())))


if __name__ == '__main__':
    main()
