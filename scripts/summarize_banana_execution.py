"""Summarize recorded robot motion and control timing without rerunning physics."""
import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pi0_repro.diagnostics import chunk_continuity, task_progress

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
DT = 0.02


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size:
        return None
    return {"median": float(np.median(a)), "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def summarize(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    config = json.loads((directory / 'run.json').read_text())['config']
    result = {"run_directory": str(directory.resolve()), "config": config, "episodes": []}
    for episode in summary['episodes']:
        stem = f"episode_{episode['episode']:03d}"
        timings = [json.loads(row) for row in (directory / f'{stem}_latency.jsonl').read_text().splitlines()]
        control_path = directory / f'{stem}_control.jsonl'
        control = [json.loads(row) for row in control_path.read_text().splitlines()] if control_path.exists() else []
        task_path = directory / f'{stem}_task.jsonl'
        tasks = [json.loads(line) for line in task_path.read_text().splitlines()] if task_path.exists() else []
        with np.load(directory / f'{stem}.npz', allow_pickle=False) as data:
            row = {"result": episode, "motion": {}}
            for name in ('requested_action', 'submitted_action', 'action', 'state'):
                q = data[name][:, JOINTS].astype(float)
                row['motion'][name] = {
                    "velocity_abs_rad_s": distribution(np.abs(np.diff(q, axis=0) / DT).ravel()),
                    "acceleration_abs_rad_s2": distribution(np.abs(np.diff(q, n=2, axis=0) / DT**2).ravel()),
                    "jerk_abs_rad_s3": distribution(np.abs(np.diff(q, n=3, axis=0) / DT**3).ravel()),
                    "joint_path_l1_rad": float(np.abs(np.diff(q, axis=0)).sum()),
                }
            row['task_progress'] = task_progress(tasks, data['action'])
            if 'action_prediction_index' in data and len(data['requested_action']):
                row['chunk_continuity'] = chunk_continuity(data)
                indices, offsets = data['action_prediction_index'], data['action_prediction_offset']
                expected = data['predicted_chunks'][indices, offsets]
                row['raw_prediction_max_error'] = float(np.max(np.abs(expected - data['requested_action'])))
                expected_steps = data['prediction_steps'][indices] + offsets
                row['max_action_timestamp_error_steps'] = int(np.max(np.abs(expected_steps - np.arange(len(indices)))))
                settings = config['simulation']
                if settings.get('temporal_ensemble'):
                    # NPZ is lazy: cache these arrays instead of decompressing
                    # whole chunks repeatedly for every control timestamp.
                    predictions = data['predicted_chunks']
                    origins = data['prediction_steps']
                    references = data['blended_action']
                    errors, counts = [], []
                    for step, latest in enumerate(indices):
                        eligible = [i for i in range(max(0, int(latest) - settings['temporal_ensemble_max_chunks'] + 1), int(latest) + 1)
                                    if 0 <= step - origins[i] < len(predictions[i])]
                        weights = np.exp(-settings['temporal_ensemble_coeff'] * np.arange(len(eligible)))
                        aligned = np.stack([predictions[i, step - origins[i], JOINTS] for i in eligible])
                        target = np.average(aligned, axis=0, weights=weights)
                        errors.append(np.max(np.abs(target - references[step, JOINTS])))
                        counts.append(len(eligible))
                    row['temporal_ensemble'] = {
                        'max_contributors': max(counts),
                        'steps_with_multiple_predictions': sum(c > 1 for c in counts),
                        'reference_reconstruction_max_error': float(max(errors)),
                        'contributor_counts_match': bool(np.array_equal(counts, data['temporal_ensemble_counts'])),
                        'newest_gripper_max_error': float(np.max(np.abs(references[:, [6, 13]] - data['requested_action'][:, [6, 13]]))),
                    }
        row['round_trip_ms'] = distribution([t['round_trip_ms'] for t in timings])
        row['server_infer_ms'] = distribution([t['server_timing']['infer_ms'] for t in timings if 'infer_ms' in t.get('server_timing', {})])
        row['payload_bytes'] = distribution([t['client_transport']['payload_bytes'] for t in timings if 'payload_bytes' in t.get('client_transport', {})])
        transported = [t['transport_timing'] for t in timings if t.get('transport_timing')]
        row['lossless_image_hashes_verified'] = all(t.get('pixel_digest_verified') for t in transported) if transported else None
        if control:
            intervals = np.diff([r['wall_elapsed_seconds'] for r in control])
            row['control_interval_ms'] = distribution(intervals * 1000)
            row['control_intervals_above_100ms'] = int(np.sum(intervals > .1))
            row['blocking_seconds_after_startup'] = float(sum(r['inference_blocked_seconds'] for r in control[1:]))
            apply_times = [r['action_apply_wall_seconds'] for r in control if 'action_apply_wall_seconds' in r]
            row['action_apply_interval_ms'] = distribution(np.diff(apply_times) * 1000)
            display_path = directory / f'{stem}_display.jsonl'
            displays = [json.loads(line) for line in display_path.read_text().splitlines()] if display_path.exists() else []
            # Exclude frames shown before the first action, including startup wait.
            running_displays = [r['wall_elapsed_seconds'] for r in displays
                                if apply_times and r['wall_elapsed_seconds'] >= apply_times[0]]
            display_intervals = np.diff(running_displays) * 1000
            row['display_interval_ms_after_startup'] = distribution(display_intervals)
            row['display_intervals_above_50ms_after_startup'] = int(np.sum(display_intervals > 50)) if len(running_displays) > 1 else None
            row['display_frames_after_startup'] = len(running_displays) if displays else None
        result['episodes'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {"scope": "recorded episodes; derivatives use simulation time, commanded targets differ from physical motion; timing follows local wall clock", "runs": [summarize(p) for p in args.runs]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
