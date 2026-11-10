"""Offline v9/control motion comparison, retaining incomplete attempts and coverage."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_motion_v9 import (BASELINE, CONTRACT_KEYS, LAYOUTS, OUTPUT, baseline_binding,
                                comparable_config, load_module, sha)

PHASES = ('approach', 'near_grasp_before_lift', 'carrying_before_box', 'inside_before_success')
DT = .02


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {'n': 0, 'median': None, 'p99': None, 'max': None}
    return {'n': len(values), 'median': float(np.median(values)),
            'p99': float(np.quantile(values, .99)), 'max': max(values)}


def transition(tasks, start_key, end_key, dt=DT):
    """Include elapsed time to the recorded end when a reached stage never completes."""
    start = next((i for i, row in enumerate(tasks) if row.get(start_key)), None)
    if start is None:
        return {'reached': False, 'completed': False, 'censored_at_recording_end': False, 'seconds': None}
    end = next((i for i in range(start, len(tasks)) if tasks[i].get(end_key)), None)
    return {'reached': True, 'completed': end is not None,
            'censored_at_recording_end': end is None,
            'seconds': float(((end - start) if end is not None else (len(tasks) - 1 - start)) * dt)}


def stage_metrics(states, tasks, thresholds, dt=DT):
    """Label global physical derivatives by the frozen metric's phase rule."""
    from pi0_repro.motion_quality import ARM_JOINTS
    q = np.asarray(states, float)[:, ARM_JOINTS]
    speed = np.max(np.abs(np.diff(q, axis=0)), axis=1) / dt
    increments = np.max(np.abs(np.diff(q, axis=0)), axis=1)
    jerk = np.max(np.abs(np.diff(q, n=3, axis=0)), axis=1) / dt**3
    active = speed[2:] > thresholds['active_joint_speed_rad_s']
    phases = np.array(['inside_before_success' if t.get('inside_box') else
                       'carrying_before_box' if t.get('ever_grasped') else
                       'near_grasp_before_lift' if t.get('near_grasp_region') else 'approach'
                       for t in tasks])
    return {phase: {'observed_seconds': float(np.sum(phases == phase) * dt),
                    'active_motion_seconds': float(np.sum(active & (phases[3:] == phase)) * dt),
                    'active_frame_max_jerk_rad_s3': distribution(jerk[active & (phases[3:] == phase)]),
                    'physical_joint_increment_rad': distribution(increments[phases[1:] == phase])}
            for phase in PHASES}


def read_attempt(row, metric, execution_summary, binding, expected_config):
    result = {key: row.get(key) for key in ('profile', 'layout', 'repeat', 'run_directory', 'infrastructure_error')}
    result['recorded_error'] = row.get('infrastructure_error')
    if not row.get('run_directory'):
        result['analysis_error'] = 'No completed run directory; attempt retained and unscored'
        return result
    directory = Path(row['run_directory'])
    try:
        receipt = json.loads((directory / 'run.json').read_text())
        if receipt['source_code_sha256'] != binding['execution_source_sha256']:
            raise ValueError('Recorded execution source does not match baseline')
        config = receipt['config']
        layout = binding['layouts'][row['layout']]
        if comparable_config(config) != comparable_config(expected_config):
            raise ValueError('Recorded execution settings do not match control')
        if config['simulation']['seed'] != layout['reset_seed'] or config['simulation']['display']:
            raise ValueError('Unexpected reset seed or display')
        if sha(Path(config['simulation']['scene_config'])) != layout['scene_sha256']:
            raise ValueError('Recorded scene does not match baseline layout')
        metadata = json.loads((directory / 'server.json').read_text())
        if {key: metadata.get(key) for key in CONTRACT_KEYS} != binding['metadata_contract']:
            raise ValueError('Recorded metadata contract does not match baseline')
        suffix = expected_config['model']['expected_checkpoint_suffix']
        if not str(metadata.get('checkpoint', '')).rstrip('/').endswith('/' + suffix):
            raise ValueError('Recorded checkpoint differs from declared profile')
        details = execution_summary.summarize(directory)
        tasks = [json.loads(s) for s in (directory / 'episode_000_task.jsonl').read_text().splitlines()]
        with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
            states, actions = data['state'], data['action']
        quality = metric.motion_quality(states, actions, tasks)
        if row.get('motion_quality') is not None and row['motion_quality'] != quality:
            raise ValueError('Recomputed frozen motion metric differs from recorded metric')
        episode = details['episodes'][0]['result']
        result.update(motion_quality=quality, task_result=episode,
                      task_error=episode.get('error'), recorded_seconds=len(tasks) * DT,
                      near_to_lift_or_recording_end=transition(tasks, 'near_grasp_region', 'ever_grasped'),
                      inside_to_success_or_recording_end=transition(tasks, 'inside_box', 'is_success'),
                      stages=stage_metrics(states, tasks, quality['thresholds']),
                      recording_sha256={name: sha(directory / name) for name in
                          ('run.json', 'server.json', 'summary.json', 'episode_000.npz', 'episode_000_task.jsonl')})
        timeout = bool(episode.get('truncated') or episode.get('local_step_limit_reached'))
        for key in ('near_to_lift_or_recording_end', 'inside_to_success_or_recording_end'):
            result[key]['timeout'] = result[key]['censored_at_recording_end'] and timeout
            result[key]['early_failure_or_abort'] = result[key]['censored_at_recording_end'] and not timeout
        result['near_to_lift_or_timeout_seconds'] = result['near_to_lift_or_recording_end']['seconds']
        result['inside_to_success_or_timeout_seconds'] = result['inside_to_success_or_recording_end']['seconds']
    except Exception as exc:
        result['analysis_error'] = f'{type(exc).__name__}: {exc}'
    return result


def aggregate(rows):
    valid = [r for r in rows if 'motion_quality' in r]
    scored = [r for r in valid if r['motion_quality']['coverage']['active_motion_scored']]
    qualities = [r['motion_quality'] for r in valid]
    stages = {}
    for phase in PHASES:
        reached = [r for r in valid if r['stages'][phase]['observed_seconds'] > 0]
        stages[phase] = {
            'attempts_observed': len(reached),
            'attempts_active': sum(r['stages'][phase]['active_motion_seconds'] > 0 for r in reached),
            'observed_seconds': distribution(r['stages'][phase]['observed_seconds'] for r in reached),
            'active_motion_seconds': distribution(r['stages'][phase]['active_motion_seconds'] for r in reached),
            'active_jerk_p99': distribution(r['stages'][phase]['active_frame_max_jerk_rad_s3']['p99'] for r in reached),
            'worst_physical_increment_rad': distribution(r['stages'][phase]['physical_joint_increment_rad']['max'] for r in reached),
            'threshold_dwell_total_seconds': distribution(sum(s['duration_seconds'] for s in r['motion_quality']['phase_pause_spans'][phase]) for r in reached),
            'threshold_dwell_max_seconds': distribution(max((s['duration_seconds'] for s in r['motion_quality']['phase_pause_spans'][phase]), default=0) for r in reached)}
    return {'attempts': len(rows), 'recordings_scored': len(valid),
            'infrastructure_or_analysis_failures': sum(bool(r.get('recorded_error') or r.get('analysis_error')) for r in rows),
            'task_errors': sum(bool(r.get('task_error')) for r in rows),
            'coverage': {name: sum(bool(q['coverage'][name]) for q in qualities) for name in
                         ('lifted_pinch', 'entered_box', 'transport_motion_comparable', 'active_motion_scored')},
            'primary': {
                'active_frame_max_jerk_p99_rad_s3': distribution(r['motion_quality']['active_frame_max_jerk_rad_s3']['p99'] for r in scored),
                'active_frame_max_jerk_peak_rad_s3': distribution(r['motion_quality']['active_frame_max_jerk_rad_s3']['max'] for r in scored),
                'worst_physical_increment_rad': distribution(q['physical_joint_increment_rad']['max'] for q in qualities),
                'threshold_dwell_total_seconds': distribution(q['low_motion_pause_total_seconds'] for q in qualities),
                'threshold_dwell_max_seconds': distribution(q['low_motion_pause_max_seconds'] for q in qualities),
                'near_to_lift_or_timeout_seconds': distribution(r['near_to_lift_or_timeout_seconds'] for r in valid),
                'inside_to_success_or_timeout_seconds': distribution(r['inside_to_success_or_timeout_seconds'] for r in valid)},
            'transition_coverage': {key: {'reached': sum(r[key]['reached'] for r in valid),
                                        'completed': sum(r[key]['completed'] for r in valid),
                                        'timeouts': sum(r[key]['timeout'] for r in valid),
                                        'early_failures_or_aborts': sum(r[key]['early_failure_or_abort'] for r in valid)}
                                    for key in ('near_to_lift_or_recording_end', 'inside_to_success_or_recording_end')},
            'stages': stages,
            'secondary': {'successes': sum(bool(r['task_result']['success']) for r in valid),
                          'attempt_denominator': len(rows)}}


def report(directory, root=ROOT):
    from pi0_repro.config import load_config
    ledger = json.loads((directory / 'comparison.json').read_text())
    plan = json.loads((directory / 'plan.json').read_text())
    if ledger['plan'] != plan or tuple(plan['layouts']) != LAYOUTS:
        raise ValueError('Evaluation ledger/plan mismatch')
    binding = baseline_binding(root)
    if binding != plan['binding']:
        raise ValueError('Baseline changed since evaluation plan was written')
    for name, digest in plan['scripts_sha256'].items():
        if sha(root / 'scripts' / name) != digest:
            raise ValueError(f'Analysis source changed since plan: {name}')
    for name, filename in plan['configurations'].items():
        if sha(root / 'configs' / filename) != plan['config_sha256'][name]:
            raise ValueError('Client configuration changed since evaluation plan')
    metric = load_module('frozen_metric_v9', root / 'src/pi0_repro/motion_quality.py')
    summary = load_module('frozen_execution_summary_v9', root / 'scripts/summarize_banana_execution.py')
    baseline = json.loads((root / BASELINE / 'comparison.json').read_text())
    control_rows = [r for r in baseline['results'] if r['profile'] == 'control']
    candidate_rows = ledger['results']
    expected = {(layout, repeat) for layout in LAYOUTS for repeat in (1, 2, 3)}
    keys = [(r['layout'], r['repeat']) for r in candidate_rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Unexpected or duplicated candidate attempts')
    if ledger['complete'] and set(keys) != expected:
        raise ValueError('Complete ledger does not contain all 18 attempts')
    groups = {name: [read_attempt(r, metric, summary, binding,
                                asdict(load_config(root / 'configs' / plan['configurations'][name]))) for r in rows]
              for name, rows in (('control', control_rows), ('v9', candidate_rows))}
    covered = {name: {(r['layout'], r['repeat']) for r in rows if
                      r.get('motion_quality', {}).get('coverage', {}).get('transport_motion_comparable')}
               for name, rows in groups.items()}
    common = covered['control'] & covered['v9']
    return {'schema': 'motion-v9-comparison-v1', 'complete': ledger['complete'],
            'expected_attempts_per_profile': 18, 'unattempted_v9': [
                {'layout': layout, 'repeat': repeat} for layout, repeat in sorted(expected - set(keys))],
            'fatal_error': ledger.get('fatal_error'), 'restore': ledger.get('restore'),
            'comparison_sha256': sha(directory / 'comparison.json'), 'plan_sha256': sha(directory / 'plan.json'),
            'groups': {name: {'all_attempts': aggregate(rows),
                             'transport_covered': aggregate([r for r in rows if (r['layout'], r['repeat']) in covered[name]]),
                             'common_transport_coverage_only': aggregate([r for r in rows if (r['layout'], r['repeat']) in common]),
                             'per_layout': {layout: aggregate([r for r in rows if r['layout'] == layout]) for layout in LAYOUTS}}
                       for name, rows in groups.items()},
            'attempts': groups,
            'scope': 'Fixed validation only, 3 repeats per layout; policy noise is not paired. All failures retained. '
                     'Primary motion metrics require stage/motion coverage beside them. Never reward stationary failures. '
                     'Transition end times include timeout or separately labelled early recording end; absent stages are null. '
                     'Threshold dwell can include deliberate holds; derivatives use simulation time, not a servo guarantee. '
                     'Success is secondary. No automatic winner, promotion, or success-rate gate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / OUTPUT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = report(args.directory)
    output = args.output or args.directory / 'summary.json'
    with output.open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')
    print(output)


if __name__ == '__main__':
    main()
