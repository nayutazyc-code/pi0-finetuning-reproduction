"""Read-only RTC10 tail audit with explicit pre-step state/post-step task clocks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from build_three_video_comparison import (find_font, load, require, selected_frames,
                                          sha, video_info, write_new)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pi0_repro.motion_quality import ARM_JOINTS, motion_quality

SOURCE = ROOT / 'outputs/v6_rtc10_replication_20261027'
OUTPUT = ROOT / 'outputs/v6_rtc10_tail_review_20261028'
DT = .02
ARRAYS = ('state', 'requested_action', 'blended_action', 'submitted_action', 'action')
FLAGS = ('near_grasp_region', 'grasped', 'ever_grasped', 'lifted', 'ever_lifted',
         'inside_box', 'released', 'supported_by_box', 'touching_table', 'stable', 'is_success')
JOINT_NAMES = {index + shift: f'vx300s_{side}/{name}' for side, shift in (('left', 0), ('right', 7))
               for index, name in enumerate(('waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'))}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def distribution(values):
    values = np.asarray(values, float)
    return None if not values.size else {'n': int(values.size), 'median': float(np.median(values)),
        'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)), 'max': float(values.max())}


def task_phase(task):
    return ('inside_before_success' if task.get('inside_box') else 'carrying_before_box' if task.get('ever_grasped')
            else 'near_grasp_before_lift' if task.get('near_grasp_region') else 'approach')


def physical_peak(states):
    q = np.asarray(states, float)[:, ARM_JOINTS]
    active = np.max(np.abs(np.diff(q, axis=0)), axis=1)[2:] / DT > .01
    jerk = np.abs(np.diff(q, n=3, axis=0)) / DT**3
    require(active.any(), 'No active-motion score; do not assign a stationary run a perfect score')
    offset, axis = np.unravel_index(np.argmax(np.where(active[:, None], jerk, -np.inf)), jerk.shape)
    return int(offset + 3), int(ARM_JOINTS[axis]), float(jerk[offset, axis]), jerk.max(axis=1), active


def select_runs(source):
    ledger = load(source / 'comparison.json')
    require(ledger.get('complete') is True and len(ledger['results']) == 18, 'Expected complete 18-run RTC10 ledger')
    require(sha(ROOT / 'src/pi0_repro/motion_quality.py') == ledger['plan']['metric_sha256'], 'Frozen metric changed')
    require({(r['layout'], r['repeat']) for r in ledger['results']} == {
        (layout, repeat) for layout in ledger['plan']['layouts'] for repeat in (1, 2, 3)}, 'Unexpected/missing repeats')
    population = []
    for row in ledger['results']:
        require(not row.get('infrastructure_error'), 'Retained infrastructure error cannot be hidden by selection')
        directory = Path(row['run_directory'])
        tasks = rows(directory / 'episode_000_task.jsonl')
        with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
            quality = motion_quality(data['state'], data['action'], tasks)
            step, axis, peak, _, _ = physical_peak(data['state'])
        require(quality == row['motion_quality'], f'Recomputed frozen metric differs: {directory}')
        require(peak == quality['active_frame_max_jerk_rad_s3']['max'], 'Peak derivative mismatch')
        population.append({'index': row['index'], 'layout': row['layout'], 'repeat': row['repeat'],
                           'run_directory': str(directory), 'active_jerk_max_rad_s3': peak,
                           'active_jerk_p99_rad_s3': quality['active_frame_max_jerk_rad_s3']['p99'],
                           'peak_state_row_and_video_frame': step, 'peak_sim_seconds': step * DT,
                           'peak_axis': axis, 'success_secondary': row['diagnostics']['episodes'][0]['result']['success'],
                           'coverage': quality['coverage']})
    by_max = sorted(population, key=lambda r: r['active_jerk_max_rad_s3'], reverse=True)
    by_p99 = sorted(population, key=lambda r: r['active_jerk_p99_rad_s3'], reverse=True)
    selected = by_max[:3]
    if by_p99[0]['index'] not in {r['index'] for r in selected}:
        selected.append(by_p99[0])
    return ledger, by_max, by_p99, selected


def amplitude(array, axis):
    q = np.asarray(array, float)
    delta = np.diff(q[:, ARM_JOINTS], axis=0)
    focus = q[:, axis]
    return {'axis_range_rad': float(np.ptp(focus)), 'axis_net_displacement_rad': float(focus[-1] - focus[0]),
            'axis_range_degrees': float(np.rad2deg(np.ptp(focus))),
            'axis_max_single_increment_rad': float(np.max(np.abs(np.diff(focus)))),
            'axis_max_single_increment_degrees': float(np.rad2deg(np.max(np.abs(np.diff(focus))))),
            'frame_max_arm_increment_rad': distribution(np.max(np.abs(delta), axis=1)),
            'frame_max_arm_speed_rad_s': distribution(np.max(np.abs(delta), axis=1) / DT),
            'frame_max_arm_acceleration_rad_s2': distribution(np.max(np.abs(np.diff(q[:, ARM_JOINTS], n=2, axis=0)), axis=1) / DT**2),
            'frame_max_arm_jerk_rad_s3': distribution(np.max(np.abs(np.diff(q[:, ARM_JOINTS], n=3, axis=0)), axis=1) / DT**3)}


def four_frame(array, start, axis):
    q = np.asarray(array[start:start + 4], float)
    return {**amplitude(q, axis), 'record_indices': list(range(start, start + 4)),
            'axis_values_rad': q[:, axis].tolist(), 'axis_increments_rad': np.diff(q[:, axis]).tolist(),
            'axis_signed_jerk_rad_s3': float(np.diff(q[:, axis], n=3)[0] / DT**3)}


def nearest(events, peak_frame):
    if not events:
        return None
    event = min(events, key=lambda e: abs(e['video_frame'] - peak_frame))
    return {**event, 'signed_seconds_from_physical_peak': (event['video_frame'] - peak_frame) * DT}


def analyze(row, source_row):
    directory = Path(row['run_directory'])
    with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    tasks = rows(directory / 'episode_000_task.jsonl')
    controls, latency = rows(directory / 'episode_000_control.jsonl'), rows(directory / 'episode_000_latency.jsonl')
    s, axis, peak, frame_jerk, active = physical_peak(arrays['state'])
    lo, hi = max(0, s - 25), min(len(tasks) - 1, s + 25)
    switches = np.flatnonzero(np.diff(arrays['action_prediction_index']) != 0) + 1
    switch_events = []
    for index in switches:
        prediction = int(arrays['action_prediction_index'][index])
        switch_events.append({'video_frame': int(index), 'control_step_index': int(index),
                              'episode_time_seconds': int(index) * DT, 'prediction_index': prediction,
                              'prediction_origin_step': int(arrays['prediction_steps'][prediction]),
                              'client_blend_length_steps': int(arrays['chunk_transition_steps'][prediction])})
    flag_events = {}
    for flag in FLAGS:
        values = np.asarray([bool(t[flag]) for t in tasks], np.int8)
        flag_events[flag] = [{'task_row_index': int(i), 'video_frame': int(i + 1),
                             'episode_time_seconds': (int(i) + 1) * DT,
                             'before': bool(values[i - 1]), 'after': bool(values[i])}
                            for i in np.flatnonzero(np.diff(values) != 0) + 1]
    previous_commands = {name: four_frame(arrays[name], s - 4, axis) for name in ARRAYS if name != 'state'}
    target = arrays['action'].astype(float)[s - 4:s]
    physical = arrays['state'].astype(float)[s - 3:s + 1]
    residual = physical - target
    local_control = [r for r in controls if lo <= r['step'] <= hi]
    causal_control = [r for r in controls if s - 4 <= r['step'] <= s - 1]
    local_latency = [r for r in latency if lo - 25 <= r['received_at_step'] <= hi + 25]
    prefix_errors = [(r.get('rtc_request') or {}).get('planned_vs_executed_prefix_max_error') for r in latency]
    prefix_values = [v for v in prefix_errors if v is not None]
    task_at_peak = tasks[s - 1]
    increment = np.max(np.abs(np.diff(arrays['state'].astype(float)[:, ARM_JOINTS], axis=0)), axis=1)
    active_values = frame_jerk[active]
    own_p99 = source_row['motion_quality']['active_frame_max_jerk_rad_s3']['p99']
    tail_frames = np.flatnonzero(active & (frame_jerk >= own_p99)) + 3
    bins = {}
    for index in tail_frames:
        phase = task_phase(tasks[index - 1])
        bins[phase] = bins.get(phase, 0) + 1
    video_path = directory / 'episode_000.mp4'
    result = {
        **row, 'source_sha256': {name: sha(directory / name) for name in
            ('episode_000.npz', 'episode_000.mp4', 'episode_000_task.jsonl', 'episode_000_control.jsonl',
             'episode_000_latency.jsonl', 'run.json', 'server.json', 'summary.json')},
        'source_video': str(video_path), 'source_code_receipt': load(directory / 'run.json')['source_code_sha256'],
        'peak': {'state_row_index': s, 'mp4_frame_index': s, 'episode_sim_seconds': s * DT,
                 'last_completed_action_step': s - 1, 'task_row_at_same_physical_time': s - 1,
                 'physics_time_including_reset_settle': task_at_peak['simulation_time_seconds'],
                 'axis': axis, 'joint_name': JOINT_NAMES[axis], 'active_jerk_rad_s3': peak,
                 'frame_max_physical_increment_rad': float(increment[s - 1]),
                 'phase_at_peak': task_phase(task_at_peak), 'task_at_peak': task_at_peak},
        'same_index_four_frame_derivatives': {name: four_frame(arrays[name], s - 3, axis) for name in ARRAYS},
        'preceding_commands_for_same_physical_sample_times': previous_commands,
        'aligned_tracking_residual': {'definition': 'state[i]-action[i-1], pre-step physical sample vs target applied in the preceding interval.',
            'state_indices': list(range(s - 3, s + 1)), 'action_indices': list(range(s - 4, s)),
            'axis_residual_rad': residual[:, axis].tolist(),
            'frame_max_arm_abs_residual_rad': np.max(np.abs(residual[:, ARM_JOINTS]), axis=1).tolist()},
        'plusminus_half_second_window': {'state_start_index': lo, 'state_end_index_inclusive': hi,
            'start_seconds': lo * DT, 'end_seconds': hi * DT,
            'amplitudes': {name: amplitude(arrays[name][lo:hi + 1], axis) for name in ARRAYS},
            'rows': [{'state_row_and_video_frame': i, 'episode_seconds': i * DT,
                      'axis_values': {name: float(arrays[name][i, axis]) for name in ARRAYS},
                      'preceding_executed_target_axis': float(arrays['action'][i - 1, axis]),
                      'right_gripper_state': float(arrays['state'][i, 13]),
                      'preceding_right_gripper_target': float(arrays['action'][i - 1, 13]),
                      'task_at_same_physical_time': {flag: tasks[i - 1][flag] for flag in FLAGS}}
                     for i in range(lo, hi + 1)]},
        'nearest_chunk_switch': nearest(switch_events, s),
        'nearest_prior_chunk_switch': nearest([e for e in switch_events if e['video_frame'] <= s], s),
        'nearest_future_chunk_switch': nearest([e for e in switch_events if e['video_frame'] > s], s),
        'chunk_switches_within_half_second': [e for e in switch_events if abs(e['video_frame'] - s) <= 25],
        'nearest_banana_robot_contact_change': nearest(flag_events['released'], s),
        'nearest_task_flag_changes': {flag: nearest(events, s) for flag, events in flag_events.items()},
        'task_flag_changes_within_half_second': {flag: [e for e in events if abs(e['video_frame'] - s) <= 25]
                                                for flag, events in flag_events.items()},
        'timing_and_prefix': {'local_control_rows': local_control,
            'causal_four_preceding_control_intervals': causal_control, 'nearby_request_responses': local_latency,
            'local_action_apply_interval_ms': distribution(np.diff([r['action_apply_wall_seconds'] for r in local_control]) * 1000),
            'local_round_trip_ms': distribution([r['round_trip_ms'] for r in local_latency]),
            'local_blocking_seconds': sum(r['inference_blocked_seconds'] for r in local_control),
            'all_run_prefix_forecast_error_max': max(prefix_values) if prefix_values else None,
            'all_run_prefix_error_receipts': len(prefix_values), 'response_receipts': len(latency),
            'all_run_execution': source_row['diagnostics']['episodes'][0]['result']['execution'],
            'declared_rtc_blend_steps': source_row['diagnostics']['config']['simulation']['rtc_blend_steps'],
            'client_chunk_blend_steps': source_row['diagnostics']['config']['simulation']['chunk_blend_steps']},
        'full_run_context': {'frozen_quality': source_row['motion_quality'],
            'task_progress': source_row['diagnostics']['episodes'][0]['task_progress'],
            'task_result': source_row['diagnostics']['episodes'][0]['result'],
            'own_p99_exceedance_count': len(tail_frames), 'own_p99_tail_phase_counts': bins,
            'active_jerk_ge_300_count': int(np.sum(active_values >= 300)),
            'active_jerk_ge_600_count': int(np.sum(active_values >= 600)),
            'own_p99_tail_state_frame_indices': tail_frames.tolist()},
    }
    return result


def contact_sheet(result, output, font_path):
    from PIL import Image, ImageDraw, ImageFont
    video = Path(result['source_video'])
    info = video_info(video)
    require(info['fps'] == 50 and info['frame_count'] == result['full_run_context']['task_result']['steps'] + 1,
            'Source video is not complete original 50fps recording')
    s = result['peak']['mp4_frame_index']
    offsets = (-25, -5, -2, -1, 0, 1, 5, 25)
    indices = [max(0, min(info['frame_count'] - 1, s + offset)) for offset in offsets]
    roi_indices = [max(0, min(info['frame_count'] - 1, s + offset)) for offset in (-5, -2, 0, 5)]
    frames = selected_frames(video, set(indices + roi_indices))
    width, height = 2048, 1430
    canvas = Image.new('RGB', (width, height), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(font_path), 19)
    small = ImageFont.truetype(str(font_path), 17)
    title = f"{result['layout']} repeat {result['repeat']} | peak {result['active_jerk_max_rad_s3']:.2f} | p99 {result['active_jerk_p99_rad_s3']:.2f} rad/s^3"
    draw.text((12, 8), title, fill='black', font=ImageFont.truetype(str(font_path), 26))
    draw.text((12, 43), f"Physical state/frame {s}, t={s*DT:.2f}s, {result['peak']['joint_name']}; task row {s-1} is at the same physical time.", fill='black', font=font)
    tasks = rows(Path(result['run_directory']) / 'episode_000_task.jsonl')
    evidence = []
    for position, index in enumerate(indices):
        x, y = (position % 4) * 512, 80 + (position // 4) * 450
        task = tasks[index - 1] if index else {}
        draw.text((x + 8, y), f'frame {index} | t={index*DT:.2f}s | offset {(index-s)*DT:+.2f}s', font=font,
                  fill=(170, 20, 20) if index == s else 'black')
        draw.text((x + 8, y + 26), f"banana-contact={int(not task.get('released', True))} box={int(bool(task.get('inside_box')))} stable={int(bool(task.get('stable')))}", font=small, fill='black')
        canvas.paste(Image.fromarray(frames[index]).resize((512, 384), Image.Resampling.LANCZOS), (x, y + 55))
        evidence.append({'mp4_frame_index': index, 'episode_sim_seconds': index * DT,
                         'offset_from_physical_peak_seconds': (index - s) * DT,
                         'task_row_index_at_same_time': index - 1 if index else None})
    roi = (275, 110, 540, 345)
    for position, index in enumerate(roi_indices):
        x, y = position * 512, 990
        draw.text((x + 8, y), f'ROI frame {index} | offset {(index-s)*DT:+.2f}s', font=font, fill='black')
        crop = Image.fromarray(frames[index]).crop(roi).resize((448, 397), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x + 32, y + 30))
    path = output / f"tail_{result['index']:02d}_{result['layout']}_repeat{result['repeat']}.jpg"
    canvas.save(path, quality=93)
    return {'source_video_info': info, 'contactsheet': str(path), 'contactsheet_sha256': sha(path),
            'extracted_frames': evidence, 'roi_xyxy': list(roi), 'visual_review': 'Pending direct view_image review.'}


def short_clip(result, output):
    import imageio_ffmpeg
    video = Path(result['source_video'])
    info = result['media']['source_video_info']
    peak = result['peak']['mp4_frame_index']
    start, end = max(0, peak - 25), min(info['frame_count'] - 1, peak + 25)
    count = end - start + 1
    path = output / f"tail_{result['index']:02d}_{result['layout']}_repeat{result['repeat']}_1x.mp4"
    log = path.with_suffix('.encode.log')
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n', '-threads', '1', '-i', str(video),
               '-vf', f'trim=start_frame={start}:end_frame={end+1},settb=expr=1/50,setpts=N',
               '-frames:v', str(count), '-r', '50', '-fps_mode', 'cfr', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '18', '-threads', '1', '-filter_threads', '1', str(path)]
    with log.open('xb') as file:
        process = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)
    require(process.returncode == 0, 'Clip encoding failed; preserve log')
    require('non-monoton' not in log.read_text().lower(), 'Clip timestamps are not monotonic')
    decoded = video_info(path)
    require(decoded['fps'] == 50 and decoded['frame_count'] == count, 'Clip fps/frame count changed')
    chosen = [0, count // 2, count - 1]
    actual, original = selected_frames(path, chosen), selected_frames(video, {start + i for i in chosen})
    errors = [float(np.mean(np.abs(actual[i].astype(float) - original[start + i].astype(float)))) for i in chosen]
    require(max(errors) < 8, 'Clip frame alignment does not match source')
    return {'path': str(path), 'sha256': sha(path), **decoded, 'source_first_frame': start, 'source_last_frame': end,
            'source_episode_start_seconds': start * DT, 'source_episode_end_seconds': end * DT,
            'scope': 'Derived short window only, original speed 50fps; full unmodified source remains available.',
            'first_middle_last_mean_absolute_rgb_error': errors, 'command': command}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--clips', action='store_true', help='Also encode original-speed short windows')
    parser.add_argument('--check-only', action='store_true', help='Recompute/rank metrics without decoding media or writing outputs')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    ledger, by_max, by_p99, selected = select_runs(source)
    if args.check_only:
        print(json.dumps({'complete': ledger['complete'], 'all_18_frozen_metrics_match': True, 'selected': selected}, indent=2))
        return
    output.mkdir(exist_ok=False)
    report = {'schema': 'rtc10-tail-review-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'builder_sha256': sha(Path(__file__)), 'helper_sha256': sha(ROOT / 'scripts/build_three_video_comparison.py'),
              'source_comparison_sha256': sha(source / 'comparison.json'),
              'metric_sha256': ledger['plan']['metric_sha256'], 'all_18_recomputed_frozen_metrics_match': True,
              'ranked_by_maximum': by_max, 'ranked_by_p99': by_p99, 'selected_events': [],
              'selection': 'Three runs with largest physical active-frame-max jerk, plus the worst-p99 run if distinct. All 18 remain in rankings.',
              'time_alignment': 'State[i] and MP4 frame i are pre-step at episode t=i*.02; task[i] is post-step '
                                'at t=(i+1)*.02 and corresponds to frame i+1. At physical peak state[s], task[s-1] '
                                'and the preceding applied action[s-1] are aligned. No old one-frame-forward convention is used.',
              'limits': ['Jerk is a third finite difference of float32-recorded physical positions; also report sub-degree increments, range and command context.',
                         'Same-index command derivatives and preceding-command derivatives are separate; equal record indices are not post-step tracking residuals.',
                         'released indicates banana contact with any robot geometry; it is not a force measurement or a unique cause.',
                         'No robot torque, contact force, solver impulse, or complete robot-box/self/table contact records are available.',
                         'Chunk changes are frequent and blends overlap. Event proximity is descriptive association, not causal identification.',
                         'Short MP4 windows preserve simulation-time speed and do not depict inference wall-clock waiting. No new policy/physics run or source change.']}
    try:
        font = find_font(None)
        source_rows = {r['index']: r for r in ledger['results']}
        for row in selected:
            result = analyze(row, source_rows[row['index']])
            report['selected_events'].append(result)
            result['selection_reason'] = 'top_three_maximum' if row['index'] in {r['index'] for r in by_max[:3]} else 'additional_worst_p99'
            result['media'] = contact_sheet(result, output, font)
            if args.clips:
                result['media']['clip'] = short_clip(result, output)
        require(sha(source / 'comparison.json') == report['source_comparison_sha256'], 'Source ledger changed during review')
        for result in report['selected_events']:
            for name, digest in result['source_sha256'].items():
                require(sha(Path(result['run_directory']) / name) == digest, 'Source evidence changed during review')
        write_new(output / 'summary.json', report)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        write_new(output / 'failure.json', report)
        raise
    print(json.dumps({'out': str(output), 'reviewed_run_count': len(selected),
                      'contact_sheets': [r['media']['contactsheet'] for r in report['selected_events']]}, indent=2))


if __name__ == '__main__':
    main()
