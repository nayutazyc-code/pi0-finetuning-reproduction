"""All-outcome media for the complete same-session RTC20/RTC10 comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from analyze_rtc_pair import metadata_valid, validate_cases
from build_three_video_comparison import (find_font, load, require, selected_frames,
                                          sha, video_info, write_new)
from build_v6_motion_review import event_frames, fit_font
from review_rtc10_tails import JOINT_NAMES, four_frame, nearest, physical_peak, rows
from pi0_repro.motion_quality import motion_quality

PROFILES = ('rtc20', 'rtc10')
LAYOUTS = ('episode_004', 'episode_012', 'episode_019', 'episode_020', 'episode_026', 'episode_029')
FPS, HEADER = 50, 96
DEFAULT_OUT = ROOT / 'outputs/v6_rtc_same_session_20261028'


def validate(out):
    """Require all outcomes and matching endpoint/source/scene evidence before media."""
    plan, ledger = load(out / 'plan.json'), load(out / 'comparison.json')
    source_rows = validate_cases(plan, ledger)
    require(plan == ledger['plan'] and not ledger.get('error'), 'Ledger/plan mismatch or recorded fatal error')
    before, after = ledger.get('endpoint_before') or {}, ledger.get('endpoint_after') or {}
    require(before.get('metadata_verified') is True and after.get('metadata_verified') is True,
            'Both endpoint verifications must be complete')
    require(before['service_identity'] == after['service_identity'] and before['metadata'] == after['metadata']
            and metadata_valid(before['metadata']), 'Service process or endpoint metadata changed')
    for name, digest in plan['source_sha256'].items():
        require(sha(ROOT / 'src/pi0_repro' / name) == digest, 'Bound execution source changed: ' + name)
    require(sha(ROOT / 'src/pi0_repro/motion_quality.py') == plan['metric_sha256'], 'Frozen metric changed')
    require(sha(out / 'runner_source.py') == plan['script_sha256'], 'Recorded runner source differs')
    for profile, name in plan['profiles'].items():
        require(sha(ROOT / 'configs' / name) == plan['profile_sha256'][profile], 'Bound profile changed')
    attempts, scenes, xmls, reference_config = [], {}, {}, None
    for row in source_rows:
        require(not row.get('infrastructure_error'), 'Retained infrastructure failure prevents complete media; do not substitute another run')
        directory = Path(row['run_directory'])
        video = Path(row['video'])
        require(video.resolve() == (directory / 'episode_000.mp4').resolve()
                and sha(video) == row['video_sha256'], 'Source video path/hash differs from ledger')
        run = load(directory / 'run.json')
        config = run['config']
        require(run['source_code_sha256'] and all(plan['source_sha256'].get(name) == digest
                for name, digest in run['source_code_sha256'].items()), 'Recorded source receipts differ')
        require(config == row['diagnostics']['config'], 'Recorded execution config differs from diagnostics')
        metadata = load(directory / 'server.json')
        require(metadata == before['metadata'] and metadata_valid(metadata), 'Run endpoint differs from verified shared service')
        episode = load(directory / 'summary.json')['episodes'][0]
        require(episode == row['diagnostics']['episodes'][0]['result'], 'Recorded task result differs')
        require(episode['seed'] == config['simulation']['seed'] == row['reset_seed'], 'Reset seed differs')
        require(config['simulation']['video_fps'] == FPS and config['simulation']['display'] is False,
                'Expected headless original 50fps recording')
        condition = 20 if row['profile'] == 'rtc20' else 10
        require(config['simulation']['rtc_blend_steps'] == condition, 'RTC condition differs from row profile')
        normalized = json.loads(json.dumps(config))
        normalized['simulation']['rtc_blend_steps'] = 20
        for key in ('scene_config', 'seed'):
            normalized['simulation'].pop(key)
        if reference_config is None:
            reference_config = normalized
        require(normalized == reference_config, 'Unplanned execution difference across profiles/repeats')
        scene = Path(config['simulation']['scene_config'])
        require(scene.resolve() == (out / (row['layout'] + '.toml')).resolve()
                and sha(scene) == row['scene_sha256'], 'Scene path/hash differs')
        require(scenes.setdefault(row['layout'], row['scene_sha256']) == row['scene_sha256'],
                'Same layout has differing scene configuration')
        xml_digest = sha(directory / 'scene.xml')
        require(xmls.setdefault(row['layout'], xml_digest) == xml_digest, 'Same layout has differing physical scene XML')
        tasks = rows(directory / 'episode_000_task.jsonl')
        with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
            states, actions = data['state'], data['action']
        require(len(tasks) == episode['steps'], 'Task recording is incomplete')
        quality = motion_quality(states, actions, tasks)
        require(quality == row['motion_quality'], 'Recomputed frozen metric differs')
        require(row.get('prefix_error') == 0 and episode['execution']['inference_underruns'] == 0,
                'Prefix integrity or inference underrun violated the frozen protocol')
        events = event_frames(tasks, states, actions)
        attempts.append({'index': row['index'], 'profile': row['profile'], 'condition_rtc_taper_steps': condition,
                         'layout': row['layout'], 'repeat': row['repeat'], 'reset_seed': row['reset_seed'],
                         'success': bool(episode['success']), 'task_error': episode.get('error'),
                         'steps': episode['steps'], 'episode_sim_seconds': episode['steps'] / FPS,
                         'source_video': str(video.resolve()), 'source_video_sha256': row['video_sha256'],
                         'run_directory': str(directory), 'events': events, 'motion_quality': quality,
                         'source_receipts_sha256': {name: sha(directory / name) for name in
                             ('run.json', 'server.json', 'summary.json', 'scene.xml', 'episode_000.npz', 'episode_000_task.jsonl')}})
    return ledger, attempts


def overview(layout, attempts, media, font_path):
    from PIL import Image, ImageDraw, ImageFont
    columns = (('near_grasp', 'Near grasp'), ('first_lifted_pinch', 'First lifted pinch'),
               ('first_box_entry', 'First box entry'), ('final', 'Final outcome'))
    width, height, row_height, top = 400, 300, 380, 76
    canvas = Image.new('RGB', (4 * width, top + 6 * row_height), (244, 246, 249))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{layout} | all 6 outcomes | RTC20 / RTC10 x 3 repeats',
              font=ImageFont.truetype(str(font_path), 27), fill='black')
    draw.text((12, 44), 'Event-aligned thumbnails; missing events are never substituted. Task row i -> MP4 frame i+1. Simulation time.',
              font=ImageFont.truetype(str(font_path), 18), fill=(45, 45, 45))
    cells = []
    for row_index, (repeat, profile) in enumerate((r, p) for r in (1, 2, 3) for p in PROFILES):
        attempt = next(a for a in attempts if (a['layout'], a['repeat'], a['profile']) == (layout, repeat, profile))
        events = attempt['events']
        selected = {events[key]['video_frame'] for key, _ in columns if events[key] is not None}
        frames = selected_frames(Path(attempt['source_video']), selected)
        y = top + row_index * row_height
        outcome = 'SUCCESS' if attempt['success'] else 'FAILURE'
        text = f'Repeat {repeat} | {profile.upper()} taper={attempt["condition_rtc_taper_steps"]} | {outcome} | {attempt["episode_sim_seconds"]:.2f} sim s | trial {attempt["index"]}'
        font, _ = fit_font(draw, text, font_path, 23, canvas.width - 24)
        draw.text((12, y), text, font=font, fill=(25, 100, 45) if attempt['success'] else (155, 35, 35))
        for col, (key, label) in enumerate(columns):
            x = col * width
            event = events[key]
            draw.text((x + 8, y + 30), label, font=ImageFont.truetype(str(font_path), 19), fill='black')
            if event is None:
                draw.rectangle((x + 2, y + 74, x + width - 2, y + 374), fill=(210, 214, 220))
                draw.text((x + 30, y + 170), 'MISSING: event not reached',
                          font=ImageFont.truetype(str(font_path), 23), fill=(130, 40, 40))
            else:
                draw.text((x + 8, y + 53), f'frame {event["video_frame"]} | t={event["episode_time_seconds"]:.2f}s',
                          font=ImageFont.truetype(str(font_path), 17), fill=(40, 40, 40))
                frame = Image.fromarray(frames[event['video_frame']]).resize((width, height), Image.Resampling.LANCZOS)
                canvas.paste(frame, (x, y + 74))
            cells.append({'index': attempt['index'], 'profile': profile, 'repeat': repeat, 'event': key, 'selection': event})
    path = media / f'{layout}_all_six_outcomes.jpg'
    canvas.save(path, quality=93)
    return {'layout': layout, 'path': str(path), 'sha256': sha(path), 'cells': cells,
            'scope': 'All six prespecified outcomes, event-aligned views; no first-success selection.'}


def choose_peaks(attempts):
    chosen = []
    for profile in PROFILES:
        scored = [a for a in attempts if a['profile'] == profile and a['motion_quality']['active_frame_max_jerk_rad_s3'] is not None]
        for criterion, field in (('largest_physical_maximum', 'max'), ('worst_physical_p99', 'p99')):
            if not scored:
                continue
            attempt = max(scored, key=lambda a: a['motion_quality']['active_frame_max_jerk_rad_s3'][field])
            prior = next((entry for entry in chosen if entry['index'] == attempt['index']), None)
            if prior:
                prior['selection_reasons'].append(criterion)
            else:
                chosen.append({**attempt, 'selection_reasons': [criterion]})
    return chosen


def peak_sheet(attempt, media, font_path):
    from PIL import Image, ImageDraw, ImageFont
    with np.load(Path(attempt['run_directory']) / 'episode_000.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    tasks = rows(Path(attempt['run_directory']) / 'episode_000_task.jsonl')
    peak, axis, value, _, _ = physical_peak(arrays['state'])
    info = attempt['video']
    offsets = (-25, -5, -2, -1, 0, 1, 5, 25)
    indices = [max(0, min(info['frame_count'] - 1, peak + offset)) for offset in offsets]
    roi_indices = [max(0, min(info['frame_count'] - 1, peak + offset)) for offset in (-5, -2, 0, 5)]
    frames = selected_frames(Path(attempt['source_video']), set(indices + roi_indices))
    canvas = Image.new('RGB', (2048, 1430), (245, 246, 249))
    draw = ImageDraw.Draw(canvas)
    title = f'{attempt["profile"].upper()} | {attempt["layout"]} r{attempt["repeat"]} | max={value:.2f} | p99={attempt["motion_quality"]["active_frame_max_jerk_rad_s3"]["p99"]:.2f} rad/s^3'
    draw.text((12, 8), title, fill='black', font=ImageFont.truetype(str(font_path), 27))
    draw.text((12, 44), f'{JOINT_NAMES[axis]} | pre-step state/frame {peak}, t={peak/FPS:.2f}s | aligned task row {peak-1}',
              fill='black', font=ImageFont.truetype(str(font_path), 19))
    evidence = []
    for position, frame in enumerate(indices):
        x, y = (position % 4) * 512, 80 + (position // 4) * 450
        task = tasks[frame - 1] if frame else {}
        draw.text((x + 8, y), f'frame {frame} | t={frame/FPS:.2f}s | offset {(frame-peak)/FPS:+.2f}s',
                  font=ImageFont.truetype(str(font_path), 19), fill=(170, 30, 30) if frame == peak else 'black')
        draw.text((x + 8, y + 27), f'banana-contact={int(not task.get("released", True))} box={int(bool(task.get("inside_box")))}',
                  font=ImageFont.truetype(str(font_path), 17), fill='black')
        canvas.paste(Image.fromarray(frames[frame]).resize((512, 384), Image.Resampling.LANCZOS), (x, y + 55))
        evidence.append({'video_frame': frame, 'episode_seconds': frame / FPS,
                         'aligned_task_row': frame - 1 if frame else None})
    roi = (275, 110, 540, 345)
    for position, frame in enumerate(roi_indices):
        x, y = position * 512, 990
        draw.text((x + 8, y), f'ROI frame {frame} | offset {(frame-peak)/FPS:+.2f}s',
                  font=ImageFont.truetype(str(font_path), 19), fill='black')
        crop = Image.fromarray(frames[frame]).crop(roi).resize((448, 397), Image.Resampling.LANCZOS)
        canvas.paste(crop, (x + 32, y + 30))
    path = media / f'{attempt["profile"]}_peak_{attempt["index"]:02d}_{attempt["layout"]}_r{attempt["repeat"]}.jpg'
    canvas.save(path, quality=93)
    switches = np.flatnonzero(np.diff(arrays['action_prediction_index']) != 0) + 1
    contacts = np.flatnonzero(np.diff(np.asarray([task['released'] for task in tasks], np.int8)) != 0) + 2
    return {'index': attempt['index'], 'profile': attempt['profile'], 'layout': attempt['layout'], 'repeat': attempt['repeat'],
            'selection_reasons': attempt['selection_reasons'], 'path': str(path), 'sha256': sha(path),
            'physical_peak': {'state_row_and_video_frame': peak, 'episode_seconds': peak / FPS,
                              'axis': axis, 'joint': JOINT_NAMES[axis], 'jerk_rad_s3': value},
            'same_index_four_frame_derivatives': {name: four_frame(arrays[name], peak - 3, axis) for name in
                ('state', 'requested_action', 'blended_action', 'submitted_action', 'action')},
            'preceding_applied_target_four_frames': four_frame(arrays['action'], peak - 4, axis) if peak >= 4 else None,
            'nearest_chunk_switch': nearest([{'video_frame': int(frame)} for frame in switches], peak),
            'nearest_banana_contact_change': nearest([{'video_frame': int(frame)} for frame in contacts], peak),
            'frames': evidence, 'roi_xyxy': list(roi),
            'scope': 'Pre-step physical peak correctly aligned to frame/state i and task i-1. Proximity is not a unique causal claim.'}


def side_by_side(attempts, media, font_path):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    selected = [next(a for a in attempts if (a['layout'], a['repeat'], a['profile']) == ('episode_019', 1, p)) for p in PROFILES]
    sizes = {(a['video']['width'], a['video']['height']) for a in selected}
    require(len(sizes) == 1, 'Source geometry differs; no resizing task video')
    width, height = sizes.pop()
    maximum = max(a['video']['frame_count'] for a in selected)
    header = Image.new('RGB', (width * 2, HEADER), (21, 27, 38))
    draw = ImageDraw.Draw(header)
    labels = []
    for column, attempt in enumerate(selected):
        y = 7
        status = 'SUCCESS' if attempt['success'] else 'FAILURE'
        extra = maximum - attempt['video']['frame_count']
        lines = [(f'{attempt["profile"].upper()} | taper {attempt["condition_rtc_taper_steps"]} | repeat 1', 27),
                 (f'{status} | end {attempt["episode_sim_seconds"]:.2f} sim s | 1x, 50fps', 21),
                 (f'Final frame frozen after {attempt["video"]["duration_seconds"]:.2f}s' if extra else 'Full original duration | no terminal padding', 19)]
        for text, size in lines:
            font, bounds = fit_font(draw, text, font_path, size, width - 24)
            x = column * width + 12
            draw.text((x, y - bounds[1]), text, fill=(240, 244, 250), font=font)
            labels.append({'profile': attempt['profile'], 'text': text,
                           'bounds_xyxy': [x, y, x + bounds[2] - bounds[0], y + bounds[3] - bounds[1]]})
            y += bounds[3] - bounds[1] + 5
        require(y <= HEADER, 'Header text would be cropped')
    header_path = media / 'episode_019_r1_header.png'
    header.save(header_path)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n', '-filter_threads', '1', '-filter_complex_threads', '1']
    for attempt in selected:
        command += ['-threads', '1', '-i', attempt['source_video']]
    command += ['-threads', '1', '-loop', '1', '-framerate', str(FPS), '-i', str(header_path)]
    filters, freezes = [], {}
    for column, attempt in enumerate(selected):
        extra = maximum - attempt['video']['frame_count']
        freezes[attempt['profile']] = extra
        filters.append(f'[{column}:v]tpad=stop_mode=clone:stop={extra},trim=end_frame={maximum},'
                       f'settb=expr=1/{FPS},setpts=N,pad=iw:ih+{HEADER}:0:{HEADER}:black[v{column}]')
    filters += ['[v0][v1]hstack=inputs=2[body]', f'[2:v]settb=expr=1/{FPS},setpts=N[header]',
                f'[body][header]overlay=0:0:shortest=1,settb=expr=1/{FPS},setpts=N,format=yuv420p[result]']
    temporary = media / '.episode_019_repeat1.building.mp4'
    command += ['-filter_complex', ';'.join(filters), '-map', '[result]', '-an', '-frames:v', str(maximum),
                '-r', str(FPS), '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-threads', '1', '-movflags', '+faststart', str(temporary)]
    with (media / 'encode.log').open('xb') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    require(completed.returncode == 0 and 'non-monoton' not in (media / 'encode.log').read_text().lower(), 'Encoding failed or timestamps invalid; preserve log')
    info = video_info(temporary)
    require(info['fps'] == FPS and info['frame_count'] == maximum
            and (info['width'], info['height']) == (2 * width, height + HEADER), 'Combined media dimensions/fps/count differ')
    destination = media / 'episode_019_repeat1_all_outcomes.mp4'
    temporary.rename(destination)
    indices = sorted({0, maximum // 2, min(a['video']['frame_count'] for a in selected) - 1, maximum - 1})
    frames = selected_frames(destination, indices)
    originals = {a['profile']: selected_frames(Path(a['source_video']),
                 {min(i, a['video']['frame_count'] - 1) for i in indices}) for a in selected}
    checks, previews = [], []
    for index in indices:
        frame = frames[index]
        preview = media / f'episode_019_r1_qa_{index:06d}.png'
        Image.fromarray(frame).save(preview)
        previews.append({'frame_index': index, 'timestamp_seconds': index / FPS, 'path': str(preview), 'sha256': sha(preview)})
        for column, attempt in enumerate(selected):
            source_index = min(index, attempt['video']['frame_count'] - 1)
            actual = frame[HEADER:HEADER + height, column * width:(column + 1) * width].astype(float)
            expected = originals[attempt['profile']][source_index].astype(float)
            error = float(np.mean(np.abs(actual - expected)))
            require(error < 8, 'Combined frame does not match aligned original source')
            checks.append({'profile': attempt['profile'], 'combined_frame': index, 'source_frame': source_index,
                           'mean_absolute_rgb_error_0_255': error})
    sheet = Image.new('RGB', (width * 2, len(indices) * (height + HEADER + 30)), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        y = position * (height + HEADER + 30)
        draw.text((10, y + 5), f'Combined frame {index} | t={index/FPS:.2f}s',
                  fill='black', font=ImageFont.truetype(str(font_path), 18))
        sheet.paste(Image.fromarray(frames[index]), (0, y + 30))
    qa = media / 'episode_019_r1_alignment_qa.jpg'
    sheet.save(qa, quality=93)
    return {'path': str(destination), 'sha256': sha(destination), **info,
            'fixed_selection': 'episode_019 repeat1, both profiles regardless of outcome',
            'left_to_right': list(PROFILES), 'original_trial_indices': [a['index'] for a in selected],
            'freeze_frames': freezes, 'header_labels': labels, 'header_sha256': sha(header_path), 'command': command,
            'qa': {'preview_frames': previews, 'alignment_checks': checks, 'contactsheet': str(qa), 'contactsheet_sha256': sha(qa)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check-only', action='store_true', help='Validate complete evidence without decoding/encoding or writing media')
    args = parser.parse_args()
    out = args.out.resolve()
    ledger, attempts = validate(out)
    selected_peaks = choose_peaks(attempts)
    if args.check_only:
        print(json.dumps({'validated_attempts': len(attempts), 'selected_peak_trial_indices': [a['index'] for a in selected_peaks]}, indent=2))
        return
    media = out / 'media'
    media.mkdir(exist_ok=False)
    font = find_font(None)
    helpers = ('analyze_rtc_pair.py', 'build_three_video_comparison.py', 'build_v6_motion_review.py', 'review_rtc10_tails.py')
    report = {'schema': 'rtc-pair-all-outcome-media-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'builder_sha256': sha(Path(__file__)), 'helper_sha256': {name: sha(ROOT / 'scripts' / name) for name in helpers},
              'source_comparison_sha256': sha(out / 'comparison.json'), 'source_plan_sha256': sha(out / 'plan.json'),
              'endpoints_identical': True, 'attempts': attempts,
              'selection': 'All36 outcomes in six overviews. Each profile maximum-spike and worst-p99 run, deduplicated. Fixed019/r1 full videos, never first-success selection.',
              'time_alignment': 'Pre-step state[i] and MP4frame i are at t=i/50. Task[i] is post-step and maps to frame i+1. A physical peak at state[s] uses task[s-1].',
              'playback': 'Original50fps, all source frames in order from common frame0, no cuts or speed changes. Shorter video final-frame clone only, explicitly labelled. Source MP4s remain unchanged.',
              'scope': 'Same-session, matched reset scenes but unpaired policy noise. All failures/missing events retained. Lift means confirmed lifted pinch (ever_grasped). '
                       'Percentile/extreme selection and event proximity do not establish unique causes or general superiority.',
              'manual_visual_review': 'Pending direct review of generated figures and aligned frames.'}
    try:
        for attempt in attempts:
            info = video_info(Path(attempt['source_video']))
            require(info['fps'] == FPS and info['frame_count'] == attempt['steps'] + 1, 'Original video lacks full50fps episode including terminal frame')
            attempt['video'] = info
        report['layout_overviews'] = [overview(layout, attempts, media, font) for layout in LAYOUTS]
        # Refresh the selected copies with their verified video metadata.
        verified = {a['index']: a for a in attempts}
        report['peak_contact_sheets'] = [peak_sheet({**verified[a['index']], 'selection_reasons': a['selection_reasons']}, media, font) for a in selected_peaks]
        report['unscored_profiles_without_peak'] = [p for p in PROFILES if not any(a['profile'] == p for a in selected_peaks)]
        report['combined_video'] = side_by_side(attempts, media, font)
        require(sha(out / 'comparison.json') == report['source_comparison_sha256'], 'Source ledger changed during build')
        for attempt in attempts:
            require(sha(Path(attempt['source_video'])) == attempt['source_video_sha256'], 'Source video changed during build')
        write_new(media / 'manifest.json', report)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        write_new(media / 'failure.json', report)
        raise
    print(json.dumps({'overviews': [r['path'] for r in report['layout_overviews']],
                      'peaks': [r['path'] for r in report['peak_contact_sheets']],
                      'video': report['combined_video']['path'], 'manifest': str(media / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
