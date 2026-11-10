"""Offline all-outcome media for the complete, fixed twelve-attempt v6 screen."""
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

DEFAULT_OUT = ROOT / 'outputs/v6_runtime_motion_20261027'
GROUPS = ('baseline', 'acc20', 'rtc30', 'wrist125')
LAYOUTS = ('episode_019', 'episode_020', 'episode_026')
LABELS = {'baseline': 'V6 baseline', 'acc20': 'V6 | acceleration 20',
          'rtc30': 'V6 | RTC blend 30', 'wrist125': 'V6 | wrist damping +25%'}
FPS, HEADER = 50, 96


def event_frames(tasks, states, actions):
    """State[i] is pre-step/frame i; task[i] is post-step/frame i+1."""
    require(len(tasks) == len(states) == len(actions) and len(states) >= 4,
            'Expected complete synchronized task/state/action recording')
    events = {'start': {'video_frame': 0, 'episode_time_seconds': 0., 'reason': 'initial settled scene'}}
    for name, key in (('near_grasp', 'near_grasp_region'), ('first_lifted_pinch', 'ever_grasped'),
                      ('first_box_entry', 'inside_box')):
        index = next((i for i, task in enumerate(tasks) if task.get(key)), None)
        events[name] = None if index is None else {
            'task_row_index': index, 'task_key': key, 'video_frame': index + 1,
            'episode_time_seconds': (index + 1) / FPS}
    q = np.asarray(states, float)[:, ARM_JOINTS]
    active = np.max(np.abs(np.diff(q, axis=0)), axis=1)[2:] * FPS > .01
    jerks = np.abs(np.diff(q, n=3, axis=0)) * FPS**3
    if active.any():
        index, axis = np.unravel_index(np.argmax(np.where(active[:, None], jerks, -np.inf)), jerks.shape)
        state_index = int(index + 3)
        events['max_active_physical_jerk'] = {
            'state_row_index': state_index, 'np_diff_n3_index': int(index),
            'video_frame': state_index, 'episode_time_seconds': state_index / FPS,
            'joint_axis': int(ARM_JOINTS[axis]), 'jerk_rad_s3': float(jerks[index, axis]),
            'neighbour_video_frames': [max(0, state_index - 2), state_index, min(len(tasks), state_index + 2)],
            'scope': 'Frozen active frame-max physical jerk; state is pre-step. No contact-causality inference.'}
    else:
        events['max_active_physical_jerk'] = None
    events['final'] = {'video_frame': len(tasks), 'episode_time_seconds': len(tasks) / FPS,
                       'task_row_index': len(tasks) - 1, 'reason': 'terminal camera frame; all outcomes retained'}
    return events


def validate(out):
    ledger, plan = load(out / 'comparison.json'), load(out / 'plan.json')
    require(ledger.get('complete') is True and ledger.get('plan') == plan,
            'Screen is not complete; no waiting/polling or partial media selection')
    require(plan['schema'] == 'v6-runtime-motion-screen-v1' and plan['attempts'] == 12
            and plan['video_fps'] == FPS, 'Unexpected screen protocol')
    expected_order = [(layout, group) for layout, groups in plan['blocks'] for group in groups]
    require(set(expected_order) == {(layout, group) for layout in LAYOUTS for group in GROUPS}
            and len(expected_order) == 12, 'Expected three fixed layouts and all four groups')
    rows = ledger['attempts']
    require(len(rows) == 12 and [(row['layout'], row['profile']) for row in rows] == expected_order,
            'Attempts are missing, duplicated or reordered')
    require([row['index'] for row in rows] == list(range(1, 13)), 'Unexpected attempt indices')
    require(sha(ROOT / 'src/pi0_repro/motion_quality.py') == plan['source_sha256']['motion_quality.py'],
            'Frozen metric source changed')
    endpoint = ledger.get('final_endpoint') or {}
    require(endpoint.get('model') == 'v6' and endpoint.get('metadata_verified') is True,
            'Final verified v6 endpoint is not recorded')
    attempts = []
    for row in rows:
        require(not row.get('infrastructure_error'), 'Infrastructure failure is retained in ledger; cannot fabricate missing media')
        directory = Path(row['run_directory'])
        video = Path(row['video'])
        require(video.resolve() == (directory / 'episode_000.mp4').resolve()
                and sha(video) == row['video_sha256'], 'Source video path/hash differs from ledger')
        run = load(directory / 'run.json')
        episode = load(directory / 'summary.json')['episodes'][0]
        require(episode == row['diagnostics']['episodes'][0]['result'] and episode['success'] == row['success'],
                'Source result differs from attempt ledger')
        require(episode['seed'] == plan['scene_seeds'][row['layout']]
                == run['config']['simulation']['seed'], 'Reset seed differs')
        require(run['config']['simulation']['video_fps'] == FPS, 'Refuse non-50fps source video')
        require(sha(Path(run['config']['simulation']['scene_config'])) == row['scene_sha256'], 'Scene digest differs')
        require(all(plan['source_sha256'][name] == digest for name, digest in run['source_code_sha256'].items()),
                'Recorded execution sources differ from plan')
        metadata = load(directory / 'server.json')
        require(metadata.get('model_family') == 'pi0' and metadata['checkpoint'].rstrip('/').endswith(
            '/banana_v6_long100_accum8000/7999'), 'Expected original-pi0 v6 in every group')
        tasks = [json.loads(line) for line in (directory / 'episode_000_task.jsonl').read_text().splitlines()]
        with np.load(directory / 'episode_000.npz', allow_pickle=False) as data:
            states, actions = data['state'], data['action']
        require(len(tasks) == episode['steps'], 'Task recording is incomplete')
        quality = motion_quality(states, actions, tasks)
        require(quality == row['motion_quality'], 'Recomputed frozen metric differs')
        events = event_frames(tasks, states, actions)
        peak = events['max_active_physical_jerk']
        require((peak is None and quality['active_frame_max_jerk_rad_s3'] is None)
                or abs(peak['jerk_rad_s3'] - quality['active_frame_max_jerk_rad_s3']['max']) < 1e-8,
                'Peak location does not reproduce frozen metric')
        attempts.append({'index': row['index'], 'layout': row['layout'], 'profile': row['profile'],
                         'success': bool(row['success']), 'task_error': episode.get('error'),
                         'steps': episode['steps'], 'episode_sim_seconds': episode['steps'] / FPS,
                         'source_video': str(video.resolve()), 'source_video_sha256': row['video_sha256'],
                         'run_directory': str(directory), 'events': events, 'motion_quality': quality,
                         'source_receipts_sha256': {name: sha(directory / name) for name in
                             ('run.json', 'server.json', 'summary.json', 'episode_000.npz', 'episode_000_task.jsonl')}})
    return ledger, attempts


def fit_font(draw, text, path, size, width):
    from PIL import ImageFont
    while size >= 13:
        font = ImageFont.truetype(str(path), size)
        bounds = draw.textbbox((0, 0), text, font=font)
        if bounds[2] - bounds[0] <= width:
            return font, bounds
        size -= 1
    raise ValueError('Review label cannot fit legibly')


def contact_sheet(layout, attempts, media, font_path):
    from PIL import Image, ImageDraw, ImageFont
    columns = [('start', 'Start'), ('near_grasp', 'Near grasp'), ('first_lifted_pinch', 'First lifted pinch'),
               ('first_box_entry', 'First box entry'), ('max_active_physical_jerk', 'Max active physical jerk'), ('final', 'Final outcome')]
    tile_width, tile_height, row_height, top = 320, 240, 322, 74
    canvas = Image.new('RGB', (tile_width * 6, top + 4 * row_height), (243, 245, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{layout} | all four V6 outcomes | simulation time | missing events stay missing',
              font=ImageFont.truetype(str(font_path), 25), fill='black')
    draw.text((12, 42), 'Task events: post-step frame i+1. Physical peak: pre-step state/frame i. Thumbnails are event-aligned, not simultaneous.',
              font=ImageFont.truetype(str(font_path), 18), fill=(40, 45, 55))
    records = []
    by_group = {attempt['profile']: attempt for attempt in attempts if attempt['layout'] == layout}
    for row_index, group in enumerate(GROUPS):
        attempt = by_group[group]
        chosen = {event['video_frame'] for event in attempt['events'].values() if event is not None}
        frames = selected_frames(Path(attempt['source_video']), chosen)
        y = top + row_index * row_height
        outcome = 'SUCCESS' if attempt['success'] else 'FAILURE'
        jerk = attempt['motion_quality']['active_frame_max_jerk_rad_s3']
        p99 = 'unscored' if jerk is None else f'{jerk["p99"]:.2f}'
        heading = f'{LABELS[group]} | {outcome} | recorded {attempt["episode_sim_seconds"]:.2f} sim s | active jerk p99={p99} rad/s^3'
        font, _ = fit_font(draw, heading, font_path, 22, canvas.width - 24)
        draw.text((12, y), heading, fill=(20, 95, 45) if attempt['success'] else (155, 35, 35), font=font)
        for col, (key, label) in enumerate(columns):
            x = col * tile_width
            event = attempt['events'][key]
            font, _ = fit_font(draw, label, font_path, 19, tile_width - 12)
            draw.text((x + 6, y + 30), label, font=font, fill='black')
            if event is None:
                draw.rectangle((x + 2, y + 74, x + tile_width - 2, y + 74 + tile_height), fill=(210, 214, 220))
                draw.text((x + 16, y + 155), 'MISSING', fill=(120, 35, 35), font=ImageFont.truetype(str(font_path), 26))
                draw.text((x + 16, y + 190), 'event not reached / unscored', fill=(45, 45, 45), font=ImageFont.truetype(str(font_path), 17))
            else:
                draw.text((x + 6, y + 52), f'frame {event["video_frame"]} | t={event["episode_time_seconds"]:.2f}s',
                          font=ImageFont.truetype(str(font_path), 17), fill=(45, 45, 45))
                image = Image.fromarray(frames[event['video_frame']]).resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                canvas.paste(image, (x, y + 74))
            records.append({'profile': group, 'event': key, 'selection': event})
    path = media / f'{layout}_all_outcomes_contactsheet.jpg'
    canvas.save(path, quality=93)
    return {'layout': layout, 'path': str(path), 'sha256': sha(path),
            'width': canvas.width, 'height': canvas.height, 'cells': records,
            'scope': 'All four outcomes, event-aligned thumbnails. Missing events are blank, never replaced by a successful event.'}


def comparison_header(attempts, path, width, height, font_path, max_frames):
    from PIL import Image, ImageDraw
    header = Image.new('RGBA', (2 * width, 2 * (height + HEADER)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(header)
    labels = []
    for panel, attempt in enumerate(attempts):
        x, y = (panel % 2) * width, (panel // 2) * (height + HEADER)
        draw.rectangle((x, y, x + width - 1, y + HEADER - 1), fill=(21, 27, 38, 255))
        status = 'SUCCESS' if attempt['success'] else 'FAILURE'
        lines = [(LABELS[attempt['profile']], 26),
                 (f'{status} | end {attempt["episode_sim_seconds"]:.2f} sim s | 1x, 50 fps', 21),
                 (f'Final frame frozen after {attempt["video"]["duration_seconds"]:.2f}s'
                  if attempt['video']['frame_count'] < max_frames else 'Full original duration | no terminal padding', 19)]
        cursor = y + 7
        for text, size in lines:
            font, bounds = fit_font(draw, text, font_path, size, width - 24)
            draw.text((x + 12, cursor - bounds[1]), text, fill=(240, 244, 250, 255), font=font)
            labels.append({'profile': attempt['profile'], 'text': text,
                           'bounds_xyxy': [x + 12, cursor, x + 12 + bounds[2] - bounds[0], cursor + bounds[3] - bounds[1]]})
            cursor += bounds[3] - bounds[1] + 5
        require(cursor <= y + HEADER, 'Video header would be vertically cropped')
    header.save(path)
    return labels


def combined_video(attempts, media, font_path):
    import imageio_ffmpeg
    group_rows = {attempt['profile']: attempt for attempt in attempts if attempt['layout'] == 'episode_019'}
    selected = [group_rows[group] for group in GROUPS]
    sizes = {(row['video']['width'], row['video']['height']) for row in selected}
    require(len(sizes) == 1, 'Refuse to resize differing source video geometries')
    width, height = sizes.pop()
    maximum = max(row['video']['frame_count'] for row in selected)
    header = media / 'episode_019_header.png'
    labels = comparison_header(selected, header, width, height, font_path, maximum)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n',
               '-filter_threads', '1', '-filter_complex_threads', '1']
    for row in selected:
        command += ['-threads', '1', '-i', row['source_video']]
    command += ['-threads', '1', '-loop', '1', '-framerate', str(FPS), '-i', str(header)]
    filters = []
    for panel, row in enumerate(selected):
        extra = maximum - row['video']['frame_count']
        row['combined_freeze_frames'] = extra
        filters.append(f'[{panel}:v]tpad=stop_mode=clone:stop={extra},trim=end_frame={maximum},'
                       f'settb=expr=1/{FPS},setpts=N,pad=iw:ih+{HEADER}:0:{HEADER}:black[v{panel}]')
    filters += ['[v0][v1]hstack=inputs=2[top]', '[v2][v3]hstack=inputs=2[bottom]',
                '[top][bottom]vstack=inputs=2[body]', f'[4:v]settb=expr=1/{FPS},setpts=N[labels]',
                f'[body][labels]overlay=0:0:shortest=1,settb=expr=1/{FPS},setpts=N,format=yuv420p[result]']
    temporary = media / '.episode_019_all_outcomes.building.mp4'
    command += ['-filter_complex', ';'.join(filters), '-map', '[result]', '-an', '-frames:v', str(maximum),
                '-r', str(FPS), '-fps_mode', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-threads', '1', '-movflags', '+faststart', str(temporary)]
    with (media / 'encode.log').open('xb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, 'Encoding failed; preserve encode.log and failure.json')
    require('non-monoton' not in (media / 'encode.log').read_text().lower(), 'Non-monotonic encoded timestamps')
    info = video_info(temporary)
    require(info['fps'] == FPS and info['frame_count'] == maximum
            and (info['width'], info['height']) == (width * 2, (height + HEADER) * 2),
            'Encoded video frame count, fps or geometry differs')
    destination = media / 'episode_019_all_outcomes.mp4'
    temporary.rename(destination)
    # Compare exact frame indices, including final-frame padding, with lossy tolerance.
    indices = sorted({0, maximum // 2, min(row['video']['frame_count'] for row in selected) - 1, maximum - 1})
    frames = selected_frames(destination, indices)
    sources = {row['profile']: selected_frames(Path(row['source_video']),
               {min(index, row['video']['frame_count'] - 1) for index in indices}) for row in selected}
    from PIL import Image, ImageDraw, ImageFont
    checks, previews = [], []
    for index in indices:
        frame = frames[index]
        preview = media / f'episode_019_qa_frame_{index:06d}.png'
        Image.fromarray(frame).save(preview)
        previews.append({'frame_index': index, 'timestamp_seconds': index / FPS, 'path': str(preview), 'sha256': sha(preview)})
        for panel, row in enumerate(selected):
            x, y = (panel % 2) * width, (panel // 2) * (height + HEADER) + HEADER
            source_index = min(index, row['video']['frame_count'] - 1)
            error = float(np.mean(np.abs(frame[y:y + height, x:x + width].astype(float)
                                        - sources[row['profile']][source_index].astype(float))))
            require(error < 8, 'Aligned combined frame differs excessively from its original')
            checks.append({'profile': row['profile'], 'combined_frame': index, 'source_frame': source_index,
                           'mean_absolute_rgb_error_0_255': error})
    sheet = Image.new('RGB', (width * 2, len(indices) * ((height + HEADER) * 2 + 32)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        y = position * ((height + HEADER) * 2 + 32)
        draw.text((10, y + 5), f'Combined frame {index} | t={index/FPS:.2f}s',
                  fill='black', font=ImageFont.truetype(str(font_path), 19))
        sheet.paste(Image.fromarray(frames[index]), (0, y + 32))
    qa_sheet = media / 'episode_019_alignment_qa.jpg'
    sheet.save(qa_sheet, quality=92)
    return {'path': str(destination), 'sha256': sha(destination), **info, 'panel_order': list(GROUPS),
            'command': command, 'header_sha256': sha(header), 'header_labels': labels,
            'freeze_frames': {row['profile']: row['combined_freeze_frames'] for row in selected},
            'qa': {'preview_frames': previews, 'contactsheet': str(qa_sheet), 'contactsheet_sha256': sha(qa_sheet),
                   'alignment_checks': checks, 'manual_visual_review': 'Pending inspection of generated previews.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--font-path', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Validate complete ledger/recordings without decoding or writing media')
    args = parser.parse_args()
    out = args.out.resolve()
    ledger, attempts = validate(out)
    if args.check_only:
        print(json.dumps({'validated_attempts': len(attempts), 'complete': ledger['complete']}, indent=2))
        return
    media = out / 'media'
    media.mkdir(exist_ok=False)
    font_path = find_font(args.font_path)
    report = {'schema': 'v6-all-outcomes-media-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'builder_sha256': sha(Path(__file__)), 'media_helper_sha256': sha(ROOT / 'scripts/build_three_video_comparison.py'),
              'source_plan_sha256': sha(out / 'plan.json'), 'source_comparison_sha256': sha(out / 'comparison.json'),
              'attempts': attempts, 'font_file': str(font_path), 'font_sha256': sha(font_path),
              'frame_mapping': 'State[i] and original MP4 frame i are pre-step at t=i/50. Task[i] is post-step '
                               'at t=(i+1)/50 and maps to MP4 frame i+1. The final MP4 frame n follows action n-1.',
              'playback': 'All four episode_019 original frame0 views start together, original 50fps simulation time, '
                          'no cuts, source-frame drops or speed changes. Only clone terminal frames to the longest source. '
                          'Each panel explicitly states its freeze start. Original MP4 bytes remain unchanged.',
              'scope': 'All twelve prespecified attempts and every success/failure retained; no first-success selection. '
                       'Contact sheets align distinct task events, while the 2x2 video aligns common time. '
                       'Missing events stay blank. Lift column means first confirmed lifted pinch (ever_grasped). '
                       'Frozen physical metric remains unchanged. Media does not establish causal or general superiority claims.'}
    try:
        for attempt in attempts:
            info = video_info(Path(attempt['source_video']))
            require(info['fps'] == FPS and info['frame_count'] == attempt['steps'] + 1,
                    'Expected all original 50fps episode frames including terminal frame')
            attempt['video'] = info
        require(len({(r['video']['width'], r['video']['height']) for r in attempts}) == 1,
                'Unexpected source geometry variation')
        report['contact_sheets'] = [contact_sheet(layout, attempts, media, font_path) for layout in LAYOUTS]
        report['combined_video'] = combined_video(attempts, media, font_path)
        require(sha(out / 'comparison.json') == report['source_comparison_sha256'], 'Source ledger changed during media build')
        for attempt in attempts:
            require(sha(Path(attempt['source_video'])) == attempt['source_video_sha256'], 'Source video changed during media build')
        write_new(media / 'manifest.json', report)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        write_new(media / 'failure.json', report)
        raise
    print(json.dumps({'attempts': len(attempts), 'contact_sheets': [r['path'] for r in report['contact_sheets']],
                      'video': report['combined_video']['path'], 'manifest': str(media / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
