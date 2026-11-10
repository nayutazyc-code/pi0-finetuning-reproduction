"""Copy three complete selected successes and build an original-speed comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'outputs/three_version_success_20261027'
MODELS = ('v6', 'v8_control', 'v9')
LABELS = {'v6': 'V6 | original pi0', 'v8_control': 'V8 | control', 'v9': 'V9 | motion'}
COPIES = {'v6': 'v6_success.mp4', 'v8_control': 'v8_control_success.mp4', 'v9': 'v9_success.mp4'}
FPS = 50


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write('\n')


def validate_ledger(out):
    """Read-only provenance checks before any copy, decode, or encoding."""
    ledger = load(out / 'comparison.json')
    plan = load(out / 'plan.json')
    require(ledger.get('complete') is True and ledger.get('plan') == plan,
            'Recording is incomplete or ledger/plan differ')
    require(plan['order'] == list(MODELS) and set(ledger['selected']) == set(MODELS),
            'Exactly the requested three versions must be selected')
    require(plan['layout'] == 'episode_019' and plan['video_fps'] == FPS,
            'Unexpected scene protocol or video rate')
    require(ledger.get('restoration', {}).get('model') == 'v8_control'
            and ledger['restoration'].get('metadata_verified') is True,
            'Recording has not finished its verified control restoration')
    require(not ledger.get('error'), 'Recording ledger retains a fatal error')
    selected = []
    for model in MODELS:
        row = ledger['selected'][model]
        attempts = [r for r in ledger['attempts'] if r['model'] == model]
        successes = [r for r in attempts if r.get('success') is True]
        require(successes and successes[0] == row and attempts[-1] == row,
                f'{model}: selected row must be the first success and final attempt')
        require([r['attempt'] for r in attempts] == list(range(1, row['attempt'] + 1)),
                f'{model}: attempts are missing or duplicated')
        require(1 <= row['attempt'] <= plan['max_attempts_per_model'] and not row.get('infrastructure_error'),
                f'{model}: selected row is invalid')
        directory = Path(row['run_directory'])
        video = Path(row['video'])
        require(video.resolve() == (directory / 'episode_000.mp4').resolve(),
                f'{model}: video does not belong to selected recording')
        require(video.is_file() and sha(video) == row['video_sha256'],
                f'{model}: source video digest does not match successful selection')
        summary = load(directory / 'summary.json')
        require(len(summary['episodes']) == 1, f'{model}: expected one complete episode')
        episode = summary['episodes'][0]
        require(episode['success'] is True and not episode.get('error')
                and not episode.get('aborted_by_window_close'), f'{model}: source episode is not a clean success')
        require(episode == row['diagnostics']['episodes'][0]['result'],
                f'{model}: source summary differs from ledger result')
        run = load(directory / 'run.json')
        config = run['config']
        require(config['simulation']['seed'] == plan['reset_seed'] and episode['seed'] == plan['reset_seed'],
                f'{model}: reset seed differs')
        require(config['simulation']['video_fps'] == FPS and config['simulation']['save_video'],
                f'{model}: original video settings differ')
        require(sha(Path(config['simulation']['scene_config'])) == plan['scene_sha256'],
                f'{model}: selected recording scene differs from plan')
        tasks = [json.loads(line) for line in (directory / 'episode_000_task.jsonl').read_text().splitlines()]
        require(len(tasks) == episode['steps'] and tasks, f'{model}: task recording is incomplete')
        final = tasks[-1]
        require(all(final[key] for key in ('inside_box', 'released', 'supported_by_box', 'stable', 'is_success'))
                and final['success_hold_steps'] >= final['required_hold_steps'],
                f'{model}: final task success is not confirmed')
        metadata = load(directory / 'server.json')
        require(metadata.get('model_family') == 'pi0' and metadata.get('action_horizon') == 100,
                f'{model}: unexpected model family or horizon')
        suffix = config['model']['expected_checkpoint_suffix']
        require(suffix and metadata['checkpoint'].rstrip('/').endswith('/' + suffix),
                f'{model}: checkpoint does not match selected client profile')
        require(metadata['norm_stats_sha256'] == config['model']['expected_norm_stats_sha256'],
                f'{model}: normalization differs')
        elapsed = episode['steps'] / FPS
        require(abs(episode['execution']['simulated_seconds'] - elapsed) < 1e-8,
                f'{model}: simulation duration does not match control steps')
        require(abs(final['simulation_time_seconds'] - row['success_seconds']) < 1e-8,
                f'{model}: final physics timestamp differs from ledger')
        selected.append({'model': model, 'attempt': row['attempt'], 'attempt_count': len(attempts),
                         'attempt_limit': plan['max_attempts_per_model'], 'source': str(video.resolve()),
                         'source_sha256': row['video_sha256'], 'copy_filename': COPIES[model],
                         'success_episode_sim_seconds': elapsed,
                         'success_physics_time_including_settle_seconds': row['success_seconds'],
                         'initial_settle_offset_seconds': row['success_seconds'] - elapsed,
                         'steps': episode['steps'], 'checkpoint': metadata['checkpoint'],
                         'source_receipts_sha256': {name: sha(directory / name) for name in
                             ('run.json', 'server.json', 'summary.json', 'episode_000_task.jsonl')},
                         'failed_attempts_retained_in_ledger': [r['attempt'] for r in attempts if not r.get('success')]})
    require(len({json.dumps(load(Path(r['source']).parent / 'run.json')['source_code_sha256'], sort_keys=True)
                 for r in selected}) == 1, 'Execution source receipts differ across versions')
    return ledger, selected


def video_info(path):
    import imageio.v2 as imageio
    reader = imageio.get_reader(path, input_params=['-threads', '1'])
    try:
        meta = reader.get_meta_data()
        first, last, count = None, None, 0
        for frame in reader:
            if first is None:
                first = frame
            last, count = frame, count + 1
        require(count > 0, f'Empty video: {path}')
        require(first.shape == last.shape and first.ndim == 3 and first.shape[2] == 3,
                f'Unexpected RGB video dimensions: {path}')
        return {'fps': float(meta['fps']), 'width': int(first.shape[1]), 'height': int(first.shape[0]),
                'frame_count': int(count), 'duration_seconds': count / float(meta['fps']),
                'container_duration_seconds': meta.get('duration'), 'codec': meta.get('codec')}
    finally:
        reader.close()


def selected_frames(path, indices):
    """Decode sequentially: compressed MP4 EOF seeking can miss a valid final frame."""
    import imageio.v2 as imageio
    wanted, frames = set(indices), {}
    reader = imageio.get_reader(path, input_params=['-threads', '1'])
    try:
        for index, frame in enumerate(reader):
            if index in wanted:
                frames[index] = frame
            if index >= max(wanted):
                break
    finally:
        reader.close()
    require(set(frames) == wanted, f'Requested frames are missing during sequential decode: {path}')
    return frames


def find_font(explicit):
    if explicit:
        require(explicit.is_file(), 'Font file does not exist')
        return explicit
    for candidate in ('/System/Library/Fonts/Supplemental/Arial.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                      '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(candidate).is_file():
            return Path(candidate)
    raise ValueError('No suitable installed font; provide --font-path without installing dependencies')


def make_header(path, selected, width, height, font_path):
    from PIL import Image, ImageDraw, ImageFont
    header = Image.new('RGB', (width * 3, height), (21, 27, 38))
    draw = ImageDraw.Draw(header)
    text_records = []
    for column, row in enumerate(selected):
        lines = [(LABELS[row['model']], 28),
                 (f"First success | attempt {row['attempt']}/{row['attempt_limit']}", 21),
                 (f"Success: {row['success_episode_sim_seconds']:.2f} sim s | 1x, 50 fps", 21)]
        y = 8
        for text, size in lines:
            while size >= 14:
                font = ImageFont.truetype(str(font_path), size)
                bounds = draw.textbbox((0, 0), text, font=font)
                if bounds[2] - bounds[0] <= width - 32:
                    break
                size -= 1
            require(size >= 14, 'Header text cannot fit without becoming illegible')
            x = column * width + 16
            draw.text((x, y - bounds[1]), text, font=font, fill=(240, 244, 250))
            text_records.append({'model': row['model'], 'text': text, 'font_size': size,
                                 'bounds_xyxy': [x, y, x + bounds[2] - bounds[0], y + bounds[3] - bounds[1]]})
            y += bounds[3] - bounds[1] + 5
        require(y <= height, 'Header text would be vertically cropped')
        if column:
            draw.line((column * width, 0, column * width, height - 1), fill=(130, 145, 165), width=2)
    header.save(path)
    return text_records


def qa_frames(combined, selected, output, width, height, header_height, count, font_path):
    """Compare aligned first/middle/final images without claiming losslessness."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    indices = sorted({0, count // 2, min(r['video']['frame_count'] for r in selected) - 1, count - 1})
    rows, endpoint_checks, frame_images = [], [], []
    combined_frames = selected_frames(combined, indices)
    source_frames = {row['model']: selected_frames(output / row['copy_filename'],
                     {min(index, row['video']['frame_count'] - 1) for index in indices}) for row in selected}
    for index in indices:
        frame = combined_frames[index]
        rows.append((index, Image.fromarray(frame)))
        evidence_path = output / f'media_qa_frame_{index:06d}.png'
        require(not evidence_path.exists(), 'QA frame evidence already exists')
        Image.fromarray(frame).save(evidence_path)
        frame_images.append({'frame_index': index, 'timestamp_seconds': index / FPS,
                             'path': str(evidence_path), 'sha256': sha(evidence_path)})
        for column, row in enumerate(selected):
            source_index = min(index, row['video']['frame_count'] - 1)
            expected = source_frames[row['model']][source_index].astype(float)
            actual = frame[header_height:header_height + height, column * width:(column + 1) * width].astype(float)
            error = float(np.mean(np.abs(actual - expected)))
            require(error < 8, 'Combined frame is not close to aligned original; inspect encoding/alignment')
            endpoint_checks.append({'model': row['model'], 'combined_frame': index, 'source_frame': source_index,
                                    'mean_absolute_rgb_error_0_255': error,
                                    'scope': 'Lossy comparison encode; aligned frame check, not a lossless identity claim.'})
    padding = 30
    sheet = Image.new('RGB', (width * 3, len(rows) * (height + header_height + padding)), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    for n, (index, frame) in enumerate(rows):
        y = n * (height + header_height + padding)
        draw.text((10, y + 6), f'Combined frame {index} | t={index/FPS:.2f}s | first / middle / shortest-end / final evidence',
                  fill='black', font=ImageFont.truetype(str(font_path), 18))
        sheet.paste(frame, (0, y + padding))
    path = output / 'media_qa_contactsheet.jpg'
    require(not path.exists(), 'QA evidence already exists')
    sheet.save(path, quality=93)
    return {'contactsheet': str(path), 'contactsheet_sha256': sha(path),
            'frame_images': frame_images,
            'combined_frame_indices': indices, 'endpoint_alignment_checks': endpoint_checks,
            'manual_visual_review': 'Pending: inspect header cropping, equal t=0 alignment and final frozen frames with view_image.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--font-path', type=Path)
    parser.add_argument('--check-only', action='store_true', help='Validate ledger/hashes; do not copy, decode or encode')
    parser.add_argument('--archive-failed-build', action='store_true',
                        help='Preserve prior failed media artifacts in a new archive before a fresh build')
    args = parser.parse_args()
    out = args.out.resolve()
    ledger, selected = validate_ledger(out)
    if args.check_only:
        print(json.dumps({'validated': True, 'selected': selected}, indent=2))
        return
    targets = [out / row['copy_filename'] for row in selected]
    targets += [out / name for name in ('side_by_side.mp4', 'side_by_side_header.png',
                                       'media_manifest.json', 'media_failure.json',
                                       'media_encode.log', 'media_qa_contactsheet.jpg', '.side_by_side.building.mp4')]
    archived = None
    if args.archive_failed_build:
        require((out / 'media_failure.json').is_file() and not (out / 'media_manifest.json').exists(),
                'Only a recorded failed media build may be archived; completed media is preserved')
        for attempt in range(1, 100):
            candidate = out / f'media_failed_build_{attempt:02d}'
            if not candidate.exists():
                candidate.mkdir()
                archived = candidate
                break
        require(archived is not None, 'Too many media failure archives')
        previous = [path for path in targets if path.exists()] + list(out.glob('media_qa_frame_*.png'))
        receipt = {'scope': 'Only artifacts made by this builder; source ledger and source recordings remain untouched.',
                   'artifacts': [{'name': path.name, 'sha256': sha(path)} for path in previous]}
        for path in previous:
            path.rename(archived / path.name)
        write_new(archived / 'archive_receipt.json', receipt)
    require(not any(path.exists() for path in targets), 'Media outputs already exist; preserve them rather than overwrite')
    manifest = {'schema': 'three-selected-success-videos-v1',
                'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'builder_sha256': sha(Path(__file__)), 'source_ledger_sha256': sha(out / 'comparison.json'),
                'source_plan_sha256': sha(out / 'plan.json'), 'layout': ledger['plan']['layout'],
                'prior_failed_build_archive': str(archived) if archived else None,
                'reset_seed': ledger['plan']['reset_seed'], 'order': list(MODELS),
                'selected': selected,
                'scope': 'First successful full example for each version, all original failed attempts retained in ledger. '
                         'Selected successes are not success-rate evidence or a general performance ranking.',
                'playback': 'Each original starts at frame 0 together, 50 fps, 1x simulation time; no cuts or speed changes. '
                            'Inference/network wall-clock waiting is not represented by the simulation-time MP4.',
                'freeze_rule': 'After a shorter original ends, clone its final frame to the longest original frame count. '
                               'The three separately copied originals are complete, unpadded, byte-identical files.'}
    try:
        import imageio_ffmpeg
        for row in selected:
            source = Path(row['source'])
            destination = out / row['copy_filename']
            with source.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            row['copy_sha256'] = sha(destination)
            require(row['copy_sha256'] == row['source_sha256'] == sha(source), 'Original or copy digest changed')
            row['video'] = video_info(destination)
            require(row['video']['fps'] == FPS, 'Source is not 50fps; refuse to retime')
            require(row['video']['frame_count'] == row['steps'] + 1,
                    'Source is missing full episode frames including its terminal frame')
        sizes = {(row['video']['width'], row['video']['height']) for row in selected}
        require(len(sizes) == 1, 'Source dimensions differ; refuse to resize task videos')
        width, height = sizes.pop()
        require(width % 2 == 0 and height % 2 == 0, 'Expected even dimensions for yuv420p')
        header_height = 96
        count = max(row['video']['frame_count'] for row in selected)
        header = out / 'side_by_side_header.png'
        font_path = find_font(args.font_path)
        text = make_header(header, selected, width, header_height, font_path)
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-filter_threads', '1', '-filter_complex_threads', '1']
        for row in selected:
            command += ['-threads', '1', '-i', str(out / row['copy_filename'])]
        command += ['-threads', '1', '-loop', '1', '-framerate', str(FPS), '-i', str(header)]
        filters = []
        for index, row in enumerate(selected):
            extra = count - row['video']['frame_count']
            row['combined_freeze_frames'] = extra
            row['combined_freeze_seconds'] = extra / FPS
            filters.append(f'[{index}:v]tpad=stop_mode=clone:stop={extra},'
                           f'trim=end_frame={count},settb=expr=1/{FPS},setpts=N,'
                           f'pad=iw:ih+{header_height}:0:{header_height}:black[v{index}]')
        filters += ['[v0][v1][v2]hstack=inputs=3[body]',
                    f'[3:v]settb=expr=1/{FPS},setpts=N[header]',
                    f'[body][header]overlay=0:0:shortest=1,settb=expr=1/{FPS},setpts=N,format=yuv420p[result]']
        temporary = out / '.side_by_side.building.mp4'
        command += ['-filter_complex', ';'.join(filters), '-map', '[result]', '-an',
                    '-frames:v', str(count), '-r', str(FPS), '-fps_mode', 'cfr', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', '18', '-threads', '1', '-movflags', '+faststart', str(temporary)]
        manifest['encoding'] = {'command': command, 'ffmpeg_executable': ffmpeg,
                                'decoder_threads_per_input': 1, 'encoder_threads': 1,
                                'filter_complex_threads': 1, 'header_text': text,
                                'timestamp_rule': 'Verified original 50fps frame order is preserved; after final-frame padding, '
                                                  'assign PTS=N with timebase 1/50. No source frames are dropped or retimed.',
                                'font_file': str(font_path), 'font_sha256': sha(font_path),
                                'header_sha256': sha(header)}
        with (out / 'media_encode.log').open('xb') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        require(completed.returncode == 0, 'Comparison encoding failed; retained media_encode.log')
        info = video_info(temporary)
        require(info['frame_count'] == count and info['fps'] == FPS
                and (info['width'], info['height']) == (3 * width, height + header_height),
                'Encoded comparison has unexpected dimensions, length or fps')
        combined = out / 'side_by_side.mp4'
        temporary.rename(combined)
        manifest['combined'] = {'path': str(combined), 'sha256': sha(combined), **info}
        manifest['qa'] = qa_frames(combined, selected, out, width, height, header_height, count, font_path)
        # Recheck source/selection immutability after encoding, which can take time.
        require(sha(out / 'comparison.json') == manifest['source_ledger_sha256'], 'Source ledger changed during build')
        for row in selected:
            require(sha(Path(row['source'])) == row['source_sha256'], 'Source video changed during build')
        write_new(out / 'media_manifest.json', manifest)
    except BaseException as exc:
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        write_new(out / 'media_failure.json', manifest)
        raise
    print(json.dumps({'original_copies': [str(out / row['copy_filename']) for row in selected],
                      'combined': str(out / 'side_by_side.mp4'), 'manifest': str(out / 'media_manifest.json'),
                      'qa': manifest['qa']['contactsheet']}, indent=2))


if __name__ == '__main__':
    main()
