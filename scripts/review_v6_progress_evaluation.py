"""Build all-outcome event sheets and one predeclared-layout comparison offline."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_three_video_comparison import find_font, selected_frames
from build_v6_motion_review import event_frames


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/v6_progress_20261028/evaluation')
    out = parser.parse_args().out.resolve()
    assert out.is_relative_to(ROOT / 'outputs/v6_progress_20261028')
    report = read(out / 'aggregate.json')
    assert report['complete'] and report['audit_passed'] and len(report['trials']) == 18
    media = out / 'media'
    media.mkdir(exist_ok=False)
    font = ImageFont.truetype(str(find_font(None)), 18)
    columns = [('start', 'Start'), ('near_grasp', 'Near grasp'),
               ('first_lifted_pinch', 'Lifted pinch'), ('first_box_entry', 'Box entry'),
               ('max_active_physical_jerk', 'Peak physical jerk'), ('final', 'Final')]
    records = []
    for row in report['trials']:
        run = Path(row['run_directory']); video = Path(row['video'])
        assert sha(video) == row['video_sha256']
        tasks = [json.loads(line) for line in (run / 'episode_000_task.jsonl').read_text().splitlines()]
        with np.load(run / 'episode_000.npz', allow_pickle=False) as data:
            events = event_frames(tasks, data['state'], data['action'])
        steps = len(tasks)
        # The writer records each even pre-step state plus the exact terminal state.
        source_steps = list(range(0, steps, 2)) + [steps]
        for event in events.values():
            if event is not None:
                step = event.pop('video_frame')
                index = next(i for i, source in enumerate(source_steps) if source >= step)
                event.update(event_control_step=step, video_frame=index,
                             recorded_control_step=source_steps[index])
        wanted = {event['video_frame'] for event in events.values() if event is not None}
        frames = selected_frames(video, wanted)
        reader = imageio.get_reader(video, input_params=['-threads', '1'])
        try:
            metadata = reader.get_meta_data(); count = sum(1 for _ in reader)
        finally:
            reader.close()
        assert count == len(source_steps) and metadata['fps'] == 25
        assert frames[0].shape == (480, 640, 3)
        records.append(dict(index=row['index'], model=row['model'], layout=row['layout'],
                            success=row['success'], source_video=str(video), source_sha256=sha(video),
                            video_frames=count, fps=25, events=events, frames=frames))
    pages = []
    for layout in dict.fromkeys(row['layout'] for row in records):
        group = [row for row in records if row['layout'] == layout]
        assert [row['model'] for row in group] == ['v6', 'control', 'recovery']
        page = Image.new('RGB', (1920, 1040), (240, 243, 247)); draw = ImageDraw.Draw(page)
        draw.text((12, 9), layout + ' | all outcomes | event-aligned, not simultaneous', font=font, fill='black')
        for j, row in enumerate(group):
            y = 45 + j * 330
            source = next(r for r in report['trials'] if r['index'] == row['index'])
            text = (f"{row['model']} | {'SUCCESS' if row['success'] else 'FAILURE'} | "
                    f"{source['simulated_seconds']:.2f} sim s | waits={source['inference_underruns']} | "
                    f"max low motion={source['quality']['low_motion_pause_max_seconds']:.2f}s")
            draw.text((12, y), text, font=font, fill='black')
            for col, (key, label) in enumerate(columns):
                x = col * 320; event = row['events'][key]
                draw.text((x + 5, y + 27), label, font=font, fill='black')
                if event is None:
                    draw.rectangle((x, y + 77, x + 318, y + 317), fill=(210, 214, 220))
                    draw.text((x + 12, y + 160), 'MISSING', font=font, fill=(150, 40, 30))
                else:
                    draw.text((x + 5, y + 51), f"source step {event['recorded_control_step']}", font=font, fill='black')
                    page.paste(Image.fromarray(row['frames'][event['video_frame']]).resize((320, 240)), (x, y + 77))
        path = media / (layout + '_all_outcomes.jpg'); page.save(path, quality=94)
        pages.append(dict(layout=layout, path=str(path), sha256=sha(path)))
    # First predeclared validation layout, never success-filtered. Freeze terminal
    # images for shorter episodes; keep the original 25fps simulation-time speed.
    chosen = [next(row for row in records if row['model'] == model and row['layout'] == 'episode_004')
              for model in ('v6', 'control', 'recovery')]
    maximum = max(row['video_frames'] for row in chosen)
    header = Image.new('RGB', (1920, 80), (20, 27, 38)); draw = ImageDraw.Draw(header)
    for j, row in enumerate(chosen):
        draw.text((j * 640 + 12, 10), f"{row['model']} | {'SUCCESS' if row['success'] else 'FAILURE'} | episode_004", font=font, fill='white')
        draw.text((j * 640 + 12, 40), f"25fps, 1x sim time; terminal frozen after {(row['video_frames']-1)/25:.2f}s", font=font, fill='white')
    headpath = media / 'comparison_header.png'; header.save(headpath)
    destination = media / 'episode_004_three_models.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-nostdin', '-n', '-loglevel', 'error',
               '-filter_threads', '1', '-filter_complex_threads', '1']
    for row in chosen:
        command += ['-threads', '1', '-i', row['source_video']]
    command += ['-loop', '1', '-i', str(headpath)]
    filters = [f'[{i}:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={maximum-row["video_frames"]}[v{i}]'
               for i, row in enumerate(chosen)]
    filters += ['[v0][v1][v2]hstack=inputs=3[body]', '[3:v][body]vstack=inputs=2[out]']
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-frames:v', str(maximum),
                '-r', '25', '-c:v', 'libx264', '-threads', '2', '-crf', '20', '-pix_fmt', 'yuv420p', str(destination)]
    subprocess.run(command, check=True)
    for row in records:
        del row['frames']
    manifest = dict(schema='v6-progress-media-review-v1', builder_sha256=sha(Path(__file__)),
                    aggregate_sha256=sha(out / 'aggregate.json'), outcomes=records, overview_pages=pages,
                    comparison_video=dict(path=str(destination), sha256=sha(destination), layout='episode_004',
                                          selection='First predeclared layout, all three outcomes, no success filtering',
                                          terminal_padding='Explicitly labelled final-frame freeze, no speed change'),
                    time_alignment='Task row i is post-step i+1. States are pre-step. 25fps frames map to even control steps plus the terminal step; events map to the first recorded step at or after their event.',
                    limits='Sampled-frame review cannot establish continuous smoothness. MP4 follows simulation time, not wall-clock inference waits. No validation frames become training labels.',
                    visual_review_pending=True)
    (media / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(pages=pages, comparison_video=str(destination)), indent=2))


if __name__ == '__main__':
    main()
