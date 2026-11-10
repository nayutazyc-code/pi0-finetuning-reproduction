"""All24 same-layout policy outcomes for one independent factorial cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_three_video_comparison import find_font, video_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/release_retreat_20261029/closed_loop_replication'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / 'outputs/release_retreat_20261029')
    read = lambda p: json.loads(p.read_text())
    audit, aggregate = read(out / 'audit.json'), read(out / 'aggregate.json')
    assert audit['passed'] and aggregate['complete']
    trials = aggregate['trials']
    assert len(trials) == 24
    dest = out / 'media'
    dest.mkdir(exist_ok=False)
    font = ImageFont.truetype(str(find_font(None)), 16)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records, pages, videos = [], [], []
    for layout in sorted({t['layout'] for t in trials}):
        rows = sorted((t for t in trials if t['layout'] == layout),
                      key=lambda t: (t['repeat'], ('v6', 'v6_damping2', 'candidate', 'candidate_damping2').index(t['model'])))
        assert len(rows) == 4
        page = Image.new('RGB', (1440, 1060), '#edf2f5')
        draw = ImageDraw.Draw(page)
        draw.text((8, 5), f'{layout} | all four policy attempts | 25fps video samples', font=font, fill='black')
        infos = []
        for j, trial in enumerate(rows):
            video = Path(trial['video'])
            assert sha(video) == trial['video_sha256']
            info = video_info(video)
            assert info['fps'] == 25
            infos.append(info)
            run = Path(trial['run_directory'])
            tasks = [json.loads(s) for s in (run / 'episode_000_task.jsonl').read_text().splitlines()]
            with np.load(run / 'episode_000.npz', allow_pickle=False) as data:
                states = data['state'][:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].astype(float)
            peak = int(np.argmax(np.max(abs(np.diff(states, n=3, axis=0)), axis=1))) + 3
            events = [('Start', 0), ('Lifted pinch', next((i + 1 for i, t in enumerate(tasks) if t.get('ever_grasped')), None)),
                      ('Box entry', next((i + 1 for i, t in enumerate(tasks) if t.get('inside_box')), None)),
                      ('Peak arm jerk', peak), ('Halfway', len(states) // 2), ('Final', len(states) - 1)]
            y = 34 + j * 255
            label = f"{trial['model']} repeat{trial['repeat']} | {'SUCCESS' if trial['success'] else 'FAILURE'} | sim{trial['simulated_seconds']:.2f}s"
            draw.text((8, y), label, font=font, fill='black')
            for k, (name, step) in enumerate(events):
                x = k * 240
                draw.text((x + 5, y + 24), name, font=font, fill='black')
                if step is None:
                    draw.text((x + 5, y + 45), 'MISSING EVENT', font=font, fill='black')
                    draw.rectangle((x, y + 68, x + 239, y + 247), fill='#cad3dc')
                    continue
                when = min(int(step * .02 * 25 + .5) / 25, (info['frame_count'] - 1) / 25)
                draw.text((x + 5, y + 45), f'sim {step*.02:.2f}s; video{when:.2f}s', font=font, fill='black')
                command = [ffmpeg, '-hide_banner', '-nostdin', '-loglevel', 'error', '-threads', '1',
                           '-ss', str(when), '-i', str(video), '-frames:v', '1', '-vf', 'scale=240:180',
                           '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
                pixels = subprocess.check_output(command)
                assert len(pixels) == 240 * 180 * 3
                page.paste(Image.frombytes('RGB', (240, 180), pixels), (x, y + 68))
            records.append(dict(index=trial['index'], model=trial['model'], repeat=trial['repeat'],
                                layout=layout, success=trial['success'], events=events, video=str(video),
                                video_sha256=trial['video_sha256'], info=info))
        sheet = dest / (layout + '.jpg')
        page.save(sheet, quality=94)
        pages.append(dict(path=str(sheet), sha256=sha(sheet)))
        header = Image.new('RGBA', (1920, 400), (0, 0, 0, 0))
        draw = ImageDraw.Draw(header)
        for j, (trial, info) in enumerate(zip(rows, infos, strict=True)):
            x, y = (j % 4) * 480, (j // 4) * 400
            draw.rectangle((x, y, x + 479, y + 39), fill=(20, 28, 38, 255))
            label = f"{trial['model']} repeat{trial['repeat']} | {'SUCCESS' if trial['success'] else 'FAILURE'}"
            draw.text((x + 4, y), label, font=font, fill='white')
            draw.text((x + 4, y + 20), f"1x sim video; terminal freeze after {info['duration_seconds']:.2f}s", font=font, fill='white')
        labels = dest / (layout + '_labels.png')
        header.save(labels)
        maximum = max(i['frame_count'] for i in infos)
        command = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'error', '-filter_threads', '1', '-filter_complex_threads', '1']
        for t in rows:
            command += ['-threads', '1', '-i', t['video']]
        command += ['-loop', '1', '-i', str(labels)]
        filters = [f'[{j}:v]setpts=PTS-STARTPTS,scale=480:360,pad=480:400:0:40:color=black,tpad=stop_mode=clone:stop={maximum-i["frame_count"]}[v{j}]' for j, i in enumerate(infos)]
        filters += ['[v0][v1][v2][v3]hstack=inputs=4[g]', '[g][4:v]overlay=0:0,fps=25[out]']
        video = dest / (layout + '_all_outcomes.mp4')
        command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-frames:v', str(maximum),
                    '-c:v', 'libx264', '-threads', '2', '-crf', '19', '-pix_fmt', 'yuv420p', str(video)]
        subprocess.run(command, check=True)
        info = video_info(video)
        assert info['frame_count'] == maximum and info['fps'] == 25
        videos.append(dict(path=str(video), sha256=sha(video), info=info, encoder_command=command))
    manifest = dict(schema='release-damping-all-outcomes-media-v1', script_sha256=sha(Path(__file__)),
                    aggregate_sha256=sha(out / 'aggregate.json'), audit_sha256=sha(out / 'audit.json'),
                    attempts=records, pages=pages, videos=videos, visual_review_pending=True,
                    scope='All24 independent replication attempts, six same-layout grids. Unpaired policy noise; simulated video time is distinct from control wall time. Terminal padding explicitly labeled; missing task events retained.')
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('COMPLETE', dest)


if __name__ == '__main__':
    main()
