"""All ten outcomes, event sheets and a fixed-speed video grid; no selection."""
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from repeat_v6_control_layout import OUT
from evaluate_v6_progress import read,sha,write
from build_three_video_comparison import find_font,selected_frames,video_info
from build_v6_motion_review import event_frames


def main():
    report=read(OUT/'aggregate.json');assert report['complete'] and report['audit_passed'] and len(report['trials'])==10
    media=OUT/'media';media.mkdir(exist_ok=False);font=ImageFont.truetype(str(find_font(None)),16)
    columns=[('start','Start'),('near_grasp','Near grasp'),('first_lifted_pinch','Lifted pinch'),
             ('first_box_entry','Box entry'),('max_active_physical_jerk','Peak physical jerk'),('final','Final')]
    records=[];frames=[]
    for row in report['trials']:
        run=Path(row['run_directory']);video=Path(row['video']);assert sha(video)==row['video_sha256']
        tasks=[json.loads(s) for s in (run/'episode_000_task.jsonl').read_text().splitlines()]
        with np.load(run/'episode_000.npz',allow_pickle=False) as data:events=event_frames(tasks,data['state'],data['action'])
        steps=list(range(0,len(tasks),2))+[len(tasks)]
        for event in events.values():
            if event is not None:
                step=event.pop('video_frame');index=next(i for i,s in enumerate(steps) if s>=step)
                event.update(event_control_step=step,video_frame=index,recorded_control_step=steps[index])
        info=video_info(video);assert info['fps']==25 and info['width']==640 and info['height']==480 and info['frame_count']==len(steps)
        wanted={e['video_frame'] for e in events.values() if e is not None};frames.append(selected_frames(video,wanted))
        records.append(dict(repeat=row['repeat'],success=row['success'],source_video=str(video),source_sha256=sha(video),
                            video=info,events=events,failure_class=row['failure_class'],simulated_seconds=row['simulated_seconds']))
    pages=[]
    for page_index in range(2):
        page=Image.new('RGB',(1440,1360),(242,245,248));draw=ImageDraw.Draw(page)
        draw.text((10,8),'Control | episode004 | identical reset | all attempts | event-aligned thumbnails',font=font,fill='black')
        for j,row in enumerate(records[page_index*5:page_index*5+5]):
            y=40+j*260;draw.text((10,y),f"Round{row['repeat']} | {'SUCCESS' if row['success'] else row['failure_class'].upper()} | {row['simulated_seconds']:.2f} sim s",font=font,fill='black')
            for k,(name,label) in enumerate(columns):
                x=k*240;event=row['events'][name];draw.text((x+4,y+24),label,font=font,fill='black')
                if event is None:
                    draw.rectangle((x,y+65,x+238,y+245),fill=(210,215,220));draw.text((x+15,y+140),'MISSING',font=font,fill=(155,40,35))
                else:
                    draw.text((x+4,y+44),f"source step {event['recorded_control_step']}",font=font,fill='black')
                    page.paste(Image.fromarray(frames[page_index*5+j][event['video_frame']]).resize((240,180)),(x,y+65))
        path=media/f'all_outcomes_{page_index+1:02d}.jpg';page.save(path,quality=94);pages.append(dict(path=str(path),sha256=sha(path)))
    # Five columns, two rows; original time and all outcomes, labelled terminal freeze.
    maximum=max(r['video']['frame_count'] for r in records)
    header=Image.new('RGBA',(1600,560),(0,0,0,0));draw=ImageDraw.Draw(header)
    small=ImageFont.truetype(str(find_font(None)),14)
    for i,row in enumerate(records):
        x=(i%5)*320;y=(i//5)*280;draw.rectangle((x,y,x+319,y+39),fill=(20,27,38,255))
        draw.text((x+5,y+2),f"Round{row['repeat']} | {'SUCCESS' if row['success'] else 'FAILURE'} | 1x sim time",font=small,fill='white')
        draw.text((x+5,y+21),f"Terminal freeze after {(row['video']['frame_count']-1)/25:.2f}s",font=small,fill='white')
    headerpath=media/'grid_header.png';header.save(headerpath);destination=media/'all10_same_layout.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n','-loglevel','error','-filter_threads','1','-filter_complex_threads','1']
    for row in records:cmd+=['-threads','1','-i',row['source_video']]
    cmd+=['-loop','1','-i',str(headerpath)]
    filters=[f'[{i}:v]setpts=PTS-STARTPTS,scale=320:240,pad=320:280:0:40:color=black,tpad=stop_mode=clone:stop={maximum-row["video"]["frame_count"]}[v{i}]' for i,row in enumerate(records)]
    filters+=[''.join(f'[v{i}]' for i in range(5))+'hstack=inputs=5[r0]',
              ''.join(f'[v{i}]' for i in range(5,10))+'hstack=inputs=5[r1]',
              '[r0][r1]vstack=inputs=2[grid]','[grid][10:v]overlay=0:0[out]']
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-frames:v',str(maximum),'-r','25',
          '-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p',str(destination)]
    subprocess.run(cmd,check=True)
    info=video_info(destination);assert info['fps']==25 and info['frame_count']==maximum and info['width']==1600 and info['height']==560
    manifest=dict(schema='v6-control-fixed-layout-media-v1',builder_sha256=sha(Path(__file__)),aggregate_sha256=sha(OUT/'aggregate.json'),
        attempts=records,overview_pages=pages,comparison_video=dict(path=str(destination),sha256=sha(destination),info=info),
        scope='All10 rounds, no outcome selection. Same start time and25fps simulation speed; final-frame freezes clearly labelled. MP4 does not reproduce wall-clock inference waits. Static event review is not continuous playback review.',
        alignment='Task row i post-step i+1; state pre-step. Even control-step frames plus terminal step; first frame at or after event.',visual_review_pending=True)
    write(media/'manifest.json',manifest);print(json.dumps(dict(pages=pages,video=str(destination)),indent=2))


if __name__=='__main__':main()
