"""Render every outcome of the single-variable IK precision experiment."""
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from experiment_ik_precision import OUT
from evaluate_v6_progress import read,sha,write
from build_three_video_comparison import find_font,video_info
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg


def main():
    audit=read(OUT/'audit.json');aggregate=read(OUT/'aggregate.json');assert audit['passed']
    media=OUT/'media';media.mkdir(exist_ok=False)
    font=ImageFont.truetype(str(find_font(None)),17)
    rows=sorted([r for g in aggregate['groups'].values() for r in g['records']],key=lambda r:(r['parent'],r['profile']))
    records=[];pages=[]
    for r in rows:
        path=Path(r['collection'])/'episode_000';tasks=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
        with np.load(path/'trajectory.npz',allow_pickle=False) as d:states=d['state'].copy()
        n=len(states)
        peak=int(np.argmax(np.max(abs(np.diff(states[:,[0,1,2,3,4,5,7,8,9,10,11,12]].astype(float),n=3)/.02**3),axis=1)))+3 if n>=4 else 0
        box=next((i+1 for i,t in enumerate(tasks) if t['inside_box']),None)
        release=next((i+1 for i,t in enumerate(tasks) if box is not None and i+1>=box and t['released']),None)
        events=[('Start',0),('Lifted pinch',next((i+1 for i,t in enumerate(tasks) if t['grasped']),None)),('Box entry',box),('Released in box',release),('Peak arm jerk',peak),('Final',n)]
        info=video_info(Path(r['video']));assert info['fps']==50 and info['frame_count']==n+1
        records.append(dict(parent=r['parent'],profile=r['profile'],role=r['role'],success=r['automatic_success'],error=r['error'],events=events,
            video=r['video'],video_sha256=sha(Path(r['video'])),info=info,collection=r['collection'],trajectory_sha256=r['trajectory_sha256']))
    def cell(page,draw,x,y,label,step,path,n):
        draw.text((x+5,y),label,font=font,fill='black')
        draw.text((x+5,y+22),'MISSING' if step is None else f'frame {step} | {step*.02:.2f}s',font=font,fill='black')
        if step is None:
            draw.rectangle((x,y+44,x+239,y+223),fill='#cad3dc');return
        p=path/'terminal.png' if step==n else path/'rgb'/f'{step:06d}.png'
        with Image.open(p) as im:page.paste(im.convert('RGB').resize((240,180)),(x,y+44))
    for half in range(2):
        for profile in ('baseline','tight_ik'):
            page=Image.new('RGB',(1440,1100),'#edf2f5');draw=ImageDraw.Draw(page)
            draw.text((8,5),f'{profile} EXPERT | all outcomes, parents {half*4}..{half*4+3} | not pi0 inference results',font=font,fill='black')
            for j,r in enumerate(r for r in records if r['profile']==profile and r['parent']//4==half):
                y=36+j*265;draw.text((8,y),f"Parent {r['parent']} | {r['role']} | {'SUCCESS' if r['success'] else 'FAILURE'} | {(r['info']['frame_count']-1)*.02:.2f}s",font=font,fill='black')
                for k,(label,step) in enumerate(r['events']):cell(page,draw,k*240,y+24,label,step,Path(r['collection'])/'episode_000',r['info']['frame_count']-1)
            dest=media/f'{profile}_parents_{half*4}_{half*4+3}.jpg';page.save(dest,quality=94);pages.append(dict(path=str(dest),sha256=sha(dest)))
    # Focus on the entire first second rather than only a favorable event.
    for half in range(2):
        page=Image.new('RGB',(1440,2150),'#edf2f5');draw=ImageDraw.Draw(page)
        draw.text((8,5),'Expert takeover samples at fixed times | baseline/tight_ik pairs | missing frames explicit',font=font,fill='black')
        for j,r in enumerate(r for r in records if r['parent']//4==half):
            y=36+j*262;draw.text((8,y),f"Parent {r['parent']} | {r['profile']} | {'SUCCESS' if r['success'] else 'FAILURE'}",font=font,fill='black')
            n=r['info']['frame_count']-1
            for k,step in enumerate((0,5,10,20,30,50)):cell(page,draw,k*240,y+24,'Takeover',step if step<=n else None,Path(r['collection'])/'episode_000',n)
        dest=media/f'takeover_parents_{half*4}_{half*4+3}.jpg';page.save(dest,quality=94);pages.append(dict(path=str(dest),sha256=sha(dest)))
    videos=[]
    for half in range(2):
        paired=[r for r in records if r['parent']//4==half];maximum=max(r['info']['frame_count'] for r in paired)
        header=Image.new('RGBA',(960,1600),(0,0,0,0));draw=ImageDraw.Draw(header)
        for k,r in enumerate(paired):
            x=k%2*480;y=k//2*400;draw.rectangle((x,y,x+479,y+39),fill=(20,28,38,255))
            draw.text((x+5,y+1),f"Parent {r['parent']} | {r['profile']} EXPERT | {'SUCCESS' if r['success'] else 'FAILURE'}",font=font,fill='white')
            draw.text((x+5,y+21),f"1x sim time; terminal freeze after {(r['info']['frame_count']-1)/50:.2f}s",font=font,fill='white')
        head=media/f'labels_{half}.png';header.save(head);dest=media/f'all_outcomes_parents_{half*4}_{half*4+3}.mp4'
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n','-loglevel','error','-filter_threads','1','-filter_complex_threads','1']
        for r in paired:cmd+=['-threads','1','-i',r['video']]
        cmd+=['-loop','1','-i',str(head)]
        filters=[f'[{k}:v]setpts=PTS-STARTPTS,scale=480:360,pad=480:400:0:40:color=black,tpad=stop_mode=clone:stop={maximum-r["info"]["frame_count"]}[v{k}]' for k,r in enumerate(paired)]
        filters += [f'[v{k*2}][v{k*2+1}]hstack=inputs=2[r{k}]' for k in range(4)]
        filters += ['[r0][r1][r2][r3]vstack=inputs=4[g]','[g][8:v]overlay=0:0,fps=25[out]']
        cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-frames:v',str(math.ceil(maximum/2)),'-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p',str(dest)]
        subprocess.run(cmd,check=True);info=video_info(dest)
        assert (info['width'],info['height'],info['fps'],info['frame_count'])==(960,1600,25,math.ceil(maximum/2))
        videos.append(dict(path=str(dest),sha256=sha(dest),info=info,encoder_command=cmd))
    write(media/'manifest.json',dict(script_sha256=sha(Path(__file__)),audit_sha256=sha(OUT/'audit.json'),aggregate_sha256=sha(OUT/'aggregate.json'),pages=pages,attempts=records,videos=videos,
        visual_review_pending=True,scope='All16 paired teacher outcomes, not learned-model performance. 25fps presentation samples original50fps at1x simulationtime; terminal freeze labeled. No successful-only selection; original RGB/video retained.'))
    print('MEDIA COMPLETE',media)


if __name__=='__main__':main()
