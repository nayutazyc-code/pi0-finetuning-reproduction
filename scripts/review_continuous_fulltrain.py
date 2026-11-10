"""All20 train-label outcome sheets and read-only video provenance."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from experiment_ik_continuous import OUT
from evaluate_v6_progress import read,sha,write
from build_three_video_comparison import find_font,video_info
from PIL import Image,ImageDraw,ImageFont
import numpy as np


def main():
    result=read(OUT/'relabel_aggregate.json');assert result['audit_passed'] and result['quality_gate_passed']
    dest=OUT/'fulltrain_media';dest.mkdir(exist_ok=False);font=ImageFont.truetype(str(find_font(None)),16);pages=[];records=[]
    for half in range(5):
        page=Image.new('RGB',(1400,1100),'#edf2f5');draw=ImageDraw.Draw(page)
        draw.text((8,6),'CONTINUOUS EXPERT labels | original train layouts | all outcomes | not learned-model performance',font=font,fill='black')
        for j,r in enumerate(result['groups']['continuous'][half*4:half*4+4]):
            path=Path(r['episode_path']);tasks=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
            with np.load(path/'trajectory.npz') as d:states=d['state'].copy()
            n=len(states);peak=int(np.argmax(np.max(abs(np.diff(states[:,[0,1,2,3,4,5,7,8,9,10,11,12]].astype(float),n=3)/.02**3),axis=1)))+3
            post=lambda key:next((i+1 for i,t in enumerate(tasks) if t[key]),None)
            box=post('inside_box');release=next((i+1 for i,t in enumerate(tasks) if box is not None and i+1>=box and t['released']),None)
            events=[('Start',0),('Lifted pinch',post('grasped')),('Transport',next((i for i,t in enumerate(tasks) if t['phase']=='transport'),None)),('Box entry',box),('Released',release),('Peak arm jerk',peak),('Final',n)]
            y=36+j*265;draw.text((8,y),f"Train {r['index']} | {r['layout_id']} | {'SUCCESS' if r['automatic_success'] else 'FAILURE'} | {n*.02:.2f}s",font=font,fill='black')
            for k,(label,step) in enumerate(events):
                x=k*200;draw.text((x+4,y+23),label,font=font,fill='black');draw.text((x+4,y+45),'MISSING' if step is None else f'{step*.02:.2f}s frame{step}',font=font,fill='black')
                if step is None:draw.rectangle((x,y+70,x+199,y+220),fill='#cad3dc');continue
                p=path/'terminal.png' if step==n else path/'rgb'/f'{step:06d}.png'
                with Image.open(p) as im:page.paste(im.convert('RGB').resize((200,150)),(x,y+70))
            info=video_info(Path(r['video']));assert info['fps']==50 and info['frame_count']==n+1
            records.append(dict(index=r['index'],layout_id=r['layout_id'],episode_path=r['episode_path'],events=events,video=r['video'],video_sha256=r['video_sha256'],info=info,trajectory_sha256=r['trajectory_sha256']))
        p=dest/f'train_{half*4:02d}_{half*4+3:02d}.jpg';page.save(p,quality=94);pages.append(dict(path=str(p),sha256=sha(p)))
    write(dest/'manifest.json',dict(script_sha256=sha(Path(__file__)),relabel_aggregate_sha256=sha(OUT/'relabel_aggregate.json'),pages=pages,records=records,visual_review_pending=True,
        scope='All20train candidate expert labels, sevenevent samples each plus completeoriginal50fps videos. Sampled evidence doesnotestablish continuousplaybackreview or policyperformance. No data acceptance bythisrenderer.'))
    print('ALL20 MEDIA COMPLETE')


if __name__=='__main__':main()
