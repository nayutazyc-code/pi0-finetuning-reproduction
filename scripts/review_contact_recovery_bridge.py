"""All eight expert-candidate outcomes, event sheets and a paired-source grid."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from evaluate_v6_progress import read,sha,write
from build_three_video_comparison import find_font,video_info
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio_ffmpeg


def main():
    out=ROOT/'outputs/contact_recovery_20261029';audit=read(out/'bridged_audit.json');assert audit['passed']
    media=out/'bridge_media';media.mkdir(exist_ok=False);font=ImageFont.truetype(str(find_font(None)),16)
    records=[];pages=[]
    for batch,name in [('clearance','clearance_candidates.json'),('bridged','bridged_candidates.json')]:
        ledger=read(out/name);selected=[]
        page=Image.new('RGB',(1440,1080),'#edf2f5');draw=ImageDraw.Draw(page)
        title='8mm clearance expert' if batch=='clearance' else '8mm clearance +0.2s initial command bridge'
        draw.text((10,8),title+' | all4 fixed train-parent states | EXPERT demonstrations, not pi0 results',font=font,fill='black')
        for index,row in enumerate(ledger['results']):
            path=Path(row['collection'])/'episode_000';meta=read(path/'episode.json');tasks=[json.loads(x) for x in (path/'steps.jsonl').read_text().splitlines()]
            with np.load(path/'trajectory.npz',allow_pickle=False) as d:states=d['state'].copy()
            arms=[0,1,2,3,4,5,7,8,9,10,11,12];peak=int(np.argmax(np.max(abs(np.diff(states[:,arms].astype(float),n=3)/.02**3),axis=1)))+3 if len(states)>3 else 0
            post=lambda key:next((i+1 for i,t in enumerate(tasks) if t.get(key)),None)
            box=post('inside_box')
            released=next((i+1 for i,t in enumerate(tasks) if box is not None and i>=box and t.get('released')),None)
            events=[('Start',0),('Lifted pinch',post('grasped')),('Box entry',box),('Released in box',released),('Peak physical jerk',peak),('Final',len(tasks))]
            y=40+index*260;draw.text((8,y),f"State{index+1} | {'SUCCESS' if row['automatic_success'] else 'FAILURE'} | {len(tasks)*.02:.2f} simulated seconds",font=font,fill='black')
            for col,(label,step) in enumerate(events):
                x=col*240;draw.text((x+5,y+23),label,font=font,fill='black')
                if step is None:
                    draw.rectangle((x,y+64,x+239,y+244),fill='#d4dadd');draw.text((x+20,y+135),'MISSING',font=font,fill='#a22b22');continue
                draw.text((x+5,y+43),f'Source frame {step}',font=font,fill='black')
                p=path/'terminal.png' if step==len(tasks) else path/'rgb'/f'{step:06d}.png'
                with Image.open(p) as im:page.paste(im.convert('RGB').resize((240,180)),(x,y+64))
            video=path/'preview.mp4';info=video_info(video)
            assert info['fps']==50 and info['frame_count']==len(tasks)+1
            record=dict(batch=batch,index=index,success=row['automatic_success'],events=events,video=str(video),video_sha256=sha(video),info=info,
                        source_collection=row['collection'],trajectory_sha256=sha(path/'trajectory.npz'))
            records.append(record);selected.append(record)
        p=media/(batch+'_all4_outcomes.jpg');page.save(p,quality=94);pages.append(dict(path=str(p),sha256=sha(p)))
    # Pair columns by identical takeover state, original on left and new on right.
    paired=[r for index in range(4) for batch in ('clearance','bridged') for r in records if r['index']==index and r['batch']==batch]
    maximum=max(r['info']['frame_count'] for r in paired)
    header=Image.new('RGBA',(960,1600),(0,0,0,0));draw=ImageDraw.Draw(header);font=ImageFont.truetype(str(find_font(None)),17)
    for k,r in enumerate(paired):
        x=(k%2)*480;y=(k//2)*400;draw.rectangle((x,y,x+479,y+39),fill=(20,28,38,255))
        draw.text((x+6,y+2),f"State{r['index']+1} | {r['batch']} expert | {'SUCCESS' if r['success'] else 'FAILURE'}",font=font,fill='white')
        draw.text((x+6,y+21),f"1x sim time; terminal freeze after {(r['info']['frame_count']-1)/50:.2f}s",font=font,fill='white')
    head=media/'comparison_labels.png';header.save(head);dest=media/'all8_expert_bridge_comparison.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n','-loglevel','error','-filter_threads','1','-filter_complex_threads','1']
    for r in paired:cmd+=['-threads','1','-i',r['video']]
    cmd+=['-loop','1','-i',str(head)]
    filters=[f'[{k}:v]setpts=PTS-STARTPTS,scale=480:360,pad=480:400:0:40:color=black,tpad=stop_mode=clone:stop={maximum-r["info"]["frame_count"]}[v{k}]' for k,r in enumerate(paired)]
    filters += [f'[v{k*2}][v{k*2+1}]hstack=inputs=2[r{k}]' for k in range(4)]
    filters += ['[r0][r1][r2][r3]vstack=inputs=4[g]','[g][8:v]overlay=0:0,fps=25[out]']
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-frames:v',str(math.ceil(maximum/2)),'-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p',str(dest)]
    subprocess.run(cmd,check=True);info=video_info(dest)
    assert info['width']==960 and info['height']==1600 and info['fps']==25 and info['frame_count']==math.ceil(maximum/2)
    import imageio.v2 as imageio
    reader=imageio.get_reader(dest,input_params=['-threads','1'])
    try:imageio.imwrite(media/'comparison_first_frame.png',reader.get_data(0))
    finally:reader.close()
    write(media/'manifest.json',dict(schema='contact-recovery-media-v1',script_sha256=sha(Path(__file__)),audit_sha256=sha(out/'bridged_audit.json'),
        pages=pages,attempts=records,video=dict(path=str(dest),sha256=sha(dest),info=info),encoder_command=cmd,visual_review_pending=True,
        scope='All8 expert candidates, four paired identical training takeover states, clearance left/clearance+bridge right. Expert labels, not learned-model inference.25fps presentation samples original50fps videos at1xsimulationtime with terminalfreezes; originals and RGB retained.'))
    print(dest)


if __name__=='__main__':main()
