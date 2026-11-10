"""Original-video frame audit and outcome-transparent previews for 12 random poses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_three_video_comparison import find_font
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    summary=read(out/'aggregate.json');ledger=read(out/'comparison.json')
    assert summary['complete'] and summary['audit_passed'] and len(ledger['results'])==12
    media=out/'media';media.mkdir(exist_ok=False)
    font=ImageFont.truetype(str(find_font(None)),17)
    manifests=[];tiles=[]
    for row,result in zip(ledger['results'],summary['trial_results'],strict=True):
        directory=Path(row['run_directory']);video=Path(row['video'])
        assert sha(video)==row['video_sha256']
        receipts=[json.loads(s) for s in (directory/'episode_000_multiview.jsonl').read_text().splitlines()]
        tasks=[json.loads(s) for s in (directory/'episode_000_task.jsonl').read_text().splitlines()]
        milestones={'initial':0}
        for name,key in [('near_grasp','near_grasp_region'),('lifted','ever_grasped'),('box','inside_box')]:
            step=next((i+1 for i,t in enumerate(tasks) if t.get(key)),None)
            milestones[name]=next((i for i,r in enumerate(receipts) if r['source_control_step']>=step),None) if step is not None else None
        milestones['final']=len(receipts)-1
        wanted={i for i in milestones.values() if i is not None};frames={};count=0
        reader=imageio.get_reader(video,input_params=['-threads','1'])
        try:
            meta=reader.get_meta_data()
            for i,frame in enumerate(reader):
                count+=1
                if i in wanted:frames[i]=frame.copy()
        finally:reader.close()
        assert meta['fps']==25 and count==len(receipts) and frames[0].shape==(720,1280,3)
        assert set(frames)==wanted
        tile=Image.new('RGB',(640,415),(16,23,29));draw=ImageDraw.Draw(tile)
        color=(111,220,170) if row['success'] else (255,153,142)
        x,y,z=result['sample_position']
        draw.text((12,7),f"{row['index']+1:02d} | {'SUCCESS' if row['success'] else 'FAILURE'} | {result['simulated_seconds']:.2f}s",font=font,fill=color)
        draw.text((12,30),f'x={x:.4f} y={y:.4f} yaw={result["sample_yaw_degrees"]:.2f} deg',font=font,fill=(215,226,235))
        tile.paste(Image.fromarray(frames[milestones['final']]).resize((640,360),Image.Resampling.LANCZOS),(0,55))
        tiles.append(tile)
        event_paths={}
        for name,index in milestones.items():
            if row['index'] in (0,4) and index is not None:
                path=media/f"pose_{row['index']+1:02d}_{name}.png"
                Image.fromarray(frames[index]).save(path);event_paths[name]=str(path)
        manifests.append({'index':row['index'],'success':row['success'],'seed':row['seed'],
            'source':str(video),'source_sha256':row['video_sha256'],'frame_count':count,'fps':25,
            'size':[1280,720],'milestone_video_indices':milestones,'saved_event_frames':event_paths,
            'milestone_source_receipts':{name:receipts[i] if i is not None else None for name,i in milestones.items()},
            'failure_class':result['failure_class'],'transport_warning':result['transport_warning']})
    pages=[]
    for start in (0,6):
        page=Image.new('RGB',(1280,3*415),(8,14,20))
        for offset,tile in enumerate(tiles[start:start+6]):page.paste(tile,((offset%2)*640,(offset//2)*415))
        path=media/f'all_outcomes_{start//6+1:02d}.jpg';page.save(path,quality=93);pages.append(str(path))
    first=ledger['results'][0]
    copy=media/'first_random_pose_multiview.mp4';shutil.copyfile(first['video'],copy)
    assert sha(copy)==first['video_sha256']
    manifest={'schema':'random-multiview-review-v1','source_comparison_sha256':sha(out/'comparison.json'),
        'builder_sha256':sha(Path(__file__)),'attempts':manifests,'overview_pages':pages,
        'first_example':{'selection':'First predeclared seed, not success-filtered. No retiming or re-encoding.',
                         'video':str(copy),'sha256':sha(copy)},
        'time_alignment':'Composite video frame maps via its receipt to an older same-snapshot front/side/wrist state. Task row i is post-step i+1. Original policy-camera video is separate.',
        'scope':'All12 outcomes retained, including one no-lift failure and an accounted transport wait. Video uses simulation time; wall-time waits are not added to MP4 duration. Sampled frames are not continuous video review.',
        'manual_review_pending':True}
    (media/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'media':str(media),'overviews':pages,'first_example':str(copy)},indent=2))


if __name__=='__main__':main()
