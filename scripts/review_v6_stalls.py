"""Offline, outcome-transparent phase and wall-gap review of recorded v6 runs.

Never converts validation policy failures into training demonstrations. Selected
events are diagnostic extremes, not representative videos or causal evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from pi0_repro.motion_quality import motion_quality
from pi0_repro.diagnostics import task_progress


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]


def interval(first,last):
    """Censored endpoints remain unavailable, never become zero-duration wins."""
    return None if first is None or last is None else max(0.,last-first)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cohort=args.cohort.resolve();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs')
    ledger=read(cohort/'comparison.json');assert ledger['complete']
    out.mkdir(parents=True,exist_ok=False)
    records=[];candidates=[]
    for row in ledger['results']:
        run=Path(row['run_directory']);tasks=rows(run/'episode_000_task.jsonl')
        controls=rows(run/'episode_000_control.jsonl')
        with np.load(run/'episode_000.npz',allow_pickle=False) as data:
            progress=task_progress(tasks,data['action'])
            quality=motion_quality(data['state'],data['action'],tasks,pause_min_seconds=.16)
        summary=read(run/'summary.json')['episodes'][0]
        near=progress['first_near_grasp_seconds'];lift=progress['first_confirmed_lifted_pinch_seconds']
        box=progress['first_inside_box_seconds'];duration=summary['steps']*.02
        record=dict(index=row['index'],run_directory=str(run),seed=summary['seed'],success=summary['success'],
            bindings={n:sha(run/n) for n in ('episode_000.npz','episode_000_task.jsonl','episode_000_control.jsonl','summary.json')},
            progress=progress,low_motion_diagnostic=quality,
            near_to_lift_seconds=interval(near,lift),lift_to_containment_seconds=interval(lift,box),
            near_to_end_censored_seconds=duration-near if near is not None and lift is None else None,
            lift_to_end_censored_seconds=duration-lift if lift is not None and box is None else None)
        records.append(record)
        # Rank completed intervals and censored exposure explicitly together for
        # inspection only; exposure is not a completed duration or good score.
        for kind,start,end in [('near_to_lift',near,lift),('post_lift_pre_containment',lift,box)]:
            if start is not None:
                finish=duration if end is None else end
                candidates.append(dict(kind=kind,index=row['index'],run_directory=str(run),
                    score_seconds=finish-start,censored=end is None,
                    center_control_step=round((start+min(finish,start+8))/2/.02),
                    interval_start_seconds=start,interval_end_seconds=finish))
        if len(controls)>1:
            gaps=np.diff([c['action_apply_wall_seconds'] for c in controls])
            i=int(np.argmax(gaps))+1
            candidates.append(dict(kind='wall_gap',index=row['index'],run_directory=str(run),
                score_seconds=float(gaps[i-1]),center_control_step=i,
                previous_control=controls[i-1],current_control=controls[i],censored=False))
    chosen=[]
    for kind in ('near_to_lift','post_lift_pre_containment','wall_gap'):
        chosen.extend(sorted((c for c in candidates if c['kind']==kind),key=lambda c:c['score_seconds'],reverse=True)[:2])
    # Media work happens only after the timed experiment has completed.
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from build_three_video_comparison import find_font
    font=ImageFont.truetype(str(find_font(None)),16)
    for event_id,event in enumerate(chosen):
        run=Path(event['run_directory']);video=run/'episode_000_multiview.mp4'
        if video.exists():
            receipts=rows(run/'episode_000_multiview.jsonl')
        else:
            video=run/'episode_000.mp4'
            if not video.exists():
                event['media_unavailable']='This control-only run intentionally has no video';continue
            fps=read(run/'run.json')['config']['simulation']['video_fps']
            receipts=[{'source_control_step':i*round(50/fps)} for i in range((read(run/'summary.json')['episodes'][0]['steps']-1)//round(50/fps)+2)]
        event['source_video']=str(video);event['source_video_sha256']=sha(video)
        center=event['center_control_step']
        desired=[max(0,center+d) for d in (-100,-50,-10,0,50,100)]
        indices=[min(range(len(receipts)),key=lambda i:abs(receipts[i]['source_control_step']-step)) for step in desired]
        frames={};reader=imageio.get_reader(video,input_params=['-threads','1'])
        try:
            metadata=reader.get_meta_data()
            for i,frame in enumerate(reader):
                if i in indices:frames[i]=frame.copy()
                if i>=max(indices):break
        finally:reader.close()
        sheet=Image.new('RGB',(1280,3*395),(16,23,29));draw=ImageDraw.Draw(sheet)
        event['sampled_frames']=[]
        for j,i in enumerate(indices):
            frame=frames[i];x=(j%2)*640;y=(j//2)*395
            draw.text((x+8,y+5),f"case {event['index']+1} {event['kind']} | source step {receipts[i]['source_control_step']}",font=font,fill=(194,226,235))
            sheet.paste(Image.fromarray(frame).resize((640,360),Image.Resampling.LANCZOS),(x,y+35))
            event['sampled_frames'].append(dict(video_frame=i,receipt=receipts[i]))
        path=out/f'event_{event_id+1:02d}_{event["kind"]}.jpg';sheet.save(path,quality=93)
        event['sheet']=str(path);event['sheet_sha256']=sha(path);event['video_fps']=metadata['fps']
    report=dict(schema='v6-stall-review-v1',source_comparison=str(cohort/'comparison.json'),
        source_comparison_sha256=sha(cohort/'comparison.json'),review_script_sha256=sha(Path(__file__)),
        outcome_count=len(records),records=records,selected_events=chosen,
        selection='Top2 near-to-lift exposures (including explicitly censored failures), top2 post-lift/pre-containment exposures, and top2 wall gaps, across every recorded outcome.',
        limits=['Phase names come from diagnostic task flags, not policy intent. Post-lift/pre-containment is not a geometric proof of being above the box.',
                'Simulation-time clips cannot reproduce wall-clock pauses. Inspect original action-application timestamps.',
                'Low-motion >=0.16s is a supplementary diagnostic, not a replacement for the frozen >=0.5s score; deliberate holds may remain.',
                'Diagnostic validation outcomes and their frames are not training data. New corrections must use train-only parent scenes.',
                'No continuous video playback review or unique physical failure cause is asserted.'],
        visual_review_pending=True)
    (out/'review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'attempts':len(records),'selected_events':len(chosen),'out':str(out)}))


if __name__=='__main__':main()
