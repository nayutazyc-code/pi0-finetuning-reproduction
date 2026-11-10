"""All16 ACT start/lift/final outcomes, preserving failures."""
from pathlib import Path
import json,hashlib,subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from build_three_video_comparison import find_font,video_info
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'outputs/act_baseline_20261029';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d=read(out/'evaluation.json');assert d['complete'] and read(out/'evaluation_audit.json')['passed'];dest=out/'media';dest.mkdir(exist_ok=False);font=ImageFont.truetype(str(find_font(None)),15);ff=imageio_ffmpeg.get_ffmpeg_exe();pages=[];videos=[]
 for start in [0,8]:
  page=Image.new('RGB',(1200,1730),'#edf2f5');draw=ImageDraw.Draw(page);draw.text((6,3),'ACT final20000 | all16 outcomes | start/lift/final samples, no omitted failures',font=font,fill='black')
  for j,r in enumerate(d['results'][start:start+8]):
   path=Path(r['run_directory']);video=path/'episode.mp4';info=video_info(video);tasks=[json.loads(s) for s in (path/'task.jsonl').read_text().splitlines()];lift=next((i for i,t in enumerate(tasks) if t['ever_grasped']),None);events=[('Start',0),('Lifted pinch',None if lift is None else (lift+1)*.02),('Final',info['duration_seconds']-.04)];y=32+j*211
   draw.text((6,y),f"#{r['index']:02d} {r['split']} {r['layout']} repeat{r['repeat']} {'SUCCESS' if r['success'] else 'FAILURE'} sim{r['simulated_seconds']:.2f}s",font=font,fill='black')
   for k,(label,t) in enumerate(events):
    x=k*400;draw.text((x+5,y+21),label+(' MISSING' if t is None else f' video{t:.2f}s'),font=font,fill='black')
    if t is None:continue
    t=min(round(t*25)/25,(info['frame_count']-1)/25);pixels=subprocess.check_output([ff,'-v','error','-threads','1','-ss',str(t),'-i',str(video),'-frames:v','1','-vf','scale=220:165','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-']);assert len(pixels)==220*165*3;page.paste(Image.frombytes('RGB',(220,165),pixels),(x,y+41))
   videos.append(dict(index=r['index'],split=r['split'],success=r['success'],path=str(video),sha256=sha(video),events=events,info=info))
  p=dest/f'outcomes_{start:02d}_{start+7:02d}.jpg';page.save(p,quality=94);pages.append(dict(path=str(p),sha256=sha(p)))
 (dest/'manifest.json').write_text(json.dumps(dict(attempts=videos,pages=pages,evaluation_sha256=sha(out/'evaluation.json'),visual_review_pending=True,scope='All16 static samples, not complete continuous playback. Offline sim time video does not reproduce synchronous network waits.'),indent=2)+'\n')
 print('COMPLETE',dest)
if __name__=='__main__':main()
