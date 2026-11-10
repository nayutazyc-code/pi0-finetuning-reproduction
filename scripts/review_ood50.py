"""Build all-outcome OOD contact sheets without success-selected omissions."""
from pathlib import Path
import hashlib,json,subprocess
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from build_three_video_comparison import find_font,video_info
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'outputs/ood50_20261029';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();a=read(out/'aggregate.json');assert a['complete'] and a['audit_passed'] and len(a['trials'])==50
 dest=out/'media';dest.mkdir(exist_ok=False);font=ImageFont.truetype(str(find_font(None)),14);ff=imageio_ffmpeg.get_ffmpeg_exe();pages=[];records=[]
 for offset in range(0,50,10):
  page=Image.new('RGB',(1200,2150),'#edf2f5');draw=ImageDraw.Draw(page);draw.text((8,4),'All50 frozen OOD outcomes | static video samples | no omitted failures',font=font,fill='black')
  for row,t in enumerate(a['trials'][offset:offset+10]):
   video=Path(t['video']);assert sha(video)==t['video_sha256'];info=video_info(video);tasks=[json.loads(s) for s in (Path(t['run_directory'])/'episode_000_task.jsonl').read_text().splitlines()];lift=next((i+1 for i,r in enumerate(tasks) if r['ever_grasped']),None);y=30+row*211
   s=t['requested_scene'];draw.text((6,y),f"#{t['index']:02d} {t['group']} {'SUCCESS' if t['success'] else 'FAILURE'} | XY {s['banana_position'][0]:.3f},{s['banana_position'][1]:.3f} yaw{s['banana_yaw_degrees']:.1f}",font=font,fill='black')
   events=[('Start',0),('Lifted pinch',None if lift is None else lift*.02),('Final',max(0,info['duration_seconds']-.04))]
   for col,(label,sec) in enumerate(events):
    x=col*400;draw.text((x+5,y+21),label+(' MISSING' if sec is None else f' video {sec:.2f}s'),font=font,fill='black')
    if sec is not None:
     sec=min(round(sec*25)/25,(info['frame_count']-1)/25);raw=subprocess.check_output([ff,'-v','error','-threads','1','-ss',str(sec),'-i',str(video),'-frames:v','1','-vf','scale=220:165','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-']);assert len(raw)==220*165*3;page.paste(Image.frombytes('RGB',(220,165),raw),(x,y+41))
   records.append(dict(index=t['index'],group=t['group'],success=t['success'],video=str(video),video_sha256=t['video_sha256'],info=info,events=events))
  p=dest/f'outcomes_{offset:02d}_{offset+9:02d}.jpg';page.save(p,quality=94);pages.append(dict(path=str(p),sha256=sha(p)))
 (dest/'manifest.json').write_text(json.dumps(dict(schema='ood50-all-outcomes-media-v1',attempts=records,pages=pages,aggregate_sha256=sha(out/'aggregate.json'),script_sha256=sha(Path(__file__)),visual_review_pending=True,scope='All50 start/lift/final samples; missing events retained. Simulation-time video does not reproduce wall waits; static sampling is not complete continuous playback.'),indent=2)+'\n')
 print('COMPLETE',dest)
if __name__=='__main__':main()
