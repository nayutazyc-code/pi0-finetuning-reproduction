"""Stack every fresh controller outcome, with source hashes and terminal freezes."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_continued_wrist_kp import OUT
from evaluate_v6_progress import read,sha,write
from build_three_video_comparison import video_info,find_font
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
import imageio.v2 as imageio


def main():
    media=OUT/'media';sources=[]
    for profile in ['baseline','kp150']:
        manifest=read(media/profile/'manifest.json');path=Path(manifest['comparison_video']['path'])
        assert sha(path)==manifest['comparison_video']['sha256'];sources.append(path)
    infos=[video_info(p) for p in sources]
    assert all(i['fps']==25 and i['width']==1600 and i['height']==560 for i in infos)
    maximum=max(i['frame_count'] for i in infos)
    header=Image.new('RGBA',(1600,1200),(0,0,0,0));draw=ImageDraw.Draw(header);font=ImageFont.truetype(str(find_font(None)),23)
    for y,text in [(0,'CONTINUED MODEL | ORIGINAL RIGHT-WRIST GAIN kp200 | 10 fresh trials'),(600,'SAME CONTINUED MODEL | CANDIDATE RIGHT-WRIST GAIN kp150 | 10 fresh trials')]:
        draw.rectangle((0,y,1599,y+39),fill=(15,23,35,255));draw.text((12,y+6),text,font=font,fill='white')
    head=media/'comparison_labels.png';header.save(head);destination=media/'all20_kp200_vs_kp150.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostdin','-n','-loglevel','error','-filter_threads','1','-filter_complex_threads','1']
    for p in sources:cmd+=['-threads','1','-i',str(p)]
    cmd+=['-loop','1','-i',str(head)]
    filters=[f'[{i}:v]setpts=PTS-STARTPTS,pad=1600:600:0:40:color=black,tpad=stop_mode=clone:stop={maximum-info["frame_count"]}[v{i}]' for i,info in enumerate(infos)]
    filters+=['[v0][v1]vstack=inputs=2[g]','[g][2:v]overlay=0:0[out]']
    cmd+=['-filter_complex',';'.join(filters),'-map','[out]','-an','-r','25','-frames:v',str(maximum),'-c:v','libx264','-threads','2','-crf','19','-pix_fmt','yuv420p',str(destination)]
    subprocess.run(cmd,check=True);info=video_info(destination)
    assert info['width']==1600 and info['height']==1200 and info['fps']==25 and info['frame_count']==maximum
    reader=imageio.get_reader(destination,input_params=['-threads','1'])
    try:imageio.imwrite(media/'comparison_first_frame.png',reader.get_data(0))
    finally:reader.close()
    write(media/'comparison_manifest.json',dict(schema='continued-wrist-kp-all20-media-v1',builder_sha256=sha(Path(__file__)),
        video=str(destination),sha256=sha(destination),info=info,encoder_command=cmd,
        sources=[dict(path=str(p),sha256=sha(p)) for p in sources],scope='All20freshtrial outcomes;kp200above,kp150below,each10trials arranged1-5above6-10.1xsimtime25fps,source terminal freezes explicit. Not noise-paired side-by-side trials or wall-time playback.',visual_review_pending=True))
    print(destination)


if __name__=='__main__':main()
