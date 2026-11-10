"""All-attempt timing/camera/motion plots; no success-only timing selection."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from evaluate_control_overruns import OUT
from evaluate_v6_progress import read,sha,write
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    report=read(OUT/'aggregate.json');assert report['audit_passed']
    out=OUT/'plots';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    colors={'baseline':'#2563a6','recover':'#bd6532'}
    labels={'baseline':'Default clock','recover':'Overrun recovery'}
    fields=[('A  Typical action interval','Trial p99 (ms)',lambda r:r['action_apply_ms']['p99']),
            ('B  Longest action interval','Trial maximum (ms)',lambda r:r['action_apply_ms']['max']),
            ('C  Environment-step envelope','Largest environment wall span (ms)',lambda r:r['environment_wall_ms']['max']),
            ('D  Physical motion remains separate','Active physical jerk p99 (rad/s³)',lambda r:r['quality']['active_frame_max_jerk_rad_s3']['p99'])]
    points=[]
    for ax,(title,ylabel,fn) in zip(axes.flat,fields):
        panel_values=[]
        for x,(key,group) in enumerate(report['groups'].items()):
            values=[]
            for i,row in enumerate(group['trials']):
                value=fn(row);values.append(value);panel_values.append(value)
                covered=row['quality']['coverage']['transport_motion_comparable']
                ax.scatter(x+(i-2.5)*.045,value,s=55,facecolors=colors[key] if covered else 'white',edgecolors=colors[key],linewidths=1.4,zorder=3)
                points.append(dict(panel=title,profile=key,repeat=row['repeat'],value=value,transport_covered=covered))
            ax.hlines(np.median(values),x-.22,x+.22,color='#242424',linewidth=1.5)
        ax.set(title=title,ylabel=ylabel,xticks=[0,1],xticklabels=list(labels.values()))
        ax.set_xlim(-.4,1.4);ax.set_ylim(0,max(panel_values)*1.15);ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Overrun deadline recovery: conditional live pilot',fontsize=17,x=.07,ha='left',y=.98)
    fig.text(.07,.928,'6+6 fresh alternating trials; same continued checkpoint, kp200 and RTC20. Policy noise unpaired.',fontsize=10)
    fig.text(.07,.023,'All attempts retained. Hollow = incomplete transport; black = all-attempt median. Video uses simulation time.',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.91));path=out/'runtime_comparison.png';fig.savefig(path,dpi=180);plt.close(fig)
    write(out/'manifest.json',dict(schema='control-overrun-runtime-plots-v1',builder_sha256=sha(Path(__file__)),
        aggregate_sha256=sha(OUT/'aggregate.json'),images=[dict(path=str(path),sha256=sha(path))],points=points,visual_review_pending=True))
    print(path)


if __name__=='__main__':main()
