"""Preserve all-state takeover motion comparison, including failed quality gates."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from evaluate_v6_progress import read,sha,write
OUT=ROOT/'outputs/contact_recovery_20261029'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    before=read(OUT/'audit.json');after=read(OUT/'bridged_audit.json');assert before['passed'] and after['passed']
    plots=OUT/'plots';plots.mkdir(exist_ok=False)
    colors={'original':'#777777','clearance':'#2563a6','bridged':'#c26b35'}
    groups={k:[r for r in (before if k!='bridged' else after)['records'] if r['batch']==k] for k in colors}
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    fields=[('A  Contact loading after expert takeover','Maximum post-step wrist effort (N m)','poststep_wrist_effort_peak_nm'),
            ('B  Takeover physical jerk remains unresolved','First-second wrist jerk peak (rad/s³)','first_second_wrist_jerk_peak')]
    points=[]
    for ax,(title,ylabel,key) in zip(axes,fields):
        for x,(name,rows) in enumerate(groups.items()):
            values=[r[key] for r in rows]
            for j,(r,value) in enumerate(zip(rows,values)):
                ax.scatter(x+(j-1.5)*.07,value,s=55,facecolors=colors[name] if r['automatic_success'] else 'white',edgecolors=colors[name],zorder=3)
                ax.annotate(str(r['index']+1),(x+(j-1.5)*.07,value),xytext=(5,4),textcoords='offset points',fontsize=9)
                points.append(dict(panel=title,batch=name,state=r['index']+1,value=value,automatic_success=r['automatic_success']))
            ax.hlines(np.median(values),x-.22,x+.22,color='black',linewidth=1.3)
        ax.set(title=title,ylabel=ylabel,xticks=[0,1,2],xticklabels=['Original expert','8mm clearance','Clearance + bridge'])
        ax.set_ylim(0,max(p['value'] for p in points if p['panel']==title)*1.2);ax.set_xlim(-.4,2.4)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Expert recovery candidates: successful completion is not a smoothness pass',fontsize=14,x=.07,ha='left',y=.99)
    fig.text(.07,.90,'Four identical train-parent states per stage; no new pi0 inference or training. Numbers identify takeover states.',fontsize=10)
    fig.text(.07,.015,'Hollow markers = failed expert task. Black lines = all4 median. All samples retained; 50Hz post-step effort omits 2ms peaks.',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.87));path=plots/'expert_contact_and_takeover.png';fig.savefig(path,dpi=170);plt.close(fig)
    write(plots/'manifest.json',dict(schema='contact-recovery-plot-v1',source_audit_sha256=sha(OUT/'audit.json'),bridge_audit_sha256=sha(OUT/'bridged_audit.json'),
        image=str(path),image_sha256=sha(path),points=points,script_sha256=sha(Path(__file__)),visual_review_pending=True))
    print(path)


if __name__=='__main__':main()
