"""Standalone diagnostic plots for frozen teacher pairs; no gate changes."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from replicate_contact_train import OUT
from evaluate_v6_progress import read,sha,write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result=read(OUT/'aggregate.json');assert result['audit_passed']
    dest=OUT/'plots';dest.mkdir(exist_ok=False)
    colors={'clearance':'#ae542e','unload':'#246d91'};files=[]
    for metric in ('wrist_jerk','wrist_effort'):
        fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True,layout='constrained')
        for parent,ax in enumerate(axes.ravel()):
            for profile,group in result['groups'].items():
                r=next(r for r in group['records'] if r['parent']==parent)
                if metric=='wrist_jerk':
                    with np.load(Path(r['collection'])/'episode_000/trajectory.npz',allow_pickle=False) as d:q=d['state'][:101,11].astype(float)
                    y=abs(np.diff(q,n=3)/.02**3);t=np.arange(3,len(q))*.02
                else:
                    with np.load(OUT/'teacher_replays'/f'case_{r["index"]:02d}'/'samples.npz',allow_pickle=False) as d:y=abs(d['poststep_wrist_effort_nm'][:100]);t=(np.arange(len(y))+1)*.02
                ax.plot(t,y,color=colors[profile],label=profile,lw=1.3)
            ax.axvspan(0,.2,color='#246d91',alpha=.08);ax.axvline(1,color='#999',lw=.8,ls=':')
            ax.set_title(f'Parent {parent} | {r["role"]}',fontsize=10)
            ax.grid(alpha=.2);ax.set_xlim(0,2)
            if parent>=6:ax.set_xlabel('Seconds since expert takeover')
            if parent%2==0:ax.set_ylabel('|wrist jerk| (rad/s³)' if metric=='wrist_jerk' else '|wrist effort| (N m)')
        axes[0,0].legend(frameon=False)
        fig.suptitle('EXPERT same-state pairs: first 2 seconds\n0–0.2 s shaded = candidate stationary unloading; not pi0 inference',fontsize=13)
        path=dest/(metric+'.png');fig.savefig(path,dpi=150);plt.close(fig);files.append(dict(path=str(path),sha256=sha(path)))
    # Box exposure and low movement are distinct; missing phases stay unavailable.
    boxrows=[]
    for parent in range(8):
        items=[('v6 learner',result['learner_box_records'][parent]['box'])]
        items += [(p,next(r for r in g['records'] if r['parent']==parent)['box']) for p,g in result['groups'].items()]
        for name,b in items:boxrows.append(dict(parent=parent,profile=name,**b))
    write(dest/'manifest.json',dict(script_sha256=sha(Path(__file__)),aggregate_sha256=sha(OUT/'aggregate.json'),plots=files,box_records=boxrows,
        scope='Diagnostic physical replay plots, original predeclared gates unchanged. First2s view guards against shifted peak. Boxgeometry/speed is not policy intent. Null means no covered boxphase, not smoothness.'))
    print('PLOTS COMPLETE',dest)


if __name__=='__main__':main()
