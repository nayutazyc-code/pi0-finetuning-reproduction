"""Complete-lift and whole-episode comparison across the three expert arms."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from experiment_ik_continuous import OUT
from evaluate_v6_progress import read,sha,write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    result=read(OUT/'aggregate.json');assert result['audit_passed'];dest=OUT/'plots';dest.mkdir(exist_ok=False)
    colors={'baseline':'#ae542e','tight_only':'#246d91','continuous':'#26764c'};files=[]
    fig,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
    for parent,ax in enumerate(axes.ravel()):
        for profile,g in result['groups'].items():
            r=next(r for r in g['records'] if r['parent']==parent)
            with np.load(Path(r['collection'])/'episode_000/trajectory.npz') as d:q=d['state'][:,11].astype(float);phases=d['phase']
            lift=np.flatnonzero(phases=='lift');mask=np.array([all(phases[k:k+4]=='lift') for k in range(len(q)-3)])
            j=abs(np.diff(q,n=3)/.02**3);t=(np.arange(len(j))+3-lift[0])*.02
            ax.plot(t[mask],j[mask],color=colors[profile],label=profile,lw=1.2)
        ax.set_title(f'Parent {parent}',fontsize=11);ax.grid(alpha=.2)
        if parent>=6:ax.set_xlabel('Seconds from lift-phase start')
        if parent%2==0:ax.set_ylabel('|wrist jerk| (rad/s³)')
    axes[0,0].legend(frameon=False)
    fig.suptitle('EXPERT full lift phase, aligned at phase start\nOriginal / precision-only / precision + continuous transit; not learned-policy inference',fontsize=13)
    p=dest/'full_lift.png';fig.savefig(p,dpi=150);plt.close(fig);files.append(dict(path=str(p),sha256=sha(p)))
    fig,axes=plt.subplots(1,4,figsize=(15,4.8),layout='constrained');profiles=list(result['groups'])
    metrics=[('active p99 (rad/s³)',lambda r:r['motion_quality']['active_frame_max_jerk_rad_s3']['p99']),
        ('worst arm jerk (rad/s³)',lambda r:r['motion_quality']['active_frame_max_jerk_rad_s3']['max']),
        ('low-motion dwell (s)',lambda r:r['motion_quality']['low_motion_pause_total_seconds']),('duration (s)',lambda r:r['frames']*.02)]
    for ax,(name,get) in zip(axes,metrics,strict=True):
        vals=np.array([[get(next(r for r in result['groups'][profile]['records'] if r['parent']==parent)) for profile in profiles] for parent in range(8)])
        for row in vals:ax.plot(range(3),row,color='#aab3bb',alpha=.6,lw=.8)
        for i,profile in enumerate(profiles):ax.scatter(np.repeat(i,8),vals[:,i],color=colors[profile],s=24);ax.plot([i-.12,i+.12],[np.median(vals[:,i])]*2,color='black',lw=2)
        ax.set_title(name);ax.set_xticks(range(3),profiles,rotation=18);ax.grid(axis='y',alpha=.2)
    fig.suptitle('EXPERT full-episode physical metrics | all8 states; black ticks = medians',fontsize=13)
    p=dest/'whole_episode_metrics.png';fig.savefig(p,dpi=150);plt.close(fig);files.append(dict(path=str(p),sha256=sha(p)))
    write(dest/'manifest.json',dict(script_sha256=sha(Path(__file__)),aggregate_sha256=sha(OUT/'aggregate.json'),plots=files,visual_review_pending=True,
        scope='All8states across3expertarms, first16reference trajectories reused fromround1. Fullphase andfulltrajectory, not clipped2s windows. Not independent24trials or a learned-model result.'))


if __name__=='__main__':main()
