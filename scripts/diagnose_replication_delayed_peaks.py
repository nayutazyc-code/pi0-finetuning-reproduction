"""Post-hoc first-three-second diagnostic; original gates and decision unchanged."""
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
    dest=OUT/'delayed_peak_diagnostic';dest.mkdir(exist_ok=False)
    fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True,layout='constrained');records=[]
    for parent,ax in enumerate(axes.ravel()):
        for profile,color in [('clearance','#ae542e'),('unload','#246d91')]:
            r=next(r for r in result['groups'][profile]['records'] if r['parent']==parent)
            path=Path(r['collection'])/'episode_000/trajectory.npz'
            with np.load(path,allow_pickle=False) as d:q=d['state'][:151,11].astype(float);phase=d['phase'].copy()
            j=abs(np.diff(q,n=3)/.02**3);time=(np.arange(len(j))+3)*.02;i=int(np.argmax(j));step=i+3
            force=np.load(OUT/'teacher_replays'/f'case_{r["index"]:02d}'/'samples.npz',allow_pickle=False)['poststep_wrist_effort_nm']
            phases=[dict(phase=str(p),step=int(k),time_seconds=float(k*.02)) for k,p in enumerate(phase) if k==0 or phase[k-1]!=p]
            records.append(dict(parent=parent,profile=profile,source_trajectory_sha256=sha(path),first_three_second_wrist_jerk_peak=float(j[i]),
                peak_time_seconds=float(time[i]),peak_phase=str(phase[step]),phase_starts=phases,
                first_0_2_second_poststep_effort_peak_nm=float(np.max(abs(force[:10])))))
            ax.plot(time,j,color=color,label=profile,lw=1.2);ax.plot(time[i],j[i],'o',color=color,ms=3)
        ax.axvline(2,color='#555',ls=':',lw=1);ax.axvspan(0,.2,color='#246d91',alpha=.08);ax.grid(alpha=.2);ax.set_xlim(0,3)
        ax.set_title(f'Parent {parent}',fontsize=11)
        if parent>=6:ax.set_xlabel('Seconds since expert takeover')
        if parent%2==0:ax.set_ylabel('|wrist jerk| (rad/s³)')
    axes[0,0].legend(frameon=False)
    fig.suptitle('POST-HOC EXPERT diagnostic: extend window to 3 s\nDotted 2 s cutoff exposes delayed lift peaks; original gates unchanged',fontsize=13)
    plot=dest/'first_three_seconds.png';fig.savefig(plot,dpi=150);plt.close(fig)
    stats={}
    for p in ('clearance','unload'):
        vals=[r['first_three_second_wrist_jerk_peak'] for r in records if r['profile']==p]
        stats[p]=dict(median=float(np.median(vals)),worst=max(vals))
    write(dest/'diagnostic.json',dict(schema='posthoc-delayed-lift-peak-diagnostic-v1',aggregate_sha256=sha(OUT/'aggregate.json'),script_sha256=sha(Path(__file__)),
        plot=dict(path=str(plot),sha256=sha(plot)),records=records,statistics=stats,
        median_ratio=stats['unload']['median']/stats['clearance']['median'],worst_ratio=stats['unload']['worst']/stats['clearance']['worst'],
        scope='Post-hoc diagnostic prompted by first2s curve review; not a replacementgate, selection criterion, new treatment or policy trial. Typical liftpeaks shift0.2s with little amplitudechange. Force sampled20ms, not2ms impulses. Originalfailure/decision unchanged.'))
    (dest/'source.py').write_bytes(Path(__file__).read_bytes());print(stats)


if __name__=='__main__':main()
