"""Standalone observed trajectory distributions and worst-spike zooms."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from compare_fixed_layout_trajectories import OUT
from evaluate_v6_progress import read,sha,write
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from review_rtc10_tails import physical_peak


def main():
    report=read(OUT/'comparison.json');assert report['audit_passed']
    media=OUT/'plots';media.mkdir(exist_ok=False)
    colors={'v6':'#2563A6','control':'#C66A32'};labels={'v6':'Original v6','control':'Continued model'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
    panels=[('A  Typical physical jerk','Active jerk p99 (rad/s³)',lambda r:r['quality']['active_frame_max_jerk_rad_s3']['p99'],False),
            ('B  Worst physical spikes','Peak jerk (rad/s³; log scale)',lambda r:r['quality']['active_frame_max_jerk_rad_s3']['max'],True),
            ('C  Physical step size','Joint increment p99 (deg / 20 ms)',lambda r:np.rad2deg(r['jumps']['physical_increment_rad']['p99']),False),
            ('D  Raw replacement disagreement','Same-time plan disagreement p99 (deg)',lambda r:np.rad2deg(r['jumps']['same_time_raw_disagreement_rad']['p99']),False)]
    fig,axes=plt.subplots(2,2,figsize=(11,8));points=[]
    for axis,(title,ylabel,fn,log) in zip(axes.flat,panels):
        for x,(model,group) in enumerate(report['groups'].items()):
            rows=group['trials'];ys=[float(fn(r)) for r in rows];covered=[]
            for i,(row,y) in enumerate(zip(rows,ys)):
                ok=row['quality']['coverage']['transport_motion_comparable'];xx=x+(i-4.5)*.025
                axis.scatter(xx,y,s=45,facecolors=colors[model] if ok else 'white',edgecolors=colors[model],linewidths=1.4,zorder=3)
                points.append(dict(panel=title,model=model,repeat=row['repeat'],value=y,covered=ok))
                if ok:covered.append(y)
            axis.hlines(np.median(ys),x-.21,x+.21,color='black',linewidth=1.6)
            axis.hlines(np.median(covered),x-.21,x+.21,color='black',linewidth=1.6,linestyle='dashed')
        axis.set(title=title,ylabel=ylabel,xticks=[0,1],xticklabels=[labels[k] for k in report['groups']])
        axis.set_xlim(-.45,1.45);axis.grid(axis='y',alpha=.18)
        if log:axis.set_yscale('log')
        else:axis.set_ylim(bottom=0)
    fig.suptitle('Same layout: typical jerk improves, worst spikes do not',fontsize=17,x=.08,ha='left',y=.98)
    fig.text(.08,.935,'10 runs per model; earlier continued-model block vs later v6 block. Samples unpaired.',fontsize=11,color='#444444')
    handles=[Line2D([0],[0],marker='o',linestyle='',color='#555555',label='Completed transport'),
             Line2D([0],[0],marker='o',linestyle='',markerfacecolor='white',color='#555555',label='Incomplete transport'),
             Line2D([0],[0],color='black',label='All-attempt median'),Line2D([0],[0],color='black',linestyle='dashed',label='Covered-only median')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.01),fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.92));distribution_path=media/'jump_distributions.png';fig.savefig(distribution_path,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex='col');peaks=[]
    for column,(model,group) in enumerate(report['groups'].items()):
        row=max(group['trials'],key=lambda r:r['quality']['active_frame_max_jerk_rad_s3']['max'])
        run=Path(row['run_directory'])
        with np.load(run/'episode_000.npz',allow_pickle=False) as data:
            q=np.asarray(data['state'],float);action=np.asarray(data['action'],float)
            step,joint,peak,_,_=physical_peak(q);assert joint==11,'Review label assumes right wrist-angle'
            lo=max(1,step-25);hi=min(len(q),step+26);indices=np.arange(lo,hi);times=(indices-step)*.02
            baseline=q[step,joint]
            axes[0,column].plot(times,np.rad2deg(q[indices,joint]-baseline),color=colors[model],label='Physical joint')
            axes[0,column].plot(times,np.rad2deg(action[indices-1,joint]-baseline),color='#666666',linestyle='dashed',label='Last applied target')
            axes[1,column].plot(times,np.rad2deg(q[indices,joint]-q[indices-1,joint]),color=colors[model])
        for ax in axes[:,column]:
            ax.axvline(0,color='#B42318',linewidth=1.3)
            for switch in row['jumps']['switches']:
                t=(switch['control_step']-step)*.02
                if times[0]<=t<=times[-1]:ax.axvline(t,color='#999999',linewidth=.7,alpha=.45)
            ax.grid(axis='y',alpha=.18)
        axes[0,column].set_title(f'{labels[model]} | round {row["repeat"]}\npeak={peak:.1f} rad/s³ at {step*.02:.2f}s',fontsize=12)
        axes[1,column].set_xlabel('Seconds relative to largest jerk sample')
        peaks.append(dict(model=model,repeat=row['repeat'],run_directory=str(run),state_index=step,joint_axis=joint,peak_rad_s3=peak))
    axes[0,0].set_ylabel('Wrist angle relative to peak (deg)');axes[1,0].set_ylabel('Wrist increment (deg / 20 ms)')
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Worst observed spike in each group: right wrist-angle',fontsize=16,x=.08,ha='left',y=.99)
    fig.text(.08,.925,'Independent extreme selection; red = jerk peak, grey = accepted chunk switches. Timing is not causality.',fontsize=10,color='#444444')
    fig.tight_layout(rect=(0,0,1,.91));peak_path=media/'worst_wrist_trajectory.png';fig.savefig(peak_path,dpi=180);plt.close(fig)
    write(media/'manifest.json',dict(schema='fixed-layout-trajectory-plots-v1',builder_sha256=sha(Path(__file__)),comparison_sha256=sha(OUT/'comparison.json'),
        images=[dict(path=str(p),sha256=sha(p)) for p in [distribution_path,peak_path]],distribution_points=points,peak_selections=peaks,
        scope='Static scientific plots from all20 recorded trajectories; unpaired sequential blocks and coverage displayed explicitly. Different extremes are illustrative, not paired failures or a cause diagnosis.',visual_review_pending=True))
    print(distribution_path);print(peak_path)


if __name__=='__main__':main()
