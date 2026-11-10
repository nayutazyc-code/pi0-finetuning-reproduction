"""Aligned contact-release plots for the three retained candidate extremes."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/release_retreat_20261029'

def main():
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    d=read(OUT/'diagnosis.json');assert d['complete']
    dest=OUT/'contact_plots';dest.mkdir(exist_ok=False)
    fig,axes=plt.subplots(4,3,figsize=(14,11),sharex='col',constrained_layout=True)
    records=[]
    for col,index in enumerate((18,19,20)):
        r=next(r for r in d['results'] if r['index']==index and r['instrumented'])
        rows=[json.loads(s) for s in (Path(r['directory'])/'substeps.jsonl').read_text().splitlines()]
        with np.load(Path(r['source'])/'episode_000.npz') as z:q=z['state'].astype(float)
        k=r['release_peak_state_index'];joint=int(np.argmax(abs(np.diff(q[:,7:13],n=3,axis=0)[k-3])))
        near=[x for x in rows if k-3<=x['control_step']<k]
        delta=np.diff([x['qacc'][joint] for x in near]);i=int(np.argmax(abs(delta)))+1;event=near[i]['pre_episode_time']
        selected=[x for x in rows if abs(x['pre_episode_time']-event)<=.08]
        times=np.array([x['pre_episode_time']-event for x in selected])*1000
        offset=selected[0]['pre_q'][joint]
        axes[0,col].plot(times,np.degrees([x['target'][joint]-offset for x in selected]),label='Target')
        axes[0,col].plot(times,np.degrees([x['pre_q'][joint]-offset for x in selected]),label='Measured')
        axes[1,col].plot(times,[x['pre_qvel'][joint] for x in selected],color='#305ca3')
        axes[2,col].plot(times,[x['qfrc_actuator'][joint] for x in selected],label='Actuator')
        axes[2,col].plot(times,[x['constraint_torque']['contact'][joint] for x in selected],label='Contact')
        axes[2,col].plot(times,[x['qfrc_passive'][joint] for x in selected],label='Passive',ls='--')
        def force(row):
            return sum(max(0.,c['force_contact_frame'][0]) for c in row['contacts'] if c['active'] and any('vx300s_right/' in n for n in c['geoms']) and any(n.startswith('box_') for n in c['geoms']))
        axes[3,col].plot(times,[force(x) for x in selected],color='#b34d24')
        axes[0,col].set_title(f"Case {index} | {r['layout']}\n{r['joint_names'][joint]}",fontsize=10)
        for ax in axes[:,col]:ax.axvline(0,color='black',lw=.7,ls=':');ax.grid(alpha=.2)
        axes[0,col].legend(fontsize=8);axes[2,col].legend(fontsize=8)
        axes[3,col].set_xlabel('ms from largest acceleration change\nin20ms-jerk finite-difference support')
        before,after=near[i-1],near[i]
        records.append(dict(index=index,source_sha256=r['source_sha256'],substeps_sha256=sha(Path(r['directory'])/'substeps.jsonl'),joint=r['joint_names'][joint],event_episode_seconds=event,step_before=before['control_step'],substep_before=before['substep'],step_after=after['control_step'],substep_after=after['substep'],target_before=before['target'][joint],target_after=after['target'][joint],acc_before=before['qacc'][joint],acc_after=after['qacc'][joint],contact_torque_before=before['constraint_torque']['contact'][joint],contact_torque_after=after['constraint_torque']['contact'][joint],box_normal_before=force(before),box_normal_after=force(after)))
    for ax,label in zip(axes[:,0],('Joint offset(deg)','Joint velocity(rad/s)','Generalized torque(N m)','Right robot/box normal(N)')):ax.set_ylabel(label)
    fig.suptitle('Frozen new-policy traces | force-aligned2ms samples around released/support events\nLocal association; no universal mechanism or closed-loop improvement claim',fontsize=13)
    path=dest/'three_release_events.png';fig.savefig(path,dpi=150);plt.close(fig)
    result=dict(schema='release-contact-aligned-plot-v1',script_sha256=sha(Path(__file__)),diagnosis_sha256=sha(OUT/'diagnosis.json'),plot=dict(path=str(path),sha256=sha(path)),records=records,visual_review_pending=True,scope='Dominant recorded50Hz arm-jerk axis per retained extreme. At2ms select largest acceleration change inside that derivative support, showing+-80ms. Preintegration forces aligned to saved preintegration state. No added physics solves; generalized torques do not alone decompose acceleration in a coupled system.')
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(path)

if __name__=='__main__':main()
