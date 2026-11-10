"""Summarize force-aligned replay events without generalizing beyond two traces."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from evaluate_v6_progress import read,sha,write
OUT=ROOT/'outputs/wrist_contact_diagnosis_20261028'
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl_cache'))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert read(OUT/'summary.json')['complete'];assert not (OUT/'analysis.json').exists()
    records={};fig,axes=plt.subplots(4,2,figsize=(11,10),sharex='col')
    for column,model in enumerate(['control','v6']):
        folder=OUT/(model+'_instrumented');summary=read(folder/'summary.json');assert summary['source_state_max_abs_error']==0
        rows=[json.loads(s) for s in (folder/'substeps.jsonl').read_text().splitlines()]
        center=summary['focus_peak_state_index']*.02
        selected=[r for r in rows if abs(r['pre_episode_time']-center)<=.12]
        peak=min(selected,key=lambda r:r['qacc'])
        def finger_table(row):
            return [c for c in row['contacts'] if 'table' in c['geoms'] and any('vx300s_right/10_' in n for n in c['geoms'])]
        releases=[]
        for a,b in zip(selected,selected[1:]):
            if finger_table(a) and not finger_table(b) and abs(b['pre_episode_time']-peak['pre_episode_time'])<=.02:
                releases.append(dict(before=a,after=b))
        residual=max(abs(r['qfrc_constraint']-sum(r['constraint_torque'].values())) for r in selected)
        records[model]=dict(source_summary_sha256=sha(folder/'summary.json'),substep_log_sha256=sha(folder/'substeps.jsonl'),
            minimum_acceleration_sample=peak,nearby_finger_table_releases=releases,
            max_constraint_decomposition_residual_nm=residual,
            scope='Specific exact frozen-action event only; aggregate contact torque includes all contacts. A table-contact release and tracking error coincide with the transient. This does not prove every peak has that cause or establish closed-loop candidate benefit.')
        times=np.array([r['pre_episode_time']-center for r in selected]);q0=min(selected,key=lambda r:abs(r['pre_episode_time']-center))['pre_q']
        axes[0,column].plot(times,np.rad2deg([r['pre_q']-q0 for r in selected]),label='Physical wrist')
        axes[0,column].plot(times,np.rad2deg([r['target']-q0 for r in selected]),linestyle='dashed',label='Command target')
        axes[1,column].plot(times,[r['pre_qvel'] for r in selected],color='#2563a6')
        axes[2,column].plot(times,[r['qfrc_actuator'] for r in selected],label='Actuator')
        axes[2,column].plot(times,[r['constraint_torque']['contact'] for r in selected],label='Contact')
        axes[2,column].plot(times,[r['qfrc_passive'] for r in selected],label='Passive',linestyle='dashed')
        axes[3,column].plot(times,[sum(abs(c['force_contact_frame'][0]) for c in finger_table(r)) for r in selected],color='#c66a32')
        axes[0,column].set_title(('Continued model, round8' if model=='control' else 'Original v6, round9')+'\nexact replay;2ms samples',fontsize=12)
        for ax in axes[:,column]:ax.axvline(0,color='#999999',linewidth=.8);ax.grid(alpha=.18)
        axes[3,column].set_xlabel('Seconds relative to recorded20ms jerk peak')
    for row in axes:
        low=min(a.get_ylim()[0] for a in row);high=max(a.get_ylim()[1] for a in row)
        for a in row:a.set_ylim(low,high)
    for ax,label in zip(axes[:,0],['Wrist angle offset(deg)','Wrist velocity(rad/s)','Wrist torque(N m)','Right finger/table normal force(N)']):ax.set_ylabel(label)
    axes[0,0].legend(fontsize=9,frameon=False);axes[2,0].legend(fontsize=9,frameon=False)
    fig.suptitle('Finger/table constraint release during wrist tracking transients',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));path=OUT/'contact_release.png';fig.savefig(path,dpi=160);plt.close(fig)
    write(OUT/'analysis.json',dict(schema='wrist-replay-contact-analysis-v1',script_sha256=sha(Path(__file__)),records=records,
        plot=dict(path=str(path),sha256=sha(path)),visual_review_pending=True))
    print(json.dumps({k:dict(qacc=v['minimum_acceleration_sample']['qacc'],release_count=len(v['nearby_finger_table_releases']),force_residual=v['max_constraint_decomposition_residual_nm']) for k,v in records.items()},indent=2))


if __name__=='__main__':main()
