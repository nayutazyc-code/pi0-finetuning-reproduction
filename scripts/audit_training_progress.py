"""Read-only phase/progress and split-lineage audit; never edits demonstrations."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from pi0_repro.motion_quality import motion_quality,ARM_JOINTS
from pi0_repro.diagnostics import task_progress


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dist(values):
    a=np.asarray(values,float)
    return None if not a.size else dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();dataset=args.dataset.resolve();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'outputs');out.mkdir(parents=True,exist_ok=False)
    manifest=read(dataset/'dataset_manifest.json');raw=Path(manifest['source_path'])
    mapping=read(dataset/'local/pi0_banana_train/source_episodes.json')
    forbidden=set()
    for split in ('validation','test'):
        for r in read(dataset/f'local/pi0_banana_{split}/source_episodes.json')['episodes']:
            forbidden.add(r['layout_id'])
    checks=[];records=[];phases=Counter();parent_counts=Counter();full_durations=[];recovery_durations=[]
    def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
    for row in mapping['episodes']:
        name=row['source_episode'];directory=raw/name;meta=read(directory/'episode.json');layout=read(directory/'layout.json')
        recovery=layout.get('recovery') or (layout.get('layout') or {}).get('recovery')
        if recovery:
            parent=recovery.get('source_layout_id')
            if parent is None:
                original=Path(recovery['source_episode'])
                original_layout=read(original/'layout.json')
                parent=original_layout.get('layout',{}).get('layout_id')
            check(name+':recovery_train_source',recovery.get('source_split')=='train')
        else:parent=row['layout_id']
        check(name+':parent_known',bool(parent));check(name+':no_holdout_parent',parent not in forbidden)
        check(name+':train_split',row['split']=='train' and meta.get('split',row['split'])=='train')
        check(name+':accepted',meta.get('review_status')=='accepted' and meta.get('eligible_for_training') is True)
        check(name+':trajectory_digest',sha(directory/'trajectory.npz')==row['source_trajectory_sha256'])
        steps=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
        with np.load(directory/'trajectory.npz',allow_pickle=False) as data:
            state=data['state'];action=data['action'];phase=data['phase'];n=len(action)
            check(name+':aligned',len(state)==len(steps)==len(phase)==row['frames'])
            check(name+':finite',np.isfinite(state).all() and np.isfinite(action).all())
            quality=motion_quality(state,action,steps,pause_min_seconds=.16)
            progress=task_progress(steps,action)
            command_delta=np.max(abs(np.diff(action[:,ARM_JOINTS],axis=0)),axis=1)
            counts=Counter(str(s) for s in phase);phases.update(counts)
            # Phase labels are expert annotations, not equivalent to physical
            # near/grasp flags or evidence that a hold should be removed.
            phase_seconds={k:v*.02 for k,v in counts.items()}
            rec=dict(episode=name,layout_id=row['layout_id'],parent_layout_id=parent,recovery_segment=bool(recovery),
                frames=n,duration_seconds=n*.02,phase_seconds=phase_seconds,progress=progress,
                low_motion_seconds=quality['low_motion_pause_total_seconds'],
                low_motion_max_seconds=quality['low_motion_pause_max_seconds'],
                low_motion_spans=quality['low_motion_pause_spans'],phase_pause_spans=quality['phase_pause_spans'],
                near_grasp_geometry_available=any('near_grasp_region' in s for s in steps),
                max_adjacent_arm_target_change_rad=float(command_delta.max()),
                bindings={'trajectory_sha256':sha(directory/'trajectory.npz'),'steps_sha256':sha(directory/'steps.jsonl'),
                          'metadata_sha256':sha(directory/'episode.json'),'layout_sha256':sha(directory/'layout.json')})
        parent_counts[parent]+=1;records.append(rec)
        (recovery_durations if recovery else full_durations).append(n*.02)
    check('frame_total',sum(r['frames'] for r in records)==manifest['datasets']['train']['frames'])
    candidates=sorted(records,key=lambda r:r['low_motion_max_seconds'],reverse=True)[:10]
    failed=[c for c in checks if not c['passed']]
    report=dict(schema='training-progress-audit-v1',audit_passed=not failed,checks=checks,failed_checks=failed,
        dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),
        train_mapping_sha256=sha(dataset/'local/pi0_banana_train/source_episodes.json'),
        script_sha256=sha(Path(__file__)),episodes=len(records),frames=sum(r['frames'] for r in records),
        complete_demonstrations=len(full_durations),recovery_segments=len(recovery_durations),
        distinct_parent_layouts=len(parent_counts),parent_multiplicity=dict(parent_counts),
        complete_duration_seconds=dist(full_durations),recovery_duration_seconds=dist(recovery_durations),
        phase_frame_counts=dict(phases),near_grasp_geometry_missing=sum(not r['near_grasp_geometry_available'] for r in records),
        records=records,review_queue=[dict(episode=r['episode'],recovery=r['recovery_segment'],max_low_motion_seconds=r['low_motion_max_seconds']) for r in candidates],
        decision='Read-only review queue. No rejection, deletion, trimming, retiming, resampling, new labels, export or training has been performed.',
        limits=['Recovery segments share parent layouts and are not independent spatial coverage.',
                'Low-motion periods can be deliberate gripping/release/settling; do not trim automatically.',
                'Old logs without near-grasp geometry report unavailable, not zero waiting.',
                'Do not mix validation policy diagnostics into training labels. New collections must retain parent-level train splits.'])
    (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','checks','parent_multiplicity')},indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
