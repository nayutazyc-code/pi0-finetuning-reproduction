"""Collect the four preselected train-only contact recovery candidates, no export."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write


def main():
    from pi0_repro.recovery import collect_recovery
    out=ROOT/'outputs/contact_recovery_20261029';selection=read(out/'selection.json')
    assert selection['complete'] and len(selection['plan']['entries'])==4
    path=out/'unload_candidates.json';assert not path.exists()
    cfg=out/'unload_expert.toml'
    assert read(out/'clearance_candidates.json')['complete']
    record=dict(schema='contact-recovery-unload-candidates-v1',selection_sha256=sha(out/'selection.json'),script_sha256=sha(Path(__file__)),
        expert_config_sha256=sha(cfg),previous_candidates_sha256=sha(out/'clearance_candidates.json'),intervention='Same8mm recovery floor plus10step(0.2s) stationary arm reference unloading before grasp motion; gripper commands preserved; same4takeoverstates, no replacements.',complete=False,results=[],scope='All4 preselected train-only takeover states. New scripted-expert candidate labels, pending review; no accepted merge, export, upload, policy change or training. Task failures retained with no replacement.')
    write(path,record);(out/'unload_collector_source.py').write_bytes(Path(__file__).read_bytes())
    for entry in selection['plan']['entries']:
        print('START',entry['index'],'step',entry['step'],flush=True)
        row=dict(entry);before=set((ROOT/'data/banana_demos').iterdir())
        try:
            collection=collect_recovery(cfg,Path(entry['rollout']),Path(entry['source_episode']),entry['step'])
            meta=read(collection/'episode_000/episode.json')
            row.update(collection=str(collection),automatic_success=meta['automatic_success'],steps=meta['steps'],error=meta.get('error'),review_status=meta.get('review_status'))
            print('END',entry['index'],row['automatic_success'],row['error'],row['steps'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}'
            row['retained_partial_directories']=[str(p) for p in set((ROOT/'data/banana_demos').iterdir())-before]
            record['results'].append(row);write(path,record);raise
        record['results'].append(row);write(path,record)
    record['complete']=True;write(path,record)


if __name__=='__main__':main()
