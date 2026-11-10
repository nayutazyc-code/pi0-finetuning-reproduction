"""Recover missing phase geometry by exact-joint-state expert replay.

New diagnostic artifacts only. Never edits/export labels or treats a replay as
an independent demonstration. Recovery snapshots are explicitly out of scope.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'repos/openpi/packages/openpi-client/src')]
os.environ.setdefault('MUJOCO_GL','glfw')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.scene import SceneConfig
    from pi0_repro.diagnostics import task_progress
    dataset=args.dataset.resolve();raw=Path(read(dataset/'dataset_manifest.json')['source_path'])
    mapping=read(dataset/'local/pi0_banana_train/source_episodes.json')
    report=dict(schema='v6-expert-progress-replay-v1',complete=False,dataset=str(dataset),
        mapping_sha256=sha(dataset/'local/pi0_banana_train/source_episodes.json'),
        script_sha256=sha(Path(__file__)),environment_sha256=sha(ROOT/'src/pi0_repro/banana_env.py'),
        results=[],scope='Train-only frozen-action diagnostic replay. Joint states and final banana position checked against recorded reference; not a full historical physical-state or image equivalence proof. No new labels/training or independent layout count.')
    def save():(out/'replay.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for row in mapping['episodes']:
            source=raw/row['source_episode'];meta=read(source/'episode.json');layout=read(source/'layout.json')
            assert row['split']=='train' and meta['review_status']=='accepted'
            recovery=layout.get('recovery') or (layout.get('layout') or {}).get('recovery')
            if recovery:
                report['results'].append(dict(episode=row['source_episode'],status='skipped_recovery_snapshot_required'));save();continue
            digest=sha(source/'trajectory.npz');assert digest==row['source_trajectory_sha256']
            with np.load(source/'trajectory.npz',allow_pickle=False) as data:actions=data['action'].copy();states=data['state'].copy()
            env=BananaBoxEnv(SceneConfig(**meta['scene']),max_episode_steps=len(actions),terminate_on_success=False,render_observations=False)
            task=[];max_error=0.;command_error=0.
            try:
                obs,_=env.reset(seed=meta['reset_seed'])
                for i,action in enumerate(actions):
                    max_error=max(max_error,float(np.max(abs(obs['agent_pos']-states[i]))))
                    obs,_,_,_,info=env.step(action)
                    command_error=max(command_error,float(np.max(abs(np.asarray(info.pop('executed_action'))-action))))
                    task.append(info)
                position_error=float(np.max(abs(np.asarray(task[-1]['banana_position'])-meta['final_info']['banana_position'])))
                assert max_error<1e-5 and command_error<1e-6 and position_error<1e-6,(row['source_episode'],max_error,command_error,position_error)
                assert sha(source/'trajectory.npz')==digest
                destination=out/(row['source_episode']+'_task.jsonl')
                destination.write_text(''.join(json.dumps(r)+'\n' for r in task))
                report['results'].append(dict(episode=row['source_episode'],status='replayed',frames=len(actions),
                    source_trajectory_sha256=digest,diagnostic_sha256=sha(destination),joint_state_max_error=max_error,
                    executed_target_max_error=command_error,final_banana_position_max_error=position_error,
                    success=task[-1]['is_success'],progress=task_progress(task,actions)))
                print('REPLAY',row['source_episode'],'state_error',max_error,flush=True)
            finally:env.close()
            save()
        report['complete']=True
    except BaseException as exc:report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:save()
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
