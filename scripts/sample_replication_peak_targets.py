"""Post-hoc local command/state samples around first-three-second wrist peaks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from replicate_contact_train import OUT
from evaluate_v6_progress import read,sha,write
import numpy as np


def main():
    dest=OUT/'delayed_peak_diagnostic';diag=read(dest/'diagnostic.json');aggregate=read(OUT/'aggregate.json');records=[]
    assert not (dest/'peak_target_samples.json').exists()
    for d in diag['records']:
        r=next(r for r in aggregate['groups'][d['profile']]['records'] if r['parent']==d['parent'])
        with np.load(Path(r['collection'])/'episode_000/trajectory.npz',allow_pickle=False) as data:
            q=data['state'][:,11].astype(float);action=data['action'][:,11].astype(float);phase=data['phase']
        k=int(round(d['peak_time_seconds']/.02));samples=[]
        for i in range(max(1,k-5),min(len(q),k+3)):
            samples.append(dict(step=i,time_seconds=i*.02,phase=str(phase[i]),measured_wrist_rad=float(q[i]),target_wrist_rad=float(action[i]),target_increment_rad=float(action[i]-action[i-1])))
        records.append(dict(parent=d['parent'],profile=d['profile'],peak_step=k,samples=samples))
    write(dest/'peak_target_samples.json',dict(script_sha256=sha(Path(__file__)),diagnostic_sha256=sha(dest/'diagnostic.json'),
        frozen_expert_sha256=sha(OUT/'frozen_source/banana_expert.py'),records=records,
        observation='Exampleparent1 has repeated targets and0.00087/0/0.00112/0/0.00138rad wrist increments near liftpeak; unload reproduces this pattern10stepslater.',
        hypothesis='Warm-started _ik returns last_q unchanged when its pose error lies within tolerances. Small Cartesian increments may therefore yield staircasejointtargets. Contribution to physicaljerk requires a separately controlled IK/planner experiment; these samples alone do not establishcausation.',
        scope='Post-hoc diagnosis only; no frozen gate changes, no traininglabels/acceptance, no implementation or model change.'))
    (dest/'target_sample_source.py').write_bytes(Path(__file__).read_bytes());print('Target samples retained')


if __name__=='__main__':main()
