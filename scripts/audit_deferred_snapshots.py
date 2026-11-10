"""Native audit of saved video states against learner states and raw RGB hashes."""
import hashlib
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_deferred_video import OUT
from evaluate_v6_progress import read,sha,write


def main():
    import mujoco
    from dm_control import mujoco as dm_mujoco
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    ledger=read(OUT/'comparison.json');assert ledger['complete']
    assert not (OUT/'snapshot_audit.json').exists()
    records=[]
    for row in ledger['results']:
        if row['profile']!='deferred':continue
        run=Path(row['run_directory']);receipt=read(run/'episode_000_video_receipt.json')
        assert receipt['complete'] and sha(run/receipt['state_file'])==receipt['state_file_sha256']
        with np.load(run/receipt['state_file'],allow_pickle=False) as d:
            snapshots=d['integration_state'].copy();steps=d['control_step'].copy();spec=int(d['state_spec'])
        with np.load(run/'episode_000.npz',allow_pickle=False) as d:q=d['state'].copy()
        assert spec==int(mujoco.mjtState.mjSTATE_INTEGRATION)
        clone=dm_mujoco.Physics.from_xml_string((run/'scene.xml').read_text())
        checks=[];max_error=0.;times=[]
        # Beginning, pre-grasp, lift/transport vicinity, placement, terminal.
        sample=sorted(set(np.linspace(0,len(snapshots)-1,12,dtype=int).tolist()))
        try:
            for i,(snapshot,step) in enumerate(zip(snapshots,steps,strict=True)):
                mujoco.mj_setState(clone.model.ptr,clone.data.ptr,snapshot,spec)
                times.append(float(clone.data.time))
                if step<len(q):
                    error=float(np.max(np.abs(BananaBoxEnv._state_from_physical(clone.data.qpos[:16])-q[step])))
                    max_error=max(max_error,error)
                if i in sample:
                    clone.forward();frame=clone.render(height=480,width=640,camera_id='top')
                    matched=hashlib.sha256(frame.tobytes()).hexdigest()==receipt['pixel_sha256'][i]
                    assert matched,(row['repeat'],i)
                    checks.append(dict(video_frame=i,control_step=int(step),raw_rgb_sha256_matches=True))
            assert max_error==0,max_error
            assert np.max(np.abs((np.asarray(times)-times[0])-(steps-steps[0])*.02))<1e-9
        finally:clone.free()
        records.append(dict(repeat=row['repeat'],snapshot_file_sha256=receipt['state_file_sha256'],
            frames=len(snapshots),compared_pre_action_frames=int(sum(steps<len(q))),pre_action_state_max_error=max_error,
            sampled_rgb_checks=checks,all_frame_timestamps_match_control_steps=True,
            boundary='Terminal frame has no next pre-action learner state; only its saved state/time/RGB is audited. All14D pre-action samples match, not an independent proof of every unrecorded live physical field.'))
        print('PASS',row['repeat'],'states',len(snapshots),'sampled RGB',len(checks),flush=True)
    write(OUT/'snapshot_audit.json',dict(schema='deferred-native-snapshot-audit-v1',passed=True,
        ledger_sha256=sha(OUT/'comparison.json'),script_sha256=sha(Path(__file__)),records=records))


if __name__=='__main__':main()
