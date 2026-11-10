"""Read-only, train-only expert target/physical-motion and phase-boundary audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def derivative_metrics(q):
    q = np.asarray(q, np.float64)[:, JOINTS]
    result = {}
    for order, name in ((1, 'velocity'), (2, 'acceleration'), (3, 'jerk')):
        values = np.abs(np.diff(q, n=order, axis=0)) / .02**order
        result[name] = {'p95': float(np.quantile(values, .95)), 'max': float(values.max())}
    return result


def audit(dataset):
    manifest_path = dataset / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    raw = Path(manifest['source_path'])
    mapping = json.loads((dataset / 'local/pi0_banana_train/source_episodes.json').read_text())
    if mapping['source_split'] != 'train' or manifest['status'] != 'complete':
        raise ValueError('Only a complete training split is accepted')
    rows = []
    for entry in mapping['episodes']:
        if entry['split'] != 'train':
            raise ValueError('Non-training source')
        episode = raw / entry['source_episode']
        path = episode / 'trajectory.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['source_trajectory_sha256']:
            raise ValueError('Source trajectory changed')
        meta = json.loads((episode / 'episode.json').read_text())
        if not meta['eligible_for_training'] or meta['review_status'] != 'accepted':
            raise ValueError('Source is not reviewed/eligible')
        with np.load(path, allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        n = entry['frames']
        assert all(arrays[k].shape == (n, 14) and np.isfinite(arrays[k]).all()
                   for k in ('state', 'next_state', 'action', 'requested_action'))
        assert np.allclose(arrays['state'][1:], arrays['next_state'][:-1], atol=1e-6, rtol=0)
        assert np.allclose(arrays['timestamp'], np.arange(n)*.02, atol=1e-7, rtol=0)
        phase = arrays['phase']
        delta = np.abs(np.diff(arrays['action'][:, JOINTS].astype(float), axis=0))
        boundary = np.flatnonzero(phase[:-1] != phase[1:])
        peaks = np.argsort(delta.max(axis=1))[-5:][::-1]
        row = {'episode': entry['source_episode'], 'frames': n, 'trajectory_sha256': digest,
               'metrics': {key: derivative_metrics(arrays[key]) for key in ('action', 'requested_action', 'state')},
               'max_target_delta_rad': float(delta.max()),
               'max_initial_target_minus_state_rad': float(np.abs(arrays['action'][0, JOINTS]-arrays['state'][0, JOINTS]).max()),
               'large_target_delta_over_0_05rad': bool(np.any(delta > .05)),
               'phase_boundaries': [{'step': int(i+1), 'from': str(phase[i]), 'to': str(phase[i+1]),
                                     'max_target_delta_rad': float(delta[i].max())} for i in boundary],
               'peak_target_steps': [{'step': int(i+1), 'from': str(phase[i]), 'to': str(phase[i+1]),
                                      'joint_index': JOINTS[int(delta[i].argmax())],
                                      'max_target_delta_rad': float(delta[i].max())} for i in peaks]}
        recovery = meta.get('recovery')
        if recovery:
            assert recovery['source_split'] == 'train'
            learner_path = Path(recovery['rollout']) / 'episode_000.npz'
            assert hashlib.sha256(learner_path.read_bytes()).hexdigest() == recovery['learner_trajectory_sha256']
            with np.load(learner_path, allow_pickle=False) as a:
                previous = a['action'][recovery['step']-1, JOINTS]
            row['recovery_command_seam_max_delta_rad'] = float(np.abs(previous-arrays['action'][0, JOINTS]).max())
        rows.append(row)
    flagged = [r['episode'] for r in rows if r['large_target_delta_over_0_05rad']]
    return {'schema': 'expert-action-audit-v1', 'dataset': str(dataset.relative_to(ROOT)),
            'dataset_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'train_episodes': len(rows), 'train_frames': sum(r['frames'] for r in rows),
            'rows': rows, 'flagged_episodes': flagged,
            'max_target_delta_rad': max(r['max_target_delta_rad'] for r in rows),
            'max_phase_boundary_delta_rad': max(b['max_target_delta_rad'] for r in rows for b in r['phase_boundaries']),
            'decision': 'review_required_before_training' if flagged else 'retain_reviewed_data_no_gross_target_jump_detected',
            'scope': '0.05 rad is a replay nomination threshold, not an IK branch proof or physical smoothness guarantee. No samples deleted or relabeled; no validation/test trajectories inspected.',
            'ik_source_sha256': hashlib.sha256((ROOT/'src/pi0_repro/banana_expert.py').read_bytes()).hexdigest(),
            'ik_source_observation': 'Current expert uses previous-solution initialization, damped least squares, bounded updates, line search and joint limits. Historical recordings do not contain solver iteration/branch histories.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.dataset.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Preserve prior audit')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
