"""One predeclared gripper-opening slew candidate on all twelve frozen traces."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('MUJOCO_GL', 'glfw')
from diagnose_release_retreat import OUT, read, sha, write


def main():
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.motion_quality import motion_quality
    from pi0_repro.scene import SceneConfig
    diagnosis = read(OUT / 'diagnosis.json')
    source_plan = read(OUT / 'diagnosis_plan.json')
    assert diagnosis['complete'] and len(diagnosis['results']) == 48
    assert all(r['exact_replay_passed'] for r in diagnosis['results'])
    cases = [r for r in source_plan['cases'] if r['model'] == 'candidate']
    out = OUT / 'opening_velocity_screen'
    out.mkdir(exist_ok=False)
    plan = dict(schema='release-opening-slew-screen-v1', opening_velocity=1.0, terminal_hold_steps=100,
                diagnosis_sha256=sha(OUT / 'diagnosis.json'), cases=cases, script_sha256=sha(Path(__file__)),
                sources={n:sha(ROOT / 'src/pi0_repro' / n) for n in ('banana_env.py', 'scene.py', 'motion_quality.py')},
                treatment='Only limit positive normalized gripper-command increments to1.0/s (both grippers). Closing and all arm commands unchanged. No geometry/contact/task-state input, no plant change. One fixed value, no sweep.',
                extension='Both arms receive the same100step terminal-action hold. Baseline must exactly reproduce original14D states through the recorded prefix. Extended states are new open-loop diagnostic data, not original recordings or model outcomes.',
                gates=dict(all12_success_retained=True, median_full_active_p99_ratio_max=.8,
                           median_prefix_active_p99_ratio_max=.8, worst_full_peak_ratio_max=.9,
                           worst_prefix_peak_ratio_max=.9, worst_increment_ratio_max=1.05,
                           max_pause_added_seconds=.1, success_time_ratio_max=1.1),
                decision='Only a passed screen may proceed to fresh closed-loop inference. No new default or learned-weight improvement is established by frozen commands.')
    write(out / 'plan.json', plan)
    (out / 'runner_source.py').write_bytes(Path(__file__).read_bytes())
    ledger = dict(complete=False, results=[])
    write(out / 'results.json', ledger)
    for item in cases:
        source = Path(item['source'])
        assert sha(source / 'episode_000.npz') == item['source_sha256']
        with np.load(source / 'episode_000.npz', allow_pickle=False) as z:
            recorded, reference = z['action'].copy(), z['state'].copy()
        actions = np.concatenate([recorded, np.repeat(recorded[-1:], 100, axis=0)])
        scene = SceneConfig(**read(source / 'episode_000_scene.json')['scene'])
        seed = read(source / 'run.json')['config']['simulation']['seed']
        for limited in (False, True):
            env = BananaBoxEnv(scene, max_episode_steps=len(actions), terminate_on_success=False, render_observations=False)
            try:
                obs, _ = env.reset(seed=seed)
                previous = env._state_from_physical(env.physics.data.ctrl).astype(float)
                states, commands, tasks = [], [], []
                for action in actions:
                    target = action.copy()
                    if limited:
                        target[[6, 13]] = np.minimum(target[[6, 13]], previous[[6, 13]] + .02)
                    states.append(obs['agent_pos'].copy())
                    obs, _, _, _, info = env.step(target)
                    previous = np.asarray(info.pop('executed_action'), float)
                    commands.append(previous.copy())
                    tasks.append(info)
                states, commands = np.asarray(states), np.asarray(commands)
                full = motion_quality(states, commands, tasks)
                prefix = motion_quality(states[:len(recorded)], commands[:len(recorded)], tasks[:len(recorded)])
                prefix_error = float(np.max(abs(states[:len(reference)] - reference)))
                if not limited:
                    assert prefix_error == 0
                    np.testing.assert_array_equal(commands[:len(recorded)].astype(np.float32), recorded)
                else:
                    initial = env._state_from_physical(env.physics.data.ctrl)  # shape only; initial command bound below
                    assert np.max(np.diff(commands[:, [6, 13]], axis=0)) <= .0200001
                    np.testing.assert_array_equal(commands[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].astype(np.float32), actions[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]])
                first_success = next(((i + 1) * .02 for i, t in enumerate(tasks) if t['is_success']), None)
                row = dict(index=item['index'], limited=limited, source=str(source), success=any(t['is_success'] for t in tasks),
                           final_success=tasks[-1]['is_success'], first_success_seconds=first_success,
                           prefix_state_error=prefix_error, quality=full, prefix_quality=prefix)
                file = out / f"case_{item['index']:02d}_{'limited' if limited else 'baseline'}.npz"
                np.savez_compressed(file, state=states, action=commands)
                row['samples'] = str(file)
                ledger['results'].append(row)
                write(out / 'results.json', ledger)
                print(item['index'], limited, row['success'], full['active_frame_max_jerk_rad_s3'], flush=True)
            finally:
                env.close()
    groups = {name:[r for r in ledger['results'] if r['limited'] == limited] for name, limited in [('baseline', False), ('limited', True)]}
    summary = {}
    for name, rows in groups.items():
        summary[name] = dict(successes=sum(r['success'] for r in rows),
            median_full_active_p99=float(np.median([r['quality']['active_frame_max_jerk_rad_s3']['p99'] for r in rows])),
            median_prefix_active_p99=float(np.median([r['prefix_quality']['active_frame_max_jerk_rad_s3']['p99'] for r in rows])),
            worst_full_peak=max(r['quality']['all_frame_max_jerk_rad_s3']['max'] for r in rows),
            worst_prefix_peak=max(r['prefix_quality']['all_frame_max_jerk_rad_s3']['max'] for r in rows),
            worst_increment=max(r['quality']['physical_joint_increment_rad']['max'] for r in rows),
            worst_pause=max(r['quality']['low_motion_pause_max_seconds'] for r in rows),
            median_success_seconds=float(np.median([r['first_success_seconds'] for r in rows if r['success']])))
    b, c = summary['baseline'], summary['limited']
    gates = []
    for key, limit in plan['gates'].items():
        if key.endswith('_ratio_max'):
            metric = key.removesuffix('_ratio_max')
            if metric == 'success_time':
                metric = 'median_success_seconds'
            gates.append(dict(name=key, baseline=b[metric], candidate=c[metric], passed=c[metric] <= b[metric] * limit))
    gates += [dict(name='all12_success', passed=c['successes'] == b['successes'] == 12),
              dict(name='worst_pause', passed=c['worst_pause'] <= b['worst_pause'] + .1)]
    ledger.update(complete=True, summaries=summary, gates=gates, passed=all(g['passed'] for g in gates))
    write(out / 'results.json', ledger)
    print(json.dumps(dict(summaries=summary, gates=gates, passed=ledger['passed']), indent=2))


if __name__ == '__main__':
    main()
