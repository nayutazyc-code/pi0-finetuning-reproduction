"""One fixed wrist-damping factor on all frozen candidate/v6 recordings."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
os.environ.setdefault('MUJOCO_GL', 'glfw')
from diagnose_release_retreat import OUT, read, sha, write


def summarize(rows):
    import numpy as np
    covered = [r for r in rows if r['prefix']['coverage']['transport_motion_comparable']]
    return dict(successes=sum(r['success'] for r in rows), coverage=len(covered),
        typical=float(np.median([r['prefix']['active_frame_max_jerk_rad_s3']['p99'] for r in rows])),
        covered_typical=float(np.median([r['prefix']['active_frame_max_jerk_rad_s3']['p99'] for r in covered])),
        worst=max(r['prefix']['all_frame_max_jerk_rad_s3']['max'] for r in rows),
        extended_worst=max(r['full']['all_frame_max_jerk_rad_s3']['max'] for r in rows),
        increment=max(r['prefix']['physical_joint_increment_rad']['max'] for r in rows),
        pause=max(r['prefix']['low_motion_pause_max_seconds'] for r in rows),
        median_success_seconds=float(np.median([r['first_success_seconds'] for r in rows if r['success']])))


def main():
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.motion_quality import motion_quality
    from pi0_repro.scene import SceneConfig
    diagnosis = read(OUT / 'diagnosis.json')
    assert diagnosis['complete'] and all(r['exact_replay_passed'] for r in diagnosis['results'])
    cases = read(OUT / 'diagnosis_plan.json')['cases']
    out = OUT / 'damping_screen'
    out.mkdir(exist_ok=False)
    plan = dict(schema='single-wrist-damping-release-screen-v1', scale=2., cases=cases,
        script_sha256=sha(Path(__file__)), source_sha256={n:sha(ROOT / 'src/pi0_repro' / n) for n in ('banana_env.py', 'scene.py', 'motion_quality.py')},
        treatment='Only existing wrist_damping_scale1to2 (six wrist/forearm joints). Original weights, commands, gains, contacts, friction and success rule retained. One value chosen before this screen, no sweep.',
        method='All12candidate plus12v6 source trajectories; two arms each, common100step terminal hold. Baseline recorded-prefix14D replay must be exact. Extended tails are new diagnostic states, not recorded policy outcomes.',
        gates=dict(candidate_typical_ratio_max=.8,candidate_covered_typical_ratio_max=.8,candidate_worst_ratio_max=.8,
            v6_typical_ratio_max=1.1,v6_covered_typical_ratio_max=1.1,v6_worst_ratio_max=1.1,
            extended_worst_ratio_max=1.1,increment_ratio_max=1.05,pause_added_seconds_max=.1,success_time_ratio_max=1.1,
            successes_and_coverage='retain at least each baseline'),
        scope='Frozen-command plant screen only; passing permits a fresh factorial closed-loop comparison, not a learned-weights improvement or default promotion.')
    write(out / 'plan.json', plan)
    (out / 'runner_source.py').write_bytes(Path(__file__).read_bytes())
    ledger = dict(complete=False, results=[])
    write(out / 'results.json', ledger)
    for item in cases:
        source = Path(item['source'])
        assert sha(source / 'episode_000.npz') == item['source_sha256']
        with np.load(source / 'episode_000.npz', allow_pickle=False) as z:
            original, reference = z['action'].copy(), z['state'].copy()
        actions = np.concatenate([original, np.repeat(original[-1:], 100, axis=0)])
        base = SceneConfig(**read(source / 'episode_000_scene.json')['scene'])
        assert base.wrist_damping_scale == 1
        seed = read(source / 'run.json')['config']['simulation']['seed']
        coefficients = None
        for scale in (1., 2.):
            env = BananaBoxEnv(replace(base, wrist_damping_scale=scale), max_episode_steps=len(actions), terminate_on_success=False, render_observations=False)
            try:
                m = env.physics.model
                arrays = {n:np.array(getattr(m, n)) for n in ('actuator_gainprm','actuator_biasprm','dof_damping','dof_frictionloss','geom_friction','geom_solref','geom_solimp','jnt_range','body_mass')}
                if coefficients is None:
                    coefficients = arrays
                else:
                    changed = np.flatnonzero(arrays['dof_damping'] != coefficients['dof_damping'])
                    assert len(changed) == 6
                    np.testing.assert_array_equal(arrays['dof_damping'][changed], coefficients['dof_damping'][changed] * 2)
                    for name in arrays:
                        if name != 'dof_damping':
                            np.testing.assert_array_equal(arrays[name], coefficients[name])
                obs, _ = env.reset(seed=seed)
                states, tasks = [], []
                for action in actions:
                    states.append(obs['agent_pos'].copy())
                    obs, _, _, _, info = env.step(action)
                    info.pop('executed_action')
                    tasks.append(info)
                states = np.asarray(states)
                error = float(np.max(abs(states[:len(original)] - reference)))
                if scale == 1:
                    assert error == 0
                row = dict(index=item['index'], model=item['model'], scale=scale, source=str(source),
                    prefix_state_error=error, success=any(t['is_success'] for t in tasks),
                    first_success_seconds=next(((i+1)*.02 for i,t in enumerate(tasks) if t['is_success']),None),
                    prefix=motion_quality(states[:len(original)],actions[:len(original)],tasks[:len(original)]),
                    full=motion_quality(states,actions,tasks))
                file = out / f"case_{item['index']:02d}_scale{int(scale)}.npz"
                np.savez_compressed(file,state=states,action=actions)
                row['samples'] = str(file)
                ledger['results'].append(row)
                write(out / 'results.json', ledger)
                print(item['index'],scale,row['success'],row['prefix']['active_frame_max_jerk_rad_s3'],flush=True)
            finally:
                env.close()
    summaries = {};checks = []
    for model in ('v6','candidate'):
        groups = {str(int(scale)):summarize([r for r in ledger['results'] if r['model']==model and r['scale']==scale]) for scale in (1.,2.)}
        summaries[model] = groups
        b,c = groups['1'],groups['2']
        for metric in ('typical','covered_typical','worst'):
            limit = plan['gates'][model+'_'+metric+'_ratio_max']
            checks.append(dict(model=model,name=metric,baseline=b[metric],candidate=c[metric],passed=c[metric]<=b[metric]*limit))
        for metric,key in [('extended_worst','extended_worst_ratio_max'),('increment','increment_ratio_max'),('median_success_seconds','success_time_ratio_max')]:
            checks.append(dict(model=model,name=metric,baseline=b[metric],candidate=c[metric],passed=c[metric]<=b[metric]*plan['gates'][key]))
        checks += [dict(model=model,name='success_and_coverage',passed=c['successes']>=b['successes'] and c['coverage']>=b['coverage']),
                   dict(model=model,name='worst_pause',passed=c['pause']<=b['pause']+.1)]
    ledger.update(complete=True,summaries=summaries,gates=checks,passed=all(c['passed'] for c in checks))
    write(out / 'results.json', ledger)
    print(json.dumps(dict(summaries=summaries,gates=checks,passed=ledger['passed']),indent=2))


if __name__ == '__main__':
    main()
