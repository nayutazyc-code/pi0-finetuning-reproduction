"""Exact full-action replay and force-aligned release/retreat diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
os.environ.setdefault('MUJOCO_GL', 'glfw')
OUT = ROOT / 'outputs/release_retreat_20261029'
EVALUATION = ROOT / 'outputs/ik_continuous_20261029/evaluation'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(p)


def replay(item, destination, instrument):
    import mujoco
    import numpy as np
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.motion_quality import motion_quality
    from pi0_repro.scene import SceneConfig
    source = Path(item['source'])
    meta = read(source / 'run.json')
    with np.load(source / 'episode_000.npz', allow_pickle=False) as z:
        actions, reference = z['action'].copy(), z['state'].copy()
    scene = SceneConfig(**read(source / 'episode_000_scene.json')['scene'])
    env = BananaBoxEnv(scene, max_episode_steps=len(actions), terminate_on_success=False, render_observations=False)
    samples, states, tasks = [], [], []
    peak = item['release_peak_state_index']
    start, end = (max(0, peak - 60), peak + 60) if peak is not None else (0, -1)
    try:
        assert env.xml + '\n' == (source / 'scene.xml').read_text()
        obs, _ = env.reset(seed=meta['config']['simulation']['seed'])
        physics, model, data = env.physics, env.physics.model, env.physics.data
        assert physics.legacy_step and int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
        joints = [int(np.flatnonzero(np.asarray(model.jnt_qposadr) == q)[0]) for q in range(8, 14)]
        dofs = np.asarray([model.jnt_dofadr[j] for j in joints], int)
        qpos = np.asarray([model.jnt_qposadr[j] for j in joints], int)
        actuators = np.asarray([np.flatnonzero(np.asarray(model.actuator_trnid)[:, 0] == j)[0] for j in joints], int)
        joint_names = [model.id2name(j, 'joint') for j in joints]
        assert all(n.startswith('vx300s_right/') for n in joint_names)
        groups = {'contact': (5, 6, 7), 'joint_limit': (3, 4), 'friction': (1, 2), 'equality': (0,)}
        original = physics.step

        def step(nstep=1):
            if not start <= env.steps <= end:
                return original(nstep=nstep)
            with physics.check_invalid_state():
                for substep in range(nstep):
                    pre_q, pre_v = data.qpos[qpos].copy(), data.qvel[dofs].copy()
                    before_time = float(data.time)
                    mujoco.mj_step2(model.ptr, data.ptr)
                    torques = {}
                    for label, kinds in groups.items():
                        weights = np.asarray(data.efc_force).copy()
                        weights[~np.isin(np.asarray(data.efc_type), kinds)] = 0
                        torque = np.zeros(model.nv)
                        mujoco.mj_mulJacTVec(model.ptr, data.ptr, torque, weights)
                        torques[label] = torque[dofs].tolist()
                    contacts = []
                    for i in range(data.ncon):
                        contact, force = data.contact[i], np.zeros(6)
                        mujoco.mj_contactForce(model.ptr, data.ptr, i, force)
                        contacts.append(dict(geoms=[model.id2name(int(g), 'geom') or '' for g in (contact.geom1, contact.geom2)],
                                             distance_m=float(contact.dist), active=bool(contact.efc_address >= 0),
                                             force_contact_frame=force.tolist()))
                    samples.append(dict(control_step=env.steps, substep=substep, pre_physics_time=before_time,
                                        pre_episode_time=env.steps * .02 + substep * .002,
                                        pre_q=pre_q.tolist(), pre_qvel=pre_v.tolist(),
                                        post_q=data.qpos[qpos].tolist(), post_qvel=data.qvel[dofs].tolist(),
                                        target=data.ctrl[actuators].tolist(), qacc=data.qacc[dofs].tolist(),
                                        actuator_force=data.actuator_force[actuators].tolist(),
                                        qfrc_actuator=data.qfrc_actuator[dofs].tolist(),
                                        qfrc_passive=data.qfrc_passive[dofs].tolist(),
                                        qfrc_bias=data.qfrc_bias[dofs].tolist(), qfrc_constraint=data.qfrc_constraint[dofs].tolist(),
                                        constraint_torque=torques, contacts=contacts))
                    mujoco.mj_step1(model.ptr, data.ptr)
        if instrument and peak is not None:
            physics.step = step
        for index, action in enumerate(actions):
            if index == start and peak is not None:
                spec = mujoco.mjtState.mjSTATE_INTEGRATION
                state = np.empty(mujoco.mj_stateSize(model.ptr, spec))
                mujoco.mj_getState(model.ptr, data.ptr, state, spec)
                np.savez_compressed(destination / 'branch_start.npz', integration=state,
                                    step=np.array(env.steps), hold_steps=np.array(env.hold_steps),
                                    ever_grasped=np.array(env.ever_grasped), ever_lifted=np.array(env.ever_lifted))
            states.append(obs['agent_pos'].copy())
            obs, _, _, _, info = env.step(action)
            info.pop('executed_action')
            tasks.append(info)
        states = np.asarray(states)
        error = float(np.max(abs(states - reference)))
        quality = motion_quality(states, actions, tasks)
        assert error == 0, f'Replay differs: {error}'
        assert np.isclose(quality['all_frame_max_jerk_rad_s3']['max'], item['recorded_global_peak'], rtol=1e-12)
        np.savez_compressed(destination / 'states.npz', state=states)
        if instrument:
            (destination / 'substeps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in samples))
        result = dict(index=item['index'], model=item['model'], layout=item['layout'], repeat=item['repeat'],
                      source=str(source), source_sha256=item['source_sha256'], instrumented=instrument,
                      source_state_max_error=error, exact_replay_passed=error == 0,
                      release_peak_state_index=peak, joint_names=joint_names, control_dt=.02, physics_dt=.002,
                      samples=len(samples), recorded_success=item['success'], replay_success=tasks[-1]['is_success'],
                      motion_quality=quality, directory=str(destination))
        write(destination / 'summary.json', result)
        return result
    finally:
        env.close()


def main():
    import numpy as np
    audit, aggregate = read(EVALUATION / 'audit.json'), read(EVALUATION / 'aggregate.json')
    assert audit['passed'] and aggregate['complete']
    OUT.mkdir(exist_ok=False)
    items = []
    for trial in aggregate['trials']:
        if trial['model'] not in ('v6', 'candidate'):
            continue
        source = Path(trial['run_directory'])
        tasks = [json.loads(s) for s in (source / 'episode_000_task.jsonl').read_text().splitlines()]
        with np.load(source / 'episode_000.npz', allow_pickle=False) as z:
            q = z['state'].astype(float)
        # State k is observed before action k; task k-1 describes that state.
        released = np.r_[False, [t['inside_box'] and t['released'] and t['supported_by_box'] for t in tasks[:-1]]]
        jerk = np.max(abs(np.diff(q[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], n=3, axis=0)), axis=1) / .02**3
        indices = np.flatnonzero(released[3:]) + 3
        peak = int(indices[np.argmax(jerk[indices - 3])]) if len(indices) else None
        items.append(dict(index=trial['index'], model=trial['model'], layout=trial['layout'], repeat=trial['repeat'],
                          source=str(source), source_sha256=sha(source / 'episode_000.npz'),
                          release_peak_state_index=peak, release_peak=None if peak is None else float(jerk[peak - 3]),
                          recorded_global_peak=trial['quality']['all_frame_max_jerk_rad_s3']['max'], success=trial['success']))
    assert len(items) == 24
    plan = dict(schema='release-retreat-exact-diagnostic-v1', at=time.time(), script_sha256=sha(Path(__file__)),
                source_sha256={n:sha(ROOT / 'src/pi0_repro' / n) for n in ('banana_env.py', 'scene.py', 'motion_quality.py')},
                evaluation_aggregate_sha256=sha(EVALUATION / 'aggregate.json'), cases=items,
                method='All12 candidate and12 v6 trajectories, including failures. Full plain and instrumented replay must exactly match all14 recorded states. Instrument peak+-60controlsteps at2ms for all6right joints. Select full released-supported-inside phase maximum before replay; missing phases explicit. No policy inference, labels, model training or plant change.',
                interpretation='Pre-integration force/Jacobian state and post-integration q/qvel separated. Source-switch membership uses action_prediction_index changes; chunk_transition_steps is blend length. This is diagnostic evidence, not a successful runtime intervention.')
    write(OUT / 'diagnosis_plan.json', plan)
    (OUT / 'diagnosis_source.py').write_bytes(Path(__file__).read_bytes())
    ledger = dict(schema='release-retreat-exact-diagnostic-results-v1', complete=False, results=[])
    write(OUT / 'diagnosis.json', ledger)
    for item in items:
        for instrument in (False, True):
            dest = OUT / f"case_{item['index']:02d}_{'instrumented' if instrument else 'plain'}"
            dest.mkdir()
            result = replay(item, dest, instrument)
            ledger['results'].append(result)
            write(OUT / 'diagnosis.json', ledger)
            print(item['index'], instrument, result['source_state_max_error'], result['samples'], flush=True)
    ledger['complete'] = True
    write(OUT / 'diagnosis.json', ledger)


if __name__ == '__main__':
    main()
