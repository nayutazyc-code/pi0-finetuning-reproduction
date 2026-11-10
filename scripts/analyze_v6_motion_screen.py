"""Audit a completed 12-attempt v6 screen without rerunning physics or inference."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from pi0_repro.config import load_config, OPENPI_COMMIT, SimulationConfig
from pi0_repro.motion_quality import motion_quality
from pi0_repro.scene import build_scene_xml, load_scene_config
from summarize_banana_execution import summarize

PROFILES = {'baseline': 'banana_v6_bridge20_pi0.toml', 'acc20': 'banana_v6_acc20_pi0.toml',
            'rtc30': 'banana_v6_rtc30_pi0.toml', 'wrist125': 'banana_v6_wrist125_pi0.toml'}
SOURCE9 = ('config.py', 'execution.py', 'jerk_reference.py', 'simulation.py', 'viewer.py',
           'remote.py', 'banana_env.py', 'scene.py', 'diagnostics.py')
NORM = '6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
SUFFIX = 'banana_v6_long100_accum8000/7999'
DATASET = ROOT / 'data/lerobot/20261024-185715-merged-ebdbe3'
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def read(path):
    return json.loads(path.read_text())


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparable(left, right):
    """Compare recomputed nested metrics, tolerating only floating roundoff."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(comparable(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(comparable(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return bool(np.isclose(left, right, rtol=1e-10, atol=1e-10))
    return left == right


def supplementary(data, tasks):
    def dist(values):
        values = np.asarray(values)
        return {'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)),
                'max': float(np.max(values))}
    state, action = data['state'].astype(float), data['action'].astype(float)
    short = motion_quality(state, action, tasks, pause_min_seconds=.16)
    return {'short_low_motion_min_seconds': .16,
            'short_low_motion_total_seconds': short['low_motion_pause_total_seconds'],
            'short_low_motion_max_seconds': short['low_motion_pause_max_seconds'],
            'short_low_motion_spans': short['low_motion_pause_spans'],
            'submitted_frame_max_jerk_rad_s3': dist(np.max(abs(np.diff(action[:, JOINTS], n=3, axis=0)), axis=1)/.02**3),
            'post_step_command_tracking_residual_rad': dist(abs(state[1:, JOINTS]-action[:-1, JOINTS]).ravel()),
            'right_wrist_axis11_increment_rad': dist(abs(np.diff(state[:, 11]))),
            'right_wrist_axis11_jerk_rad_s3': dist(abs(np.diff(state[:, 11], n=3))/.02**3)}


def metadata_valid(metadata):
    return (metadata.get('model_family') == 'pi0' and metadata.get('policy_config') == 'pi0_banana_lora'
            and metadata.get('action_dim') == 14 and metadata.get('action_horizon') == 100
            and metadata.get('action_semantics') == 'absolute_aloha_joint_targets'
            and metadata.get('openpi_commit') == OPENPI_COMMIT and metadata.get('denoise_steps') == 10
            and metadata.get('checkpoint', '').endswith('/' + SUFFIX)
            and metadata.get('norm_stats_sha256') == NORM
            and metadata.get('rtc', {}).get('schema') == 'pi0-rtc-v1'
            and metadata.get('transport_proxy', {}).get('kind') == 'lossless-png')


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def limiter_treatment(data, settings, scene_file):
    """Rebuild commands with pure projection; never reset or step physics."""
    from pi0_repro.banana_env import BananaBoxEnv
    from pi0_repro.execution import JointTargetLimiter
    env = BananaBoxEnv(load_scene_config(scene_file), max_episode_steps=len(data['action']),
                       terminate_on_success=False, render_observations=False)
    commands = {}
    try:
        for name, acceleration in [('recorded', settings.max_joint_acceleration), ('baseline40', 40.)]:
            candidate = copy.deepcopy(settings)
            candidate.max_joint_acceleration = acceleration
            limiter = JointTargetLimiter(data['state'][0], candidate, env.action_space.low, env.action_space.high)
            output = []
            for target in data['blended_action']:
                command = env.project_action(limiter.limit(target))
                output.append(command)
                limiter.update_executed(command)
            commands[name] = np.asarray(output)
        original = data['action']
        actual_error = np.max(abs(commands['recorded']-original), axis=1)
        baseline_error = np.max(abs(commands['baseline40']-original), axis=1)
        return {'scope': 'Frozen references, state[0] and XML control bounds; env.project_action only. No reset, physics step, inference or new closed-loop trial.',
                'recorded_acceleration_limit_rad_s2': settings.max_joint_acceleration,
                'actual_limiter_reconstruction_bitwise_equal': bool(np.array_equal(commands['recorded'], original)),
                'actual_limiter_reconstruction_max_error_rad': float(actual_error.max()),
                'baseline40_reconstruction_bitwise_equal': bool(np.array_equal(commands['baseline40'], original)),
                'baseline40_max_command_difference_rad': float(baseline_error.max()),
                'baseline40_changed_frames_gt_1e_7': int((baseline_error > 1e-7).sum()),
                'recorded_command_max_acceleration_rad_s2': float(np.max(abs(np.diff(original[:, JOINTS], n=2, axis=0)))/.02**2),
                'physics_steps': env.steps,
                'limiter_treatment': ('inactive_limiter' if settings.max_joint_acceleration == 20. and not (baseline_error > 1e-7).any()
                                      else 'command_changed_vs_baseline40' if (baseline_error > 1e-7).any() else 'matches_baseline40')}
    finally:
        env.close()


def profile_summary(rows):
    covered = [row for row in rows if row['coverage']['transport_motion_comparable']]
    def statistics_for(subset):
        return {'count': len(subset),
                'median_active_jerk_p99_rad_s3': median([r['active_jerk_p99_rad_s3'] for r in subset]),
                'worst_active_jerk_peak_rad_s3': max([r['active_jerk_max_rad_s3'] for r in subset if r['active_jerk_max_rad_s3'] is not None], default=None),
                'worst_all_frame_jerk_peak_rad_s3': max([r['all_frame_jerk_max_rad_s3'] for r in subset], default=None),
                'worst_physical_increment_rad': max([r['physical_increment_max_rad'] for r in subset], default=None),
                'median_original_pause_ge_0_5s_total_seconds': median([r['pause_ge_0_5s_seconds'] for r in subset]),
                'median_supplementary_pause_ge_0_16s_total_seconds': median([r['pause_ge_0_16s_seconds'] for r in subset]),
                'median_simulated_duration_seconds': median([r['duration_seconds'] for r in subset]),
                'median_active_motion_seconds': median([r['active_motion_seconds'] for r in subset]),
                'median_arm_path_l1_rad': median([r['arm_path_l1_rad'] for r in subset]),
                'median_first_inside_box_seconds': median([r['task_progress']['first_inside_box_seconds'] for r in subset]),
                'median_near_to_lift_seconds': median([r['task_progress']['near_to_lifted_pinch_seconds'] for r in subset]),
                'median_carry_to_box_seconds': median([r['carry_to_box_seconds'] for r in subset]),
                'median_blocking_seconds_after_startup': median([r['blocking_seconds_after_startup'] for r in subset])}
    return {'all_attempts': statistics_for(rows), 'transport_covered_only': statistics_for(covered),
            'lifted_pinch_count': sum(r['coverage']['lifted_pinch'] for r in rows),
            'entered_box_count': sum(r['coverage']['entered_box'] for r in rows),
            'success_count': sum(r['success'] for r in rows),
            'active_motion_scored_count': sum(r['coverage']['active_motion_scored'] for r in rows),
            'incomplete_transport_attempt_indices': [r['index'] for r in rows if not r['coverage']['transport_motion_comparable']]}


def gate(candidate, control, audit_passed):
    """Frozen numeric gate; video review remains a separate required condition."""
    c, b = candidate['all_attempts'], control['all_attempts']
    cc, bc = candidate['transport_covered_only'], control['transport_covered_only']
    checks = {
        'median_active_jerk_p99_improves_ge_10_percent': (
            c['median_active_jerk_p99_rad_s3'] is not None and b['median_active_jerk_p99_rad_s3'] is not None
            and b['median_active_jerk_p99_rad_s3'] > 0
            and c['median_active_jerk_p99_rad_s3'] <= .9*b['median_active_jerk_p99_rad_s3']),
        'worst_physical_increment_le_110_percent_control': (
            c['worst_physical_increment_rad'] is not None and b['worst_physical_increment_rad'] is not None
            and c['worst_physical_increment_rad'] <= 1.1*b['worst_physical_increment_rad']),
        'median_short_dwell_increase_le_0_2_seconds': (
            c['median_supplementary_pause_ge_0_16s_total_seconds'] is not None
            and b['median_supplementary_pause_ge_0_16s_total_seconds'] is not None
            and c['median_supplementary_pause_ge_0_16s_total_seconds'] <= b['median_supplementary_pause_ge_0_16s_total_seconds']+.2),
        'transport_covered_count_ge_control': cc['count'] >= bc['count'],
        'covered_median_duration_le_110_percent_control': (
            cc['median_simulated_duration_seconds'] is not None and bc['median_simulated_duration_seconds'] is not None
            and cc['median_simulated_duration_seconds'] <= 1.1*bc['median_simulated_duration_seconds'])}
    # Incomplete failures still appear in the original all-attempt aggregation;
    # their low derivatives never suffice to claim improved transport motion.
    covered_jerk_supported = (cc['median_active_jerk_p99_rad_s3'] is not None
        and bc['median_active_jerk_p99_rad_s3'] is not None
        and cc['median_active_jerk_p99_rad_s3'] <= .9*bc['median_active_jerk_p99_rad_s3'])
    return {'predeclared_numeric_checks': checks, 'predeclared_numeric_pass': all(checks.values()),
            'audit_passed': audit_passed, 'covered_only_jerk_improvement_supported': covered_jerk_supported,
            'provisional_candidate_pending_video_review': bool(all(checks.values()) and audit_passed and covered_jerk_supported),
            'video_review': 'Required for all 12 attempts; this script does not assert a visual review.',
            'promotion': 'No automatic promotion. Retain baseline pending review; small unpaired screen only.'}


def analyze(out):
    plan, ledger = read(out/'plan.json'), read(out/'comparison.json')
    expected = [(layout, profile) for layout, profiles in plan['blocks'] for profile in profiles]
    if (not ledger.get('complete') or len(ledger.get('attempts', [])) != 12
            or len(expected) != 12 or plan.get('attempts') != 12):
        raise ValueError('Screen incomplete; do not write aggregates until all 12 attempts finish')
    if [(r['layout'], r['profile']) for r in ledger['attempts']] != expected:
        raise ValueError('Attempt order differs from the frozen 12-attempt plan')
    if [r['index'] for r in ledger['attempts']] != list(range(1, 13)):
        raise ValueError('Attempt indices are not exactly 1 through 12')
    checks = []
    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})
    check('ledger_plan_matches_frozen_plan', ledger['plan'] == plan)
    check('schema_and_unpaired_scope', plan['schema'] == 'v6-runtime-motion-screen-v1' and plan['noise_paired'] is False)
    check('frozen_runner_copy_hash', sha(out/'runner_source.py') == plan['script_sha256'])
    check('frozen_selector_copy_hash', sha(out/'selector_source.py') == plan['selector_sha256'])
    warm_requests = read(out/'warmup/requests.json')['requests']
    if plan['warmup_scope'].startswith('Six requests'):
        last_rtts = [r['round_trip_ms'] for r in warm_requests[-4:]]
        check('six_request_warmup_transport_preflight', len(warm_requests) == 6 and len(last_rtts) == 4
              and max(last_rtts) < 1200 and statistics.median(last_rtts) < 800,
              final_four_round_trip_ms=last_rtts)
        check('frozen_runner_ipqos_none', 'IPQoS=none' in (out/'runner_source.py').read_text())
    else:
        check('two_request_warmup', len(warm_requests) == 2)
    for name, expected_digest in plan['source_sha256'].items():
        check('current_source:' + name, sha(ROOT/'src/pi0_repro'/name) == expected_digest)
    mapping = DATASET/'local/pi0_banana_validation/source_episodes.json'
    check('validation_mapping_hash', sha(mapping) == plan['validation_mapping_sha256'])
    allowed = {r['source_episode'] for r in read(mapping)['episodes'] if r['split'] == 'validation'}
    raw = Path(read(DATASET/'dataset_manifest.json')['source_path'])
    configs = {p: asdict(load_config(ROOT/'configs'/f)) for p, f in PROFILES.items()}
    base = configs['baseline']
    for profile, config in configs.items():
        check('frozen_profile_hash:' + profile, sha(ROOT/'configs'/PROFILES[profile]) == plan['profile_sha256'][profile])
        normalized = json.loads(json.dumps(config))
        if profile == 'acc20':
            check('acc20_single_value', normalized['simulation']['max_joint_acceleration'] == 20.)
            normalized['simulation']['max_joint_acceleration'] = 40.
        elif profile == 'rtc30':
            check('rtc30_single_value', normalized['simulation']['rtc_blend_steps'] == 30)
            normalized['simulation']['rtc_blend_steps'] = 20
        elif profile == 'wrist125':
            check('wrist_scene_profile', Path(normalized['simulation']['scene_config']).name == 'banana_demo_scene_wrist125.toml')
            normalized['simulation']['scene_config'] = base['simulation']['scene_config']
        check('config_single_factor:' + profile, normalized == base)
    baseline_scene = asdict(load_scene_config(ROOT/'configs/banana_demo_scene.toml'))
    wrist_scene = asdict(load_scene_config(ROOT/'configs/banana_demo_scene_wrist125.toml'))
    check('wrist125_damping_value', wrist_scene['wrist_damping_scale'] == 1.25)
    wrist_scene['wrist_damping_scale'] = 1.
    check('wrist_scene_single_factor', wrist_scene == baseline_scene)
    selected = ledger.get('selection') or {}
    check('selection_v6_metadata', selected.get('metadata_verified') is True and metadata_valid(selected.get('metadata', {})))
    final = ledger.get('final_endpoint') or {}
    check('final_v6_metadata', final.get('metadata_verified') is True and metadata_valid(final.get('metadata', {})))
    attempts = []
    for row in ledger['attempts']:
        start = len(checks)
        index, profile, layout = row['index'], row['profile'], row['layout']
        run = Path(row['run_directory']).resolve()
        check(f'{index}:no_infrastructure_error', not row.get('infrastructure_error'))
        check(f'{index}:validation_layout', layout in allowed)
        recorded = read(run/'run.json')
        check(f'{index}:source9', recorded['source_code_sha256'] == {name: plan['source_sha256'][name] for name in SOURCE9})
        check(f'{index}:recorded_upstream_commit', recorded['openpi_commit'] == OPENPI_COMMIT)
        metadata = read(run/'server.json')
        check(f'{index}:server_metadata_checkpoint_norm_rtc', metadata_valid(metadata))
        check(f'{index}:server_matches_selection', metadata == selected.get('metadata'))
        config = recorded['config']
        wanted = json.loads(json.dumps(configs[profile]))
        scene_file = out/f'{layout}_{"wrist125" if profile == "wrist125" else "normal"}.toml'
        wanted['simulation'].update(scene_config=str(scene_file), display=False, seed=plan['scene_seeds'][layout])
        wanted['connection']['port'] = config['connection']['port']
        check(f'{index}:run_config_matches_profile_and_overrides', config == wanted)
        check(f'{index}:scene_hash', sha(scene_file) == row['scene_sha256'])
        layout_record = read(raw/layout/'layout.json')
        check(f'{index}:layout_reset_seed', layout_record['reset_seed'] == plan['scene_seeds'][layout])
        scene = asdict(load_scene_config(scene_file))
        check(f'{index}:recorded_scene_matches_file', read(run/'scene.json')['scene'] == json.loads(json.dumps(scene)))
        check(f'{index}:recorded_scene_xml_matches_config',
              (run/'scene.xml').read_text().rstrip() == build_scene_xml(load_scene_config(scene_file)).rstrip())
        # Compare normalized dataclasses so omitted defaults cannot hide a second factor.
        normal_scene = asdict(load_scene_config(out/f'{layout}_normal.toml'))
        original_layout_scene = dict(layout_record['scene'], wrist_damping_scale=1.)
        expected_scene = asdict(load_scene_config(out/f'{layout}_normal.toml'))
        check(f'{index}:normal_scene_matches_layout_fields', all(comparable(normal_scene[k], v) for k, v in original_layout_scene.items()))
        expected_scene['wrist_damping_scale'] = 1.25 if profile == 'wrist125' else 1.
        check(f'{index}:runtime_scene_single_factor', scene == expected_scene)
        tasks = json_rows(run/'episode_000_task.jsonl')
        latency = json_rows(run/'episode_000_latency.jsonl')
        details = summarize(run)['episodes'][0]
        with np.load(run/'episode_000.npz', allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        quality = motion_quality(data['state'], data['action'], tasks)
        extra = supplementary(data, tasks)
        treatment = limiter_treatment(data, SimulationConfig(**config['simulation']), scene_file)
        check(f'{index}:actual_limiter_reconstruction', treatment['actual_limiter_reconstruction_bitwise_equal'])
        check(f'{index}:limiter_diagnostic_no_physics_steps', treatment['physics_steps'] == 0)
        check(f'{index}:motion_first_v1_recomputed', comparable(quality, row['motion_quality']))
        check(f'{index}:supplementary_recomputed', comparable(extra, {k: row['extra'][k] for k in extra}))
        check(f'{index}:raw_prediction_and_timestamp_zero', details.get('raw_prediction_max_error') == 0
              and details.get('max_action_timestamp_error_steps') == 0)
        prefix_by_step = {int(step): i for i, step in enumerate(data['rtc_request_steps'])}
        prefix_errors, digest_ok, received_prefix_count = [], True, 0
        for request in latency:
            rtc = request.get('rtc_request')
            if not rtc:
                continue
            received_prefix_count += 1
            origin = request['step']
            slot = prefix_by_step[origin]
            prefix = data['rtc_prefixes'][slot]
            digest_ok &= hashlib.sha256(prefix.tobytes()).hexdigest() == rtc['prefix_sha256']
            end = min(request['received_at_step'], origin+rtc['valid_steps'])
            error = float(np.max(abs(data['action'][origin:end] - prefix[:end-origin]))) if end > origin else 0.
            prefix_errors.append(error)
            digest_ok &= (int(data['rtc_valid_steps'][slot]) == rtc['valid_steps']
                          and error == rtc['planned_vs_executed_prefix_max_error']
                          and rtc['blend_steps'] == config['simulation']['rtc_blend_steps']
                          and request.get('rtc', {}).get('schema') == 'pi0-rtc-v1')
        check(f'{index}:prefix_reconstruction', received_prefix_count > 0 and digest_ok and max(prefix_errors, default=0.) == 0.,
              received_prefix_count=received_prefix_count, max_error=max(prefix_errors, default=None),
              sent_prefix_count=len(prefix_by_step))
        check(f'{index}:recorded_prefix_error_zero', row.get('max_prefix_forecast_error') == 0)
        check(f'{index}:lossless_png_pixel_digests', details.get('lossless_image_hashes_verified') is True)
        check(f'{index}:video_hash', sha(run/'episode_000.mp4') == row['video_sha256'])
        result = details['result']
        check(f'{index}:finished_without_runtime_error', not result.get('error') and not result.get('aborted_by_window_close'))
        check(f'{index}:success_agreement', row['success'] == quality['coverage']['success'] == result['success'])
        active = quality['active_frame_max_jerk_rad_s3']
        progress = details['task_progress']
        carry = (progress['first_inside_box_seconds'] - progress['first_confirmed_lifted_pinch_seconds']
                 if progress['first_inside_box_seconds'] is not None and progress['first_confirmed_lifted_pinch_seconds'] is not None else None)
        attempts.append({'index': index, 'profile': profile, 'layout': layout, 'run_directory': str(run),
            'success': row['success'], 'coverage': quality['coverage'], 'duration_seconds': len(data['action'])*.02,
            'active_jerk_p99_rad_s3': active['p99'] if active else None,
            'active_jerk_max_rad_s3': active['max'] if active else None,
            'all_frame_jerk_max_rad_s3': quality['all_frame_max_jerk_rad_s3']['max'],
            'physical_increment_max_rad': quality['physical_joint_increment_rad']['max'],
            'pause_ge_0_5s_seconds': quality['low_motion_pause_total_seconds'],
            'pause_ge_0_16s_seconds': extra['short_low_motion_total_seconds'],
            'active_motion_seconds': quality['active_motion_seconds'], 'arm_path_l1_rad': quality['arm_path_l1_rad'],
            'motion_first_v1': quality, 'supplementary_ge_0_16s': extra, 'task_progress': progress,
            'carry_to_box_seconds': carry, 'blocking_seconds_after_startup': details.get('blocking_seconds_after_startup'),
            'action_apply_interval_ms': details.get('action_apply_interval_ms'),
            'chunk_continuity': details.get('chunk_continuity'), 'video': str(run/'episode_000.mp4'),
            'limiter_treatment': treatment,
            'failure_class': None if quality['coverage']['transport_motion_comparable'] else
                ('never_lifted_pinch' if not quality['coverage']['lifted_pinch'] else 'lifted_but_never_entered_box'),
            'audit_passed': all(c['passed'] for c in checks[start:])})
    ports = {read(Path(row['run_directory'])/'run.json')['config']['connection']['port'] for row in attempts}
    check('same_local_port_all_attempts', len(ports) == 1)
    profiles = {profile: profile_summary([r for r in attempts if r['profile'] == profile]) for profile in PROFILES}
    audit_passed = all(check['passed'] for check in checks)
    per_layout = {}
    for layout, _ in plan['blocks']:
        rows = {r['profile']: r for r in attempts if r['layout'] == layout}
        base_row = rows['baseline']
        per_layout[layout] = {'noise_paired': False, 'attempts': {p: r['index'] for p, r in rows.items()}, 'candidate_minus_control': {}}
        for profile in (p for p in PROFILES if p != 'baseline'):
            candidate = rows[profile]
            metrics = ('active_jerk_p99_rad_s3', 'physical_increment_max_rad', 'pause_ge_0_5s_seconds',
                       'pause_ge_0_16s_seconds', 'duration_seconds', 'active_motion_seconds', 'arm_path_l1_rad')
            per_layout[layout]['candidate_minus_control'][profile] = {
                'both_transport_covered': candidate['coverage']['transport_motion_comparable'] and base_row['coverage']['transport_motion_comparable'],
                'candidate_failure_class': candidate['failure_class'], 'control_failure_class': base_row['failure_class'],
                **{key: candidate[key]-base_row[key] if candidate[key] is not None and base_row[key] is not None else None for key in metrics}}
    gates = {p: gate(profiles[p], profiles['baseline'], audit_passed) for p in PROFILES if p != 'baseline'}
    acc20_rows = [r for r in attempts if r['profile'] == 'acc20']
    inactive_acc20 = all(r['limiter_treatment']['limiter_treatment'] == 'inactive_limiter' for r in acc20_rows)
    gates['acc20']['treatment_identification'] = ('inactive_limiter/no_identifiable_treatment' if inactive_acc20
        else 'Command difference observed; stochastic closed-loop differences still do not establish causation.')
    gates['acc20']['inactive_limiter_attempt_indices'] = [r['index'] for r in acc20_rows
        if r['limiter_treatment']['limiter_treatment'] == 'inactive_limiter']
    if inactive_acc20:
        gates['acc20']['provisional_candidate_pending_video_review'] = False
        gates['acc20']['promotion'] = 'Not an effective parameter candidate: acc20 commands equal baseline40 on every recorded acc20 reference. Do not attribute score/success differences to the unused limiter.'
    aggregate = {'schema': 'v6-runtime-motion-screen-analysis-v1', 'complete_attempt_count': len(attempts),
        'scope': 'All 12 attempts, three validation layouts per profile, unpaired policy noise. No retries or first-success selection; no general superiority claim.',
        'frozen_plan_sha256': sha(out/'plan.json'), 'analyzer_sha256': sha(Path(__file__)),
        'audit_passed': audit_passed, 'profiles': profiles, 'per_layout': per_layout, 'attempts': attempts,
        'predeclared_gate_text': plan['promotion_gate'], 'gates': gates,
        'provisional_candidates_pending_video_review': [p for p, g in gates.items() if g['provisional_candidate_pending_video_review']],
        'recommendation': 'Retain baseline until required all-video review. Numeric candidates remain provisional.',
        'metric_boundaries': ['motion-first-v1 original pauses >=0.5s remain unchanged.',
            'Supplementary pauses >=0.16s are separate and use simulation time; intentional holds may be included.',
            'Gate duration means total simulated trial duration restricted to transport-covered attempts.',
            'Gate active jerk uses all active-scored attempts, with a separately disclosed covered-only support guard.',
            'Incomplete transport derivatives remain descriptive and do not establish smoother completed motion.',
            'State samples are pre-step and task rows post-step; tracking residual pairs next state with prior action.',
            'No independent repetitions per profile/layout; per-layout differences do not pair policy noise.']}
    return aggregate, {'schema': 'v6-runtime-motion-screen-audit-v1', 'passed': audit_passed, 'checks': checks,
                       'failed_checks': [c['name'] for c in checks if not c['passed']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('Analysis output must be within project outputs')
    for filename in ('aggregate.json', 'analysis_audit.json'):
        if (out/filename).exists():
            raise FileExistsError('Retain existing generated analysis: ' + str(out/filename))
    aggregate, audit = analyze(out)
    for filename, value in [('aggregate.json', aggregate), ('analysis_audit.json', audit)]:
        temporary = out/(filename+'.tmp')
        with temporary.open('x') as stream:
            stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        # Publish a complete file without replacing an existing result, even if
        # another analyst finished between the initial check and this write.
        os.link(temporary, out/filename)
        temporary.unlink()
    print(json.dumps({'audit_passed': audit['passed'], 'attempts': aggregate['complete_attempt_count'],
                      'provisional_candidates_pending_video_review': aggregate['provisional_candidates_pending_video_review'],
                      'aggregate': str(out/'aggregate.json'), 'failed_checks': audit['failed_checks']}))


if __name__ == '__main__':
    main()
