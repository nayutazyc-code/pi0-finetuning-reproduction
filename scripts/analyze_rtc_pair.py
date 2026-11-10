"""Read-only audit of the complete same-session v6 RTC20/RTC10 comparison."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from pi0_repro.config import load_config, OPENPI_COMMIT
from pi0_repro.motion_quality import motion_quality
from pi0_repro.scene import SceneConfig, build_scene_xml, load_scene_config
from summarize_banana_execution import summarize

SCHEMA = 'v6-rtc-same-session-v1'
PROFILES = ('rtc20', 'rtc10')
LAYOUTS = ('episode_004', 'episode_012', 'episode_019', 'episode_020', 'episode_026', 'episode_029')
SOURCE9 = ('config.py', 'execution.py', 'jerk_reference.py', 'simulation.py', 'viewer.py',
           'remote.py', 'banana_env.py', 'scene.py', 'diagnostics.py')
NORM_SHA = '6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
CHECKPOINT_SUFFIX = 'banana_v6_long100_accum8000/7999'
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
SELECTION_GATE = {'all_and_covered_median_p99_ratio_max': .90, 'worst_max_jerk_ratio_max': 1.10,
    'worst_physical_increment_ratio_max': 1.10, 'worst_pause_ge_0_5s_add_seconds_max': .10,
    'median_pause_ge_0_16s_total_add_seconds_max': .10, 'covered_count_min': 'baseline',
    'covered_median_simulated_duration_ratio_max': 1.10, 'humanvideo_review_required': True}
DATASET = ROOT/'data/lerobot/20261024-185715-merged-ebdbe3'


def read(path):
    return json.loads(path.read_text())


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return bool(np.isclose(left, right, rtol=1e-10, atol=1e-10))
    return left == right


def distribution(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {'n': 0, 'median': None, 'p95': None, 'max': None, 'min': None}
    return {'n': len(values), 'median': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'max': max(values), 'min': min(values)}


def validate_cases(plan, ledger):
    """Return only after every predeclared attempt, including failures, is present."""
    if plan.get('schema') != SCHEMA or plan.get('noise_paired') is not False:
        raise ValueError('Expected the frozen same-session schema with unpaired policy noise')
    cases, results = plan.get('cases', []), ledger.get('results', [])
    if not ledger.get('complete') or len(cases) != 36 or len(results) != 36:
        raise ValueError('Comparison incomplete; all 36 predeclared attempts are required before analysis')
    if len({(case['profile'], case['layout'], case['repeat']) for case in cases}) != 36:
        raise ValueError('Duplicate predeclared trial identity')
    expected = {(profile, layout, repeat) for profile in PROFILES for layout in LAYOUTS for repeat in (1, 2, 3)}
    if {(case['profile'], case['layout'], case['repeat']) for case in cases} != expected:
        raise ValueError('Cases do not cover the frozen six layouts, three repeats and two profiles')
    if plan.get('layouts') != list(LAYOUTS) or plan.get('repeats') != [1, 2, 3] or plan.get('attempts') != 36:
        raise ValueError('Frozen layout/repeat/update budget differs from the agreed experiment')
    fields = ('index', 'profile', 'layout', 'repeat', 'reset_seed')
    for index, (case, row) in enumerate(zip(cases, results)):
        if case['index'] != index or any(case[key] != row.get(key) for key in fields):
            raise ValueError('Results differ from frozen case order or reset seed')
    counterbalanced = []
    for repeat in (1, 2, 3):
        for layout_index, layout in enumerate(LAYOUTS):
            order = PROFILES if (repeat-1+layout_index) % 2 == 0 else tuple(reversed(PROFILES))
            counterbalanced.extend((profile, layout, repeat) for profile in order)
    if [(case['profile'], case['layout'], case['repeat']) for case in cases] != counterbalanced:
        raise ValueError('Frozen case order does not match the alternating AB/BA reset blocks')
    return results


def extra_metrics(data, tasks):
    def dist(values):
        values = np.asarray(values)
        return {'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)),
                'max': float(np.max(values))}
    states, actions = data['state'].astype(float), data['action'].astype(float)
    short = motion_quality(states, actions, tasks, pause_min_seconds=.16)
    return {'short_low_motion_min_seconds': .16,
            'short_low_motion_total_seconds': short['low_motion_pause_total_seconds'],
            'short_low_motion_max_seconds': short['low_motion_pause_max_seconds'],
            'short_low_motion_spans': short['low_motion_pause_spans'],
            'submitted_frame_max_jerk_rad_s3': dist(np.max(abs(np.diff(actions[:, JOINTS], n=3, axis=0)), axis=1)/.02**3),
            'post_step_command_tracking_residual_rad': dist(abs(states[1:, JOINTS]-actions[:-1, JOINTS]).ravel()),
            'right_wrist_axis11_increment_rad': dist(abs(np.diff(states[:, 11]))),
            'right_wrist_axis11_jerk_rad_s3': dist(abs(np.diff(states[:, 11], n=3))/.02**3)}


def metadata_valid(metadata):
    return (metadata.get('model_family') == 'pi0' and metadata.get('policy_config') == 'pi0_banana_lora'
            and metadata.get('action_dim') == 14 and metadata.get('action_horizon') == 100
            and metadata.get('action_semantics') == 'absolute_aloha_joint_targets'
            and metadata.get('openpi_commit') == OPENPI_COMMIT and metadata.get('denoise_steps') == 10
            and metadata.get('checkpoint', '').endswith('/'+CHECKPOINT_SUFFIX)
            and metadata.get('norm_stats_sha256') == NORM_SHA
            and metadata.get('rtc', {}).get('schema') == 'pi0-rtc-v1'
            and metadata.get('transport_proxy', {}).get('kind') == 'lossless-png')


METRICS = ('active_jerk_p99_rad_s3', 'active_jerk_max_rad_s3', 'all_frame_jerk_max_rad_s3',
           'physical_increment_p99_rad', 'physical_increment_max_rad',
           'pause_ge_0_5s_seconds', 'pause_ge_0_16s_seconds', 'pause_ge_0_5s_max_seconds', 'pause_ge_0_16s_max_seconds',
           'duration_seconds', 'active_motion_seconds', 'arm_path_l1_rad',
           'near_to_lift_seconds', 'carry_to_box_seconds', 'first_inside_box_seconds',
           'round_trip_median_ms', 'round_trip_p95_ms', 'round_trip_max_ms',
           'server_infer_median_ms', 'inference_underruns', 'blocking_seconds_after_startup')


def trial_summary(rows):
    """All attempts stay in the denominator; no-motion failures stay unscored."""
    covered = [row for row in rows if row.get('coverage', {}).get('transport_motion_comparable')]
    def subset(selected):
        return {'n': len(selected), 'metrics': {key: distribution([row.get(key) for row in selected]) for key in METRICS},
                'active_scored_n': sum(bool(row.get('coverage', {}).get('active_motion_scored')) for row in selected),
                'lifted_pinch_n': sum(bool(row.get('coverage', {}).get('lifted_pinch')) for row in selected),
                'entered_box_n': sum(bool(row.get('coverage', {}).get('entered_box')) for row in selected),
                'success_n': sum(row.get('success') is True for row in selected),
                'audit_passed_n': sum(row.get('audit_passed') is True for row in selected),
                'infrastructure_error_n': sum(bool(row.get('infrastructure_error')) for row in selected)}
    return {'all_trials': subset(rows), 'transport_covered_only': subset(covered),
            'incomplete_transport_attempt_indices': [row['index'] for row in rows if not row.get('coverage', {}).get('transport_motion_comparable')],
            'unavailable_metric_attempt_indices': [row['index'] for row in rows if row.get('metrics_unavailable')]}


def matched_resets(rows):
    result = []
    for repeat in (1, 2, 3):
        for layout in LAYOUTS:
            matched = {row['profile']: row for row in rows if row['layout'] == layout and row['repeat'] == repeat}
            if set(matched) != set(PROFILES):
                raise ValueError('Missing matched reset arm')
            base, candidate = matched['rtc20'], matched['rtc10']
            delta = {key: candidate.get(key)-base.get(key)
                     if candidate.get(key) is not None and base.get(key) is not None else None for key in METRICS}
            result.append({'layout': layout, 'repeat': repeat, 'reset_seed': base['reset_seed'],
                           'matching_reset_seed': base['reset_seed'] == candidate['reset_seed'],
                           'same_scene': base.get('scene') is not None and base.get('scene') == candidate.get('scene'),
                           'noise_paired': False, 'control_index': base['index'], 'candidate_index': candidate['index'],
                           'both_transport_covered': bool(base.get('coverage', {}).get('transport_motion_comparable')
                                                        and candidate.get('coverage', {}).get('transport_motion_comparable')),
                           'rtc10_minus_rtc20': delta,
                           'control_failure_class': base.get('failure_class'), 'candidate_failure_class': candidate.get('failure_class')})
    return result


def assess_selection(profiles, audit_passed):
    """Evaluate every frozen gate, retaining unknown values as insufficient."""
    control, candidate = profiles['rtc20'], profiles['rtc10']
    checks = []
    def compare(name, observed, reference, *, ratio=None, allowance=None, available=True):
        complete = observed is not None and reference is not None and available
        limit = reference*ratio if reference is not None and ratio is not None else (
            reference+allowance if reference is not None and allowance is not None else reference)
        checks.append({'name': name, 'observed': observed, 'control': reference, 'limit': limit,
                       'passed': bool(observed <= limit) if complete else None})
    for subset in ('all_trials', 'transport_covered_only'):
        base, changed = control[subset], candidate[subset]
        a, b = changed['metrics']['active_jerk_p99_rad_s3'], base['metrics']['active_jerk_p99_rad_s3']
        compare(subset+'_median_active_p99_improves_ge_10_percent', a['median'], b['median'], ratio=.9,
                available=a['n'] == changed['n'] and b['n'] == base['n'] and base['n'] > 0 and changed['n'] > 0)
    base, changed = control['all_trials'], candidate['all_trials']
    for name, metric, statistic, ratio, allowance in (
        ('worst_active_jerk_peak_le_110_percent', 'active_jerk_max_rad_s3', 'max', 1.1, None),
        ('worst_all_frame_jerk_peak_le_110_percent', 'all_frame_jerk_max_rad_s3', 'max', 1.1, None),
        ('worst_physical_increment_le_110_percent', 'physical_increment_max_rad', 'max', 1.1, None),
        ('max_original_dwell_increase_le_0_10_seconds', 'pause_ge_0_5s_max_seconds', 'max', None, .1),
        ('median_short_dwell_increase_le_0_10_seconds', 'pause_ge_0_16s_seconds', 'median', None, .1)):
        a, b = changed['metrics'][metric], base['metrics'][metric]
        compare(name, a[statistic], b[statistic], ratio=ratio, allowance=allowance,
                available=a['n'] == changed['n'] and b['n'] == base['n'] and base['n'] > 0 and changed['n'] > 0)
    bc, cc = control['transport_covered_only'], candidate['transport_covered_only']
    checks.append({'name': 'transport_covered_count_ge_control', 'observed': cc['n'], 'control': bc['n'],
                   'passed': cc['n'] >= bc['n']})
    a, b = cc['metrics']['duration_seconds'], bc['metrics']['duration_seconds']
    compare('covered_median_duration_le_110_percent', a['median'], b['median'], ratio=1.1,
            available=a['n'] == cc['n'] and b['n'] == bc['n'] and bc['n'] > 0 and cc['n'] > 0)
    missing = any(check['passed'] is None for check in checks)
    passed = audit_passed and not missing and all(check['passed'] for check in checks)
    return {'status': ('insufficient_evidence' if missing or not audit_passed else
                       'provisional_candidate_pending_video_review' if passed else 'retain_rtc20'),
            'provisional_numeric_candidate': bool(passed), 'audit_passed': audit_passed,
            'checks': checks, 'human_video_review_required': True, 'automatic_config_change': False,
            'scope': 'Multi-metric descriptive gate only; no significance, equivalence or general superiority claim.',
            'success_secondary': {'rtc20_successes': base['success_n'], 'rtc10_successes': changed['success_n'],
                                  'success_count_is_selection_gate': False}}


def reconstructed_prefix(data, latency, blend_steps):
    slots = {int(step): index for index, step in enumerate(data['rtc_request_steps'])}
    errors, valid, checked = [], len(slots) == len(data['rtc_request_steps']), 0
    for response in latency:
        declared = response.get('rtc', {})
        valid &= (declared.get('schema') == 'pi0-rtc-v1' and declared.get('guidance') == 'clean_estimate_vjp'
                  and declared.get('native_dimensions') == 14 and declared.get('padded_dimensions_guided') is False
                  and response.get('chunk_shape') == [100, 14])
        rtc = response.get('rtc_request')
        if not rtc:
            continue
        checked += 1
        origin = response['step']
        prefix = data['rtc_prefixes'][slots[origin]]
        end = min(response['received_at_step'], origin+rtc['valid_steps'])
        error = float(np.max(abs(data['action'][origin:end]-prefix[:end-origin]))) if end > origin else 0.
        errors.append(error)
        valid &= (hashlib.sha256(prefix.tobytes()).hexdigest() == rtc['prefix_sha256']
                  and int(data['rtc_valid_steps'][slots[origin]]) == rtc['valid_steps']
                  and error == rtc['planned_vs_executed_prefix_max_error']
                  and rtc['blend_steps'] == blend_steps
                  and declared.get('valid_steps') == rtc['valid_steps']
                  and declared.get('delay_steps') == rtc['delay_steps']
                  and declared.get('guidance_max') == rtc['guidance_max'] == .5
                  and declared.get('constrained_steps') == min(rtc['valid_steps'], rtc['delay_steps']+blend_steps))
    return {'passed': bool(valid and checked > 0 and max(errors, default=0.) == 0.),
            'received_prefixes_checked': checked, 'sent_prefixes': len(slots),
            'max_recomputed_prefix_error': max(errors, default=None)}


def atomic_new(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    os.link(temporary, path)
    temporary.unlink()


def analyze(out):
    plan, ledger = read(out/'plan.json'), read(out/'comparison.json')
    rows = validate_cases(plan, ledger)
    checks = []
    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})
    check('ledger_plan_matches_frozen_plan', plan == ledger['plan'])
    check('frozen_multi_metric_selection_gate', plan['selection_gate'] == SELECTION_GATE)
    for name, expected in plan['source_sha256'].items():
        check('source:'+name, sha(ROOT/'src/pi0_repro'/name) == expected)
    check('source9_and_motion_metric_bound', set(SOURCE9+('motion_quality.py',)) <= set(plan['source_sha256']))
    check('frozen_motion_metric_hash', sha(ROOT/'src/pi0_repro/motion_quality.py') == plan['metric_sha256']
          == plan['source_sha256']['motion_quality.py'])
    check('frozen_runner_hash', sha(out/'runner_source.py') == plan['script_sha256'])
    service_literal = next(node.value.value for node in ast.parse((out/'runner_source.py').read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SERVICE_CODE' for target in node.targets))
    check('frozen_service_checker_copy', (out/'service_check_source.py').read_text() == service_literal)
    check('frozen_helper_hash', sha(out/'screen_helper_source.py') == plan['helper_sha256'])
    mapping = DATASET/'local/pi0_banana_validation/source_episodes.json'
    check('frozen_validation_mapping_hash', sha(mapping) == plan['validation_mapping_sha256'])
    allowed = {row['source_episode'] for row in read(mapping)['episodes'] if row['split'] == 'validation'}
    check('validation_only_layouts', set(LAYOUTS) <= allowed)
    raw = Path(read(DATASET/'dataset_manifest.json')['source_path'])
    layouts = {layout: read(raw/layout/'layout.json') for layout in LAYOUTS}
    preflight = read(out/'preflight/requests.json')
    measured = [row['round_trip_ms'] for row in preflight['requests'][2:]]
    check('six_request_no_step_transport_preflight', len(preflight['requests']) == 6 and len(measured) == 4
          and np.isfinite(measured).all() and min(measured) >= 0
          and max(measured) < 1200 and np.median(measured) < 800,
          measured_round_trip_ms=measured)
    check('frozen_repeat_warmup_counts', plan['repeat_warmup_counts'] == [2, 3, 4])
    for repeat in (1, 2, 3):
        for profile in PROFILES:
            warm = read(out/f'warmup_r{repeat}_{profile}/requests.json')
            check(f'warmup_count:r{repeat}:{profile}', warm['profile'] == profile and len(warm['requests']) == repeat+1)
    configs = {profile: asdict(load_config(ROOT/'configs'/filename)) for profile, filename in plan['profiles'].items()}
    control, candidate = configs['rtc20'], json.loads(json.dumps(configs['rtc10']))
    check('rtc20_is_twenty', control['simulation']['rtc_blend_steps'] == 20)
    check('rtc10_is_ten', candidate['simulation']['rtc_blend_steps'] == 10)
    candidate['simulation']['rtc_blend_steps'] = 20
    check('profiles_have_only_rtc_blend_difference', candidate == control)
    for profile, filename in plan['profiles'].items():
        check('profile_hash:'+profile, sha(ROOT/'configs'/filename) == plan['profile_sha256'][profile])
    for endpoint in ('endpoint_before', 'endpoint_after'):
        record = ledger.get(endpoint) or {}
        check(endpoint+'_v6_metadata', record.get('metadata_verified') is True and metadata_valid(record.get('metadata', {})))
    check('same_service_process_and_checkpoint', ledger['endpoint_before'].get('service_identity')
          == ledger['endpoint_after'].get('service_identity')
          and set(ledger['endpoint_before'].get('service_identity', {})) >= {'pid', 'starttime_ticks', 'checkpoint'})
    check('service_identity_checkpoint_matches_metadata', all(
        ledger[endpoint].get('service_identity', {}).get('checkpoint') == ledger[endpoint].get('metadata', {}).get('checkpoint')
        for endpoint in ('endpoint_before', 'endpoint_after')))
    check('service_identity_has_positive_process_fields', all(
        type(ledger[endpoint].get('service_identity', {}).get(field)) is int
        and ledger[endpoint]['service_identity'][field] > 0
        for endpoint in ('endpoint_before', 'endpoint_after') for field in ('pid', 'starttime_ticks')))
    normalized, run_ports = [], []
    for row in rows:
        beginning = len(checks)
        index, profile = row['index'], row['profile']
        record = {key: row[key] for key in ('index', 'profile', 'layout', 'repeat', 'reset_seed')}
        record.update(run_directory=row.get('run_directory'), infrastructure_error=row.get('infrastructure_error'),
                      success=False, coverage={}, failure_class='metrics_unavailable', metrics_unavailable=True)
        try:
            directory = Path(row['run_directory']).resolve()
            run = read(directory/'run.json')
            config = run['config']
            run_ports.append(config['connection']['port'])
            check(f'{index}:no_infrastructure_error', not row.get('infrastructure_error'))
            check(f'{index}:source9', run['source_code_sha256'] == {name: plan['source_sha256'][name] for name in SOURCE9})
            check(f'{index}:pinned_upstream_commit', run['openpi_commit'] == OPENPI_COMMIT)
            metadata = read(directory/'server.json')
            check(f'{index}:server_checkpoint_norm_rtc_png', metadata_valid(metadata))
            check(f'{index}:server_same_before_endpoint', metadata == ledger['endpoint_before']['metadata'])
            wanted = json.loads(json.dumps(configs[profile]))
            wanted['connection']['port'] = config['connection']['port']
            wanted['simulation'].update(display=False, seed=row['reset_seed'], scene_config=config['simulation']['scene_config'])
            check(f'{index}:config_only_planned_overrides', wanted == config)
            scene_file = Path(config['simulation']['scene_config'])
            check(f'{index}:frozen_layout_scene_path', scene_file.resolve() == (out/(row['layout']+'.toml')).resolve())
            check(f'{index}:recorded_scene_hash', sha(scene_file) == row['scene_sha256'])
            check(f'{index}:frozen_layout_reset_seed', row['reset_seed'] == layouts[row['layout']]['reset_seed'])
            scene_settings = load_scene_config(scene_file)
            scene = json.loads(json.dumps(asdict(scene_settings)))
            original_scene = layouts[row['layout']]['scene']
            check(f'{index}:scene_matches_validation_layout', equal(scene, asdict(SceneConfig(**original_scene))))
            check(f'{index}:recorded_scene_matches_config', read(directory/'scene.json')['scene'] == scene)
            check(f'{index}:scene_xml_matches_config', (directory/'scene.xml').read_text().rstrip() == build_scene_xml(scene_settings).rstrip())
            tasks = json_rows(directory/'episode_000_task.jsonl')
            latency = json_rows(directory/'episode_000_latency.jsonl')
            all_details = summarize(directory)
            details = all_details['episodes'][0]
            check(f'{index}:cached_diagnostics_recomputed', equal(all_details, row['diagnostics']))
            with np.load(directory/'episode_000.npz', allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            quality = motion_quality(data['state'], data['action'], tasks)
            extra = extra_metrics(data, tasks)
            check(f'{index}:motion_first_v1_recomputed', equal(quality, row['motion_quality']))
            check(f'{index}:supplementary_metrics_recomputed', equal(extra, {key: row['extra'][key] for key in extra}))
            check(f'{index}:raw_prediction_and_timestamp_zero', details.get('raw_prediction_max_error') == 0
                  and details.get('max_action_timestamp_error_steps') == 0)
            indices, offsets = data['action_prediction_index'], data['action_prediction_offset']
            check(f'{index}:raw_prediction_and_timestamp_direct_reconstruction',
                  np.isfinite(data['predicted_chunks']).all()
                  and np.array_equal(data['predicted_chunks'][indices, offsets], data['requested_action'])
                  and np.array_equal(data['prediction_steps'][indices]+offsets, np.arange(len(indices))))
            prefix = reconstructed_prefix(data, latency, config['simulation']['rtc_blend_steps'])
            check(f'{index}:prefix_hash_actual_forecast_and_rtc_response', prefix['passed'],
                  **{key: value for key, value in prefix.items() if key != 'passed'})
            check(f'{index}:recorded_prefix_error_zero', row['prefix_error'] == 0)
            check(f'{index}:lossless_png_hashes', details.get('lossless_image_hashes_verified') is True)
            check(f'{index}:video_hash', sha(directory/'episode_000.mp4') == row['video_sha256'])
            result = details['result']
            check(f'{index}:no_runtime_error', not result.get('error') and not result.get('aborted_by_window_close'))
            check(f'{index}:result_seed_and_recording_length', result.get('seed') == row['reset_seed']
                  and result.get('steps') == len(data['action']))
            check(f'{index}:success_coverage_agreement', result['success'] == quality['coverage']['success'])
            check(f'{index}:no_inference_underrun', result.get('execution', {}).get('inference_underruns') == 0)
            active, progress = quality['active_frame_max_jerk_rad_s3'], details['task_progress']
            rtt = [r['round_trip_ms'] for r in latency]
            infer = [r['server_timing']['infer_ms'] for r in latency if 'infer_ms' in r.get('server_timing', {})]
            check(f'{index}:latency_metrics_finite_and_complete', bool(rtt) and len(infer) == len(rtt)
                  and np.isfinite(rtt).all() and np.isfinite(infer).all() and min(rtt) >= 0 and min(infer) >= 0
                  and details.get('blocking_seconds_after_startup') is not None
                  and np.isfinite(details['blocking_seconds_after_startup']))
            coverage = quality['coverage']
            record.update(scene=scene, success=result['success'], coverage=coverage, metrics_unavailable=False,
                failure_class=None if coverage['transport_motion_comparable'] else
                    ('never_lifted_pinch' if not coverage['lifted_pinch'] else 'lifted_but_never_entered_box'),
                active_jerk_p99_rad_s3=active['p99'] if active else None,
                active_jerk_max_rad_s3=active['max'] if active else None,
                all_frame_jerk_max_rad_s3=quality['all_frame_max_jerk_rad_s3']['max'],
                physical_increment_p99_rad=quality['physical_joint_increment_rad']['p99'],
                physical_increment_max_rad=quality['physical_joint_increment_rad']['max'],
                pause_ge_0_5s_seconds=quality['low_motion_pause_total_seconds'],
                pause_ge_0_5s_max_seconds=quality['low_motion_pause_max_seconds'],
                pause_ge_0_16s_seconds=extra['short_low_motion_total_seconds'],
                pause_ge_0_16s_max_seconds=extra['short_low_motion_max_seconds'],
                duration_seconds=len(data['action'])*.02, active_motion_seconds=quality['active_motion_seconds'],
                arm_path_l1_rad=quality['arm_path_l1_rad'], near_to_lift_seconds=progress['near_to_lifted_pinch_seconds'],
                first_inside_box_seconds=progress['first_inside_box_seconds'],
                carry_to_box_seconds=(progress['first_inside_box_seconds']-progress['first_confirmed_lifted_pinch_seconds']
                    if progress['first_inside_box_seconds'] is not None and progress['first_confirmed_lifted_pinch_seconds'] is not None else None),
                round_trip_median_ms=distribution(rtt)['median'], round_trip_p95_ms=distribution(rtt)['p95'],
                round_trip_max_ms=max(rtt, default=None), server_infer_median_ms=distribution(infer)['median'],
                inference_underruns=result.get('execution', {}).get('inference_underruns'),
                blocking_seconds_after_startup=details.get('blocking_seconds_after_startup'),
                motion_first_v1=quality, supplementary_ge_0_16s=extra, task_progress=progress,
                chunk_continuity=details.get('chunk_continuity'), prefix_audit=prefix,
                video=str(directory/'episode_000.mp4'), video_sha256=row['video_sha256'])
        except (KeyError, ValueError, IndexError, OSError, TypeError) as error:
            check(f'{index}:analysis_available', False, error=f'{type(error).__name__}: {error}')
            record['analysis_error'] = f'{type(error).__name__}: {error}'
        record['audit_passed'] = all(item['passed'] for item in checks[beginning:])
        normalized.append(record)
    matched = matched_resets(normalized)
    check('same_local_forward_port_all_trials', len(set(run_ports)) == 1 and len(run_ports) == 36)
    for pair in matched:
        check(f"matched_reset:{pair['layout']}:{pair['repeat']}", pair['matching_reset_seed'] and pair['same_scene'])
    summaries = {profile: trial_summary([r for r in normalized if r['profile'] == profile]) for profile in PROFILES}
    passed = all(item['passed'] for item in checks)
    covered_pairs = [pair for pair in matched if pair['both_transport_covered']]
    report = {'schema': 'v6-rtc-same-session-analysis-v1', 'attempt_count': len(normalized), 'audit_passed': passed,
              'frozen_plan_sha256': sha(out/'plan.json'), 'analyzer_sha256': sha(Path(__file__)),
              'profiles': summaries, 'matched_resets': matched, 'attempts': normalized,
              'matched_reset_difference_summary': {
                  'all_matched_resets': {'n': len(matched), 'rtc10_minus_rtc20': {
                      key: distribution([pair['rtc10_minus_rtc20'][key] for pair in matched]) for key in METRICS}},
                  'both_transport_covered_only': {'n': len(covered_pairs), 'rtc10_minus_rtc20': {
                      key: distribution([pair['rtc10_minus_rtc20'][key] for pair in covered_pairs]) for key in METRICS}}},
              'scope': 'Six layouts, three repetitions, same session and matched reset scenes; policy noise is unpaired. No significance or equivalence claim.',
              'selection': assess_selection(summaries, passed), 'frozen_selection_gate': plan['selection_gate'],
              'metric_boundaries': ['All 36 attempts remain, including incomplete transport and infrastructure failures.',
                  'Motion-first-v1 >=0.5s dwell and supplementary >=0.16s dwell remain separate.',
                  'Inactive/no-motion cases have null active jerk; no-motion failures are never scored as perfect smoothness.',
                  'All-trial and transport-covered-only metrics are separate; failed transport cannot establish improved completed motion.',
                  'Each metric reports its available denominator; missing values do not become zero.',
                  'Worst peaks are reported alongside typical active p99, physical increments, dwell, phases, coverage and latency.',
                  'Derivatives and phase durations use simulation time; wall-clock latency and underruns are separate.',
                  'State is pre-step and task is post-step; next state is compared with prior action for tracking residual.']}
    return report, {'schema': 'v6-rtc-same-session-audit-v1', 'passed': passed, 'checks': checks,
                    'failed_checks': [item['name'] for item in checks if not item['passed']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('Analysis directory must be inside project outputs')
    if any((out/name).exists() for name in ('aggregate.json', 'audit.json')):
        raise FileExistsError('Existing generated analysis must be retained')
    report, audit = analyze(out)
    atomic_new(out/'aggregate.json', report)
    atomic_new(out/'audit.json', audit)
    print(json.dumps({'attempts': report['attempt_count'], 'audit_passed': audit['passed'],
                      'failed_checks': audit['failed_checks'], 'aggregate': str(out/'aggregate.json')}))


if __name__ == '__main__':
    main()
