"""Offline analysis checks; no inference, rendering or physics simulation."""
import copy
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

SPEC = importlib.util.spec_from_file_location('rtc_pair_analysis',
    Path(__file__).resolve().parents[1]/'scripts/analyze_rtc_pair.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def plan_and_results():
    cases = []
    for repeat in (1, 2, 3):
        for index, layout in enumerate(analysis.LAYOUTS):
            profiles = analysis.PROFILES if (repeat-1+index) % 2 == 0 else tuple(reversed(analysis.PROFILES))
            for profile in profiles:
                cases.append({'index': len(cases), 'profile': profile, 'layout': layout,
                              'repeat': repeat, 'reset_seed': index+100})
    plan = {'schema': analysis.SCHEMA, 'layouts': list(analysis.LAYOUTS), 'repeats': [1, 2, 3],
            'attempts': 36, 'noise_paired': False, 'cases': cases}
    return plan, {'plan': plan, 'complete': True, 'results': copy.deepcopy(cases)}


def metric_row(profile, index=0):
    values = {key: 1. for key in analysis.METRICS}
    values.update({'profile': profile, 'index': index, 'layout': 'episode_004', 'repeat': 1,
        'reset_seed': 100, 'scene': {'wrist_damping_scale': 1.}, 'success': True, 'audit_passed': True,
        'coverage': {'transport_motion_comparable': True, 'active_motion_scored': True,
                     'lifted_pinch': True, 'entered_box': True},
        'active_jerk_p99_rad_s3': 100., 'active_jerk_max_rad_s3': 200., 'all_frame_jerk_max_rad_s3': 210.,
        'physical_increment_max_rad': .01, 'pause_ge_0_5s_max_seconds': .5,
        'pause_ge_0_16s_seconds': 1., 'duration_seconds': 20.})
    if profile == 'rtc10':
        values['active_jerk_p99_rad_s3'] = 89.
    return values


class RtcPairAnalysisTests(unittest.TestCase):
    def summaries(self, *, change=None):
        base = [metric_row('rtc20', i) for i in range(18)]
        candidate = [metric_row('rtc10', i+18) for i in range(18)]
        if change:
            change(base, candidate)
        return {'rtc20': analysis.trial_summary(base), 'rtc10': analysis.trial_summary(candidate)}

    def test_requires_complete_exact_counterbalanced_36_cases(self):
        plan, ledger = plan_and_results()
        self.assertEqual(len(analysis.validate_cases(plan, ledger)), 36)
        ledger['complete'] = False
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            analysis.validate_cases(plan, ledger)
        ledger['complete'] = True
        ledger['results'][0]['reset_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'case order or reset seed'):
            analysis.validate_cases(plan, ledger)

    def test_duplicate_and_missing_cases_are_not_silently_dropped(self):
        plan, ledger = plan_and_results()
        plan['cases'][1] = copy.deepcopy(plan['cases'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            analysis.validate_cases(plan, ledger)

    def test_frozen_gate_requires_every_metric_and_keeps_success_secondary(self):
        profiles = self.summaries(change=lambda base, candidate: [row.update(success=False) for row in candidate])
        result = analysis.assess_selection(profiles, True)
        self.assertTrue(result['provisional_numeric_candidate'])
        self.assertFalse(result['automatic_config_change'])
        self.assertTrue(result['human_video_review_required'])
        self.assertEqual(result['success_secondary']['rtc10_successes'], 0)

    def test_active_and_all_frame_tail_peaks_each_block_promotion(self):
        for key, value in [('active_jerk_max_rad_s3', 221.), ('all_frame_jerk_max_rad_s3', 232.)]:
            with self.subTest(key=key):
                profiles = self.summaries(change=lambda base, candidate: candidate[0].update({key: value}))
                result = analysis.assess_selection(profiles, True)
                self.assertFalse(result['provisional_numeric_candidate'])
                self.assertEqual(result['status'], 'retain_rtc20')

    def test_dwell_or_coverage_regression_blocks_lower_typical_jerk(self):
        for key, value in [('pause_ge_0_5s_max_seconds', .61), ('pause_ge_0_16s_seconds', 1.11)]:
            with self.subTest(key=key):
                profiles = self.summaries(change=lambda base, candidate: [row.update({key: value}) for row in candidate])
                self.assertFalse(analysis.assess_selection(profiles, True)['provisional_numeric_candidate'])
        profiles = self.summaries(change=lambda base, candidate: candidate[0]['coverage'].update(transport_motion_comparable=False))
        self.assertFalse(analysis.assess_selection(profiles, True)['provisional_numeric_candidate'])

    def test_stationary_failure_or_missing_values_are_insufficient_not_zero(self):
        def make_failure(base, candidate):
            candidate[0].update(active_jerk_p99_rad_s3=None, active_jerk_max_rad_s3=None, success=False)
            candidate[0]['coverage'].update(transport_motion_comparable=False, active_motion_scored=False,
                                           lifted_pinch=False, entered_box=False)
        profiles = self.summaries(change=make_failure)
        result = analysis.assess_selection(profiles, True)
        self.assertEqual(profiles['rtc10']['all_trials']['n'], 18)
        self.assertEqual(profiles['rtc10']['all_trials']['metrics']['active_jerk_p99_rad_s3']['n'], 17)
        self.assertEqual(profiles['rtc10']['transport_covered_only']['n'], 17)
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertFalse(result['provisional_numeric_candidate'])

    def test_audit_failure_blocks_numeric_candidate(self):
        self.assertEqual(analysis.assess_selection(self.summaries(), False)['status'], 'insufficient_evidence')

    def test_matched_reset_differences_are_explicitly_unpaired(self):
        plan, ledger = plan_and_results()
        rows = [{**metric_row(case['profile']), **case} for case in ledger['results']]
        pairs = analysis.matched_resets(rows)
        self.assertEqual(len(pairs), 18)
        self.assertTrue(all(pair['matching_reset_seed'] and pair['same_scene'] for pair in pairs))
        self.assertTrue(all(pair['noise_paired'] is False for pair in pairs))
        self.assertEqual(pairs[0]['rtc10_minus_rtc20']['active_jerk_p99_rad_s3'], -11.)

    def test_prefix_response_parameters_hash_and_actual_targets_checked(self):
        prefix = np.zeros((1, 100, 14), np.float32)
        data = {'rtc_prefixes': prefix, 'rtc_request_steps': np.array([0]), 'rtc_valid_steps': np.array([100]),
                'action': np.zeros((5, 14), np.float32)}
        request = {'valid_steps': 100, 'delay_steps': 2, 'blend_steps': 10, 'guidance_max': .5,
                   'prefix_sha256': analysis.hashlib.sha256(prefix[0].tobytes()).hexdigest(),
                   'planned_vs_executed_prefix_max_error': 0.}
        declared = {'schema': 'pi0-rtc-v1', 'guidance': 'clean_estimate_vjp', 'native_dimensions': 14,
                    'padded_dimensions_guided': False, 'valid_steps': 100, 'delay_steps': 2,
                    'constrained_steps': 12, 'guidance_max': .5}
        latency = [{'step': 0, 'received_at_step': 2, 'rtc_request': request, 'rtc': declared, 'chunk_shape': [100, 14]}]
        self.assertTrue(analysis.reconstructed_prefix(data, latency, 10)['passed'])
        declared['constrained_steps'] = 22
        self.assertFalse(analysis.reconstructed_prefix(data, latency, 10)['passed'])
        declared['constrained_steps'] = 12
        data['action'][0, 0] = .01
        self.assertFalse(analysis.reconstructed_prefix(data, latency, 10)['passed'])

    def test_original_and_short_pause_definitions_remain_separate(self):
        q = np.zeros((30, 14), np.float32)
        tasks = [{} for _ in q]
        old = analysis.motion_quality(q, q, tasks)
        extra = analysis.extra_metrics({'state': q, 'action': q}, tasks)
        self.assertEqual(old['thresholds']['pause_min_seconds'], .5)
        self.assertEqual(extra['short_low_motion_min_seconds'], .16)
        self.assertIsNone(old['active_frame_max_jerk_rad_s3'])

    def test_new_outputs_are_published_without_replacing_existing_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'aggregate.json'
            analysis.atomic_new(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                analysis.atomic_new(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})

    def test_full_36_trial_reader_and_audit_in_temporary_fixture(self):
        """Use the real NPZ/JSONL summarizer, with no policy or environment."""
        original_root = analysis.ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            out, dataset, raw = root/'outputs/fixture', root/'dataset', root/'raw'
            out.mkdir(parents=True)
            sources = {}
            for name in analysis.SOURCE9+('motion_quality.py',):
                path = root/'src/pi0_repro'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_root/'src/pi0_repro'/name, path)
                sources[name] = analysis.sha(path)
            profile_files = {'rtc20': 'banana_v6_bridge20_pi0.toml', 'rtc10': 'banana_v6_rtc10_pi0.toml'}
            for filename in profile_files.values():
                path = root/'configs'/filename
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_root/'configs'/filename, path)
            mapping = dataset/'local/pi0_banana_validation/source_episodes.json'
            mapping.parent.mkdir(parents=True)
            mapping.write_text(json.dumps({'episodes': [{'source_episode': layout, 'split': 'validation'} for layout in analysis.LAYOUTS]}))
            (dataset/'dataset_manifest.json').write_text(json.dumps({'source_path': str(raw)}))
            scene = asdict(analysis.SceneConfig(controller_profile='position_pd'))
            for i, layout in enumerate(analysis.LAYOUTS):
                directory = raw/layout
                directory.mkdir(parents=True)
                (directory/'layout.json').write_text(json.dumps({'scene': scene, 'reset_seed': i+100}))
                (out/(layout+'.toml')).write_text('[scene]\n'+'\n'.join(f'{k}={json.dumps(v)}' for k, v in scene.items() if v is not None))
            (out/'runner_source.py').write_text('SERVICE_CODE = "fixture checker"\n')
            (out/'service_check_source.py').write_text('fixture checker')
            (out/'screen_helper_source.py').write_text('fixture helper')
            for directory, profile, count in [('preflight', 'rtc20', 6)]+[
                (f'warmup_r{repeat}_{profile}', profile, repeat+1) for repeat in (1, 2, 3) for profile in analysis.PROFILES]:
                path = out/directory
                path.mkdir()
                (path/'requests.json').write_text(json.dumps({'profile': profile, 'requests': [{'round_trip_ms': 100}]*count}))
            plan, ledger = plan_and_results()
            plan.update(profiles=profile_files, source_sha256=sources, metric_sha256=sources['motion_quality.py'],
                selection_gate=analysis.SELECTION_GATE, repeat_warmup_counts=[2, 3, 4],
                profile_sha256={p: analysis.sha(root/'configs'/f) for p, f in profile_files.items()},
                script_sha256=analysis.sha(out/'runner_source.py'), helper_sha256=analysis.sha(out/'screen_helper_source.py'),
                validation_mapping_sha256=analysis.sha(mapping))
            metadata = {'model_family': 'pi0', 'policy_config': 'pi0_banana_lora', 'action_dim': 14, 'action_horizon': 100,
                'action_semantics': 'absolute_aloha_joint_targets', 'openpi_commit': analysis.OPENPI_COMMIT,
                'denoise_steps': 10, 'checkpoint': '/fixture/'+analysis.CHECKPOINT_SUFFIX, 'norm_stats_sha256': analysis.NORM_SHA,
                'rtc': {'schema': 'pi0-rtc-v1'}, 'transport_proxy': {'kind': 'lossless-png'}}
            endpoint = {'metadata': metadata, 'metadata_verified': True,
                        'service_identity': {'pid': 123, 'starttime_ticks': 456, 'checkpoint': metadata['checkpoint']}}
            ledger.update(endpoint_before=endpoint, endpoint_after=copy.deepcopy(endpoint))
            for row in ledger['results']:
                directory = root/f"run_{row['index']}"
                directory.mkdir()
                config = asdict(analysis.load_config(root/'configs'/profile_files[row['profile']]))
                scene_file = out/(row['layout']+'.toml')
                config['simulation'].update(scene_config=str(scene_file), display=False, seed=row['reset_seed'])
                config['connection']['port'] = 18031
                (directory/'run.json').write_text(json.dumps({'config': config, 'openpi_commit': analysis.OPENPI_COMMIT,
                    'source_code_sha256': {name: sources[name] for name in analysis.SOURCE9}}))
                (directory/'server.json').write_text(json.dumps(metadata))
                (directory/'scene.json').write_text(json.dumps({'scene': scene}))
                (directory/'scene.xml').write_text('fixture XML\n')
                (directory/'episode_000.mp4').write_bytes(b'video fixture')
                actions = np.zeros((6, 14), np.float32)
                states = actions.copy()
                states[:, 0] = np.arange(6)*.001
                prefix = np.zeros((1, 100, 14), np.float32)
                data = dict(state=states, action=actions, requested_action=actions, submitted_action=actions, blended_action=actions,
                    predicted_chunks=np.zeros((1, 100, 14), np.float32), prediction_steps=np.array([0]),
                    action_prediction_index=np.zeros(6, int), action_prediction_offset=np.arange(6),
                    rtc_prefixes=prefix, rtc_request_steps=np.array([0]), rtc_valid_steps=np.array([100]))
                np.savez(directory/'episode_000.npz', **data)
                tasks = [{'near_grasp_region': True, 'ever_grasped': True, 'inside_box': True, 'is_success': True} for _ in range(6)]
                (directory/'episode_000_task.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
                request = {'valid_steps': 100, 'delay_steps': 2, 'blend_steps': config['simulation']['rtc_blend_steps'],
                    'guidance_max': .5, 'prefix_sha256': analysis.hashlib.sha256(prefix[0].tobytes()).hexdigest(),
                    'planned_vs_executed_prefix_max_error': 0.}
                response = {'schema': 'pi0-rtc-v1', 'guidance': 'clean_estimate_vjp', 'native_dimensions': 14,
                    'padded_dimensions_guided': False, 'valid_steps': 100, 'delay_steps': 2,
                    'constrained_steps': 2+request['blend_steps'], 'guidance_max': .5}
                timing = {'step': 0, 'received_at_step': 2, 'round_trip_ms': 300., 'rtc_request': request, 'rtc': response,
                    'chunk_shape': [100, 14], 'server_timing': {'infer_ms': 150.}, 'transport_timing': {'pixel_digest_verified': True}}
                (directory/'episode_000_latency.jsonl').write_text(json.dumps(timing)+'\n')
                (directory/'episode_000_control.jsonl').write_text(''.join(json.dumps({'wall_elapsed_seconds': i*.02,
                    'action_apply_wall_seconds': i*.02, 'inference_blocked_seconds': 0.})+'\n' for i in range(6)))
                (directory/'summary.json').write_text(json.dumps({'episodes': [{'episode': 0, 'steps': 6, 'success': True, 'seed': row['reset_seed'],
                    'execution': {'inference_underruns': 0}}]}))
                row.update(run_directory=str(directory), scene_sha256=analysis.sha(scene_file), prefix_error=0.,
                    motion_quality=analysis.motion_quality(states, actions, tasks), extra=analysis.extra_metrics(data, tasks),
                    diagnostics=analysis.summarize(directory), video_sha256=analysis.sha(directory/'episode_000.mp4'))
            (out/'plan.json').write_text(json.dumps(plan))
            (out/'comparison.json').write_text(json.dumps(ledger))
            with mock.patch.object(analysis, 'ROOT', root), mock.patch.object(analysis, 'DATASET', dataset), \
                 mock.patch.object(analysis, 'build_scene_xml', return_value='fixture XML'):
                result, audit = analysis.analyze(out)
            self.assertTrue(audit['passed'], audit['failed_checks'])
            self.assertEqual(result['attempt_count'], 36)
            self.assertEqual(result['profiles']['rtc20']['all_trials']['n'], 18)
            self.assertEqual(result['matched_reset_difference_summary']['both_transport_covered_only']['n'], 18)
            self.assertFalse((out/'aggregate.json').exists())


if __name__ == '__main__':
    unittest.main()
