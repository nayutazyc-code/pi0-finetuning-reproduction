"""Audit all twelve deferred-video attempts, timing, coverage and saved frames."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_deferred_video import OUT, SUFFIX, SEED
from evaluate_v6_progress import read, sha, write
from analyze_rtc_pair import equal
from analyze_v6_progress_evaluation import summarize
from profile_v6_runtime import inspect_run, distribution
from trajectory_jump_metrics import trajectory_jumps
from build_three_video_comparison import video_info


def main():
    ledger=read(OUT/'comparison.json');plan=read(OUT/'plan.json')
    assert ledger['complete'] and len(ledger['results'])==plan['attempts']==12
    assert not (OUT/'aggregate.json').exists()
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('ledger_plan',ledger['plan']==plan)
    check('runner_sha',sha(OUT/'runner_source.py')==plan['script_sha256'])
    check('restored_v6',ledger['restoration']['verified'] and ledger['restoration']['metadata']['checkpoint'].endswith('/banana_v6_long100_accum8000/7999'))
    check('restored_norm',ledger['restoration']['metadata']['norm_stats_sha256']==plan['config']['model']['expected_norm_stats_sha256'])
    check('public_route',ledger['route_before']==ledger['route_after'] and not ledger['route_after'].startswith(('utun','tun','tap')))
    left=plan['config_by_profile']['baseline'];right=plan['config_by_profile']['deferred']
    check('only_config_variable',all(equal(left[k],right[k]) for k in ('model','connection')) and
          all(equal(v,right['simulation'][k]) for k,v in left['simulation'].items() if k not in ('scene_config','defer_video_render')))
    for name,digest in plan['source_sha256'].items():check('frozen:'+name,sha(OUT/'frozen_source'/name)==digest)
    groups={k:[] for k in ('baseline','deferred')};scenes=[];xmls=[]
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        i=row['index'];kind=row['profile'];run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json')
        check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        check(f'{i}:config',equal(meta['config'],plan['config_by_profile'][kind]))
        check(f'{i}:model',server['checkpoint'].endswith('/'+SUFFIX) and server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'{i}:norm',server['norm_stats_sha256']==plan['config']['model']['expected_norm_stats_sha256'])
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for name,digest in meta[field].items():check(f'{i}:source:{name}',plan['source_sha256'][name]==digest)
        check(f'{i}:route',row['route_interface']==ledger['route_before'])
        details=inspect_run(run);check(f'{i}:recomputed',equal(details,row['diagnostics']))
        check(f'{i}:prefix_timestamp',details['prefix_reconstruction']['passed'] and details['summary']['raw_prediction_max_error']==0 and details['summary']['max_action_timestamp_error_steps']==0)
        result=details['summary']['result'];ex=result['execution']
        check(f'{i}:no_error',not row.get('operational_error') and not result['error'] and not result['aborted_by_window_close'] and result['seed']==SEED)
        check(f'{i}:gain_unchanged',read(run/'episode_000_scene.json')['scene']['right_wrist_angle_kp_scale']==1.)
        scenes.append(read(run/'episode_000_scene.json'));xmls.append((run/'scene.xml').read_text())
        controls=[json.loads(s) for s in (run/'episode_000_control.jsonl').read_text().splitlines()]
        n=result['steps'];steps=list(range(0,n,2))+[n]
        info=video_info(run/'episode_000.mp4')
        check(f'{i}:video_count_rate',info['fps']==25 and info['width']==640 and info['height']==480 and info['frame_count']==len(steps))
        receipt=ex['deferred_top_video']
        if kind=='deferred':
            check(f'{i}:deferred_complete',receipt['complete'] and receipt['frames']==len(steps) and receipt['frame_control_steps']==steps)
            check(f'{i}:receipt',equal(receipt,read(run/'episode_000_video_receipt.json')))
            check(f'{i}:snapshot_sha',sha(run/receipt['state_file'])==receipt['state_file_sha256'])
            check(f'{i}:snapshot_steps',len(receipt['pixel_sha256'])==len(steps))
            check(f'{i}:snapshot_xml',__import__('hashlib').sha256(xmls[-1][:-1].encode()).hexdigest()==receipt['xml_sha256'])
            with np.load(run/receipt['state_file'],allow_pickle=False) as data:
                check(f'{i}:snapshot_data',np.isfinite(data['integration_state']).all() and data['control_step'].tolist()==steps)
            # Only fresh request observations render the main-process camera.
            camera_steps=[c['step'] for c in controls if c['camera_seconds']>0]
            latency=[json.loads(s) for s in (run/'episode_000_latency.jsonl').read_text().splitlines()]
            request_steps=[r['step'] for r in latency]
            # A canceled outstanding request can exist after the final action.
            check(f'{i}:live_camera_only_requests',set(request_steps).issubset(camera_steps) and abs(len(camera_steps)-len(request_steps))<=1)
        else:check(f'{i}:legacy_video_path',receipt is None and not (run/'episode_000_video_states.npz').exists())
        with np.load(run/'episode_000.npz',allow_pickle=False) as d:jumps=trajectory_jumps(d)
        intervals=max(0,len(controls)-1)
        groups[kind].append(dict(index=i,repeat=row['repeat'],profile=kind,model='control',layout='episode_004',run_directory=str(run),
            success=result['success'],simulated_seconds=ex['simulated_seconds'],quality=details['motion_quality'],jumps=jumps,task_progress=result['task_progress'],
            inference_underruns=ex['inference_underruns'],inference_wait_seconds=ex['underrun_wait_seconds'],
            action_apply_ms=details['action_apply_ms'],intervals=intervals,gaps_over50=details['gaps_over_50ms'],gaps_over100=details['gaps_over_100ms'],
            camera_renders=ex['camera_renders'],camera_seconds=ex['stage_totals']['camera_seconds'],
            camera_thread_cpu_seconds=ex['stage_totals']['camera_thread_cpu_seconds'],deferred_video=receipt,
            camera_per_call_ms=distribution([c['camera_seconds']*1000 for c in controls if c['camera_seconds']>0]),
            longest_gaps=details['longest_gaps'],video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4'),
            failure_class='success' if result['success'] else 'no_lift' if not details['motion_quality']['coverage']['lifted_pinch'] else 'no_box_entry' if not details['motion_quality']['coverage']['entered_box'] else 'placement_not_confirmed'))
    check('same_scene_xml',len(set(xmls))==1)
    check('same_seed_scene',all(equal(s,scenes[0]) for s in scenes))
    summaries={k:summarize(v) for k,v in groups.items()}
    for k,rows in groups.items():
        total=sum(r['intervals'] for r in rows)
        summaries[k].update(total_intervals=total,gaps_over50=sum(r['gaps_over50'] for r in rows),
            gaps_over100=sum(r['gaps_over100'] for r in rows),
            gaps_over50_per1000=1000*sum(r['gaps_over50'] for r in rows)/total,
            median_camera_seconds=float(np.median([r['camera_seconds'] for r in rows])),
            median_camera_renders=float(np.median([r['camera_renders'] for r in rows])),
            median_camera_per_call_ms=float(np.median([r['camera_per_call_ms']['median'] for r in rows])),
            total_camera_seconds=sum(r['camera_seconds'] for r in rows),
            median_post_control_video_seconds=None if k=='baseline' else float(np.median([r['deferred_video']['post_control_seconds'] for r in rows])))
    base,candidate=summaries['baseline'],summaries['deferred'];tests=[];gate=plan['gate']
    for key,limit in [('median_trial_apply_p99_ms',gate['median_trial_apply_p99_ratio_max']),('worst_apply_interval_ms',gate['worst_apply_interval_ratio_max']),('gaps_over50_per1000',gate['gap_over50_rate_ratio_max']),('covered_median_active_jerk_p99',gate['covered_jerk_p99_ratio_max']),('covered_median_simulated_seconds',gate['covered_duration_ratio_max'])]:
        a,b=base[key],candidate[key]
        tests.append(dict(metric=key,baseline=a,candidate=b,ratio=None if a is None or b is None or a==0 else b/a,passed=a is not None and b is not None and b<=a*limit))
    tests.append(dict(metric='coverage',baseline=base['transport_covered'],candidate=candidate['transport_covered'],passed=candidate['transport_covered']>=base['transport_covered']>0))
    tests.append(dict(metric='candidate_underruns',candidate=candidate['total_running_inference_underruns'],passed=candidate['total_running_inference_underruns']==0))
    failed=[c for c in checks if not c['passed']];passed=not failed and all(t['passed'] for t in tests)
    report=dict(schema='deferred-video-ab-summary-v1',complete=True,audit_passed=not failed,groups={k:dict(summary=summaries[k],trials=v) for k,v in groups.items()},
        gate_checks=tests,decision='promising_requires_replication' if passed else 'retain_existing_default',
        scope='Fresh6+6counterbalanced trials on one deliberately selected layout and continued checkpoint; unpaired policy noise. Headless only, all outcomes retained. Deferred video preserves simulation-time frames, not wall-clock interruptions. No retraining/default promotion.')
    (OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'aggregate.json',report);write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(OUT/'comparison.json')))
    print(json.dumps(dict(checks=len(checks),failed_checks=failed,summaries=summaries,gate_checks=tests,decision=report['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
