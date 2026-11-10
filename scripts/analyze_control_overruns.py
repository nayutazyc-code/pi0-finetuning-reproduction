"""All-outcome clock/phase/GC audit of the fresh conditional overrun pilot."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_control_overruns import OUT,SUFFIX,SEED
from evaluate_v6_progress import read,sha,write
from analyze_rtc_pair import equal
from analyze_v6_progress_evaluation import summarize
from profile_v6_runtime import inspect_run,distribution
from trajectory_jump_metrics import trajectory_jumps
from build_three_video_comparison import video_info
from pi0_repro.pacing import next_apply_deadline


def main():
    ledger=read(OUT/'comparison.json');plan=read(OUT/'plan.json')
    assert ledger['complete'] and len(ledger['results'])==plan['attempts']==12
    assert not (OUT/'aggregate.json').exists()
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('plan',ledger['plan']==plan);check('runner',sha(OUT/'runner_source.py')==plan['script_sha256'])
    native=ROOT/'outputs/control_overrun_probe_20261029/summary.json'
    amendment=ROOT/'outputs/control_wakeup_diagnosis_20261029/native_screen_amendment.json'
    check('native_bound',sha(native)==plan['native_probe_sha256'] and not read(native)['passed'] and not plan['native_original_gate_passed'])
    check('amendment_bound',sha(amendment)==plan['native_scope_amendment_sha256'])
    check('restored_v6',ledger['restoration']['verified'] and ledger['restoration']['metadata']['checkpoint'].endswith('/banana_v6_long100_accum8000/7999'))
    check('restored_norm',ledger['restoration']['metadata']['norm_stats_sha256']==plan['config']['model']['expected_norm_stats_sha256'])
    check('route',ledger['route_before']==ledger['route_after'] and not ledger['route_after'].startswith(('utun','tun','tap')))
    a,b=plan['config_by_profile']['baseline'],plan['config_by_profile']['recover']
    check('one_variable',all(equal(a[k],b[k]) for k in ('model','connection')) and all(equal(v,b['simulation'][k]) for k,v in a['simulation'].items() if k not in ('scene_config','recover_control_overruns')))
    for name,digest in plan['source_sha256'].items():check('source:'+name,sha(OUT/'frozen_source'/name)==digest)
    groups={k:[] for k in ('baseline','recover')};scenes=[];xmls=[];phase_diagnosis=[]
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        i=row['index'];kind=row['profile'];run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json')
        check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        check(f'{i}:config',equal(meta['config'],plan['config_by_profile'][kind]))
        check(f'{i}:pi0',server['checkpoint'].endswith('/'+SUFFIX) and server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'{i}:norm',server['norm_stats_sha256']==plan['config']['model']['expected_norm_stats_sha256'])
        check(f'{i}:route',row['route_interface']==ledger['route_before'])
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for name,digest in meta[field].items():check(f'{i}:source:{name}',plan['source_sha256'][name]==digest)
        details=inspect_run(run);check(f'{i}:recomputed',equal(details,row['diagnostics']))
        check(f'{i}:prefix_time',details['prefix_reconstruction']['passed'] and details['summary']['raw_prediction_max_error']==0 and details['summary']['max_action_timestamp_error_steps']==0)
        result=details['summary']['result'];ex=result['execution'];n=result['steps']
        check(f'{i}:no_error',not row.get('operational_error') and not result['error'] and not result['aborted_by_window_close'] and result['seed']==SEED)
        scenes.append(read(run/'episode_000_scene.json'));xmls.append((run/'scene.xml').read_text())
        check(f'{i}:kp_unchanged',scenes[-1]['scene']['right_wrist_angle_kp_scale']==1.)
        s=meta['config']['simulation'];check(f'{i}:unchanged_runtime',not s['defer_video_render'] and not s['reuse_policy_camera'] and s['trace_control_timing'] and not s['display'] and s['rtc_blend_steps']==20 and s['chunk_blend_steps']==20)
        controls=[json.loads(x) for x in (run/'episode_000_control.jsonl').read_text().splitlines()]
        trace=read(run/'episode_000_runtime_trace.json');check(f'{i}:trace_schema',trace['schema']=='control-runtime-trace-v1')
        phase_names={'physics_step','forward','task_info','observation'}
        phase_keys=set(controls[0]['environment_phases'])
        check(f'{i}:phase_fields',phase_keys==phase_names and all(set(c['environment_phases'])==phase_names for c in controls))
        check(f'{i}:phase_spans',all(all(p['wall_seconds']>=0 and p['thread_cpu_seconds']>=0 and p['process_cpu_seconds']>=0 and abs(p['end_perf_counter']-p['start_perf_counter']-p['wall_seconds'])<1e-9 for p in c['environment_phases'].values()) and sum(p['wall_seconds'] for p in c['environment_phases'].values())<=c['environment_step_seconds']+1e-6 for c in controls))
        clock_error=[];overruns=[]
        for c in controls:
            applied,scheduled,finished=c['action_apply_wall_seconds'],c['scheduled_deadline_wall_seconds'],c['deadline_update_wall_seconds']
            expected=next_apply_deadline(scheduled,applied,finished,c['deadline_lateness_seconds'],recover_overrun=kind=='recover')
            clock_error.append(abs(expected-c['next_deadline_wall_seconds']))
            nominal=applied+.02 if c['deadline_lateness_seconds']>.002 else scheduled+.02
            if finished>=nominal:
                overruns.append(dict(step=c['step'],apply_to_update_seconds=finished-applied,
                    next_deadline_delay_after_finish=c['next_deadline_wall_seconds']-finished,
                    recovered_without_new_short_interval=c['next_deadline_wall_seconds']>=applied+.02-1e-8))
        check(f'{i}:clock_reconstruction',max(clock_error)<1e-8)
        if kind=='recover':check(f'{i}:no_new_catchup_burst',all(r['recovered_without_new_short_interval'] for r in overruns))
        info=video_info(run/'episode_000.mp4')
        check(f'{i}:video',info['fps']==25 and info['width']==640 and info['height']==480 and info['frame_count']==len(range(0,n,2))+1)
        check(f'{i}:no_deferred_video',ex['deferred_top_video'] is None)
        with np.load(run/'episode_000.npz',allow_pickle=False) as data:jumps=trajectory_jumps(data)
        gaps=np.diff([c['action_apply_wall_seconds'] for c in controls]);worst=int(np.argmax(gaps))+1
        env_worst=max(range(n),key=lambda k:controls[k]['environment_step_seconds'])
        cpus=[c['environment_step_process_cpu_seconds'] for c in controls]
        origin=trace['origin_perf_counter']
        events=[]
        for step in sorted(set((worst-1,worst,env_worst))):
            c=controls[step]
            lo=c['action_apply_wall_seconds'];hi=lo+c['environment_step_seconds']
            gc_overlap=[g for g in trace['gc_events'] if g['start_wall_seconds']<hi and g['end_wall_seconds']>lo]
            events.append(dict(step=step,control=c,gc_overlapping_environment=gc_overlap,
                phases_relative_to_origin={k:{**p,'start_wall_seconds':p['start_perf_counter']-origin,'end_wall_seconds':p['end_perf_counter']-origin} for k,p in c['environment_phases'].items()}))
        phase_diagnosis.append(dict(index=i,profile=kind,repeat=row['repeat'],worst_gap_step=worst,worst_gap_ms=float(gaps[worst-1]*1000),events=events,gc_events=trace['gc_events'],unfinished_gc_callbacks=trace['unfinished_gc_callbacks'],deadline_overruns=overruns))
        groups[kind].append(dict(index=i,repeat=row['repeat'],profile=kind,model='control',layout='episode_004',run_directory=str(run),
            success=result['success'],simulated_seconds=ex['simulated_seconds'],quality=details['motion_quality'],jumps=jumps,task_progress=result['task_progress'],
            inference_underruns=ex['inference_underruns'],inference_wait_seconds=ex['underrun_wait_seconds'],action_apply_ms=details['action_apply_ms'],
            intervals=len(gaps),gaps_over50=details['gaps_over_50ms'],gaps_over100=details['gaps_over_100ms'],
            oversleep_ms=distribution([c['pacing_oversleep_seconds']*1000 for c in controls[1:] if c['pacing_requested_seconds']>0]),
            environment_wall_ms=distribution([c['environment_step_seconds']*1000 for c in controls]),
            environment_cpu_ms=distribution([c['environment_step_thread_cpu_seconds']*1000 for c in controls]),
            phase_wall_ms={k:distribution([c['environment_phases'][k]['wall_seconds']*1000 for c in controls]) for k in phase_names},
            clock_reconstruction_max_error_seconds=max(clock_error),deadline_overruns=len(overruns),
            gc_collections=len(trace['gc_events']),gc_wall_seconds=sum(g['wall_seconds'] for g in trace['gc_events']),
            failure_class='success' if result['success'] else 'no_lift' if not details['motion_quality']['coverage']['lifted_pinch'] else 'no_box_entry' if not details['motion_quality']['coverage']['entered_box'] else 'placement_not_confirmed',
            video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4')))
    check('same_scene',all(equal(s,scenes[0]) for s in scenes));check('same_xml',len(set(xmls))==1)
    summaries={k:summarize(v) for k,v in groups.items()}
    for k,rows in groups.items():
        total=sum(r['intervals'] for r in rows)
        summaries[k].update(total_intervals=total,gaps_over50=sum(r['gaps_over50'] for r in rows),gaps_over100=sum(r['gaps_over100'] for r in rows),
            gaps_over50_per1000=1000*sum(r['gaps_over50'] for r in rows)/total,deadline_overruns=sum(r['deadline_overruns'] for r in rows),
            gc_collections=sum(r['gc_collections'] for r in rows),median_trial_oversleep_p99_ms=float(np.median([r['oversleep_ms']['p99'] for r in rows])))
    base,candidate=summaries['baseline'],summaries['recover'];gate=plan['gate'];tests=[]
    for key,limit in [('median_trial_apply_p99_ms',gate['median_trial_apply_p99_ratio_max']),('worst_apply_interval_ms',gate['worst_apply_interval_ratio_max']),('gaps_over50_per1000',gate['gap_over50_rate_ratio_max']),('covered_median_active_jerk_p99',gate['covered_jerk_p99_ratio_max']),('worst_active_jerk_peak',gate['worst_jerk_peak_ratio_max']),('covered_median_simulated_seconds',gate['covered_duration_ratio_max'])]:
        a,b=base[key],candidate[key];tests.append(dict(metric=key,baseline=a,candidate=b,ratio=None if a is None or b is None or a==0 else b/a,passed=a is not None and b is not None and b<=a*limit))
    tests.append(dict(metric='coverage',baseline=base['transport_covered'],candidate=candidate['transport_covered'],passed=candidate['transport_covered']>=base['transport_covered']>0))
    tests.append(dict(metric='candidate_underruns',candidate=candidate['total_running_inference_underruns'],passed=candidate['total_running_inference_underruns']==0))
    failed=[c for c in checks if not c['passed']];eligible=not failed and all(t['passed'] for t in tests)
    report=dict(schema='control-overrun-live-summary-v1',complete=True,audit_passed=not failed,groups={k:dict(summary=summaries[k],trials=v) for k,v in groups.items()},
        gate_checks=tests,decision='promising_requires_replication' if eligible else 'retain_existing_default',native_original_gate_passed=False,
        scope='Conditional fresh6+6AB/BA trials, one selected validation layout, same continued checkpoint/RTC20/kp200/original video. Common detailed diagnostics, unpaired noise, no injected sleeps. Native>=15ms screen remains failed, no historical pooling or default promotion.')
    (OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'aggregate.json',report);write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(OUT/'comparison.json')))
    write(OUT/'phase_diagnosis.json',dict(trials=phase_diagnosis,scope='Recorded overlapping stage/process/thread/GC envelopes. GC on another thread and wall-minus-threadCPU are not unique OS/GPU/GIL cause diagnoses. No injected waits in these trials.'))
    print(json.dumps(dict(checks=len(checks),failed_checks=failed,summaries=summaries,gate_checks=tests,decision=report['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
