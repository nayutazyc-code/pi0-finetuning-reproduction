"""Recompute/audit every outcome in a frozen v6 runtime diagnostic."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from profile_v6_runtime import inspect_run,distribution
from analyze_rtc_pair import equal,metadata_valid


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    assert not (out/'audit.json').exists() and not (out/'aggregate.json').exists()
    plan=read(out/'plan.json');ledger=read(out/'comparison.json');checks=[]
    def check(name,result):checks.append(dict(name=name,passed=bool(result)))
    check('complete',ledger['complete']);check('ledger_plan',ledger['plan']==plan)
    check('case_count',len(ledger['results'])==plan['attempts'])
    check('runner_frozen',sha(out/'runner_source.py')==plan['script_sha256'])
    check('same_service',ledger['endpoint_before']['service_identity']==ledger['endpoint_after']['service_identity'])
    check('model_metadata',metadata_valid(ledger['endpoint_before']['metadata']))
    for n,digest in plan['source_sha256'].items():check('source:'+n,sha(out/'frozen_source'/n)==digest)
    groups=defaultdict(list);all_intervals=defaultdict(list);sampled_resets=defaultdict(list);trials=[]
    for planned,row in zip(plan['cases'],ledger['results'],strict=True):
        index=row['index'];run=Path(row['run_directory']);meta=read(run/'run.json')
        check(f'{index}:case',all(row[k]==v for k,v in planned.items()))
        expected=json.loads(json.dumps(plan['profiles'][row['profile']]))
        expected['simulation']['seed']=row['seed']
        check(f'{index}:config',equal(meta['config'],expected))
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for name,digest in meta[field].items():check(f'{index}:source:{name}',plan['source_sha256'][name]==digest)
        check(f'{index}:server',equal(read(run/'server.json'),ledger['endpoint_before']['metadata']))
        actual=inspect_run(run)
        check(f'{index}:recomputed',equal(actual,row['diagnostics']))
        check(f'{index}:prefix',actual['prefix_reconstruction']['passed'])
        check(f'{index}:timestamp',actual['summary']['max_action_timestamp_error_steps']==0)
        check(f'{index}:prediction',actual['summary']['raw_prediction_max_error']==0)
        check(f'{index}:no_operational_error',not row.get('operational_error') and actual['summary']['result']['error'] is None)
        controls=rows(run/'episode_000_control.jsonl');times=np.array([r['action_apply_wall_seconds'] for r in controls])
        check(f'{index}:monotonic',bool((np.diff(times)>0).all()))
        check(f'{index}:row_count',len(controls)==actual['summary']['result']['steps'])
        sampled_resets[row['seed']].append(read(run/'episode_000_scene.json'))
        s=actual['summary']['result'];settings=meta['config']['simulation']
        check(f'{index}:original_video_presence',(run/'episode_000.mp4').exists()==settings['save_video'])
        check(f'{index}:composite_video_presence',(run/'episode_000_multiview.mp4').exists()==settings['record_multiview'])
        stages={n:distribution([r.get(n,0)*1000 for r in controls[1:] if r.get(n,0)>0])
                for n in ('camera_seconds','environment_step_seconds','display_seconds','video_submit_seconds',
                          'presentation_exchange_seconds','pacing_wait_seconds','target_limiter_seconds','pre_apply_prepare_seconds','post_apply_seconds')}
        # Baseline old logs lack loop envelopes: unavailable, not zero.
        envelopes=[]
        if all('loop_finish_wall_seconds' in r for r in controls):
            for i in range(1,len(controls)):
                prev,current=controls[i-1],controls[i]
                parts=[prev['loop_finish_wall_seconds']-prev['action_apply_wall_seconds'],
                       current['loop_begin_wall_seconds']-prev['loop_finish_wall_seconds'],
                       current['action_apply_wall_seconds']-current['loop_begin_wall_seconds']]
                envelopes.append(dict(step=i,previous_post_apply_ms=parts[0]*1000,
                    between_loops_ms=parts[1]*1000,current_pre_apply_ms=parts[2]*1000,
                    reconstructed_gap_ms=sum(parts)*1000))
            check(f'{index}:envelope_reconstruction',all(abs(e['reconstructed_gap_ms']-(times[e['step']]-times[e['step']-1])*1000)<1e-6 for e in envelopes))
        trial=dict(index=index,profile=row['profile'],seed=row['seed'],run_directory=str(run),
            success=s['success'],steps=s['steps'],simulated_seconds=s['execution']['simulated_seconds'],
            inference_underruns=s['execution']['inference_underruns'],wait_seconds=s['execution']['underrun_wait_seconds'],
            control_wall_seconds=s['execution']['control_wall_seconds'],action_apply_ms=actual['action_apply_ms'],
            gaps_over_50ms=actual['gaps_over_50ms'],gaps_over_100ms=actual['gaps_over_100ms'],
            stage_nonzero_call_ms=stages,
            camera_ms_per_render=s['execution']['stage_totals']['camera_seconds']*1000/s['execution']['camera_renders'],
            progress=s.get('task_progress'),motion=actual['motion_quality'],
            longest_gap_envelopes=sorted(envelopes,key=lambda e:e['reconstructed_gap_ms'],reverse=True)[:5])
        groups[row['profile']].append(trial);trials.append(trial)
        all_intervals[row['profile']].extend(np.diff(times)*1000)
    for seed,resets in sampled_resets.items():check(f'reset:{seed}',all(equal(r,resets[0]) for r in resets))
    profiles={name:dict(attempts=len(rs),successes=sum(r['success'] for r in rs),
        pooled_action_apply_ms=distribution(all_intervals[name]),
        median_trial_p99_ms=float(np.median([r['action_apply_ms']['p99'] for r in rs])),
        median_camera_ms_per_render=float(np.median([r['camera_ms_per_render'] for r in rs])),
        total_gaps_over_50ms=sum(r['gaps_over_50ms'] for r in rs),
        total_gaps_over_100ms=sum(r['gaps_over_100ms'] for r in rs),
        total_underruns=sum(r['inference_underruns'] for r in rs)) for name,rs in groups.items()}
    failed=[c for c in checks if not c['passed']]
    audit=dict(schema='v6-runtime-cost-audit-v1',passed=not failed,checks=checks,failed_checks=failed,
               comparison_sha256=sha(out/'comparison.json'),analyzer_sha256=sha(Path(__file__)))
    aggregate=dict(schema='v6-runtime-cost-summary-v1',complete=ledger['complete'],audit_passed=not failed,
        profiles=profiles,trials=trials,
        scope='Diagnostic selected seeds, one stochastic policy attempt per workload/seed. All outcomes retained, no causal population inference. Pooled percentiles weight longer episodes; per-trial medians are also reported.',
        timing_scope='Wall-clock intervals differ from simulation-time motion. Conditional nonzero stage-call statistics exclude non-call steps; nested timing fields cannot be added. Startup/final waits are outside running control.',
        decision='No automatic profile or checkpoint promotion. Camera-lifecycle speed alone does not establish less perceived jitter or fewer physical contact peaks.')
    for name,data in [('audit.json',audit),('aggregate.json',aggregate)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(audit_passed=not failed,checks=len(checks),failed_checks=failed,profiles=profiles),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
