"""Audit every fixed-layout attempt without historical pooling or promotion."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from repeat_v6_control_layout import OUT,SUFFIX,SEED
from evaluate_v6_progress import read,sha,write
from analyze_v6_progress_evaluation import summarize
from analyze_rtc_pair import equal
from profile_v6_runtime import inspect_run


def main():
    plan=read(OUT/'plan.json');ledger=read(OUT/'comparison.json')
    assert not (OUT/'aggregate.json').exists() and not (OUT/'audit.json').exists()
    assert ledger['complete'] and len(ledger['results'])==plan['attempts']==10
    checks=[]
    def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
    check('plan',ledger['plan']==plan);check('frozen_runner',sha(OUT/'runner_source.py')==plan['script_sha256'])
    check('restoration',ledger['restoration']['verified'] and ledger['restoration']['metadata']['checkpoint'].endswith('/banana_v6_long100_accum8000/7999'))
    check('route',ledger['route_before']==ledger['route_after'] and not ledger['route_after'].startswith(('utun','tun','tap')))
    check('scene_file',sha(OUT/'episode_004.toml')==plan['scene_sha256'])
    prior=ROOT/'outputs/v6_progress_20261028/evaluation_network_retest/comparison.json'
    check('prior_binding',sha(prior)==plan['prior_comparison_sha256'])
    previous=next(r for r in read(prior)['results'] if r['model']=='control' and r['layout']=='episode_004')
    old_config=read(Path(previous['run_directory'])/'run.json')['config']
    old_scene=read(Path(previous['run_directory'])/'episode_000_scene.json')
    pinned_norm=plan['config']['model']['expected_norm_stats_sha256']
    for n,d in plan['source_sha256'].items():check('frozen_source:'+n,sha(OUT/'frozen_source'/n)==d)
    trials=[]
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        index=row['index'];run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json')
        check(f'{index}:case',all(row[k]==v for k,v in case.items()))
        check(f'{index}:no_error',not row.get('operational_error'))
        check(f'{index}:checkpoint',server['checkpoint'].endswith('/'+SUFFIX) and server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'{index}:norm',server['norm_stats_sha256']==pinned_norm)
        check(f'{index}:config',equal(meta['config'],plan['config']))
        check(f'{index}:same_reset',equal(read(run/'episode_000_scene.json'),old_scene))
        same=[json.loads(json.dumps(c)) for c in [meta['config'],old_config]]
        for c in same:c['connection'].pop('port');c['simulation'].pop('scene_config')
        check(f'{index}:prior_config',equal(*same))
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for n,d in meta[field].items():check(f'{index}:source:{n}',plan['source_sha256'][n]==d)
        details=inspect_run(run);check(f'{index}:recomputed',equal(details,row['diagnostics']))
        check(f'{index}:prefix',details['prefix_reconstruction']['passed'])
        check(f'{index}:prediction_timestamp',details['summary']['raw_prediction_max_error']==0 and details['summary']['max_action_timestamp_error_steps']==0)
        result=details['summary']['result'];check(f'{index}:result',result['seed']==SEED and result['error'] is None and not result['aborted_by_window_close'])
        progress=result['task_progress'];coverage=details['motion_quality']['coverage']
        failure=('success' if result['success'] else 'no_lift' if not coverage['lifted_pinch'] else 'no_box_entry' if not coverage['entered_box'] else 'placement_not_confirmed')
        trials.append(dict(index=index,repeat=row['repeat'],model='control',layout='episode_004',run_directory=str(run),success=result['success'],
            simulated_seconds=result['execution']['simulated_seconds'],inference_underruns=result['execution']['inference_underruns'],
            inference_wait_seconds=result['execution']['underrun_wait_seconds'],action_apply_ms=details['action_apply_ms'],quality=details['motion_quality'],
            task_progress=progress,failure_class=failure,video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4')))
    failed=[c for c in checks if not c['passed']]
    aggregate=dict(schema='v6-control-fixed-layout-repeat-summary-v1',complete=ledger['complete'],audit_passed=not failed,
        summary=summarize(trials),trials=trials,
        scope='Ten fresh attempts on one deliberately selected validation layout, identical simulator reset. Policy noise unpaired, no historical pooling. Empirical outcomes do not establish overall task success probability or cross-model superiority. No training/default change.',
        decision='observation_only_no_automatic_promotion')
    audit=dict(passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(OUT/'comparison.json'),analyzer_sha256=sha(Path(__file__)))
    (OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    write(OUT/'aggregate.json',aggregate);write(OUT/'audit.json',audit)
    print(json.dumps(dict(audit_passed=not failed,checks=len(checks),summary=aggregate['summary'],
                         attempts=[dict(repeat=r['repeat'],success=r['success'],failure_class=r['failure_class'],seconds=r['simulated_seconds']) for r in trials]),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
