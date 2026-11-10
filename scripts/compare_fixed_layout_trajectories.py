"""Observed jump comparison: ten earlier control runs and ten fresh v6 runs."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_v6_progress import read,sha,write
from analyze_rtc_pair import equal
from profile_v6_runtime import inspect_run
from trajectory_jump_metrics import trajectory_jumps

OUT=ROOT/'outputs/v6_fixed_layout_jump_comparison_20261028'
COHORTS=dict(v6=ROOT/'outputs/v6_baseline_repeats_20261028',control=ROOT/'outputs/v6_control_repeats_20261028')


def median(values):
    values=[v for v in values if v is not None]
    return None if not values else float(np.median(values))


def summarize(rows):
    def pick(fn):return [fn(r) for r in rows]
    optional=lambda r:(r['jumps']['same_time_raw_disagreement_rad'] or {}).get('p99')
    return dict(n=len(rows),successes=sum(r['success'] for r in rows),covered=sum(r['quality']['coverage']['transport_motion_comparable'] for r in rows),
        median_active_jerk_p99=median(pick(lambda r:(r['quality']['active_frame_max_jerk_rad_s3'] or {}).get('p99'))),
        worst_active_jerk_peak=max([r['quality']['active_frame_max_jerk_rad_s3']['max'] for r in rows if r['quality']['active_frame_max_jerk_rad_s3'] is not None],default=None),
        median_physical_increment_p99_rad=median(pick(lambda r:r['jumps']['physical_increment_rad']['p99'])),
        worst_physical_increment_rad=max(pick(lambda r:r['jumps']['physical_increment_rad']['max']),default=None),
        median_physical_velocity_change_p99_rad_s=median(pick(lambda r:r['jumps']['physical_velocity_change_rad_s']['p99'])),
        median_same_timestamp_raw_disagreement_p99_rad=median(pick(optional)),
        median_executed_switch_increment_p99_rad=median(pick(lambda r:(r['jumps']['command_switch_increments_rad']['action'] or {}).get('p99'))),
        worst_same_timestamp_raw_disagreement_rad=max([r['jumps']['same_time_raw_disagreement_rad']['max'] for r in rows if r['jumps']['same_time_raw_disagreement_rad'] is not None],default=None),
        running_underruns=sum(r['inference_underruns'] for r in rows),
        worst_low_motion_pause_seconds=max(pick(lambda r:r['quality']['low_motion_pause_max_seconds']),default=None))


def ratios(control,baseline):
    fields=('median_active_jerk_p99','worst_active_jerk_peak','median_physical_increment_p99_rad','worst_physical_increment_rad',
            'median_physical_velocity_change_p99_rad_s','median_same_timestamp_raw_disagreement_p99_rad','median_executed_switch_increment_p99_rad')
    return {key:None if baseline[key] is None or baseline[key]==0 or control[key] is None else control[key]/baseline[key] for key in fields}


def main():
    assert not OUT.exists(),'Retain existing comparison'
    checks=[]
    def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
    plans={k:read(v/'plan.json') for k,v in COHORTS.items()}
    check('control_bound_in_baseline_plan',plans['v6']['reference_control_ledger_sha256']==sha(COHORTS['control']/'comparison.json'))
    check('same_sources',plans['v6']['source_sha256']==plans['control']['source_sha256'])
    check('same_scene_file',plans['v6']['scene_sha256']==plans['control']['scene_sha256'])
    configs=[json.loads(json.dumps(plans[k]['config'])) for k in COHORTS]
    for c in configs:c['connection'].pop('port');c['simulation'].pop('scene_config');c['model'].pop('expected_checkpoint_suffix')
    check('same_execution_config',equal(*configs))
    groups={};scenes=[]
    for model,directory in COHORTS.items():
        aggregate=read(directory/'aggregate.json');audit=read(directory/'audit.json');ledger=read(directory/'comparison.json')
        check(model+':complete',aggregate['complete'] and aggregate['audit_passed'] and audit['passed'] and len(aggregate['trials'])==10)
        check(model+':ledger_binding',audit['ledger_sha256']==sha(directory/'comparison.json'))
        check(model+':restored_v6',ledger['restoration']['verified'])
        rows=[]
        for trial in aggregate['trials']:
            row=dict(trial);run=Path(row['run_directory']);prefix=f'{model}:{row["repeat"]}'
            check(prefix+':recomputed',equal(inspect_run(run)['motion_quality'],row['quality']))
            check(prefix+':video',sha(Path(row['video']))==row['video_sha256'])
            scenes.append(read(run/'episode_000_scene.json'))
            with np.load(run/'episode_000.npz',allow_pickle=False) as data:row['jumps']=trajectory_jumps(data)
            check(prefix+':physical_increment',equal(row['jumps']['physical_increment_rad']['max'],row['quality']['physical_joint_increment_rad']['max']))
            rows.append(row)
        groups[model]=dict(trials=rows,all_attempts=summarize(rows),covered_only=summarize([r for r in rows if r['quality']['coverage']['transport_motion_comparable']]))
    check('all20_same_reset_scene',all(equal(scene,scenes[0]) for scene in scenes))
    failed=[c for c in checks if not c['passed']]
    all_ratio=ratios(groups['control']['all_attempts'],groups['v6']['all_attempts'])
    covered_ratio=ratios(groups['control']['covered_only'],groups['v6']['covered_only'])
    gate=plans['v6']['motion_screen']
    usable=not failed and groups['control']['covered_only']['n']>0 and groups['v6']['covered_only']['n']>0
    typical=usable and all_ratio['median_active_jerk_p99'] is not None and covered_ratio['median_active_jerk_p99'] is not None and all_ratio['median_active_jerk_p99']<=gate['typical_active_jerk_p99_ratio_max'] and covered_ratio['median_active_jerk_p99']<=gate['typical_active_jerk_p99_ratio_max']
    tails=usable and all_ratio['worst_active_jerk_peak']<=gate['worst_physical_jerk_peak_ratio_max'] and all_ratio['worst_physical_increment_rad']<=gate['worst_joint_increment_ratio_max']
    OUT.mkdir();(OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes());(OUT/'jump_metrics_source.py').write_bytes((ROOT/'scripts/trajectory_jump_metrics.py').read_bytes())
    report=dict(schema='v6-vs-control-fixed-layout-observed-jumps-v1',audit_passed=not failed,groups=groups,
        control_over_v6_ratios=dict(all_attempts=all_ratio,covered_only=covered_ratio),screen=dict(typical_jerk_improved_at_least10percent=bool(typical),worst_peak_and_increment_within10percent=bool(tails)),
        cohorts={k:dict(directory=str(v),aggregate_sha256=sha(v/'aggregate.json'),ledger_sha256=sha(v/'comparison.json')) for k,v in COHORTS.items()},
        scope='Two sequential blocks, earliercontrol10 and lateroriginalv6ten, same chosen validation layout/reset and execution; policy noise unpaired and time effects remain. Failed coverage stays explicit. This is observed trajectory evidence, not a causal/generalized model ranking. No default change.',
        definitions='Physical increments use20ms pre-step joint samples; jerk is the frozen active frame-max arm metric. Summary p99s are medians of per-trial p99s. Raw same-timestamp disagreement excludes expired old-plan overlap; command boundary increments compare adjacent timestamps and include nominal trajectory motion. No grippers.')
    write(OUT/'comparison.json',report);write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed))
    print(json.dumps(dict(audit_passed=not failed,checks=len(checks),groups={k:{n:r[n] for n in ['all_attempts','covered_only']} for k,r in groups.items()},ratios=report['control_over_v6_ratios'],screen=report['screen']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
