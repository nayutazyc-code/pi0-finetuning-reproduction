"""Audit fresh single-factor closed-loop A/B; retain failed screen and outcomes."""
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_continued_wrist_kp import OUT,SUFFIX,SEED
from evaluate_v6_progress import read,sha,write
from analyze_v6_progress_evaluation import summarize
from analyze_rtc_pair import equal
from profile_v6_runtime import inspect_run
from trajectory_jump_metrics import trajectory_jumps
import numpy as np


def main():
    ledger=read(OUT/'comparison.json');plan=read(OUT/'plan.json')
    assert ledger['complete'] and len(ledger['results'])==plan['attempts']==20
    assert not (OUT/'aggregate.json').exists()
    checks=[]
    def check(name,passed):checks.append(dict(name=name,passed=bool(passed)))
    check('plan',ledger['plan']==plan);check('runner',sha(OUT/'runner_source.py')==plan['script_sha256'])
    check('v6_restored',ledger['restoration']['verified'] and ledger['restoration']['metadata']['checkpoint'].endswith('/banana_v6_long100_accum8000/7999'))
    check('network',ledger['route_before']==ledger['route_after'] and not ledger['route_after'].startswith(('utun','tun','tap')))
    check('negative_screen_retained',not read(ROOT/'outputs/wrist_kp_replay_screen_20261028/summary.json')['passed'])
    check('scope_amendment',sha(ROOT/'outputs/wrist_kp_replay_screen_20261028/continued_only_amendment.json')==plan['amendment_sha256'])
    for name,digest in plan['source_sha256'].items():check('frozen:'+name,sha(OUT/'frozen_source'/name)==digest)
    groups={key:[] for key in ['baseline','kp150']};scenes=[];xmls={}
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        i=row['index'];profile=row['profile'];run=Path(row['run_directory']);cfg=read(run/'run.json');server=read(run/'server.json')
        check(f'{i}:case',all(row[k]==v for k,v in case.items()))
        check(f'{i}:config',equal(cfg['config'],plan['config_by_profile'][profile]))
        check(f'{i}:model',server['checkpoint'].endswith('/'+SUFFIX) and server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'{i}:norm',server['norm_stats_sha256']==plan['config']['model']['expected_norm_stats_sha256'])
        for field in ['source_code_sha256','optional_source_code_sha256']:
            for n,d in cfg[field].items():check(f'{i}:source:{n}',plan['source_sha256'][n]==d)
        details=inspect_run(run);check(f'{i}:recomputed',equal(details,row['diagnostics']))
        check(f'{i}:prefix_timestamp',details['prefix_reconstruction']['passed'] and details['summary']['raw_prediction_max_error']==0 and details['summary']['max_action_timestamp_error_steps']==0)
        result=details['summary']['result'];check(f'{i}:no_error',result['error'] is None and not row.get('operational_error') and not result['aborted_by_window_close'] and result['seed']==SEED)
        scene=read(run/'episode_000_scene.json');check(f'{i}:scale',scene['scene'].pop('right_wrist_angle_kp_scale')==(.75 if profile=='kp150' else 1.))
        scene['reset']['scene'].pop('right_wrist_angle_kp_scale');scenes.append(scene)
        xml=(run/'scene.xml').read_text();xmls.setdefault(profile,xml);check(f'{i}:same_profile_xml',xmls[profile]==xml)
        with np.load(run/'episode_000.npz',allow_pickle=False) as data:jumps=trajectory_jumps(data)
        groups[profile].append(dict(index=i,repeat=row['repeat'],profile=profile,model='control',layout='episode_004',run_directory=str(run),
            success=result['success'],simulated_seconds=result['execution']['simulated_seconds'],
            inference_underruns=result['execution']['inference_underruns'],inference_wait_seconds=result['execution']['underrun_wait_seconds'],
            action_apply_ms=details['action_apply_ms'],quality=details['motion_quality'],task_progress=result['task_progress'],jumps=jumps,
            failure_class='success' if result['success'] else 'no_lift' if not details['motion_quality']['coverage']['lifted_pinch'] else 'no_box_entry' if not details['motion_quality']['coverage']['entered_box'] else 'placement_not_confirmed',
            video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4')))
    # Realized settled positions may move by tiny floating point amounts under
    # different servo gains. Compare requested geometry/seed separately from
    # measured settling, rather than pretending both physical trajectories match.
    scene_geometry=[dict(scene=s['scene'],seed=s['reset']['seed'],sample=s['reset']['sample']) for s in scenes]
    check('same_requested_geometry_seed',all(equal(s,scene_geometry[0]) for s in scene_geometry))
    left,right=ET.fromstring(xmls['baseline']),ET.fromstring(xmls['kp150']);changes=[]
    for a,b in zip(left.iter(),right.iter(),strict=True):
        if a.attrib!=b.attrib:
            changes.append((a,b));check('only_xml_kp',a.tag==b.tag=='position' and a.get('joint')==b.get('joint')=='vx300s_right/wrist_angle' and a.get('kp')=='200' and float(b.get('kp'))==150)
            b.set('kp',a.get('kp'))
    check('one_xml_change',len(changes)==1 and ET.tostring(left)==ET.tostring(right))
    summaries={k:summarize(rows) for k,rows in groups.items()};base,candidate=summaries['baseline'],summaries['kp150'];gate=plan['gate'];tests=[]
    for key,limit in [('median_active_jerk_p99',gate['typical_active_jerk_ratio_max']),('covered_median_active_jerk_p99',gate['typical_active_jerk_ratio_max']),('worst_active_jerk_peak',gate['worst_active_jerk_peak_ratio_max']),('worst_joint_increment_rad',gate['worst_physical_increment_ratio_max'])]:
        a,b=base[key],candidate[key];tests.append(dict(metric=key,baseline=a,candidate=b,ratio=None if a is None or b is None or a==0 else b/a,passed=a is not None and b is not None and b<=a*limit))
    tests.append(dict(metric='low_motion_pause',baseline=base['worst_low_motion_pause_seconds'],candidate=candidate['worst_low_motion_pause_seconds'],passed=candidate['worst_low_motion_pause_seconds']<=base['worst_low_motion_pause_seconds']+gate['low_motion_pause_add_seconds_max']))
    tests.append(dict(metric='transport_coverage',baseline=base['transport_covered'],candidate=candidate['transport_covered'],passed=candidate['transport_covered']>=base['transport_covered']>0))
    failed=[c for c in checks if not c['passed']];eligible=not failed and all(t['passed'] for t in tests)
    report=dict(schema='continued-right-wrist-kp-ab-summary-v1',complete=True,audit_passed=not failed,groups={k:dict(summary=summaries[k],trials=v) for k,v in groups.items()},
        gate_checks=tests,decision='promising_requires_replication' if eligible else 'retain_unchanged_controller',
        scope='Fresh10versus10counterbalanced AB/BA order, same continued-model process and requested scene/reset. Noise unpaired. One simulated position-servo kp differs. Historical replays/cohorts not pooled; no originalv6 quality/generalization claim or automatic default change.')
    (OUT/'analyzer_source.py').write_bytes(Path(__file__).read_bytes());write(OUT/'aggregate.json',report);write(OUT/'audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(OUT/'comparison.json')))
    print(json.dumps(dict(audit_passed=not failed,checks=len(checks),failed_checks=failed,summaries=summaries,gate_checks=tests,decision=report['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
