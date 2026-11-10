"""Audit the48-case damping/weights factorial without pooling old cohorts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from profile_v6_runtime import inspect_run
from analyze_rtc_pair import equal


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def median(values):
    vs=[v for v in values if v is not None]
    return None if not vs else float(np.median(vs))
def maximum(values):
    vs=[v for v in values if v is not None]
    return None if not vs else float(max(vs))


def summarize(trials):
    covered=[r for r in trials if r['quality']['coverage']['transport_motion_comparable']]
    p99=lambda r:(r['quality']['active_frame_max_jerk_rad_s3'] or {}).get('p99')
    peak=lambda r:(r['quality']['active_frame_max_jerk_rad_s3'] or {}).get('max')
    return dict(attempts=len(trials),successes=sum(r['success'] for r in trials),
        scored_active_motion=sum(r['quality']['coverage']['active_motion_scored'] for r in trials),
        transport_covered=len(covered),median_active_jerk_p99=median([p99(r) for r in trials]),
        covered_median_active_jerk_p99=median([p99(r) for r in covered]),worst_active_jerk_peak=maximum([peak(r) for r in trials]),
        worst_joint_increment_rad=maximum([r['quality']['physical_joint_increment_rad']['max'] for r in trials]),
        worst_low_motion_pause_seconds=maximum([r['quality']['low_motion_pause_max_seconds'] for r in trials]),
        covered_median_simulated_seconds=median([r['simulated_seconds'] for r in covered]),
        median_low_motion_pause_total_seconds=median([r['quality']['low_motion_pause_total_seconds'] for r in trials]),
        worst_all_frame_jerk_peak=maximum([r['quality']['all_frame_max_jerk_rad_s3']['max'] for r in trials]),
        covered_median_wall_seconds=median([r['control_wall_seconds'] for r in covered]),
        total_running_inference_underruns=sum(r['inference_underruns'] for r in trials),
        total_running_wait_seconds=sum(r['inference_wait_seconds'] for r in trials),
        median_trial_apply_p99_ms=median([r['action_apply_ms']['p99'] for r in trials]),
        worst_apply_interval_ms=maximum([r['action_apply_ms']['max'] for r in trials]))


def compare(candidate,reference,gate):
    checks=[]
    def ratio(name,key,limit):
        a,b=reference.get(key),candidate.get(key)
        checks.append(dict(name=name,reference=a,candidate=b,passed=a is not None and b is not None and b<=a*limit))
    ratio('all_attempt_typical_physical_jerk','median_active_jerk_p99',gate['median_active_jerk_p99_ratio_max'])
    ratio('covered_typical_physical_jerk','covered_median_active_jerk_p99',gate['covered_median_active_jerk_p99_ratio_max'])
    ratio('worst_active_jerk_peak','worst_active_jerk_peak',gate['worst_peak_ratio_max'])
    ratio('worst_physical_increment','worst_joint_increment_rad',gate['worst_increment_ratio_max'])
    a,b=reference['worst_low_motion_pause_seconds'],candidate['worst_low_motion_pause_seconds']
    checks.append(dict(name='worst_low_motion_pause',reference=a,candidate=b,passed=a is not None and b is not None and b<=a+gate['worst_low_motion_pause_add_seconds_max']))
    a,b=reference['transport_covered'],candidate['transport_covered']
    checks.append(dict(name='transport_coverage',reference=a,candidate=b,passed=b>=a and a>0))
    ratio('covered_duration','covered_median_simulated_seconds',gate['covered_duration_ratio_max'])
    ratio('worst_all_frame_peak','worst_all_frame_jerk_peak',gate['worst_all_frame_peak_ratio_max'])
    ratio('covered_wall_duration','covered_median_wall_seconds',gate['covered_wall_duration_ratio_max'])
    a,b=reference['median_low_motion_pause_total_seconds'],candidate['median_low_motion_pause_total_seconds']
    checks.append(dict(name='median_total_pause',reference=a,candidate=b,passed=a is not None and b is not None and b<=a+gate['median_total_pause_add_seconds_max']))
    return dict(passed=all(c['passed'] for c in checks),checks=checks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=ROOT/'outputs/release_retreat_20261029/closed_loop');args=p.parse_args()
    out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    assert not (out/'aggregate.json').exists() and not (out/'audit.json').exists()
    ledger=read(out/'comparison.json');plan=read(out/'plan.json');checks=[]
    def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
    check('completed48',ledger['complete'] and len(ledger['results'])==plan['attempts']==48)
    check('ledger_plan',ledger['plan']==plan);check('frozen_runner',sha(out/'runner_source.py')==plan['script_sha256'])
    check('restored_v6',ledger.get('restoration',{}).get('verified'))
    for n,d in plan['source_sha256'].items():check('source:'+n,sha(out/'frozen_source'/n)==d)
    groups=defaultdict(list);trials=[];scenes=defaultdict(list)
    for case,row in zip(plan['cases'],ledger['results'],strict=True):
        index=row['index'];run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json')
        check(f'{index}:case',all(row[k]==v for k,v in case.items()))
        check(f'{index}:checkpoint',server['checkpoint'].endswith('/'+plan['models'][row['model']]['suffix']))
        check(f'{index}:pi0_h100',server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
        check(f'{index}:norm',server['norm_stats_sha256']==read(ROOT/'outputs/ik_continuous_20261029/training_plan_train_only.json')['norm_stats_sha256'])
        for field in ('source_code_sha256','optional_source_code_sha256'):
            for n,d in meta[field].items():check(f'{index}:source:{n}',plan['source_sha256'][n]==d)
        s=meta['config']['simulation'];check(f'{index}:execution',s['rtc_blend_steps']==20 and s['chunk_blend_steps']==20 and s['video_fps']==25 and not s['display'] and not s['reuse_policy_camera'] and not s['display_in_worker'])
        actual=inspect_run(run);check(f'{index}:recomputed',equal(actual,row['diagnostics']))
        check(f'{index}:prefix',actual['prefix_reconstruction']['passed'])
        check(f'{index}:prediction_timestamp',actual['summary']['raw_prediction_max_error']==0 and actual['summary']['max_action_timestamp_error_steps']==0)
        result=actual['summary']['result'];check(f'{index}:no_operational_error',not row.get('operational_error') and result['error'] is None)
        scene=read(run/'episode_000_scene.json');check(f'{index}:damping_scale',scene['scene']['wrist_damping_scale']==plan['models'][row['model']]['damping_scale']);scene['scene'].pop('wrist_damping_scale');scene['reset']['scene'].pop('wrist_damping_scale');scenes[row['layout']].append(scene)
        trial=dict(index=index,model=row['model'],layout=row['layout'],run_directory=str(run),success=result['success'],
            simulated_seconds=result['execution']['simulated_seconds'],control_wall_seconds=result['execution']['control_wall_seconds'],block=row['block'],repeat=row['repeat'],inference_underruns=result['execution']['inference_underruns'],
            inference_wait_seconds=result['execution']['underrun_wait_seconds'],action_apply_ms=actual['action_apply_ms'],
            quality=actual['motion_quality'],task_progress=result['task_progress'],
            video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4'))
        groups[row['model']].append(trial);trials.append(trial)
    for layout,rs in scenes.items():check('same_reset:'+layout,len(rs)==8 and all(equal(r,rs[0]) for r in rs))
    import xml.etree.ElementTree as ET
    xml_groups=defaultdict(list)
    for row in ledger['results']:
        xml_groups[row['layout']].append((plan['models'][row['model']]['damping_scale'],Path(row['run_directory'])/'scene.xml'))
    expected={f'vx300s_{side}/{kind}' for side in ('left','right') for kind in ('forearm_roll','wrist_angle','wrist_rotate')}
    for layout,entries in xml_groups.items():
        original=next(path.read_text() for scale,path in entries if scale==1)
        for scale,path in entries:
            if scale==1:
                check('xml_identical_base:'+str(path),path.read_text()==original)
                continue
            left,right=ET.fromstring(original),ET.fromstring(path.read_text());changes=[]
            for a,b in zip(left.iter(),right.iter(),strict=True):
                if a.attrib!=b.attrib:
                    changes.append(a.get('name'));check('damping_only_attribute:'+str(path)+':'+str(a.get('name')),a.tag==b.tag=='joint' and a.get('name') in expected and {k for k in a.attrib if a.get(k)!=b.get(k)}=={'damping'} and float(b.get('damping'))==float(a.get('damping'))*2)
                    b.set('damping',a.get('damping'))
            check('six_damping_changes:'+str(path),set(changes)==expected and len(changes)==6)
            check('remaining_xml_identical:'+str(path),ET.tostring(left)==ET.tostring(right))
    failed=[r for r in checks if not r['passed']];models={k:summarize(v) for k,v in groups.items()}
    comparisons={role:compare(models['candidate_damping2'],models[role],plan['gate']) for role in plan['gate']['references']}
    factorial={}
    for profile,reference in [('v6_damping2','v6'),('candidate_damping2','candidate')]:
        factorial[profile]=dict(reference=reference,comparison=compare(models[profile],models[reference],plan['gate']))
    for role,comparison in comparisons.items():
        b,c=models[role],models['candidate_damping2']
        for metric,key in [('median_trial_apply_p99_ms','median_apply_p99_ratio_max'),('worst_apply_interval_ms','worst_apply_ratio_max')]:
            comparison['checks'].append(dict(name=metric,reference=b[metric],candidate=c[metric],passed=c[metric]<=b[metric]*plan['gate'][key]))
        comparison['checks'].append(dict(name='running_inference_underruns',reference=b['total_running_inference_underruns'],candidate=c['total_running_inference_underruns'],passed=c['total_running_inference_underruns']<=b['total_running_inference_underruns']))
    floor=max(plan['gate']['minimum_successes'],min(models[r]['successes'] for r in plan['gate']['references'])-plan['gate']['success_shortfall_vs_weaker_reference_max'])
    for comparison in comparisons.values():
        comparison['checks'].append(dict(name='secondary_success_floor',candidate=models['candidate_damping2']['successes'],minimum=floor,passed=models['candidate_damping2']['successes']>=floor))
        comparison['passed']=all(c['passed'] for c in comparison['checks'])
    eligible=not failed and all(c['passed'] for c in comparisons.values())
    report=dict(schema='release-damping-factorial-summary-v1',complete=ledger['complete'],audit_passed=not failed,
        models=models,trials=trials,comparisons=comparisons,factorial=factorial,
        decision='promising_requires_replication' if eligible else 'retain_v6_reference',
        scope='Six selected validation layouts twice per weights/damping profile, reversed blocks and unpaired noise/latency. Damping is a physical-plant intervention; weight effects need same-damping comparison. Physical derivatives use simulation time; network waits are distinct. All failures retained; heldout test not used. No automatic default promotion.')
    audit=dict(schema='release-damping-factorial-audit-v1',passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(out/'comparison.json'),analyzer_sha256=sha(Path(__file__)))
    (out/'analyzer_source.py').write_bytes(Path(__file__).read_bytes())
    for name,value in [('aggregate.json',report),('audit.json',audit)]: (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(audit_passed=not failed,checks=len(checks),failed_checks=failed,models=models,decision=report['decision']),indent=2))
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
