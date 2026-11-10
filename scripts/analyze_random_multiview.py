"""Audit a complete seeded random-pose suite without further policy calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from pi0_repro.motion_quality import motion_quality
from analyze_rtc_pair import equal,metadata_valid,reconstructed_prefix


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def dist(values):
    return {'median':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':float(max(values))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();assert out.is_relative_to(ROOT/'outputs')
    assert not (out/'aggregate.json').exists() and not (out/'audit.json').exists()
    ledger=read(out/'comparison.json');plan=read(out/'plan.json')
    assert ledger['complete'] and len(ledger['results'])==12 and plan['attempts']==12
    checks=[]
    def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
    check('ledger_plan',ledger['plan']==plan)
    check('runner_copy',sha(out/'runner_source.py')==plan['script_sha256'])
    for index,part in enumerate(plan.get('parts',[])):
        directory=Path(part['path'])
        check(f'phase{index}:comparison',sha(directory/'comparison.json')==part['comparison_sha256'])
        check(f'phase{index}:plan',sha(directory/'plan.json')==part['plan_sha256'])
        check(f'phase{index}:runner',sha(directory/'runner_source.py')==part['runner_sha256'])
    for name,digest in plan['source_sha256'].items():check('frozen_source:'+name,sha(ROOT/'src/pi0_repro'/name)==digest)
    check('same_v6_process',ledger['endpoint_before']['service_identity']==ledger['endpoint_after']['service_identity'])
    check('same_v6_metadata',ledger['endpoint_before']['metadata']==ledger['endpoint_after']['metadata'])
    check('model_before_after',all(metadata_valid(ledger[k]['metadata']) for k in ['endpoint_before','endpoint_after']))
    spec=importlib.util.spec_from_file_location('random_execution_summary',ROOT/'scripts/summarize_banana_execution.py')
    summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)
    result=[]
    ages=[];exchange=[];apply_intervals=[];control_compute=[]
    for expected,row in zip(plan['cases'],ledger['results'],strict=True):
        index=row['index'];directory=Path(row['run_directory']);run=read(directory/'run.json')
        check(f'{index}:case',all(row[k]==expected[k] for k in expected))
        known_transport=(row.get('accepted_transport_warning') is True and row.get('infrastructure_error')=='AssertionError: Inference underrun; retain and abort')
        check(f'{index}:no_unaccounted_infra','infrastructure_error' not in row or known_transport)
        check(f'{index}:seed',run['config']['simulation']['seed']==row['seed'])
        check(f'{index}:RTC20',run['config']['simulation']['rtc_blend_steps']==20 and run['config']['simulation']['chunk_blend_steps']==20)
        check(f'{index}:original_model_camera',run['config']['model']['send_full_resolution'] and run['config']['simulation']['display_layout']=='multiview')
        check(f'{index}:model_metadata',metadata_valid(read(directory/'server.json')))
        for name,digest in run['source_code_sha256'].items():check(f'{index}:source:'+name,digest==plan['source_sha256'][name])
        for name in ('multiview.py','random_pose.py','async_multiview.py'):
            check(f'{index}:optional_source:'+name,run['optional_source_code_sha256'][name]==plan['source_sha256'][name])
        reset=read(directory/'episode_000_scene.json')
        check(f'{index}:realized_sample',reset['scene']==expected['planned_scene'] and reset['reset']==row['reset'])
        check(f'{index}:fixed_box',reset['scene']['box_position']==[.13,.72,0.])
        scene=reset['scene'];x,y,z=scene['banana_position'];yaw=scene['banana_yaw_degrees']
        check(f'{index}:sample_bounds',.16<=x<=.20 and .46<=y<=.50 and z==.045 and 0<=yaw<=20)
        check(f'{index}:damping_baseline',scene['wrist_damping_scale']==1.)
        tasks=rows(directory/'episode_000_task.jsonl');latency=rows(directory/'episode_000_latency.jsonl')
        with np.load(directory/'episode_000.npz',allow_pickle=False) as data:
            check(f'{index}:motion_recomputed',equal(motion_quality(data['state'],data['action'],tasks),row['motion_quality']))
            check(f'{index}:raw_action',np.array_equal(data['predicted_chunks'][data['action_prediction_index'],data['action_prediction_offset']],data['requested_action']))
            check(f'{index}:timestamp',np.array_equal(data['prediction_steps'][data['action_prediction_index']]+data['action_prediction_offset'],np.arange(len(data['action']))))
            check(f'{index}:RTC_prefix',reconstructed_prefix(data,latency,20)['passed'])
        details=summary.summarize(directory)
        check(f'{index}:diagnostics_recomputed',equal(details,row['diagnostics']))
        ep=details['episodes'][0]['result'];success=bool(ep['success'])
        check(f'{index}:success_flag',success==row['success'])
        if success:check(f'{index}:full_success',all(tasks[-1][k] for k in ['inside_box','released','supported_by_box','stable','is_success']) and tasks[-1]['success_hold_steps']>=tasks[-1]['required_hold_steps'])
        check(f'{index}:underrun_accounted',ep['execution']['inference_underruns']==0 or
            row.get('transport_warning',{}).get('underruns')==ep['execution']['inference_underruns'])
        check(f'{index}:video_hash',sha(directory/'episode_000_multiview.mp4')==row['video_sha256'])
        check(f'{index}:top_video_hash',sha(directory/'episode_000.mp4')==row['policy_video_sha256'])
        presentation=rows(directory/'episode_000_multiview.jsonl')
        check(f'{index}:video_sample_steps',[r['target_control_step'] for r in presentation]==list(range(0,ep['steps'],2))+[ep['steps']])
        check(f'{index}:snapshot_alignment',all(0<=r['source_control_step']<=r['target_control_step'] and r['frame_age_control_steps']==r['target_control_step']-r['source_control_step'] for r in presentation))
        check(f'{index}:terminal_current',presentation[-1]['source_control_step']==ep['steps'])
        ages.extend(r['frame_age_control_steps'] for r in presentation)
        control=rows(directory/'episode_000_control.jsonl')
        apply_intervals.extend(np.diff([r['action_apply_wall_seconds'] for r in control])*1000)
        exchange.extend(r['presentation_exchange_seconds']*1000 for r in control if r['presentation_exchange_seconds'])
        progress=details['episodes'][0]['task_progress'];quality=row['motion_quality']
        failure=None if success else ('no_confirmed_lift' if not quality['coverage']['lifted_pinch'] else 'not_entered_box' if not quality['coverage']['entered_box'] else 'not_complete_after_entry')
        result.append({'index':index,'seed':row['seed'],'sample_position':[x,y,z],'sample_yaw_degrees':yaw,
            'settled_pose':reset['reset']['settled'],'success':success,'failure_class':failure,'steps':ep['steps'],
            'simulated_seconds':ep['execution']['simulated_seconds'],'run_directory':str(directory),
            'video':row['video'],'task_progress':progress,'motion_quality':quality,'frame_age_steps':row['display_frame_age_steps'],
            'control_apply_interval_ms':details['episodes'][0]['action_apply_interval_ms'],
            'inference_underruns':ep['execution']['inference_underruns'],
            'underrun_wait_seconds':ep['execution']['underrun_wait_seconds'],
            'transport_warning':row.get('transport_warning'),'original_wrapper_error':row.get('infrastructure_error')})
    passed=all(c['passed'] for c in checks)
    aggregate={'schema':'v6-random-planar-multiview-summary-v1','complete':True,'audit_passed':passed,
               'attempts':12,'successes':sum(r['success'] for r in result),'failures':[r['index'] for r in result if not r['success']],
               'trial_results':result,'control_apply_interval_ms_pooled':dist(apply_intervals),
               'display_exchange_ms_pooled':dist(exchange),'display_source_age_control_steps_pooled':dist(ages),
               'inference_underruns_total':sum(r['inference_underruns'] for r in result),
               'transport_warning_attempt_indices':[r['index'] for r in result if r['transport_warning']],
               'protocol_note':plan.get('protocol_note','Single uninterrupted phase.'),
               'scope':'12 predeclared fresh pose combinations inside the original XY/yaw acquisition envelope, one attempt each; fixed box, v6 RTC20, no training. Not a whole-table, arbitrary3D or population success guarantee.',
               'display_timing':'Composite may repeat an older full snapshot. Source-step age is in 20ms simulated control steps; frame_age_wall_seconds is age since publication, not full capture-to-display latency. Original policy video is separate.'}
    audit={'schema':'v6-random-planar-multiview-audit-v1','passed':passed,'checks':checks,'failed_checks':[c['name'] for c in checks if not c['passed']],
           'source_comparison_sha256':sha(out/'comparison.json'),'analyzer_sha256':sha(Path(__file__))}
    for name,value in [('aggregate.json',aggregate),('audit.json',audit)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'attempts':12,'successes':aggregate['successes'],'audit_passed':passed,'checks':len(checks),'failed':audit['failed_checks'],
                      'control_apply_interval_ms':aggregate['control_apply_interval_ms_pooled'],'display_age_steps':aggregate['display_source_age_control_steps_pooled']},indent=2))
    if not passed:raise SystemExit('Audit failed; inspect retained evidence, do not promote a success claim')


if __name__=='__main__':main()
