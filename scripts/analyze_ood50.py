"""Audit every frozen OOD outcome; no outcome filtering or pooled cohorts."""
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from profile_v6_runtime import inspect_run
from analyze_rtc_pair import equal


def main():
 out=ROOT/'outputs/ood50_20261029';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 plan=read(out/'plan.json');ledger=read(out/'comparison.json');checks=[];trials=[]
 def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
 check('complete50',ledger['complete'] and len(ledger['results'])==50)
 check('plan_equal',ledger['plan']==plan);check('runner_hash',sha(out/'runner_source.py')==plan['script_sha256']);check('restored_v6',ledger.get('restoration',{}).get('verified'))
 for n,h in plan['source_sha256'].items():check('frozen_source:'+n,sha(out/'frozen_source'/n)==h)
 for case,row in zip(plan['cases'],ledger['results'],strict=True):
  i=row['index'];run=Path(row['run_directory']);meta=read(run/'run.json');server=read(run/'server.json');scene=read(run/'episode_000_scene.json');actual=inspect_run(run)
  check(f'{i}:case',all(row[k]==v for k,v in case.items()));check(f'{i}:no_error',not row.get('operational_error') and actual['summary']['result']['error'] is None)
  check(f'{i}:checkpoint',server['checkpoint'].endswith('/banana_v6_ik_candidate1000/999'));check(f'{i}:pi0',server['model_family']=='pi0' and server['action_horizon']==100 and server['denoise_steps']==10)
  check(f'{i}:norm',server['norm_stats_sha256']==read(ROOT/'outputs/ik_continuous_20261029/training_plan_train_only.json')['norm_stats_sha256'])
  s=meta['config']['simulation'];check(f'{i}:execution',s['rtc_blend_steps']==s['chunk_blend_steps']==20 and not s['display'] and s['video_fps']==25 and s['max_steps']==2000)
  expected=plan['layouts'][case['layout']];check(f'{i}:seed',s['seed']==expected['reset_seed']);check(f'{i}:damping',scene['scene']['wrist_damping_scale']==2)
  for k,v in expected['scene'].items():check(f'{i}:scene:'+k,equal(scene['scene'][k],v))
  for field in ('source_code_sha256','optional_source_code_sha256'):
   for n,h in meta[field].items():check(f'{i}:source:'+n,h==plan['source_sha256'][n])
  check(f'{i}:recomputed',equal(actual,row['diagnostics']));check(f'{i}:prefix',actual['prefix_reconstruction']['passed']);check(f'{i}:timestamps',actual['summary']['raw_prediction_max_error']==actual['summary']['max_action_timestamp_error_steps']==0)
  # Actual settled reset is kept in the scene receipt, not inferred from requested XY.
  reset=scene['reset'];check(f'{i}:reset_seed',reset['seed']==expected['reset_seed'])
  x,y,_=reset['settled']['position'];w,qx,qy,qz=reset['settled']['quaternion_wxyz'];yaw=math.degrees(math.atan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz)))
  if case['group']=='position':ood=not(.16<=x<=.20 and .46<=y<=.50)
  elif case['group']=='yaw':ood=not 0<=yaw<=20
  else:ood=scene['scene']['light_scale']<.85 or scene['scene']['light_scale']>1.15
  check(f'{i}:realized_ood',ood)
  result=actual['summary']['result'];trials.append(dict(index=i,group=case['group'],layout=case['layout'],success=result['success'],ever_lifted=result['ever_lifted'],final_inside_box=result['final_inside_box'],run_directory=str(run),video=str(run/'episode_000.mp4'),video_sha256=sha(run/'episode_000.mp4'),simulated_seconds=result['execution']['simulated_seconds'],underruns=result['execution']['inference_underruns'],wait_seconds=result['execution']['underrun_wait_seconds'],motion=actual['motion_quality'],requested_scene=expected['scene'],actual_reset=reset))
 def summary(rs):
  n=len(rs);k=sum(r['success'] for r in rs)
  return dict(attempts=n,successes=k,failures=n-k,success_fraction=k/n if n else None,running_underruns=sum(r['underruns'] for r in rs),running_wait_seconds=sum(r['wait_seconds'] for r in rs))
 failed=[c for c in checks if not c['passed']];report=dict(schema='ood50-summary-v1',complete=ledger['complete'],audit_passed=not failed,total=summary(trials),groups={g:summary([r for r in trials if r['group']==g]) for g in plan['groups']},trials=trials,scope='Descriptive success for this deliberately balanced50-scene fine-tuning OOD suite. One unpaired stochastic policy attempt per layout. Not a population rate or pure-weight evaluation; includes damping2. No training/tuning, no pooling with prior ID evaluations.')
 for name,d in [('aggregate.json',report),('audit.json',dict(passed=not failed,checks=checks,failed_checks=failed,ledger_sha256=sha(out/'comparison.json'),analyzer_sha256=sha(Path(__file__))))]:
  p=out/name;assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps(dict(total=report['total'],groups=report['groups'],checks=len(checks),failed=failed),indent=2))
 if failed:raise SystemExit(1)

if __name__=='__main__':main()
