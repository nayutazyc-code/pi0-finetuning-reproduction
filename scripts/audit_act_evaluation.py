"""Recompute the fixed ACT evaluation counts and interface checks."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
 out=ROOT/'outputs/act_baseline_20261029';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();d=read(out/'evaluation.json');p=read(out/'evaluation_plan.json');checks=[]
 def check(name,value):checks.append(dict(name=name,passed=bool(value)))
 check('complete16',d['complete'] and len(d['results'])==p['attempts']==16);check('frozen_plan',d['plan']==p);check('runner',sha(ROOT/'scripts/evaluate_act_baseline.py')==p['runner_sha256'])
 for n,h in p['source_sha256'].items():check(n,sha(ROOT/'src/pi0_repro'/n)==h)
 for c,r in zip(p['cases'],d['results'],strict=True):
  i=c['index'];path=Path(r['run_directory']);s=read(path/'server.json');reset=read(path/'reset.json');task=[json.loads(line) for line in (path/'task.jsonl').read_text().splitlines()]
  check(f'{i}:case',all(r[k]==v for k,v in c.items()));check(f'{i}:no_error',not r.get('error'));check(f'{i}:actual_act',s['model_family']=='act' and s['step']==20000 and s['action_dim']==14 and s['action_horizon']==100 and s['checkpoint_sha256']==d['checkpoint_sha256']);check(f'{i}:reset_seed',reset['seed']==c['seed']);check(f'{i}:no_te',s['temporal_ensemble'] is False)
  with np.load(path/'trajectory.npz',allow_pickle=False) as a:
   check(f'{i}:arrays',a['state'].shape==a['action'].shape==a['requested_action'].shape==(r['steps'],14) and all(np.isfinite(a[k]).all() for k in ['state','action','requested_action','rtt_ms']));check(f'{i}:queries',len(a['rtt_ms'])==r['inference_calls']==(r['steps']+24)//25)
  check(f'{i}:task_alignment',len(task)==r['steps'] and all(t['control_step']==j+1 for j,t in enumerate(task)));check(f'{i}:success',task[-1]['is_success']==r['success']);check(f'{i}:duration',r['simulated_seconds']==r['steps']*.02)
 groups={s:dict(successes=sum(r['success'] for r in d['results'] if r['split']==s),attempts=sum(r['split']==s for r in d['results'])) for s in ['train','validation']};check('counts',groups==d['groups']);failed=[c for c in checks if not c['passed']];report=dict(passed=not failed,checks=checks,failed_checks=failed,groups=groups,checkpoint_sha256=d['checkpoint_sha256'],evaluation_sha256=sha(out/'evaluation.json'),scope='Training-layout memorization checks and12validation attempts separate. Not matched pi0 timing, OOD, population reliability or proof of complete data quality.')
 q=out/'evaluation_audit.json';assert not q.exists();q.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(passed=report['passed'],checks=len(checks),failed=failed,groups=groups),indent=2))
 if failed:raise SystemExit(1)
if __name__=='__main__':main()
