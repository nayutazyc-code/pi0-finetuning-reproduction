"""Apply the frozen runtime adoption gate; never changes runtime defaults."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def evaluate(aggregate,gate):
    base=[r for r in aggregate['trials'] if r['profile']=='legacy_multiview']
    candidate=[r for r in aggregate['trials'] if r['profile']=='worker_multiview']
    checks=[]
    def check(name,b,c,passed):checks.append(dict(name=name,baseline=b,candidate=c,passed=bool(passed)))
    if not base or not candidate or not aggregate['audit_passed']:
        return dict(decision='insufficient_evidence',checks=checks,scope='Complete audited baseline and candidate required')
    def median(rs,path):
        values=[]
        for r in rs:
            for key in path:r=r.get(key) if isinstance(r,dict) else None
            if r is not None:values.append(r)
        return float(np.median(values)) if len(values)==len(rs) and values else None
    def ratio_check(name,b,c,limit):
        check(name,b,c,b is not None and c is not None and c<=b*limit)
    ratio_check('median_trial_apply_p99_ms',median(base,['action_apply_ms','p99']),median(candidate,['action_apply_ms','p99']),gate['median_trial_apply_p99_ratio_max'])
    ratio_check('worst_apply_interval_ms',max(r['action_apply_ms']['max'] for r in base),max(r['action_apply_ms']['max'] for r in candidate),gate['worst_apply_interval_ratio_max'])
    rates=[sum(r['gaps_over_50ms'] for r in rs)/sum(r['steps']-1 for r in rs)*1000 for rs in (base,candidate)]
    ratio_check('gap_over50_rate_per1000_intervals',*rates,gate['gap_over50_rate_per1000_steps_ratio_max'])
    check('running_underruns',sum(r['inference_underruns'] for r in base),sum(r['inference_underruns'] for r in candidate),
          sum(r['inference_underruns'] for r in candidate)<=gate['running_underruns_max'])
    covered=lambda rs:[r for r in rs if r['motion']['coverage']['transport_motion_comparable']]
    bc,cc=covered(base),covered(candidate)
    check('transport_coverage',len(bc),len(cc),len(cc)>=len(bc) and bool(bc))
    ratio_check('covered_median_simulated_seconds',median(bc,['simulated_seconds']),median(cc,['simulated_seconds']),gate['covered_median_simulated_duration_ratio_max'])
    ratio_check('covered_median_active_jerk_p99',median(bc,['motion','active_frame_max_jerk_rad_s3','p99']),median(cc,['motion','active_frame_max_jerk_rad_s3','p99']),gate['covered_median_active_jerk_p99_ratio_max'])
    missing=not bc or not cc
    return dict(decision='insufficient_evidence' if missing else 'eligible_for_review' if all(c['passed'] for c in checks) else 'retain_existing_default',
                checks=checks,failed_checks=[c['name'] for c in checks if not c['passed']],
                scope='Descriptive selected-seed validation with unpaired policy noise. Eligibility does not prove population benefit or authorize automatic default changes. All attempts and coverage remain visible.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();plan=json.loads((out/'plan.json').read_text());aggregate=json.loads((out/'aggregate.json').read_text())
    result=evaluate(aggregate,plan['runtime_adoption_gate'])
    result.update(aggregate_sha256=hashlib.sha256((out/'aggregate.json').read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
                  selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path=out/'selection.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
