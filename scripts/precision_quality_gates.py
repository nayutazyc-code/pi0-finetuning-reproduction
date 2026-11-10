"""Shared evaluator preserving the first IK experiment's twelve gate definitions."""
import numpy as np


def score(base,candidate,g):
    tests=[]
    def compare(name,a,b,stat,limit,additive=False):
        available=bool(a and b and all(v is not None for v in a+b))
        a=float(stat(a)) if available else None;b=float(stat(b)) if available else None
        tests.append(dict(metric=name,baseline=a,candidate=b,ratio=None if not available or not a else b/a,
            difference=None if not available else b-a,passed=available and b<=(a+limit if additive else a*limit)))
    for name,key,stat,threshold in [('first_three_second_wrist_peak_median','first_three_second_wrist_peak',np.median,g['first_three_second_wrist_peak_median_ratio_max']),
        ('full_lift_wrist_peak_median','full_lift_wrist_peak',np.median,g['full_lift_wrist_peak_median_ratio_max']),
        ('first_second_wrist_peak_worst','first_second_wrist_peak',max,g['first_second_wrist_peak_worst_ratio_max'])]:
        compare(name,[r[key] for r in base],[r[key] for r in candidate],stat,threshold)
    get=lambda r,k:None if r['motion_quality'] is None or r['motion_quality']['active_frame_max_jerk_rad_s3'] is None else r['motion_quality']['active_frame_max_jerk_rad_s3'][k]
    compare('whole_active_arm_jerk_p99_median',[get(r,'p99') for r in base],[get(r,'p99') for r in candidate],np.median,g['whole_active_arm_jerk_p99_median_ratio_max'])
    compare('whole_arm_jerk_worst',[get(r,'max') for r in base],[get(r,'max') for r in candidate],max,g['whole_arm_jerk_worst_ratio_max'])
    compare('duration_median',[r['frames']*.02 for r in base],[r['frames']*.02 for r in candidate],np.median,g['duration_median_ratio_max'])
    for name,stat,limit in [('low_motion_pause_total_seconds',np.median,g['idle_total_median_added_seconds_max']),('low_motion_pause_max_seconds',max,g['idle_max_worst_added_seconds_max'])]:
        compare(name,[None if r['motion_quality'] is None else r['motion_quality'][name] for r in base],[None if r['motion_quality'] is None else r['motion_quality'][name] for r in candidate],stat,limit,True)
    compare('box_to_containment_median',[r['box']['first_above_to_containment_seconds'] for r in base],[r['box']['first_above_to_containment_seconds'] for r in candidate],np.median,g['box_to_containment_median_ratio_max'])
    records=base+candidate
    tests.append(dict(metric='all16_success_structural_transport_coverage',passed=len(records)==16 and all(r['automatic_success'] and r['structural_pass'] and r['motion_quality'] is not None and r['motion_quality']['coverage']['transport_motion_comparable'] for r in records)))
    ratios=[None if get(a,'p99') is None or get(b,'p99') is None or get(a,'p99')==0 else get(b,'p99')/get(a,'p99') for a,b in zip(base,candidate,strict=True)]
    tests.append(dict(metric='individual_active_arm_jerk_p99',ratios=ratios,passed=all(r is not None and r<=g['individual_active_arm_jerk_p99_ratio_max'] for r in ratios)))
    improved=sum(a['full_lift_wrist_peak'] is not None and b['full_lift_wrist_peak'] is not None and b['full_lift_wrist_peak']<=.8*a['full_lift_wrist_peak'] for a,b in zip(base,candidate,strict=True))
    tests.append(dict(metric='pairs_with_20pct_full_lift_improvement',count=improved,passed=improved>=g['minimum_pairs_with_20pct_full_lift_improvement']))
    return tests
