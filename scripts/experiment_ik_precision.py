"""Frozen single-variable recovery-expert IK precision experiment, never runtime."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write

OUT=ROOT/'outputs/ik_precision_20261029'
PRIOR=ROOT/'outputs/contact_recovery_replication_20261029'


def prepare():
    from pi0_repro.banana_expert import load_expert_config
    selection=read(PRIOR/'takeovers.json');prior=read(PRIOR/'plan.json');old=read(PRIOR/'teacher_candidates.json')
    assert selection['complete'] and len(selection['entries'])==8
    assert not OUT.exists(),'Preserve existing experiment; no in-place retries'
    sources=dict(prior['source_sha256'],**old['auxiliary_source_sha256'])
    assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in sources.items())
    OUT.mkdir();frozen=OUT/'frozen_source';frozen.mkdir()
    for n in sources:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    source=Path(prior['expert_configs']['unload']['path']);assert sha(source)==prior['expert_configs']['unload']['sha256']
    baseline=source.read_text();assert baseline.count('ik_position_tolerance = 0.0002')==1
    (OUT/'baseline.toml').write_text(baseline)
    (OUT/'tight_ik.toml').write_text(baseline.replace('ik_position_tolerance = 0.0002','ik_position_tolerance = 0.00002'))
    configs={p:OUT/(p+'.toml') for p in ('baseline','tight_ik')}
    a,b=[asdict(load_expert_config(configs[p])) for p in configs]
    assert a['ik_position_tolerance']==.0002 and b['ik_position_tolerance']==.00002
    b['ik_position_tolerance']=a['ik_position_tolerance'];assert a==b
    cases=[]
    for e in selection['entries']:
        for profile in (('baseline','tight_ik') if e['parent']%2==0 else ('tight_ik','baseline')):
            cases.append(dict(index=len(cases),parent=e['parent'],entry_index=e['index'],profile=profile))
    plan=dict(schema='ik-position-precision-single-variable-v1',status='predeclared_before_collection',
        source_selection_sha256=sha(PRIOR/'takeovers.json'),source_acquisition_sha256=sha(PRIOR/'acquisition.json'),entries=selection['entries'],cases=cases,
        source_sha256=sources,runner_sha256=sha(Path(__file__)),source_parents=prior['parents'],
        configs={p:dict(path=str(v),sha256=sha(v),expert_config=asdict(load_expert_config(v))) for p,v in configs.items()},
        treatment='Only IK position tolerance0.0002 to0.00002m. Same8mm floor and10step stationaryunload; rotationtol0.003, IK damping/seed, motiontiming/controller/targets remainconfiguredidentically.',
        gates=dict(first_three_second_wrist_peak_median_ratio_max=.8,full_lift_wrist_peak_median_ratio_max=.8,whole_active_arm_jerk_p99_median_ratio_max=.8,
            whole_arm_jerk_worst_ratio_max=1.05,first_second_wrist_peak_worst_ratio_max=1.05,duration_median_ratio_max=1.10,
            idle_total_median_added_seconds_max=.10,idle_max_worst_added_seconds_max=.10,
            box_to_containment_median_ratio_max=1.10,individual_active_arm_jerk_p99_ratio_max=1.10,
            all16_success_structural_transport_coverage=True,minimum_pairs_with_20pct_full_lift_improvement=6),
        metrics='50Hz recorded physical states, third difference /0.02^3. Full lift phase requires all4 samples inlift; wholeepisode metrics retained. First3s avoids known2s cutoff but never replaces fullphase/fulltrajectory checks. Lowmotion >=0.5s excludes stable-successhold; does not establishpolicyintent.',
        boundary='Eight existing trainingparents already used for expert diagnosis; no new independent or heldout layout evidence. Fresh paired deterministicteachers, alternatingorder, allfailures retained, noretries/replacements. No new learner inference duringround1.',
        training_condition='Only complete all gates, exactreplay/lineage/source audit and explicitreview permit new label acceptance/export and unchanged-batch/updates data-only controlled continuation. Failed originalunload gates stayfailed. New expert candidate pass alone doesnotpromote a VLA model.',
        stop='If first-round gatesfail, no training. No posthoc tolerance search or relaxed thresholds within thisexperiment; preservefailure and reassessthisroute.')
    write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    print('PREDECLARED',OUT,flush=True)
    return plan


def collect(plan):
    from pi0_repro.recovery import collect_recovery
    assert not (OUT/'candidates.json').exists()
    ledger=dict(schema='ik-position-precision-candidates-v1',complete=False,plan_sha256=sha(OUT/'plan.json'),results=[])
    write(OUT/'candidates.json',ledger)
    for case in plan['cases']:
        assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
        e=plan['entries'][case['entry_index']];cfg=plan['configs'][case['profile']];assert sha(Path(cfg['path']))==cfg['sha256']
        parent=next(p for p in plan['source_parents'] if p['parent']==e['parent']);source=Path(e['source_episode'])
        assert sha(source/'episode.json')==parent['source_metadata_sha256'] and sha(source/'trajectory.npz')==parent['source_trajectory_sha256']
        row=dict(case,role=e['role'],step=e['step'],source_layout_id=e['source_layout_id']);ledger['results'].append(row);write(OUT/'candidates.json',ledger)
        print('START',case['parent'],case['profile'],flush=True)
        existing={p.name for p in (ROOT/'data/banana_demos').iterdir()}
        try:
            directory=collect_recovery(Path(cfg['path']),Path(e['rollout']),source,e['step']);meta=read(directory/'episode_000/episode.json')
            row.update(collection=str(directory),automatic_success=meta['automatic_success'],error=meta['error'],steps=meta['steps'])
            print('END',case['parent'],case['profile'],row['automatic_success'],row['error'],row['steps'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}';row['retained_partial_directories']=[str(p) for p in (ROOT/'data/banana_demos').iterdir() if p.name not in existing];raise
        finally:write(OUT/'candidates.json',ledger)
    ledger['complete']=True;write(OUT/'candidates.json',ledger);print('COLLECTION COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--collect-prepared',action='store_true');args=parser.parse_args()
    plan=read(OUT/'plan.json') if args.collect_prepared else prepare()
    if not args.prepare_only:collect(plan)
