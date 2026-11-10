"""Second bounded expert experiment: precision plus continuous high-corner transit."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from evaluate_v6_progress import read,sha,write
FIRST=ROOT/'outputs/ik_precision_20261029'
OUT=ROOT/'outputs/ik_continuous_20261029'


def prepare():
    from pi0_repro.banana_expert import load_expert_config
    old=read(FIRST/'plan.json');result=read(FIRST/'aggregate.json');assert result['audit_passed'] and not result['quality_gate_passed']
    assert not OUT.exists();OUT.mkdir();frozen=OUT/'frozen_source';frozen.mkdir()
    sources={n:sha(ROOT/'src/pi0_repro'/n) for n in [*old['source_sha256'],'expert_path.py']}
    for n,d in old['source_sha256'].items():
        if n!='banana_expert.py':assert sources[n]==d
    for n in sources:(frozen/n).write_bytes((ROOT/'src/pi0_repro'/n).read_bytes())
    configs={}
    for name,source,radius in [('baseline',FIRST/'baseline.toml',0),('tight_only',FIRST/'tight_ik.toml',0),('continuous',FIRST/'tight_ik.toml',.01)]:
        path=OUT/(name+'.toml');path.write_text(source.read_text()+f'\ncontinuous_transport_radius = {radius}\n')
        configs[name]=dict(path=str(path),sha256=sha(path),expert_config=asdict(load_expert_config(path)))
    a,b=[dict(configs[p]['expert_config']) for p in ('tight_only','continuous')];b['continuous_transport_radius']=0;assert a==b
    plan=dict(schema='precision-continuous-expert-v1',status='predeclared_before_collection',entries=old['entries'],source_parents=old['source_parents'],
        first_plan_sha256=sha(FIRST/'plan.json'),first_audit_sha256=sha(FIRST/'audit.json'),first_aggregate_sha256=sha(FIRST/'aggregate.json'),
        source_sha256=sources,runner_sha256=sha(Path(__file__)),configs=configs,gates=old['gates'],
        references={p:result['groups'][p]['records'] for p in ('baseline','tight_ik')},
        treatment='Retain IKpositiontol0.00002m, same8mm floor/unload10. Add opt-in0.01m C2 virtual-distance corner joins through lift/transport/lower, smooth0.5s velocity ramps and cruise <=min(existing precision/transit cartesian speed). Deliberate grasp/release/retreat remain existing paths.',
        attribution='Round1 isolated IK precision with strong jerk reduction but failed2lowmotion gates. Round2 changes only transit mode relativetight-only; original0.0002baseline and tight-only recorded references retained. No original gate is waived; same12gate dictionary vsoriginalbaseline applies tocombined candidate.',
        reference_proof='Regenerate parent0 baseline and tight-only once with newmodeoff; arrays must exactlymatch first-round frozenreferences before anycandidate. Existing8physical reference replay audits remainbound, not claimed freshmodel trials.',
        scope='Same8dependent trainingstates, not heldout/generalization evidence. First-round failure retained; no new tolerance search, no fallback or runtime feature enabled.',
        stop='Failed combined gates forbid acceptance/export/training. Any later data preparation needs its own bound plan; first conditional fulltrain regeneration remainsnotrun.')
    write(OUT/'plan.json',plan);(OUT/'runner_source.py').write_bytes(Path(__file__).read_bytes());print('PREDECLARED',OUT,flush=True)


def collect():
    import numpy as np
    from pi0_repro.recovery import collect_recovery
    plan=read(OUT/'plan.json');assert not (OUT/'candidates.json').exists()
    ledger=dict(schema='precision-continuous-candidates-v1',complete=False,plan_sha256=sha(OUT/'plan.json'),compatibility=[],results=[]);write(OUT/'candidates.json',ledger)
    for profile,refprofile in [('baseline','baseline'),('tight_only','tight_ik')]:
        e=plan['entries'][0];ref=plan['references'][refprofile][0]
        print('COMPATIBILITY',profile,flush=True)
        directory=collect_recovery(Path(plan['configs'][profile]['path']),Path(e['rollout']),Path(e['source_episode']),e['step'])
        with np.load(directory/'episode_000/trajectory.npz',allow_pickle=False) as a,np.load(Path(ref['collection'])/'episode_000/trajectory.npz',allow_pickle=False) as b:
            same=all(np.array_equal(a[k],b[k]) for k in ('state','action','next_state','phase'))
        ledger['compatibility'].append(dict(profile=profile,collection=str(directory),exact_arrays=same));write(OUT/'candidates.json',ledger);assert same,'Disabled mode changed generation; preserve control and stop'
    for e in plan['entries']:
        assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
        cfg=plan['configs']['continuous'];assert sha(Path(cfg['path']))==cfg['sha256']
        row=dict(index=len(ledger['results']),parent=e['parent'],entry_index=e['index'],profile='continuous',source_layout_id=e['source_layout_id']);ledger['results'].append(row);write(OUT/'candidates.json',ledger)
        print('START',e['parent'],'continuous',flush=True)
        existing={p.name for p in (ROOT/'data/banana_demos').iterdir()}
        try:
            directory=collect_recovery(Path(cfg['path']),Path(e['rollout']),Path(e['source_episode']),e['step']);meta=read(directory/'episode_000/episode.json')
            row.update(collection=str(directory),automatic_success=meta['automatic_success'],error=meta['error'],steps=meta['steps'])
            print('END',e['parent'],row['automatic_success'],row['error'],row['steps'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}';row['retained_partial_directories']=[str(p) for p in (ROOT/'data/banana_demos').iterdir() if p.name not in existing];raise
        finally:write(OUT/'candidates.json',ledger)
    ledger['complete']=True;write(OUT/'candidates.json',ledger);print('COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.collect:collect()
    else:prepare()
