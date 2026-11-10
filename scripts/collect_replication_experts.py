"""Paired same-state clearance/unload teachers; no acceptance/export/training."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','glfw')
from replicate_contact_train import OUT
from evaluate_v6_progress import read,sha,write


def main():
    from pi0_repro.recovery import collect_recovery
    from pi0_repro.banana_expert import load_expert_config
    from dataclasses import asdict
    plan=read(OUT/'plan.json');selection=read(OUT/'takeovers.json');assert selection['complete']
    assert not (OUT/'teacher_candidates.json').exists()
    configs={k:Path(v['path']) for k,v in plan['expert_configs'].items()}
    for k,p in configs.items():assert sha(p)==plan['expert_configs'][k]['sha256']
    a,b=[asdict(load_expert_config(configs[k])) for k in ('clearance','unload')]
    assert a['recovery_unload_steps']==0 and b['recovery_unload_steps']==10
    b['recovery_unload_steps']=0;assert a==b,'Only the stationary-unload setting may differ'
    cases=[]
    for e in selection['entries']:
        for profile in (('clearance','unload') if e['parent']%2==0 else ('unload','clearance')):
            cases.append(dict(index=len(cases),parent=e['parent'],takeover_index=e['index'],profile=profile))
    auxiliary={'layouts.py':sha(ROOT/'src/pi0_repro/layouts.py')}
    (OUT/'frozen_source/layouts.py').write_bytes((ROOT/'src/pi0_repro/layouts.py').read_bytes())
    report=dict(schema='eight-train-paired-teacher-candidates-v1',complete=False,plan_sha256=sha(OUT/'plan.json'),takeovers_sha256=sha(OUT/'takeovers.json'),auxiliary_source_sha256=auxiliary,
        script_sha256=sha(Path(__file__)),cases=cases,results=[],scope='Same-state scriptedteacher pairs, not paired stochastic learned policies. Alltaskfailures/missingstates retained. No acceptance/export/training.')
    write(OUT/'teacher_candidates.json',report);(OUT/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    for case in cases:
        e=selection['entries'][case['takeover_index']]
        assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
        assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in auxiliary.items())
        row=dict(case,role=e['role'],source_episode=e['source_episode'],rollout=e['rollout'],step=e['step'],source_split='train')
        report['results'].append(row);write(OUT/'teacher_candidates.json',report)
        print('START',case['parent'],case['profile'],'takeover',e['step'],e['role'],flush=True)
        existing={p.name for p in (ROOT/'data/banana_demos').iterdir()}
        try:
            directory=collect_recovery(configs[case['profile']],Path(e['rollout']),Path(e['source_episode']),e['step'])
            meta=read(directory/'episode_000/episode.json')
            row.update(collection=str(directory),automatic_success=meta['automatic_success'],error=meta.get('error'),steps=meta['steps'])
            print('END',case['parent'],case['profile'],row['automatic_success'],row['error'],row['steps'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}'
            row['retained_partial_directories']=[str(p) for p in (ROOT/'data/banana_demos').iterdir() if p.name not in existing]
            raise
        finally:write(OUT/'teacher_candidates.json',report)
    report['complete']=True;write(OUT/'teacher_candidates.json',report)
    print('COMPLETE',len(report['results']),'teacher candidates; pending qualitygate/review')


if __name__=='__main__':main()
