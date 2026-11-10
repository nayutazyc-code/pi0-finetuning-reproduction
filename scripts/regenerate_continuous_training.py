"""Conditionally regenerate original20 train demonstrations with the combined expert."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
os.environ.setdefault('MUJOCO_GL','glfw')
from experiment_ik_continuous import OUT
from evaluate_v6_progress import read,sha,write


def prepare():
    from dataclasses import asdict
    from pi0_repro.banana_expert import ExpertConfig,load_expert_config
    base=ROOT/'data/lerobot/20261024-185715-merged-ebdbe3';raw=Path(read(base/'dataset_manifest.json')['source_path'])
    rows=read(base/'local/pi0_banana_train/source_episodes.json')['episodes'];entries=[];original_configs=[]
    for r in rows:
        p=raw/r['source_episode'];meta=read(p/'episode.json');layout=read(p/'layout.json');rec=meta.get('recovery')
        assert meta['split']=='train' and meta['eligible_for_training'] and meta['review_status']=='accepted'
        assert sha(p/'trajectory.npz')==r['source_trajectory_sha256']
        if rec:continue
        original=Path(meta['merged_from']['batch']);c=read(original/'collection.json')['expert_config']
        original_configs.append(c)
        entries.append(dict(source_episode=str(p),source_metadata_sha256=sha(p/'episode.json'),trajectory_sha256=r['source_trajectory_sha256'],
            layout_id=meta['layout_id'],layout=layout['layout'],frames=meta['steps'],split='train'))
    assert len(entries)==20
    assert not (OUT/'relabel_plan.json').exists()
    groups=[]
    for e,old in zip(entries,original_configs,strict=True):
        group=next((i for i,g in enumerate(groups) if g['source_config']==old),None)
        if group is None:
            group=len(groups);profiles={}
            for name,config in [('original',old),('precision',dict(old,ik_position_tolerance=.00002,continuous_transport_radius=.01))]:
                path=OUT/f'full_{group:02d}_{name}.toml';path.write_text('[expert]\n'+'\n'.join(k+' = '+json.dumps(v) for k,v in config.items() if v is not None)+'\n')
                profiles[name]=dict(path=str(path),sha256=sha(path),config=asdict(load_expert_config(path)))
            a,b=[dict(profiles[p]['config']) for p in ('original','precision')]
            assert a['ik_position_tolerance']==ExpertConfig().ik_position_tolerance==.0015 and b['ik_position_tolerance']==.00002
            b['ik_position_tolerance']=a['ik_position_tolerance'];b['continuous_transport_radius']=a['continuous_transport_radius'];assert a==b
            groups.append(dict(source_config=old,profiles=profiles,compatibility_entry=len([x for x in entries if 'group' in x])))
        e['group']=group
    plan=dict(schema='original-train-continuous-regeneration-v1',status='conditional_on_combined_round2_pass',base_dataset=str(base),base_raw=str(raw),
        base_manifest_sha256=sha(base/'dataset_manifest.json'),base_train_mapping_sha256=sha(base/'local/pi0_banana_train/source_episodes.json'),entries=entries,
        config_groups=groups,source_sha256=read(OUT/'plan.json')['source_sha256'],script_sha256=sha(Path(__file__)),
        control='First mapped fulltrain episode in eachoriginal configgroup regenerated once with original defaults and must reproduce original recorded arrays exactly; otherwise stop beforecandidate generation.',
        intervention='Regenerate all20fulltrain layouts with positiontolerance0.0015→0.00002m and continuous_transport_radius0→0.01m. All other original config settings remainidentical, including approach/descend/rotation/gripper/retreat. Global cruise changes timing only in lift/transport/lower. No unload/clearance changes on ordinary episodes. Two existing base recoverysegments and all validation/test episodes remain unchanged.',
        gates=dict(all20_success_and_structural=True,median_whole_active_jerk_p99_ratio_max=.8,worst_whole_arm_jerk_ratio_max=1.10,median_duration_ratio_max=1.10,
            individual_active_jerk_p99_ratio_max=1.10,median_idle_total_added_seconds_max=.10,worst_idle_max_added_seconds_max=.10),
        scope='Same original20 trainlayouts; not new coverage or heldout evidence. Training data preparation within fixed combined expert experiment, no parameter search. Preserve allfailures; no replacement/resampling. Dataset gate stillrequired beforeacceptance/export.',
        sampling='Later controlledtraining preserves original per-layout/per-phase sampling probability mass; replace only original20 train demonstrations, retain2original recovery/heldoutdata. No additional recoverymix/sampler intervention.',
        training='Matched fresh v6-initialized control/candidate, same originalweightedFM H100 norm/seed/batch8xaccum2/effective16/1000optimizerupdates. Common final-onlysave schedule may avoid insufficient two-snapshotstorage; this doesnot change updatecount/objective.')
    write(OUT/'relabel_plan.json',plan);(OUT/'regeneration_source.py').write_bytes(Path(__file__).read_bytes());print('PREPARED conditional fulltrain regeneration')


def collect():
    import numpy as np
    from pi0_repro.demonstrations import collect_demos
    assert read(OUT/'aggregate.json')['quality_gate_passed'],'Combined round2 failed; no regeneration/training'
    plan=read(OUT/'relabel_plan.json');assert not (OUT/'relabel_candidates.json').exists()
    assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
    ledger=dict(schema='fulltrain-precision-candidates-v1',complete=False,plan_sha256=sha(OUT/'relabel_plan.json'),results=[])
    write(OUT/'relabel_candidates.json',ledger)
    ledger['legacy_compatibility']=[]
    for group in plan['config_groups']:
        first=plan['entries'][group['compatibility_entry']];control=collect_demos(Path(group['profiles']['original']['path']),layout_plan=[first['layout']],dataset_metadata={'kind':'legacy_generation_compatibility_check','source_split':'train'})
        original=Path(first['source_episode'])/'trajectory.npz';fresh=control/'episode_000/trajectory.npz'
        with np.load(original,allow_pickle=False) as a,np.load(fresh,allow_pickle=False) as b:
            same=all(np.array_equal(a[k],b[k]) for k in ('state','action','next_state','phase'))
        ledger['legacy_compatibility'].append(dict(collection=str(control),exact_arrays=same,source_trajectory_sha256=sha(original),fresh_trajectory_sha256=sha(fresh)));write(OUT/'relabel_candidates.json',ledger)
        assert same,'Current default generation differs from original; retain control and stop'
    print('LEGACY GENERATION EXACT',flush=True)
    for i,e in enumerate(plan['entries']):
        assert all(sha(ROOT/'src/pi0_repro'/n)==d for n,d in plan['source_sha256'].items())
        assert sha(Path(e['source_episode'])/'episode.json')==e['source_metadata_sha256']
        row=dict(index=i,layout_id=e['layout_id'],source_episode=e['source_episode']);ledger['results'].append(row);write(OUT/'relabel_candidates.json',ledger)
        print('RELABEL START',i,e['layout_id'],flush=True)
        existing={p.name for p in (ROOT/'data/banana_demos').iterdir()}
        try:
            profile=plan['config_groups'][e['group']]['profiles']['precision'];assert sha(Path(profile['path']))==profile['sha256']
            directory=collect_demos(Path(profile['path']),layout_plan=[e['layout']],dataset_metadata={'kind':'original_train_layout_continuous_regeneration','source_split':'train','source_episode':e['source_episode'],'source_trajectory_sha256':e['trajectory_sha256']})
            meta=read(directory/'episode_000/episode.json');row.update(collection=str(directory),automatic_success=meta['automatic_success'],error=meta['error'],steps=meta['steps'])
            print('RELABEL END',i,row['automatic_success'],row['error'],row['steps'],flush=True)
        except BaseException as exc:
            row['operational_error']=f'{type(exc).__name__}: {exc}';row['retained_partial_directories']=[str(p) for p in (ROOT/'data/banana_demos').iterdir() if p.name not in existing];raise
        finally:write(OUT/'relabel_candidates.json',ledger)
    ledger['complete']=True;write(OUT/'relabel_candidates.json',ledger);print('FULLTRAIN REGENERATION COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--collect',action='store_true');args=p.parse_args()
    if args.collect:collect()
    else:prepare()
