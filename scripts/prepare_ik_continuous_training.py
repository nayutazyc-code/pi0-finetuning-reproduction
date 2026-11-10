"""Bind the reviewed exported data to two matched original-pi01000-update arms."""
from pathlib import Path
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts')]
from experiment_ik_continuous import OUT
from evaluate_v6_progress import read,sha,write


def main():
    prepared=read(OUT/'data_preparation.json');assert prepared['status']=='exported_pending_training_preflight'
    assert read(OUT/'relabel_aggregate.json')['quality_gate_passed'] and prepared['sampling_phase_masses_preserved']
    assert not (OUT/'training_plan_train_only.json').exists()
    base=Path(prepared['base_dataset']);candidate=Path(prepared['dataset']);arms=[]
    original=(ROOT/'configs/banana_finetune_v6_progress_control.toml').read_text()
    assert original.count('save_interval = 500')==1 and original.count('assets_v6_progress_norm')==1
    original=original.replace('save_interval = 500','save_interval = 1000').replace('assets_v6_progress_norm','assets_ik_continuous_norm')
    for label,ds,sam,volume in [('control',base,ROOT/'assets_quality/banana_sampling.json','/root/pi0-ik-control-checkpoints'),('candidate',candidate,Path(prepared['sampler']),'/root/pi0-ik-continuous-checkpoints')]:
        text=original.replace('data/lerobot/20261024-185715-merged-ebdbe3',str(ds.relative_to(ROOT))).replace('assets_quality/banana_sampling.json',str(sam.relative_to(ROOT))).replace('/root/pi0-progress-control-checkpoints',volume)
        cfg=OUT/f'train_{label}.toml';cfg.write_text(text);settings=tomllib.loads(text)
        assert settings['train']['batch_size']==8 and settings['train']['gradient_accumulation_steps']==2 and settings['train']['num_train_steps']==1000 and settings['train']['save_interval']==1000
        repo=ds/'local/pi0_banana_train';rows=read(repo/'source_episodes.json')['episodes']
        assert len(rows)==22 and all(r['split']=='train' for r in rows)
        files=[p for p in repo.rglob('*') if p.is_file() and (p.relative_to(repo).parts[0] in ('data','meta') or p.name in ('source_episodes.json','conversion_audit.json'))]
        arms.append(dict(arm=label,experiment=f'banana_v6_ik_{label}1000',config=str(cfg.relative_to(ROOT)),config_sha256=sha(cfg),
            dataset=str(ds.relative_to(ROOT)),dataset_manifest_sha256=sha(ds/'dataset_manifest.json'),sampler=str(sam.relative_to(ROOT)),sampler_sha256=sha(sam),
            train_episodes=22,train_frames=sum(r['frames'] for r in rows),dataset_files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
    source_names=list(read(ROOT/'outputs/v6_progress_20261028/training_plan_train_only.json')['source_sha256'])+[str((OUT/n).relative_to(ROOT)) for n in ('cloud_training_job.py','cloud_policy_select.py')]
    plan=dict(schema='v6-ik-continuous-training-train-only-v1',status='prepared_not_deployed',initial_checkpoint='checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999',
        norm_stats_sha256='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269',arms=arms,
        source_sha256={n:sha(ROOT/n) for n in source_names},upstream_train_sha256=sha(ROOT/'repos/openpi/scripts/train.py'),
        heldout_lineage_local_verification=prepared['heldout_lineage_local_verification'],data_preparation_sha256=sha(OUT/'data_preparation.json'),review_sha256=sha(OUT/'fulltrain_primary_review.json'),
        intervention='Replace20fulltrain expert demonstrations with tighterIK+continuoustransit, preserve same layouts/splits/2legacyrecoveries and exact original per-layout/per-phase samplingprobability mass. No addedloss/runtimeintervention.',
        held_fixed='Fresh originalpi0 H100 v6 initializer/norm, pureweightedFM, strictLoRA+actionprojections, ROI, seed54, batch8xaccum2 effective16,1000updates,warm100,LR5e-6→1e-6. Bothfinal-onlysave1000, nointermediate500; save schedulingdoesnotchangeoptimizerupdates.',
        disk_policy='Different root/data volumes, each one final checkpoint with20pct initializer-size margin plus2GiBreserve; no olddeletions. Guard duringjob; partialfailure retained.',
        next='Aftercommitted finals and restoredv6metadata, freshsame-session learned-policy comparison; expertpass doesnotpromote model.')
    write(OUT/'training_plan_train_only.json',plan);(OUT/'training_preparer_source.py').write_bytes(Path(__file__).read_bytes());print('TRAINING PREPARED',[(a['arm'],a['train_frames']) for a in arms])


if __name__=='__main__':main()
