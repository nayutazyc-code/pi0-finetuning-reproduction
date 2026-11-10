"""Prepare two v6-initialized pure-FM1000-update arms; no deployment or training."""
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'outputs/v6_progress_20261028';prepared=read(out/'data_preparation.json')
    assert prepared['status']=='exported_pending_training_preflight'
    base=ROOT/prepared['base_dataset'];dataset=ROOT/prepared['dataset'];raw=Path(prepared['raw_merged'])
    before=read(base/'local/pi0_banana_train/source_episodes.json')['episodes'];rows=read(dataset/'local/pi0_banana_train/source_episodes.json')['episodes']
    assert len(before)==22 and len(rows)==33
    for a,b in zip(before,rows[:22],strict=True):assert all(a[k]==b[k] for k in ('layout_id','frames','source_trajectory_sha256','split'))
    for split in ('validation','test'):
        keys=lambda p:[(r['layout_id'],r['source_trajectory_sha256'],r['frames']) for r in read(p/f'local/pi0_banana_{split}/source_episodes.json')['episodes']]
        assert keys(base)==keys(dataset)
    base_sampler=ROOT/'assets_quality/banana_sampling.json';old=np.asarray(read(base_sampler)['weights'],float)
    assert len(old)==sum(r['frames'] for r in before)
    infos=[]
    allowed={r['layout_id'] for r in before}
    for row in rows[22:]:
        ep=raw/row['source_episode'];m=read(ep/'episode.json');rec=m['recovery']
        assert m['review_status']=='accepted' and m['eligible_for_training'] and rec['source_split']=='train'
        assert rec['source_layout_id'] in allowed and rec['learner_checkpoint'].endswith('/banana_v6_long100_accum8000/7999')
        assert rec['replay_state_max_error']<=1e-5 and not rec['independent_layout']
        assert sha(ep/'trajectory.npz')==row['source_trajectory_sha256']
        infos.append((row,rec['source_layout_id']))
    total=sum(r['frames'] for r in rows);parents={p for _,p in infos};weights=(old/old.sum()*.8*total).tolist()
    for row,parent in infos:
        siblings=sum(p==parent for _,p in infos)
        with np.load(raw/row['source_episode']/'trajectory.npz',allow_pickle=False) as d:
            # Explicit precision weighting only within new recovery mass; base
            # relative weights retain exactly80%. No new model loss term.
            local=np.where(np.isin(d['phase'],['descend','close_gripper','lower_into_box','release']),2.,1.)
        weights.extend((local/local.sum()*.2/len(parents)/siblings*total).tolist())
    assert len(weights)==total and min(weights)>0 and np.isfinite(weights).all()
    sampler=ROOT/'assets_v6_progress/banana_sampling.json';sampler.parent.mkdir(exist_ok=False)
    sampler.write_text(json.dumps({'repo_id':'local/pi0_banana_train','split':'train','weights':weights,
        'dataset_manifest_sha256':sha(dataset/'dataset_manifest.json'),'group_target_mass':{'base':.8,'new_recovery':.2},
        'new_recovery_segments':len(infos),'new_recovery_parent_layouts':len(parents),
        'method':'Preserve old relative weights at80%; new20% balanced by original train parent and sibling state, precision phase2x within segment. Not independent spatial coverage.'},indent=2)+'\n')
    original=(ROOT/'outputs/long_chunk_20261025/selected_training.toml').read_text();original=original[original.index('[data]'):]
    substitutions={'num_train_steps = 8000':'num_train_steps = 1000','seed = 46':'seed = 54','warmup_steps = 400':'warmup_steps = 100',
        'peak_lr = 0.00002':'peak_lr = 0.000005','decay_lr = 0.000002':'decay_lr = 0.000001','save_interval = 1000':'save_interval = 500',
        'keep_period = 2000':'keep_period = 1000000','banana_v5_quality_norm1000/999':'banana_v6_long100_accum8000/7999',
        'assets_base_dir = "assets_quality_norm"':'assets_base_dir = "assets_v6_progress_norm"'}
    for a,b in substitutions.items():assert original.count(a)==1,a;original=original.replace(a,b)
    arms=[]
    for label,ds,sam,volume in [('control',base,base_sampler,'/root/pi0-progress-control-checkpoints'),('recovery',dataset,sampler,'/root/pi0-progress-recovery-checkpoints')]:
        text=original.replace('directory = "data/lerobot/20261024-185715-merged-ebdbe3"',f'directory = "{ds.relative_to(ROOT)}"').replace('weights_file = "assets_quality/banana_sampling.json"',f'weights_file = "{sam.relative_to(ROOT)}"').replace('checkpoint_base_dir = "checkpoints"',f'checkpoint_base_dir = "{volume}"')
        cfg=ROOT/f'configs/banana_finetune_v6_progress_{label}.toml';assert not cfg.exists();cfg.write_text('# Matched original-pi0 v6 continuation; same objective/effective16/1000updates.\n'+text)
        mapping=read(ds/'local/pi0_banana_train/source_episodes.json')['episodes']
        repo=ds/'local/pi0_banana_train'
        files=[p for p in repo.rglob('*') if p.is_file() and
               (p.relative_to(repo).parts[0] in ('data','meta') or p.name in ('source_episodes.json','conversion_audit.json'))]
        for split in ('validation','test'):files.append(ds/f'local/pi0_banana_{split}/source_episodes.json')
        arms.append({'arm':label,'experiment':f'banana_v6_progress_{label}1000','config':str(cfg.relative_to(ROOT)),'config_sha256':sha(cfg),
            'dataset':str(ds.relative_to(ROOT)),'dataset_manifest_sha256':sha(ds/'dataset_manifest.json'),
            'sampler':str(sam.relative_to(ROOT)),'sampler_sha256':sha(sam),'train_episodes':len(mapping),'train_frames':sum(r['frames'] for r in mapping),
            'dataset_files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}})
    sources=['server/banana_model.py','server/banana_config.py','server/banana_accumulation.py','server/banana_sampling.py','server/banana_temporal.py','server/banana_temporal_data.py','server/banana_finetune.py','server/banana_probe.py','server/banana_roi.py','server/motion_norm_audit.py','server/temporal_preflight.py','server/progress_train_job.py','server/progress_policy_select.py']
    plan={'schema':'v6-progress-matched-training-v1','status':'prepared_not_deployed','initial_checkpoint':'checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999',
        'norm_stats_sha256':'6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269','arms':arms,
        'source_sha256':{n:sha(ROOT/n) for n in sources},'preparation_sha256':sha(Path(__file__)),
        'intervention':'New reviewed v6 recovery states plus explicit parent-balanced20% sampling mass; not an isolated change of data membership alone.',
        'held_fixed':'Original pi0 H100, same v6 initializer and norm, weightedFM with no added temporal/near-term loss, seed54, micro8xaccum2 effective16,1000updates,warm100,LR5e-6→1e-6,unchangedcameraROI/plant.',
        'candidate_counts':{'initial_attempts':11,'initial_auto_successes':10,'separate_reapproach_attempts':1,'accepted_new_segments':11,'rejected_retained':1,'new_independent_layouts':0,'new_recovery_parent_layouts':len(parents)},
        'disk_policy':'Separate root/data filesystem checkpoints; two initializer-sized snapshots plus2GiB reserve on each active volume, no old deletions.',
        'evaluation':'After completion, compare v6/control/recovery on the same6validation layouts withRTC20 and motion/phase coverage primary. No automatic promotion; preserve every failure/network wait. Test split unused.'}
    path=out/'training_plan.json';assert not path.exists();path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k not in ('arms','source_sha256')},indent=2));print('ARMS',[(a['arm'],a['train_episodes'],a['train_frames']) for a in arms])


if __name__=='__main__':main()
