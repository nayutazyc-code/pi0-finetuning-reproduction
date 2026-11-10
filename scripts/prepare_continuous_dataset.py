"""Explicitly review passing fulltrain labels, version replacements and export."""
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime
import uuid
import hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
OUT=ROOT/'outputs/ik_continuous_20261029'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    import numpy as np
    from pi0_repro.demo_review import review_demo
    from pi0_repro.lerobot_export import export_dataset
    assert read(OUT/'aggregate.json')['quality_gate_passed'] and read(OUT/'relabel_aggregate.json')['quality_gate_passed']
    receipt=read(OUT/'fulltrain_primary_review.json');assert receipt['approve_data_acceptance'] and receipt['relabel_aggregate_sha256']==sha(OUT/'relabel_aggregate.json')
    plan=read(OUT/'relabel_plan.json');ledger=read(OUT/'relabel_candidates.json');base=Path(plan['base_dataset']);oldraw=Path(plan['base_raw'])
    rows=read(base/'local/pi0_banana_train/source_episodes.json')['episodes'];replacements={}
    if (OUT/'data_preparation.json').exists():
        preparation=read(OUT/'data_preparation.json');assert preparation['status']=='reviewed_versioned_pending_export'
        raw=Path(preparation['raw_merged']);assert raw.is_dir()
    else:
        from pi0_repro.layouts import write_dataset_index
        for row in ledger['results']:
            ep=Path(row['collection'])/'episode_000';meta=read(ep/'episode.json');evidence=ep.parent/'precision_review.json'
            proof=dict(episode=meta['episode'],layout_id=meta['layout_id'],trajectory_sha256=sha(ep/'trajectory.npz'),decision='accepted',
                primary_review_sha256=sha(OUT/'fulltrain_primary_review.json'),quality_audit_sha256=sha(OUT/'relabel_audit.json'),reason='User-delegated review after fixed numerical gates, exact physical replay and sampled start/lift/transport/box/release/peak/final inspection. Only this train episode accepted.')
            write(evidence,proof);review_demo(ep,accept=True,reviewer='assistant',reason=proof['reason'],evidence_path=evidence)
            replacements[meta['layout_id']]=ep
        assert len(replacements)==20
        raw=ROOT/'data/banana_demos'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-ik-replaced-'+uuid.uuid4().hex[:6]);assert not raw.exists()
        shutil.copytree(oldraw,raw);collection=read(raw/'collection.json');episodes=[]
        for path in sorted(raw.glob('episode_*/episode.json')):
            previous=read(path);layout=previous['layout_id']
            if layout in replacements:
                source=replacements[layout];fresh=read(source/'episode.json');index=previous['episode'];target=path.parent
                # This directory belongs to the just-created versioned copy, not the
                # original dataset. No source episode is removed or overwritten.
                assert target.parent==raw and not target.samefile(oldraw/target.name)
                shutil.rmtree(target);shutil.copytree(source,target)
                evidence=read(source.parent/fresh['assistant_review']['evidence']);evidence['episode']=index
                evidence['merged_from']=dict(path=str(source.parent/fresh['assistant_review']['evidence']),sha256=sha(source.parent/fresh['assistant_review']['evidence']))
                proofpath=f'review_evidence/ik_episode_{index:03d}.json';write(raw/proofpath,evidence)
                fresh.update(episode=index,merged_from=dict(batch=str(source.parent),episode=0,episode_metadata_sha256=sha(source/'episode.json')),
                    replaced_from=dict(batch=str(oldraw),episode=index,trajectory_sha256=sha(oldraw/target.name/'trajectory.npz')))
                fresh['assistant_review']={**fresh['assistant_review'],'evidence':proofpath};write(target/'episode.json',fresh);episodes.append(fresh)
            else:
                assert sha(path)==sha(oldraw/path.relative_to(raw));assert sha(path.parent/'trajectory.npz')==sha(oldraw/path.parent.relative_to(raw)/'trajectory.npz');episodes.append(previous)
        collection.update(source='reviewed_train_label_replacement_v1',source_batches=[dict(source=str(oldraw),collection_sha256=sha(oldraw/'collection.json'))]+[dict(source=r['collection'],collection_sha256=sha(Path(r['collection'])/'collection.json')) for r in ledger['results']],
            replacement_plan_sha256=sha(OUT/'relabel_plan.json'),expert_labels_only=True,split_policy='Original preassigned memberships unchanged;20fulltrain replacements,2legacy recoveries/heldoutunchanged')
        write(raw/'collection.json',collection);write(raw/'summary.json',dict(source=collection['source'],episodes=episodes,accepted_demonstrations=sum(e['eligible_for_training'] for e in episodes)))
        write_dataset_index(raw)
        preparation=dict(status='reviewed_versioned_pending_export',raw_merged=str(raw),base_dataset=str(base),replaced_fulltrain=20,unchanged_recovery=2)
        write(OUT/'data_preparation.json',preparation)
    dataset=export_dataset(raw)
    newrows=read(dataset/'local/pi0_banana_train/source_episodes.json')['episodes'];assert len(newrows)==len(rows)==22
    oldw=np.asarray(read(ROOT/'assets_quality/banana_sampling.json')['weights'],float);start=0;neww=[];mass_checks=[]
    for old,new in zip(rows,newrows,strict=True):
        assert old['layout_id']==new['layout_id'] and old['split']==new['split']=='train'
        with np.load(oldraw/old['source_episode']/'trajectory.npz') as d:oldphase=d['phase']
        with np.load(raw/new['source_episode']/'trajectory.npz') as d:newphase=d['phase']
        span=oldw[start:start+old['frames']];start+=old['frames'];weights=np.empty(new['frames'])
        assert set(oldphase)==set(newphase)
        for phase in np.unique(oldphase):
            a=oldphase==phase;b=newphase==phase;assert len(np.unique(span[a]))==1
            mass=float(span[a].sum());weights[b]=mass/int(b.sum());mass_checks.append(dict(layout_id=old['layout_id'],phase=str(phase),old_mass=mass,new_mass=float(weights[b].sum())))
        neww.extend(weights.tolist())
    assert start==len(oldw) and len(neww)==sum(r['frames'] for r in newrows) and np.isfinite(neww).all() and min(neww)>0
    assert all(np.isclose(c['old_mass'],c['new_mass'],rtol=1e-12,atol=1e-12) for c in mass_checks)
    assets=ROOT/'assets_ik_continuous';assets.mkdir(exist_ok=False);sampler=assets/'banana_sampling.json'
    write(sampler,dict(repo_id='local/pi0_banana_train',split='train',weights=neww,dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),
        method='Preserve original per-layout/per-phase probability mass exactly; original within-phase weights are constant. No new recoverymix or spatialmembership.',mass_checks=mass_checks))
    heldout={}
    for split in ('validation','test'):
        fields=lambda p:[(r['layout_id'],r['source_trajectory_sha256'],r['frames'],r['split']) for r in read(p/f'local/pi0_banana_{split}/source_episodes.json')['episodes']]
        a=fields(base);b=fields(dataset);assert a==b
        import hashlib
        digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
        heldout[split]=dict(base_digest=digest(a),candidate_digest=digest(b),episodes=len(a))
    for old,new in zip(rows[-2:],newrows[-2:],strict=True):assert old['layout_id']==new['layout_id'] and old['source_trajectory_sha256']==new['source_trajectory_sha256'] and old['frames']==new['frames']
    preparation.update(status='exported_pending_training_preflight',dataset=str(dataset),sampler=str(sampler),dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),sampler_sha256=sha(sampler),train_episodes=22,train_frames=sum(r['frames'] for r in newrows),
        heldout_lineage_local_verification=dict(passed=True,**heldout),sampling_phase_masses_preserved=True)
    write(OUT/'data_preparation.json',preparation);(OUT/'data_preparation_source.py').write_bytes(Path(__file__).read_bytes());print('PREPARED',json.dumps(preparation,indent=2))


if __name__=='__main__':main()
