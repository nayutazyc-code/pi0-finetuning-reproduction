"""Upload only the reviewed candidate/train metadata, then dispatch its guard."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
REMOTE_ROOT='/root/pi0-repro'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--upload-only',action='store_true');args=parser.parse_args()
    out=ROOT/'outputs/v6_progress_20261028';plan=json.loads((out/'training_plan_train_only.json').read_text())
    assert plan['schema']=='v6-progress-matched-training-train-only-v2'
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    import re
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3']
    def remote(code,payload):
        r=subprocess.run(ssh+[target,shlex.join([REMOTE_ROOT+'/openpi/.venv/bin/python','-u','-c',code])],input=json.dumps(payload),capture_output=True,text=True,timeout=60)
        if r.returncode:raise RuntimeError(r.stderr[-4000:])
        return json.loads(r.stdout)
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for arm in plan['arms']:
        assert sha(ROOT/arm['config'])==arm['config_sha256']
        assert sha(ROOT/arm['sampler'])==arm['sampler_sha256']
        for name,digest in arm['dataset_files_sha256'].items():assert sha(ROOT/name)==digest,name
    code=r'''
import hashlib,json,os,shutil,sys
from pathlib import Path
root=Path('/root/pi0-repro');plan=json.load(sys.stdin)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=root/'outputs/v6_progress_20261028';out.mkdir(exist_ok=True)
assert not (out/'cloud_training/status.json').exists(),'Existing job retained'
norm=root/plan['initial_checkpoint']/'assets/local/pi0_banana_train/norm_stats.json'
assert sha(norm)==plan['norm_stats_sha256']
assert (root/plan['initial_checkpoint']/'_CHECKPOINT_METADATA').is_file()
for name,digest in plan['source_sha256'].items():
 if name not in ('server/progress_train_job.py','server/progress_policy_select.py'):
  assert sha(root/name)==digest,'Existing reused server source differs: '+name
control=plan['arms'][0];candidate=plan['arms'][1]
assert sha(root/control['dataset']/'dataset_manifest.json')==control['dataset_manifest_sha256']
assert sha(root/control['sampler'])==control['sampler_sha256']
for name,digest in control['dataset_files_sha256'].items():assert sha(root/name)==digest,'Existing base data mismatch: '+name
destination=root/candidate['dataset'];sentinel=destination/'.progress_manifest_expected'
if destination.exists():assert sentinel.is_file() and sentinel.read_text()==candidate['dataset_manifest_sha256'],'Unknown destination; refuse overwrite'
else:
 destination.mkdir(parents=True);sentinel.write_text(candidate['dataset_manifest_sha256'])
 # Reuse only immutable base parquet bytes. Rsync uses temporary rename, never
 # --inplace; a differing candidate file breaks the link, leaving base intact.
 base=root/control['dataset']/'local/pi0_banana_train'
 for source in base.glob('data/chunk-*/episode_*.parquet'):
  target=destination/'local/pi0_banana_train'/source.relative_to(base)
  target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
print(json.dumps({'status':'remote_base_and_sources_verified','candidate_destination':candidate['dataset'],'heldout_contents_uploaded':False}))
'''
    preparation=remote(code,plan);(out/'deployment_preflight.json').write_text(json.dumps(preparation,indent=2)+'\n')
    candidate=plan['arms'][1]
    files=list(plan['source_sha256'])+[arm['config'] for arm in plan['arms']]+[candidate['sampler'],str((out/'training_plan_train_only.json').relative_to(ROOT)),candidate['dataset']+'/dataset_manifest.json']
    files+=list(candidate['dataset_files_sha256'])
    filelist=out/'deployment_files.txt';filelist.write_text('\n'.join(sorted(set(files)))+'\n')
    # Omit -t: unchanged hard-linked base files retain their original timestamps.
    command=['rsync','-rl','--checksum','--partial','--relative','--files-from='+str(filelist),'-e',shlex.join(ssh),'./',target+':'+REMOTE_ROOT+'/']
    with (out/'upload.log').open('ab') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Upload failed; no training dispatched; see upload.log')
    verify=r'''
import hashlib,json,sys
from pathlib import Path
root=Path('/root/pi0-repro');plan=json.load(sys.stdin);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in plan['source_sha256'].items():assert sha(root/name)==digest,name
for arm in plan['arms']:
 assert sha(root/arm['config'])==arm['config_sha256'];assert sha(root/arm['sampler'])==arm['sampler_sha256']
 assert sha(root/arm['dataset']/'dataset_manifest.json')==arm['dataset_manifest_sha256']
 for name,digest in arm['dataset_files_sha256'].items():assert sha(root/name)==digest,name
print(json.dumps({'all_upload_hashes_verified':True,'base_data_unchanged':True}))
'''
    verified=remote(verify,plan);(out/'upload_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    if args.upload_only:print(json.dumps(verified));return
    dispatch=r'''
import json,subprocess
from pathlib import Path
root=Path('/root/pi0-repro');out=root/'outputs/v6_progress_20261028/cloud_training'
out.mkdir(exist_ok=True);assert not (out/'status.json').exists()
with (out/'supervisor.log').open('xb') as log:
 p=subprocess.Popen([str(root/'openpi/.venv/bin/python'),'-u',str(root/'server/progress_train_job.py'),'--root',str(root)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'status':'dispatched_preflight_not_yet_passed'}))
'''
    result=remote(dispatch,{});(out/'training_dispatch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
