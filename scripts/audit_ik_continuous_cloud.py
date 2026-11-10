"""Read-only cloud source/data/final-checkpoint audit for the matched pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/ik_continuous_20261029'
CODE = r'''
import hashlib,json,sys,time,tomllib
from pathlib import Path
root=Path('/root/pi0-repro');payload=json.load(sys.stdin);plan=payload['plan'];checks=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,condition):checks.append(dict(name=name,passed=bool(condition)))
for name,digest in plan['source_sha256'].items():check('source:'+name,sha(root/name)==digest)
check('upstream_train',sha(root/'openpi/scripts/train.py')==plan['upstream_train_sha256'])
check('initializer_norm',sha(root/plan['initial_checkpoint']/'assets/local/pi0_banana_train/norm_stats.json')==plan['norm_stats_sha256'])
finals=[]
for arm in plan['arms']:
 check('config:'+arm['arm'],sha(root/arm['config'])==arm['config_sha256'])
 check('sampler:'+arm['arm'],sha(root/arm['sampler'])==arm['sampler_sha256'])
 check('manifest:'+arm['arm'],sha(root/arm['dataset']/'dataset_manifest.json')==arm['dataset_manifest_sha256'])
 for name,digest in arm['dataset_files_sha256'].items():check('data:'+name,sha(root/name)==digest)
 if payload['final']:
  cfg=tomllib.loads((root/arm['config']).read_text());final=Path(cfg['paths']['checkpoint_base_dir'])/'pi0_banana_lora'/arm['experiment']/'999'
  check('committed_final:'+arm['arm'],(final/'_CHECKPOINT_METADATA').is_file() and (final/'params/_METADATA').is_file())
  check('final_norm:'+arm['arm'],sha(final/'assets/local/pi0_banana_train/norm_stats.json')==plan['norm_stats_sha256'])
  finals.append(dict(arm=arm['arm'],checkpoint=str(final),bytes=sum(p.stat().st_size for p in final.rglob('*') if p.is_file())))
status=json.loads((root/'outputs/ik_continuous_20261029/cloud_training/status.json').read_text())
if payload['final']:
 check('supervisor_completed',status['status']=='both_trained_pending_validation')
 check('two_trained_arms',len(status['arms'])==2 and all(a['status']=='trained_pending_validation' for a in status['arms']))
result=dict(schema='ik-cloud-immutable-training-audit-v1',at=time.time(),final=payload['final'],passed=all(c['passed'] for c in checks),checks=checks,finals=finals,status=status['status'],heldout_contents_read=False)
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads((OUT / 'training_plan_train_only.json').read_text())
    target = (ROOT / 'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', target)
    ssh = ['ssh', '-i', str(Path.home() / '.ssh/id_ed25519'), '-o', 'IdentitiesOnly=yes',
           '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'ConnectTimeout=10', '-o', 'IPQoS=none']
    command = ssh + [target, shlex.join(['/root/pi0-repro/openpi/.venv/bin/python', '-B', '-c', CODE])]
    p = subprocess.run(command, input=json.dumps(dict(plan=plan, final=not args.data_only)),
                       text=True, capture_output=True, timeout=60)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    result = json.loads(p.stdout)
    result['training_plan_sha256'] = hashlib.sha256((OUT / 'training_plan_train_only.json').read_bytes()).hexdigest()
    result['reader_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination = OUT / ('cloud_data_stability.json' if args.data_only else 'cloud_training_audit.json')
    assert not destination.exists(), 'Preserve previous audit'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], checks=len(result['checks']), status=result['status'],
                         failed_checks=[c for c in result['checks'] if not c['passed']], finals=result['finals']), indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
