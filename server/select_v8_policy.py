"""Select one committed evaluation policy, preserving all checkpoints and training logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--model',choices=('v6','control','temporal'),required=True)
args=parser.parse_args();root=args.root.resolve();out=root/'outputs/v8_temporal_20261026'
if args.model=='v6':
    config=root/'outputs/long_chunk_20261025/selected_training.toml'
    checkpoint=root/'checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999'
else:
    status=json.loads((out/'status.json').read_text())
    if status['status']!='both_trained_pending_validation':raise RuntimeError('Training has not completed')
    config=root/f'configs/banana_finetune_v8_{args.model}.toml'
    checkpoint=root/f'checkpoints/pi0_banana_lora/banana_v8_{args.model}32_3000/2999'
if not (checkpoint/'params/_METADATA').is_file() or not (checkpoint/'_CHECKPOINT_METADATA').is_file():
    raise ValueError('Checkpoint not committed')
digest=hashlib.sha256((checkpoint/'assets/local/pi0_banana_train/norm_stats.json').read_bytes()).hexdigest()
if digest!='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269':raise ValueError('Unexpected normalization')
policies=[]
for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        parts=cmdline.read_bytes().split(b'\0')
        if str(root/'server/banana_finetune.py').encode() in parts:raise RuntimeError('Training process remains active')
        if str(root/'server/serve_pi0.py').encode() in parts:policies.append(int(cmdline.parent.name))
    except OSError:pass
if len(policies)>1:raise RuntimeError('Multiple policy processes; do not guess ownership')
for pid in policies:os.kill(pid,signal.SIGTERM)
for _ in range(30):
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if used<2048:break
    time.sleep(1)
if used>=2048:raise RuntimeError('GPU allocations remain after stopping selected policy')
env=dict(os.environ);env.update(PI0_RTC='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (out/f'validation_{args.model}_serve.log').open('ab') as log:
    p=subprocess.Popen(['bash',str(root/'server/run_banana_serve.sh'),str(config),str(checkpoint),'10'],
        cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'model':args.model,'pid':p.pid,'checkpoint':str(checkpoint),'started_at_unix':time.time(),
        'norm_stats_sha256':digest,'scope':'Dispatch only; verify metadata separately.'}
(out/'selected_policy_process.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
