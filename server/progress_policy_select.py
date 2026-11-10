"""Select committed progress-pilot policy; never change training or delete weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import tomllib


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--model',choices=('v6','control','recovery'),required=True)
    args=parser.parse_args();root=args.root.resolve();out=root/'outputs/v6_progress_20261028';plan=json.loads((out/'training_plan_train_only.json').read_text())
    registry={'v6':(root/'outputs/long_chunk_20261025/selected_training.toml',root/plan['initial_checkpoint'])}
    for arm in plan['arms']:
        config=root/arm['config'];s=tomllib.loads(config.read_text())
        registry[arm['arm']]=(config,Path(s['paths']['checkpoint_base_dir'])/'pi0_banana_lora'/arm['experiment']/'999')
    if args.model!='v6':
        status=json.loads((out/'cloud_training/status.json').read_text());assert status['status']=='both_trained_pending_validation'
    config,checkpoint=registry[args.model]
    assert (checkpoint/'_CHECKPOINT_METADATA').is_file() and (checkpoint/'params/_METADATA').is_file()
    assert hashlib.sha256((checkpoint/'assets/local/pi0_banana_train/norm_stats.json').read_bytes()).hexdigest()==plan['norm_stats_sha256']
    policies=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            cmd=[x.decode() for x in path.read_bytes().split(b'\0') if x]
            assert str(root/'server/banana_finetune.py') not in cmd,'Training process active'
            if str(root/'server/serve_pi0.py') in cmd:
                assert cmd[cmd.index('--checkpoint')+1] in {str(v[1]) for v in registry.values()},'Unexpected policy ownership'
                policies.append(int(path.parent.name))
        except OSError:pass
    assert len(policies)<=1,'Multiple policies; do not guess ownership'
    for pid in policies:os.kill(pid,signal.SIGTERM)
    for _ in range(30):
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if used<2048:break
        time.sleep(1)
    assert used<2048,'Other GPU allocations remain'
    env=dict(os.environ);env.update(PI0_RTC='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    with (out/f'eval_{args.model}_serve.log').open('ab') as log:
        p=subprocess.Popen(['bash',str(root/'server/run_banana_serve.sh'),str(config),str(checkpoint),'10'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'model':args.model,'pid':p.pid,'checkpoint':str(checkpoint),'started_at':time.time(),'status':'dispatched_metadata_not_yet_verified'}
    with (out/'policy_transitions.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
