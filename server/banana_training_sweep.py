"""Measure microbatches in isolated GPU processes, then launch a versioned run."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exp-name',default='banana_v6_long100_accum8000')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.exp_name):raise ValueError('Invalid experiment name')
    interpreter=root/'openpi/.venv/bin/python'
    env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled',
             HF_HOME=str(root/'cache/huggingface'),OPENPI_DATA_HOME=str(root/'cache/openpi'),XDG_CACHE_HOME=str(root/'cache'))
    record=dict(status='diagnosing',created_at=datetime.now().astimezone().isoformat(),
                action_horizon=100,effective_batch=16,optimizer_updates=8000,results=[],reserve_mib=2048)
    def save():(out/'sweep.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    diagnostic=[str(interpreter),str(root/'server/banana_accumulation_diagnostic.py'),str(out/'accumulation-diagnostic.json')]
    with (out/'accumulation-diagnostic.log').open('w') as log:
        r=subprocess.run(diagnostic,cwd=root/'openpi',env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        record.update(status='blocked',reason='Accumulation numerical diagnostic failed');save();return
    text=args.config.read_text()
    for micro in (1,2,4,8):
        accum=16//micro;config=out/f'probe_micro{micro}.toml'
        candidate=re.sub(r'^batch_size\s*=.*$',f'batch_size = {micro}',text,flags=re.M)
        candidate=re.sub(r'^gradient_accumulation_steps\s*=.*$',f'gradient_accumulation_steps = {accum}',candidate,flags=re.M)
        config.write_text(candidate)
        result=out/f'probe_micro{micro}.json';log=out/f'probe_micro{micro}.log'
        record.update(status='probing',current_micro_batch=micro);save()
        command=[str(interpreter),'-u',str(root/'server/banana_finetune.py'),'probe','--root',str(root),
                 '--config',str(config),'--exp-name',f'{args.exp_name}_probe{micro}',
                 '--probe-updates','3','--probe-output',str(result)]
        with log.open('w') as stream:r=subprocess.run(command,cwd=root/'openpi',env=env,stdout=stream,stderr=subprocess.STDOUT)
        row=json.loads(result.read_text()) if result.exists() else dict(status='process_failed')
        row.update(micro_batch_size=micro,accumulation_steps=accum,exit_code=r.returncode,log=str(log))
        peak,total=row.get('gpu_peak_used_mib'),row.get('gpu_total_mib')
        row['eligible']=bool(r.returncode==0 and row.get('status')=='passed' and peak is not None
                             and total-peak>=record['reserve_mib'])
        record['results'].append(row);save()
        if r.returncode:
            tail=log.read_text(errors='replace')[-16000:]
            if 'RESOURCE_EXHAUSTED' in tail or 'out of memory' in tail.lower() or 'OOM' in tail:
                row['oom']=True;save();break
            record.update(status='blocked',reason='Non-memory probe failure; inspect logs');save();return
        if not row['eligible']:break
    eligible=[r for r in record['results'] if r['eligible']]
    if not eligible:
        record.update(status='blocked',reason='No finite probe retained 2 GiB GPU headroom');save();return
    best=min(eligible,key=lambda r:r['steady_update_seconds'])
    selected=out/'selected_training.toml'
    selected.write_text(re.sub(r'^gradient_accumulation_steps\s*=.*$',f'gradient_accumulation_steps = {best["accumulation_steps"]}',
        re.sub(r'^batch_size\s*=.*$',f'batch_size = {best["micro_batch_size"]}',text,flags=re.M),flags=re.M))
    record.update(status='selected',selected_micro_batch=best['micro_batch_size'],
        selected_accumulation_steps=best['accumulation_steps'],selected_config=str(selected),
        estimated_training_hours=best['estimated_training_hours']);save()
    if args.launch:
        record.update(status='training',training_log=str(out/'training.log'));save()
        command=['bash',str(root/'server/run_banana_train.sh'),args.exp_name,'--config',str(selected)]
        with (out/'training.log').open('w') as stream:r=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status='completed' if r.returncode==0 else 'training_failed',training_exit_code=r.returncode);save()


if __name__=='__main__':main()
