"""Guarded sequential matched original-pi0 training; restore v6 after both arms/failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import tomllib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    out=root/'outputs/v8_temporal_20261026';plan=json.loads((out/'plan.json').read_text())
    # A held process lock prevents two invocations racing between preflight and training.
    import fcntl
    lock=(out/'job.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json'
    if status_path.exists():
        raise FileExistsError('Retain existing job; inspect or use an explicitly versioned new experiment')
    for item in plan['arms']:
        if (root/'checkpoints/pi0_banana_lora'/item['experiment']).exists():
            raise FileExistsError('Candidate checkpoint directory already exists')
    status={'pid':os.getpid(),'status':'preflight','started_at':time.time(),'arms':[],
            'training_order':[a['arm'] for a in plan['arms']]}
    def update(**values):
        status.update(values);temp=status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path)
    stopped=[]
    python=root/'openpi/.venv/bin/python'
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled',
        HF_HOME=str(root/'cache/huggingface'),OPENPI_DATA_HOME=str(root/'cache/openpi'),XDG_CACHE_HOME=str(root/'cache'),
        PYTHONPATH=str(root/'server')+':'+str(root/'openpi/src'))
    def run(command, logname, *, cpu=False):
        env=dict(environment)
        if cpu:env['JAX_PLATFORMS']='cpu'
        fixed=('OMP_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS','XLA_PYTHON_CLIENT_PREALLOCATE',
               'HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','WANDB_MODE','HF_HOME','OPENPI_DATA_HOME','XDG_CACHE_HOME','PYTHONPATH')
        shell='if [ -f env.sh ]; then source env.sh; fi; export '+ ' '.join(k+'='+shlex.quote(env[k]) for k in fixed)
        shell+='; exec '+shlex.join([str(v) for v in command])
        with (out/logname).open('xb') as log:
            return subprocess.run(['bash','-lc',shell],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    try:
        update()
        for name,digest in plan['source_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Source hash mismatch: '+name)
        audit=out/'expert_audit.json'
        assert sha(audit)==plan['expert_audit_sha256']
        assert json.loads(audit.read_text())['decision']=='retain_reviewed_data_no_gross_target_jump_detected'
        dataset=root/plan['dataset'];assert sha(dataset/'dataset_manifest.json')==plan['dataset_manifest_sha256']
        assert sha(root/plan['sampler'])==plan['sampler_sha256']
        norm=root/plan['initial_checkpoint']/'assets/local/pi0_banana_train/norm_stats.json'
        assert sha(norm)==plan['norm_stats_sha256']
        assert (root/plan['initial_checkpoint']/'params/_METADATA').is_file()
        settings=[]
        for arm in plan['arms']:
            path=root/arm['config'];assert sha(path)==arm['config_sha256']
            config=tomllib.loads(path.read_text());settings.append(config)
            assert config['train']['batch_size']==8 and config['train']['gradient_accumulation_steps']==4
            assert config['train']['num_train_steps']==3000 and config['model']['action_horizon']==100
            assets=root/config['paths']['assets_base_dir']/'pi0_banana_lora/local/pi0_banana_train/norm_stats.json'
            assets.parent.mkdir(parents=True,exist_ok=True)
            if assets.exists() and assets.read_bytes()!=norm.read_bytes():raise ValueError('Existing normalization differs')
            assets.write_bytes(norm.read_bytes())
        a,b=json.loads(json.dumps(settings[0])),json.loads(json.dumps(settings[1]))
        assert a['model'].pop('temporal_loss_weight')==0 and b['model'].pop('temporal_loss_weight')==.1
        assert a==b
        if run([python,root/'tests/test_temporal_loss.py'],'unit_tests.log',cpu=True):
            raise RuntimeError('Temporal numerical/gradient tests failed')
        if run([python,root/'server/temporal_preflight.py','--root',root,'--config',root/plan['arms'][1]['config'],
                '--output',out/'reader_audit.json'],'reader_audit.log',cpu=True):
            raise RuntimeError('Real-reader/mask audit failed')
        # Stop only this application's exact policy command; keep proxy/VS Code intact.
        for cmdline in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if str(root/'server/serve_pi0.py').encode() in cmdline.read_bytes().split(b'\0'):
                    pid=int(cmdline.parent.name);os.kill(pid,signal.SIGTERM);stopped.append(pid)
            except (OSError,ValueError):pass
        update(status='waiting_for_idle_gpu',stopped_policy_pids=stopped)
        for _ in range(30):
            used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if used<2048:break
            time.sleep(1)
        if used>=2048:raise RuntimeError('Other GPU allocations remain')
        # Both probes must pass before either full training arm starts.
        for arm in plan['arms']:
            label=arm['arm'];update(status='probing',active_arm=label)
            command=[python,'-u',root/'server/banana_finetune.py','probe','--root',root,'--config',root/arm['config'],
                     '--exp-name',arm['experiment']+'_probe','--probe-updates','3','--probe-output',out/(label+'_probe.json')]
            if run(command,label+'_probe.log'):raise RuntimeError(label+' GPU probe failed')
            probe=json.loads((out/(label+'_probe.json')).read_text())
            if probe.get('status')!='passed' or probe.get('actual_action_shape')!=[32,100,32]:raise RuntimeError('Invalid probe result')
            if probe.get('gpu_peak_used_mib') is None or probe['gpu_total_mib']-probe['gpu_peak_used_mib']<2048:
                raise RuntimeError('Insufficient 2048MiB GPU headroom')
            status['arms'].append({'arm':label,'experiment':arm['experiment'],'status':'probe_passed_queued',
                                   'probe':{k:probe[k] for k in ('steady_update_seconds','gpu_peak_used_mib','gpu_total_mib')}})
            update()
        for index,arm in enumerate(plan['arms']):
            label=arm['arm'];status['arms'][index]['status']='training';update(status='training',active_arm=label)
            command=[python,'-u',root/'server/banana_finetune.py','train','--root',root,'--config',root/arm['config'],
                     '--exp-name',arm['experiment']]
            if run(command,label+'_training.log'):raise RuntimeError(label+' training failed')
            final=root/'checkpoints/pi0_banana_lora'/arm['experiment']/'2999'
            if not (final/'_CHECKPOINT_METADATA').is_file() or not (final/'params/_METADATA').is_file():
                raise RuntimeError('Expected committed final2999 missing')
            status['arms'][index].update(status='trained_pending_validation',checkpoint=str(final))
            update()
        update(status='both_trained_pending_validation',finished_at=time.time())
    except BaseException as exc:
        update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time())
        raise
    finally:
        if stopped:
            command=['bash',root/'server/run_banana_serve.sh',root/'outputs/long_chunk_20261025/selected_training.toml',
                     root/plan['initial_checkpoint'],'10']
            env=dict(environment);env['PI0_RTC']='1'
            with (out/'restore_v6.log').open('ab') as log:
                p=subprocess.Popen([str(x) for x in command],cwd=root,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(restore_v6_attempt_pid=p.pid,restore_v6_verified=False)


if __name__=='__main__':
    main()
