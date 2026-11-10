"""Checked matched v6 continuations, separate volumes, restore original serving."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
import tomllib

NORM_SHA='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
INITIAL='checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999'
VOLUMES={'control':Path('/root/pi0-progress-control-checkpoints'),
         'recovery':Path('/root/pi0-progress-recovery-checkpoints')}
RESERVE=2*1024**3
REQUIRED_SOURCES={'server/banana_model.py','server/banana_config.py','server/banana_accumulation.py',
    'server/banana_sampling.py','server/banana_temporal.py','server/banana_temporal_data.py',
    'server/banana_finetune.py','server/banana_probe.py','server/banana_roi.py',
    'server/motion_norm_audit.py','server/temporal_preflight.py','server/progress_train_job.py',
    'server/progress_policy_select.py'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def require(x,message):
    if not x:raise ValueError(message)
def local(root,value):
    p=Path(value);require(not p.is_absolute() and '..' not in p.parts,'Expected relative project path')
    p=(root/p).resolve();require(p.is_relative_to(root.resolve()),'Project path escape');return p
def existing_parent(p):
    while not p.exists():p=p.parent
    return p


def validate(root,plan):
    require(plan['schema']=='v6-progress-matched-training-train-only-v2','Wrong training schema')
    require(plan['initial_checkpoint']==INITIAL and plan['norm_stats_sha256']==NORM_SHA,'Initializer/norm changed')
    require([a['arm'] for a in plan['arms']]==['control','recovery'],'Expected fixed two-arm order')
    require(REQUIRED_SOURCES<=set(plan['source_sha256']),'Training sources not fully bound')
    for name,digest in plan['source_sha256'].items():require(sha(local(root,name))==digest,'Source hash mismatch: '+name)
    norm=root/INITIAL/'assets/local/pi0_banana_train/norm_stats.json'
    require(sha(norm)==NORM_SHA,'Initializer normalization mismatch')
    require((root/INITIAL/'_CHECKPOINT_METADATA').is_file() and (root/INITIAL/'params/_METADATA').is_file(),'Initializer not committed')
    initial_bytes=sum(p.stat().st_size for p in (root/INITIAL).rglob('*') if p.is_file())
    settings=[];capacities={}
    heldout=plan['heldout_lineage_local_verification']
    require(heldout['passed'] and all(heldout[s]['base_digest']==heldout[s]['candidate_digest'] for s in ('validation','test')),'Local heldout verification missing/different')
    for arm in plan['arms']:
        label=arm['arm'];config=local(root,arm['config']);require(sha(config)==arm['config_sha256'],'Config hash mismatch')
        require(arm['experiment']==f'banana_v6_progress_{label}1000','Unexpected experiment path')
        s=tomllib.loads(config.read_text());settings.append(s);m=s['model'];t=s['train'];paths=s['paths']
        require(m['family']=='pi0' and m['config_name']=='pi0_banana_lora' and m['action_horizon']==100,'Only original pi0 H100')
        require(m['strict_lora_only'] is True and m['train_action_projections'] is True and m['weighted_flow_loss'] is True,'Training parameter scope changed')
        require(m['camera_crop']==[275,110,540,345] and m['image_augmentation'] is False,'Image semantics changed')
        require(not m.get('temporal_training',False) and m.get('temporal_loss_weight',0)==0 and m.get('near_term_steps',0)==0,'Unexpected loss intervention')
        require(t['batch_size']==8 and t['gradient_accumulation_steps']==2 and t['num_train_steps']==1000,'Expected1000updates/effective16')
        require(t['save_interval']==500 and t['keep_period']==1000000 and t['require_initial_norm_match'] is True,'Retention or norm guard changed')
        require(paths['initial_checkpoint']==INITIAL and Path(paths['checkpoint_base_dir'])==VOLUMES[label],'Checkpoint boundary changed')
        require(paths['assets_base_dir']=='assets_v6_progress_norm','Unexpected normalization destination')
        require(s['data']['directory']==arm['dataset'] and s['sampling']['weights_file']==arm['sampler'],'Data/sampler changed')
        ds=local(root,arm['dataset']);require(sha(ds/'dataset_manifest.json')==arm['dataset_manifest_sha256'],'Dataset manifest changed')
        require(ds.is_relative_to(root/'data/lerobot'),'Dataset outside data boundary')
        mapping=read(ds/'local/pi0_banana_train/source_episodes.json');rs=mapping['episodes']
        require(mapping['source_split']=='train' and all(r['split']=='train' and r['lerobot_episode_index']==i for i,r in enumerate(rs)),'Non-training or noncontiguous mapping')
        require(len(rs)==arm['train_episodes'] and sum(r['frames'] for r in rs)==arm['train_frames'],'Frame counts changed')
        sampler=local(root,arm['sampler']);require(sha(sampler)==arm['sampler_sha256'],'Sampler changed');w=read(sampler)
        require(w['split']=='train' and w['repo_id']=='local/pi0_banana_train' and len(w['weights'])==arm['train_frames'],'Sampler alignment changed')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in w['weights']),'Invalid sampler weights')
        if w.get('dataset_manifest_sha256'):require(w['dataset_manifest_sha256']==arm['dataset_manifest_sha256'],'Sampler/manifest mismatch')
        for name,digest in arm['dataset_files_sha256'].items():require(sha(local(root,name))==digest,'Dataset file changed: '+name)
        parquet={str(p.relative_to(root)) for p in (ds/'local/pi0_banana_train').glob('data/chunk-*/episode_*.parquet')}
        require(bool(parquet) and parquet=={n for n in arm['dataset_files_sha256'] if n.endswith('.parquet')},'Unbound/missing training parquet')
        require(all('/local/pi0_banana_train/' in n for n in arm['dataset_files_sha256']),'Only training dataset files may be deployed')
        checkpoint=VOLUMES[label]/m['config_name']/arm['experiment'];require(not checkpoint.exists(),'Candidate already exists')
        parent=existing_parent(VOLUMES[label]);dev=os.stat(parent).st_dev
        capacities.setdefault(dev,{'free':shutil.disk_usage(parent).free,'arms':[]})['arms'].append(label)
    a,b=json.loads(json.dumps(settings[0])),json.loads(json.dumps(settings[1]))
    for s in (a,b):s['data'].pop('directory');s['sampling'].pop('weights_file');s['paths'].pop('checkpoint_base_dir')
    require(a==b,'Arms differ beyond dataset/sampler/checkpoint volume')
    for c in capacities.values():
        # If unexpectedly sharing a filesystem, allow one prior final plus the
        # active arm's two-snapshot peak. Never delete older work to make room.
        c['required']=(1+len(c['arms']))*initial_bytes+RESERVE
        require(c['free']>=c['required'],'Insufficient two-snapshot+reserve disk capacity')
    return settings,norm,{'initializer_bytes':initial_bytes,'filesystems':list(capacities.values()),'heldout_lineage_locally_verified':True,'heldout_contents_uploaded':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    parent=root/'outputs/v6_progress_20261028';plan=read(parent/'training_plan_train_only.json');out=parent/'cloud_training';out.mkdir(exist_ok=True)
    lock=(out/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=out/'status.json';require(not status_path.exists(),'Existing job status retained; no duplicate launch')
    status={'pid':os.getpid(),'status':'preflight','started_at':time.time(),'arms':[]};stopped=[];child=None
    def update(**kwargs):
        status.update(kwargs);temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(status_path)
    def interrupted(signum,frame):raise KeyboardInterrupt('Training supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    python=root/'openpi/.venv/bin/python';env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',WANDB_MODE='disabled',HF_HOME=str(root/'cache/huggingface'),OPENPI_DATA_HOME=str(root/'cache/openpi'),XDG_CACHE_HOME=str(root/'cache'),PYTHONPATH=str(root/'server')+':'+str(root/'openpi/src'))
    def run(command,logname,*,cpu=False):
        nonlocal child
        current=dict(env)
        if cpu:current['JAX_PLATFORMS']='cpu'
        keys=['OMP_NUM_THREADS','MKL_NUM_THREADS','JAX_PLATFORMS','XLA_PYTHON_CLIENT_PREALLOCATE','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','WANDB_MODE','HF_HOME','OPENPI_DATA_HOME','XDG_CACHE_HOME','PYTHONPATH']
        shell='if [ -f env.sh ]; then source env.sh; fi; export '+' '.join(k+'='+shlex.quote(current[k]) for k in keys)+'; exec '+shlex.join([str(x) for x in command])
        with (out/logname).open('xb') as log:
            child=subprocess.Popen(['bash','-lc',shell],cwd=root,env=current,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while child.poll() is None:
                    if any(shutil.disk_usage(existing_parent(v)).free<RESERVE for v in VOLUMES.values()):raise RuntimeError('Disk reserve crossed during job')
                    time.sleep(5)
                return child.returncode
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                child=None
    try:
        update();settings,norm,capacity=validate(root,plan);update(capacity=capacity)
        for arm,s in zip(plan['arms'],settings,strict=True):
            destination=root/s['paths']['assets_base_dir']/'pi0_banana_lora/local/pi0_banana_train/norm_stats.json'
            destination.parent.mkdir(parents=True,exist_ok=True)
            require(not destination.exists() or destination.read_bytes()==norm.read_bytes(),'Existing assets differ')
            destination.write_bytes(norm.read_bytes())
            label=arm['arm'];update(status='reader_and_norm_preflight',active_arm=label)
            require(run([python,root/'server/temporal_preflight.py','--root',root,'--config',root/arm['config'],'--output',out/(label+'_reader.json')],label+'_reader.log',cpu=True)==0,'H100 real-reader preflight failed')
            require(run([python,root/'server/motion_norm_audit.py','--root',root,'--config',root/arm['config'],'--output',out/(label+'_norm.json')],label+'_norm.log',cpu=True)==0,'Normalization compatibility failed')
        policies=[]
        for path in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                cmd=[x.decode() for x in path.read_bytes().split(b'\0') if x]
                require(str(root/'server/banana_finetune.py') not in cmd,'Another training process active')
                if str(root/'server/serve_pi0.py') in cmd:
                    require(cmd[cmd.index('--checkpoint')+1]==str(root/INITIAL),'Unexpected served checkpoint')
                    policies.append(int(path.parent.name))
            except OSError:pass
        require(len(policies)==1,'Expected exactly one original v6 policy before training')
        for pid in policies:os.kill(pid,signal.SIGTERM);stopped.append(pid)
        update(status='waiting_for_gpu',stopped_policy_pids=stopped)
        for _ in range(30):
            used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
            if used<2048:break
            time.sleep(1)
        require(used<2048,'Other GPU allocations remain')
        for arm in plan['arms']:
            label=arm['arm'];update(status='probing',active_arm=label)
            command=[python,'-u',root/'server/banana_finetune.py','probe','--root',root,'--config',root/arm['config'],'--exp-name',arm['experiment']+'_probe','--probe-updates','3','--probe-output',out/(label+'_probe.json')]
            require(run(command,label+'_probe.log')==0,'Real-data GPU probe failed')
            probe=read(out/(label+'_probe.json'));require(probe['status']=='passed' and probe['actual_action_shape']==[16,100,32],'Invalid GPU probe shape/status')
            require(probe['gpu_total_mib']-probe['gpu_peak_used_mib']>=2048,'GPU headroom below2048MiB')
            status['arms'].append(dict(arm=label,experiment=arm['experiment'],status='probe_passed',probe=probe));update()
        for i,arm in enumerate(plan['arms']):
            label=arm['arm'];status['arms'][i]['status']='training';update(status='training',active_arm=label)
            require(run([python,'-u',root/'server/banana_finetune.py','train','--root',root,'--config',root/arm['config'],'--exp-name',arm['experiment']],label+'_training.log')==0,'Training failed')
            final=VOLUMES[label]/'pi0_banana_lora'/arm['experiment']/'999'
            require((final/'_CHECKPOINT_METADATA').is_file() and (final/'params/_METADATA').is_file(),'Committed final999 missing')
            status['arms'][i].update(status='trained_pending_validation',checkpoint=str(final));update()
        update(status='both_trained_pending_validation',finished_at=time.time())
    except BaseException as exc:update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at=time.time());raise
    finally:
        if stopped:
            current=dict(env);current['PI0_RTC']='1'
            with (out/'restore_v6.log').open('ab') as log:
                process=subprocess.Popen(['bash',str(root/'server/run_banana_serve.sh'),str(root/'outputs/long_chunk_20261025/selected_training.toml'),str(root/INITIAL),'10'],cwd=root,env=current,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            update(restore_v6_attempt_pid=process.pid,restore_v6_verified=False)


if __name__=='__main__':main()
