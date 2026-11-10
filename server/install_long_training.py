"""Start the authorized training experiment after a checksummed SSH deployment."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time


def main():
    root=Path('/root/pi0-repro');out=root/'outputs/long_chunk_20261025'
    out.mkdir(parents=True,exist_ok=True)
    if not (root/'checkpoints/pi0_banana_lora/banana_v5_quality_norm1000/999/params').exists():
        raise FileNotFoundError('Committed v5/999 initializer is unavailable')
    if not (root/'data/lerobot/20261024-185715-merged-ebdbe3/dataset_manifest.json').exists():
        raise FileNotFoundError('Merged dataset is unavailable')
    pid_path=out/'supervisor_pid.json'
    if pid_path.exists():
        pid=json.loads(pid_path.read_text())['pid']
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and b'banana_training_sweep.py' in cmd.read_bytes():
            print('Existing training supervisor remains active; no duplicate launched');return
    stopped=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if str(root/'server/serve_pi0.py').encode() in path.read_bytes():
                pid=int(path.parent.name)
                if pid!=os.getpid():os.kill(pid,signal.SIGTERM);stopped.append(pid)
        except (OSError,ValueError):pass
    for _ in range(10):
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if used<1024:break
        time.sleep(1)
    if used>=1024:raise RuntimeError('GPU still has other allocations; no training started')
    (out/'deployment.json').write_text(json.dumps({'stopped_policy_pids':stopped,'idle_gpu_used_mib':used,
        'scope':'Original pi0 training; existing v3 checkpoint retained; no upstream edits'},indent=2)+'\n')
    f=(out/'status_http.log').open('ab')
    http=subprocess.Popen([sys.executable,'-m','http.server','8000','--bind','127.0.0.1','--directory',str(out)],
        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'status_http_pid.json').write_text(json.dumps({'pid':http.pid}))
    command='cd /root/pi0-repro; if [ -f env.sh ]; then source env.sh; fi; exec openpi/.venv/bin/python -u server/banana_training_sweep.py --root /root/pi0-repro --config /root/pi0-repro/configs/banana_finetune_long100.toml --output /root/pi0-repro/outputs/long_chunk_20261025 --launch'
    log=(out/'supervisor.log').open('ab')
    process=subprocess.Popen(['bash','-lc',command],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    pid_path.write_text(json.dumps({'pid':process.pid}))
    print('DEPLOYED_AND_SWEEP_STARTED',process.pid,flush=True)


if __name__=='__main__':main()
