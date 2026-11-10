"""Bounded foreground status watch for this active task; never starts a job."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    deadline=time.monotonic()+3600;errors=0
    while time.monotonic()<deadline:
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/status_v6_progress.py')],cwd=ROOT,capture_output=True,text=True,timeout=40)
        if r.returncode:
            errors+=1;print(json.dumps({'status_read_error':r.stderr[-500:],'consecutive_errors':errors}),flush=True)
            if errors>=3:raise SystemExit(1)
        else:
            errors=0;data=json.loads(r.stdout);label=data.get('active_arm');tail=data.get('log_tails',{}).get(str(label)+'_training',[])
            steps=[int(m.group(1)) for line in tail for m in re.finditer(r'\bStep\s+(\d+):',line)]
            compact={'status':data['status'],'active_arm':label,'elapsed_seconds':round(data.get('elapsed_seconds',0)),
                     'latest_logged_step':max(steps) if steps else None,'gpu':data.get('gpu'),
                     'arms':[(a['arm'],a['status']) for a in data.get('arms',[])],'log_tail':tail[-1:]}
            if data.get('error'):compact['error']=data['error']
            print(json.dumps(compact),flush=True)
            if data['status'] in ('both_trained_pending_validation','failed'):return
        time.sleep(30)
    raise TimeoutError('Bounded watch ended; inspect live status before continuing')


if __name__=='__main__':main()
