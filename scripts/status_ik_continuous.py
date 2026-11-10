"""Read compact training/preflight status, never raw weights or norm assets."""
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CODE=r'''
import json,subprocess,time,re
from pathlib import Path
root=Path('/root/pi0-repro');out=root/'outputs/ik_continuous_20261029/cloud_training'
read=lambda p:json.loads(p.read_text())
status=read(out/'status.json') if (out/'status.json').exists() else {'status':'not_started'}
result={k:v for k,v in status.items() if k!='arms'}
result['arms']=[{k:v for k,v in row.items() if k!='probe'} | {'probe':{k:v for k,v in row.get('probe',{}).items() if k in ('status','actual_action_shape','steady_update_seconds','gpu_peak_used_mib','gpu_total_mib')}} for row in status.get('arms',[])]
if status.get('started_at'):result['elapsed_seconds']=time.time()-status['started_at']
for name in ('control_reader','candidate_reader','control_norm','candidate_norm','control_probe','candidate_probe'):
 p=out/(name+'.json')
 if p.exists():
  d=read(p);result[name]={k:v for k,v in d.items() if k in ('status','actual_action_shape','steady_update_seconds','gpu_peak_used_mib','gpu_total_mib','checked_samples','training_frames','norm_stats_sha256','actions','states')}
def safe_lines(text):
 result=[]
 for line in text.splitlines():
  # The upstream loader logs full NormStats inside a multiline DataConfig.
  # Only positive scalar-progress/error allowlist lines may leave the host.
  if any(token in line for token in ('array(', 'NormStats', 'norm_stats', 'q99=', 'q01=', 'mean=', 'std=')):continue
  if re.search(r'\bStep\s+\d+:|\bloss=|Traceback|(?:Error|Exception):|checkpoint.*(?:saved|finished|committed)|\bTraining:\s*\d+%',line,re.IGNORECASE):
   result.append(line[:1200])
 return result[-4:]
logs={}
for name in ('supervisor','control_reader','candidate_reader','control_probe','candidate_probe','control_training','candidate_training','restore_v6'):
 p=out/(name+'.log')
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-5000));text=f.read().decode(errors='replace')
  logs[name]=safe_lines(text)
result['log_tails']=logs
try:result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
except Exception as e:result['gpu_error']=str(e)
print(json.dumps(result))
'''


def main():
    target=(ROOT/'outputs/random_multiview_connection/ssh_target.txt').read_text().strip()
    assert re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',target)
    ssh=['ssh','-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ControlMaster=no','-o','ControlPath=none','-o','ConnectTimeout=10','-o','IPQoS=none']
    r=subprocess.run(ssh+[target,shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',CODE])],text=True,capture_output=True,timeout=30)
    if r.returncode:raise RuntimeError(r.stderr[-2000:])
    data=json.loads(r.stdout);out=ROOT/'outputs/ik_continuous_20261029'
    (out/'cloud_status_latest.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
