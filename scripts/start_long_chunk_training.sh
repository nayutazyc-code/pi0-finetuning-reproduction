#!/usr/bin/env bash
# Run from an ordinary local terminal; SSH asks the user for authentication.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 - "$PROJECT_ROOT" "$@" <<'PY'
from pathlib import Path
import hashlib,io,json,shlex,subprocess,sys,zipfile
root=Path(sys.argv[1]);explicit=sys.argv[2] if len(sys.argv)>2 else None
options=[];targets=[]
if explicit:targets=[explicit]
else:
    rows=subprocess.check_output(['ps','-axo','comm=,args=']).decode('utf-8','replace')
    for line in rows.splitlines():
        parts=line.strip().split(None,1)
        if len(parts)!=2 or parts[0].rsplit('/',1)[-1]!='ssh':continue
        args=shlex.split(parts[1]);i=1;opts=[]
        while i<len(args):
            item=args[i]
            if not item.startswith('-'):
                targets.append(item);options=opts;break
            if len(item)==2 and item[1] in 'BbcDEeFIiJLlmOopQRSWw':
                if item not in ['-D','-L','-R','-E','-O','-Q','-W']:opts.extend(args[i:i+2])
                i+=2
            else:
                if item not in ['-v','-vv','-vvv','-T','-N','-f']:opts.append(item)
                i+=1
if len(set(targets))!=1:raise SystemExit('Pass the existing SSH Host alias as the sole argument; no destination is stored in this script.')
files=['server/banana_config.py','server/banana_accumulation.py','server/banana_finetune.py',
       'server/banana_probe.py','server/banana_accumulation_diagnostic.py','server/banana_training_sweep.py',
       'server/install_long_training.py','configs/banana_finetune_long100.toml']
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files:archive.writestr(name,(root/name).read_bytes())
data=buffer.getvalue();digest=hashlib.sha256(data).hexdigest()
remote="""from pathlib import Path
import sys,hashlib,io,zipfile,subprocess,json
data=sys.stdin.buffer.read()
assert hashlib.sha256(data).hexdigest()==sys.argv[1], 'Upload checksum mismatch'
root=Path('/root/pi0-repro');assert (root/'openpi/.venv/bin/python').exists()
out=root/'outputs/long_chunk_20261025';out.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(io.BytesIO(data));items={i.filename:z.read(i) for i in z.infolist()}
assert all(n.startswith(('server/banana_','server/install_long_training.py','configs/banana_finetune_long100')) and '..' not in n for n in items)
for name,content in items.items():
 p=root/name;backup=out/'deploy_backup'/name
 if p.exists() and not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(p.read_bytes())
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
(out/'deployed_sha256.json').write_text(json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in items.items()},indent=2)+'\\n')
subprocess.run([sys.executable,str(root/'server/install_long_training.py')],check=True)
"""
command='python3 -c '+shlex.quote(remote)+' '+shlex.quote(digest)
print('Uploading the training adapter through the current SSH configuration. Enter authentication only in the SSH prompt.',flush=True)
result=subprocess.run(['ssh',*options,'-o','BatchMode=no','-o','ConnectTimeout=15',targets[0],command],input=data)
raise SystemExit(result.returncode)
PY
