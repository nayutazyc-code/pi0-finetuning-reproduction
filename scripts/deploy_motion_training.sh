#!/usr/bin/env bash
# Deploy the reviewed train-only motion pilot and dispatch its guarded supervisor.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Supply the existing SSH alias}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
OUT=outputs/motion_first_20261026
PLAN="$OUT/training_plan.json"
[[ -f "$PLAN" ]] || { echo 'Prepare the reviewed training plan first.' >&2; exit 1; }
PLAN_DIGEST="$(.venv/bin/python - "$PLAN" <<'PY'
from pathlib import Path
import hashlib,sys
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
bash scripts/deploy_v7_recovery.sh "$REMOTE_TARGET" --auth-only
TASK_SOCKET="$(cat outputs/v6_recovery_20261026/ssh_control_socket.txt)"
REMOTE_ROOT=/root/pi0-repro
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
out=Path('outputs/motion_first_20261026');plan=json.loads((out/'training_plan.json').read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root=Path.cwd().resolve()
def under(value):
    path=Path(value)
    if path.is_absolute() or '..' in path.parts or not (root/path).resolve().is_relative_to(root):
        raise ValueError('Invalid project-relative input path')
    return path
for key in ('config','dataset','base_dataset','sampler','initial_checkpoint'):
    under(plan[key])
for name,digest in plan['source_sha256'].items():
    path=under(name)
    assert sha(path)==digest,name
assert sha(plan['config'])==plan['config_sha256']
assert sha(Path(plan['dataset'])/'dataset_manifest.json')==plan['dataset_manifest_sha256']
assert sha(plan['sampler'])==plan['sampler_sha256']
files=list(plan['source_sha256'])+[plan['config'],plan['sampler'],str(out/'training_plan.json')]
files += ['server/motion_train_job.py','server/motion_train_status.py']
files += [str(Path(plan['dataset'])/'dataset_manifest.json')]
for split in ('validation','test'):
    files.append(str(Path(plan['dataset'])/f'local/pi0_banana_{split}/source_episodes.json'))
repo=Path(plan['dataset'])/'local/pi0_banana_train'
files += [str(path) for path in sorted(repo.rglob('*')) if path.is_file() and
          (path.relative_to(repo).parts[0] in ('data','meta') or path.name in ('source_episodes.json','conversion_audit.json'))]
(out/'motion_deploy_files.txt').write_text('\n'.join(dict.fromkeys(files))+'\n')
PY
# Send only the plan first so remote preparation can validate its destination.
ssh -S "$TASK_SOCKET" -o BatchMode=yes "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT' '$PLAN_DIGEST'" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]);out=root/'outputs/motion_first_20261026'
if (out/'cloud_training/status.json').exists():raise FileExistsError('Existing motion job retained; inspect status')
plan=out/'training_plan.json'
if plan.exists() and hashlib.sha256(plan.read_bytes()).hexdigest()!=sys.argv[2]:
    raise ValueError('Existing motion training plan differs; preserve it')
out.mkdir(parents=True,exist_ok=True)
PY
rsync -rlpt --checksum -e "ssh -S $TASK_SOCKET -o BatchMode=yes" "$PLAN" "$REMOTE_TARGET:$REMOTE_ROOT/$OUT/"
ssh -S "$TASK_SOCKET" -o BatchMode=yes "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT'" <<'PY'
from pathlib import Path
import hashlib,json,shutil,sys
root=Path(sys.argv[1]);out=root/'outputs/motion_first_20261026';plan=json.loads((out/'training_plan.json').read_text())
def under(value):
    p=Path(value)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid project-relative path')
    p=(root/p).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('Path escapes root')
    return p
if (out/'cloud_training/status.json').exists():raise FileExistsError('Do not redeploy a started job')
dataset=under(plan['dataset']);manifest=dataset/'dataset_manifest.json'
if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest()!=plan['dataset_manifest_sha256']:
    raise ValueError('Existing destination manifest differs')
train=dataset/'local/pi0_banana_train';base=under(plan['base_dataset'])/'local/pi0_banana_train'
if not train.exists():
    train.parent.mkdir(parents=True,exist_ok=True)
    if base.exists():shutil.copytree(base,train)
    else:train.mkdir()
backup=out/'previous_server_sources';backup.mkdir(exist_ok=True)
for name in plan['source_sha256']:
    source=under(name)
    if source.is_file() and source.parts[-2]=='server':
        target=backup/source.name
        if not target.exists():shutil.copy2(source,target)
(root/'tests').mkdir(exist_ok=True)
(out/'cloud_training').mkdir(exist_ok=True)
PY
rsync -rlpt --checksum --relative --files-from="$OUT/motion_deploy_files.txt" \
    -e "ssh -S $TASK_SOCKET -o BatchMode=yes" ./ "$REMOTE_TARGET:$REMOTE_ROOT/"
ssh -S "$TASK_SOCKET" -o BatchMode=yes "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT'" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]);out=root/'outputs/motion_first_20261026/cloud_training'
if (out/'status.json').exists():raise FileExistsError('Do not launch duplicate motion training')
with (out/'supervisor.log').open('xb') as log:
    process=subprocess.Popen([str(root/'openpi/.venv/bin/python'),'-u',str(root/'server/motion_train_job.py'),'--root',str(root)],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Motion training supervisor dispatched:',process.pid)
PY
