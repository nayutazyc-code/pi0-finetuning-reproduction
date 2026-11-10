#!/usr/bin/env bash
# Deploy and dispatch the already audited matched training plan; no credential storage.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Supply the existing SSH alias}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
bash scripts/deploy_v7_recovery.sh "$REMOTE_TARGET" --auth-only
TASK_SOCKET="$(cat outputs/v6_recovery_20261026/ssh_control_socket.txt)"
REMOTE_ROOT=/root/pi0-repro
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
out=Path('outputs/v8_temporal_20261026');plan=json.loads((out/'plan.json').read_text())
for name,digest in plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((out/'expert_audit.json').read_bytes()).hexdigest()==plan['expert_audit_sha256']
files=list(plan['source_sha256'])+[a['config'] for a in plan['arms']]
files += [str(out/'plan.json'),str(out/'expert_audit.json'),plan['sampler']]
(out/'deploy_files.txt').write_text('\n'.join(files)+'\n')
PY
ssh -S "$TASK_SOCKET" -o BatchMode=yes "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT'" <<'PY'
from pathlib import Path
import shutil,sys
root=Path(sys.argv[1]);out=root/'outputs/v8_temporal_20261026'
if (out/'status.json').exists():raise FileExistsError('Existing v8 job retained; use the status helper')
out.mkdir(parents=True,exist_ok=True)
backup=out/'previous_server_sources';backup.mkdir(exist_ok=True)
for name in ('banana_model.py','banana_config.py','banana_accumulation.py','banana_sampling.py'):
    target=backup/name
    if not target.exists():shutil.copy2(root/'server'/name,target)
(root/'tests').mkdir(exist_ok=True)
PY
rsync -rlpt --checksum --relative --files-from=outputs/v8_temporal_20261026/deploy_files.txt \
    -e "ssh -S $TASK_SOCKET -o BatchMode=yes" ./ "$REMOTE_TARGET:$REMOTE_ROOT/"
ssh -S "$TASK_SOCKET" -o BatchMode=yes "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT'" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]);out=root/'outputs/v8_temporal_20261026'
if (out/'status.json').exists():raise FileExistsError('Do not launch duplicate training')
with (out/'supervisor.log').open('xb') as log:
    p=subprocess.Popen([str(root/'openpi/.venv/bin/python'),'-u',str(root/'server/temporal_train_job.py'),'--root',str(root)],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Matched training supervisor dispatched:',p.pid)
PY
