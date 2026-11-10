#!/usr/bin/env bash
# User-invoked setup only: add a dedicated public key, preserving existing keys.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Supply the existing SSH alias}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] || exit 2
if [[ "${2:-}" != '--install-key' ]]; then
  echo 'Explicit --install-key is required. This creates a dedicated local key and adds only its public key remotely.'
  exit 2
fi
umask 077
TASK_KEY="$HOME/.ssh/id_ed25519"
mkdir -p "$HOME/.ssh"
if [[ -e "$TASK_KEY" || -e "$TASK_KEY.pub" ]]; then
  [[ -f "$TASK_KEY" && -f "$TASK_KEY.pub" && ! -L "$TASK_KEY" && ! -L "$TASK_KEY.pub" ]] || {
    echo 'Incomplete or linked existing task key; refusing to replace it.' >&2; exit 1;
  }
else
  ssh-keygen -q -t ed25519 -N '' -C pi0-motion-task -f "$TASK_KEY"
fi
chmod 600 "$TASK_KEY"
chmod 644 "$TASK_KEY.pub"
DERIVED_PUBLIC="$(ssh-keygen -y -f "$TASK_KEY" | awk '{print $1 " " $2}')"
STORED_PUBLIC="$(awk '{print $1 " " $2}' "$TASK_KEY.pub")"
[[ "$DERIVED_PUBLIC" == "$STORED_PUBLIC" ]] || { echo 'Task key pair does not match.' >&2; exit 1; }
REMOTE_COMMAND="$(.venv/bin/python - "$TASK_KEY.pub" <<'PY'
from pathlib import Path
import shlex,sys
public=Path(sys.argv[1]).read_text().strip()
code=r'''
from pathlib import Path
import base64,os,shutil,sys,time
key=sys.argv[1]
parts=key.split()
if len(parts)!=3 or parts[0]!='ssh-ed25519' or parts[2]!='pi0-motion-task':
    raise ValueError('Unexpected task public key')
base64.b64decode(parts[1],validate=True)
folder=Path.home()/'.ssh'
if folder.is_symlink():raise ValueError('Refusing symlinked SSH directory')
folder.mkdir(mode=0o700,exist_ok=True)
path=folder/'authorized_keys'
if path.is_symlink():raise ValueError('Refusing symlinked authorized_keys')
old=path.read_text() if path.exists() else ''
if any(parts[1] in line.split() for line in old.splitlines()):
    print('Task public key already present; existing entries retained.')
else:
    if path.exists():
        backup=folder/('authorized_keys.before-pi0-motion-'+str(time.time_ns()))
        shutil.copy2(path,backup)
        backup.chmod(0o600)
    options='restrict,port-forwarding,permitopen="127.0.0.1:8000",permitopen="127.0.0.1:8001"'
    temporary=folder/('authorized_keys.pi0-motion-'+str(os.getpid())+'.tmp')
    with temporary.open('x') as stream:
        os.chmod(temporary,0o600)
        stream.write(old+('' if not old or old.endswith('\n') else '\n')+options+' '+key+'\n')
    os.replace(temporary,path)
    print('Task public key added; existing entries retained. Backup saved when applicable.')
'''
print(shlex.join(['/root/pi0-repro/openpi/.venv/bin/python','-B','-c',code,public]))
PY
)"
echo 'Authenticate once to add the dedicated public key. No password is saved.'
ssh -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o IPQoS=none "$REMOTE_TARGET" "$REMOTE_COMMAND"
ssh -i "$TASK_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ControlMaster=no -o ControlPath=none \
  -o ConnectTimeout=15 -o PreferredAuthentications=publickey "$REMOTE_TARGET" 'true'
printf '%s\n' "$TASK_KEY" > outputs/v6_recovery_20261026/ssh_identity_file.txt
bash scripts/reconnect_motion_cloud.sh "$REMOTE_TARGET"
echo 'Dedicated key authentication verified. No dataset uploaded or training started.'
