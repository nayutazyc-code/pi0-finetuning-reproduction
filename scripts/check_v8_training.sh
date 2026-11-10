#!/usr/bin/env bash
# Read aggregate remote status; failed connections never replace a good snapshot.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Supply the existing SSH alias}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
OUT="${PI0_STATUS_OUTPUT_DIR:-outputs/v8_temporal_20261026}"
TARGET_DIGEST="$(.venv/bin/python -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$REMOTE_TARGET")"
[[ "$(cat outputs/v6_recovery_20261026/ssh_target_digest.txt)" == "$TARGET_DIGEST" ]] || {
  echo 'SSH target differs from the authenticated task connection; authenticate that target first.' >&2
  exit 2
}
TASK_SOCKET="$(cat outputs/v6_recovery_20261026/ssh_control_socket.txt)"
POLL_TEMP="$(mktemp "$OUT/status-poll.XXXXXX")"
trap 'rm -f "$POLL_TEMP" "$POLL_TEMP.stderr"' EXIT
if ssh -S "$TASK_SOCKET" -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE_TARGET" \
  '/root/pi0-repro/openpi/.venv/bin/python /root/pi0-repro/server/temporal_train_status.py' \
  > "$POLL_TEMP" 2> "$POLL_TEMP.stderr"; then
  .venv/bin/python - "$POLL_TEMP" "$OUT" <<'PY'
from pathlib import Path
from datetime import datetime
import json,sys
path,out=Path(sys.argv[1]),Path(sys.argv[2]);data=json.loads(path.read_text())
if not isinstance(data,dict) or 'status' not in data:raise ValueError('Invalid status response')
data['_snapshot_provenance']={'freshness':'successful_remote_observation','observed_at':datetime.now().astimezone().isoformat()}
path.write_text(json.dumps(data,indent=2)+'\n');path.replace(out/'training_status.json')
(out/'latest_connection_check.json').write_text(json.dumps({'checked_at':data['_snapshot_provenance']['observed_at'],
    'status':'ssh_status_poll_succeeded','remote_training_state':data['status'],'actions':'Read-only status request; no job mutation.'},indent=2)+'\n')
print(json.dumps(data,ensure_ascii=False,indent=2))
PY
else
  .venv/bin/python - "$OUT" <<'PY'
from pathlib import Path
from datetime import datetime
import json,sys
out=Path(sys.argv[1]);record={'checked_at':datetime.now().astimezone().isoformat(),
    'status':'ssh_status_poll_failed','remote_training_state':'unverified',
    'last_successful_snapshot_preserved':True,'actions':'Read-only request; no job mutation.'}
(out/'latest_connection_check.json').write_text(json.dumps(record,indent=2)+'\n')
print('SSH status check failed; the last successful snapshot was preserved. Restore authentication before checking again.',file=sys.stderr)
PY
  exit 1
fi
