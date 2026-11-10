#!/usr/bin/env bash
# Replace this task's unresponsive SSH master; never upload or launch a job.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Supply the existing SSH alias}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_.@-]+$ ]] || exit 2
OUT=outputs/v6_recovery_20261026
TARGET_DIGEST="$(.venv/bin/python -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$REMOTE_TARGET")"
if [[ -f "$OUT/ssh_control_socket.txt" && -f "$OUT/ssh_target_digest.txt" ]] && \
   [[ "$(cat "$OUT/ssh_target_digest.txt")" == "$TARGET_DIGEST" ]]; then
  OLD_SOCKET="$(cat "$OUT/ssh_control_socket.txt")"
  case "$OLD_SOCKET" in
    /private/tmp/pi0-v7-*/ssh|/private/tmp/pi0-motion-*/ssh)
      ssh -S "$OLD_SOCKET" -O exit "$REMOTE_TARGET" 2>/dev/null || true
      ;;
    *) echo 'Unrecognized task socket; refusing to close it.' >&2; exit 1 ;;
  esac
fi
CONTROL_DIR="$(mktemp -d /private/tmp/pi0-motion-XXXXXX)"
chmod 700 "$CONTROL_DIR"
CONTROL_SOCKET="$CONTROL_DIR/ssh"
IDENTITY_ARGS=()
if [[ -f "$OUT/ssh_identity_file.txt" ]]; then
  TASK_IDENTITY="$(cat "$OUT/ssh_identity_file.txt")"
  [[ -f "$TASK_IDENTITY" && ! -L "$TASK_IDENTITY" ]] || { echo 'Task SSH identity unavailable.' >&2; exit 1; }
  IDENTITY_ARGS=(-i "$TASK_IDENTITY" -o IdentitiesOnly=yes)
fi
ssh "${IDENTITY_ARGS[@]}" -M -S "$CONTROL_SOCKET" -o ControlPersist=3600 \
  -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -o IPQoS=none -fnN "$REMOTE_TARGET"
if ! .venv/bin/python - "$CONTROL_SOCKET" "$REMOTE_TARGET" <<'PY'
import subprocess,sys
try:
    result=subprocess.run(['ssh','-S',sys.argv[1],'-o','BatchMode=yes',sys.argv[2],'true'],timeout=25)
except subprocess.TimeoutExpired:
    raise SystemExit('SSH authenticated, but the remote command timed out; task connection was not saved.')
raise SystemExit(result.returncode)
PY
then
  ssh -S "$CONTROL_SOCKET" -O exit "$REMOTE_TARGET" 2>/dev/null || true
  exit 1
fi
printf '%s\n' "$CONTROL_SOCKET" > "$OUT/ssh_control_socket.txt"
printf '%s\n' "$TARGET_DIGEST" > "$OUT/ssh_target_digest.txt"
echo 'Remote round trip verified. No data uploaded, files deleted, or training started.'
