#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ARGS=()
FRESH_SEED=true
SSH_TARGET="${PI0_SSH_TARGET:-}"
LOCAL_PORT=""
PROFILE_CONFIG="$PROJECT_DIR/configs/banana_v6_random_multiview_pi0.toml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo "Usage: bash scripts/run_random_multiview.sh [--episodes N] [--seed N] [--runtime-optimized] [--ssh-target ALIAS | --port PORT]"
      echo "Default: new random seed; reuse the saved SSH alias when available, otherwise local port 8001."
      exit 0 ;;
    --runtime-optimized)
      PROFILE_CONFIG="$PROJECT_DIR/configs/banana_v6_random_multiview_runtime_pi0.toml"; shift ;;
    --ssh-target)
      [[ $# -ge 2 ]] || { echo 'Missing SSH alias' >&2; exit 2; }
      SSH_TARGET="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { echo 'Missing local port' >&2; exit 2; }
      LOCAL_PORT="$2"; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    --seed)
      [[ $# -ge 2 ]] || { echo 'Missing seed' >&2; exit 2; }
      FRESH_SEED=false; RUN_ARGS+=("$1" "$2"); shift 2 ;;
    *) RUN_ARGS+=("$1"); shift ;;
  esac
done
SAVED_TARGET="$PROJECT_DIR/outputs/random_multiview_connection/ssh_target.txt"
if [[ -z "$SSH_TARGET" && -z "$LOCAL_PORT" && -f "$SAVED_TARGET" ]]; then
  SSH_TARGET="$(cat "$SAVED_TARGET")"
fi
if [[ -n "$SSH_TARGET" ]]; then
  [[ -z "$LOCAL_PORT" ]] || { echo 'Use --ssh-target or an existing --port, not both' >&2; exit 2; }
  [[ "$SSH_TARGET" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ]] || { echo 'Invalid SSH alias' >&2; exit 2; }
  LOCAL_PORT="$("$PROJECT_DIR/.venv/bin/python" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
  ssh -i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=10 -o IPQoS=none \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -N -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:8001" "$SSH_TARGET" &
  TUNNEL_PID=$!
  cleanup() { kill "$TUNNEL_PID" 2>/dev/null || true; wait "$TUNNEL_PID" 2>/dev/null || true; }
  trap cleanup EXIT
  "$PROJECT_DIR/.venv/bin/python" - "$LOCAL_PORT" "$TUNNEL_PID" <<'PY'
import os, socket, sys, time
port, pid = map(int, sys.argv[1:])
deadline = time.monotonic() + 15
while time.monotonic() < deadline:
    os.kill(pid, 0)
    with socket.socket() as connection:
        connection.settimeout(.3)
        if connection.connect_ex(('127.0.0.1', port)) == 0:
            break
    time.sleep(.1)
else:
    raise SystemExit('Own SSH forwarding was not ready within 15 seconds')
PY
  RUN_ARGS+=(--port "$LOCAL_PORT")
fi
COMMAND=(bash "$PROJECT_DIR/scripts/run_mac.sh" run
  --config "$PROFILE_CONFIG" --display)
if [[ "$FRESH_SEED" == true ]]; then COMMAND+=(--fresh-seed); fi
if [[ ${#RUN_ARGS[@]} -gt 0 ]]; then COMMAND+=("${RUN_ARGS[@]}"); fi
"${COMMAND[@]}"
