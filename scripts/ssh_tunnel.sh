#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 4 ]]; then
  printf '%s\n' '用法：bash scripts/ssh_tunnel.sh user@server [ssh_port=22] [local_port=8000] [remote_port=8000]' >&2
  exit 2
fi
SSH_TARGET="$1"
SSH_PORT="${2:-22}"
LOCAL_PORT="${3:-8000}"
REMOTE_PORT="${4:-8000}"
if [[ "$SSH_TARGET" == -* || "$SSH_TARGET" == *[[:space:]]* ]]; then
  printf '%s\n' 'SSH 地址格式应为 user@server。' >&2
  exit 2
fi
for PORT_VALUE in "$SSH_PORT" "$LOCAL_PORT" "$REMOTE_PORT"; do
  if [[ ! "$PORT_VALUE" =~ ^[0-9]+$ ]] || (( 10#$PORT_VALUE < 1 || 10#$PORT_VALUE > 65535 )); then
    printf '%s\n' '端口必须为 1–65535 的整数。' >&2
    exit 2
  fi
done
printf '建立隧道：127.0.0.1:%s → 服务器 127.0.0.1:%s；保持此终端打开。\n' "$LOCAL_PORT" "$REMOTE_PORT"
SSH_OPTIONS=(-N -C -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3)
if [[ -n "${PI0_SSH_IDENTITY:-}" ]]; then
  SSH_OPTIONS+=(-i "$PI0_SSH_IDENTITY" -o IdentitiesOnly=yes)
fi
exec ssh "${SSH_OPTIONS[@]}" \
  -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" -p "$SSH_PORT" "$SSH_TARGET"
