#!/usr/bin/env bash
# Runs in the existing cloud environment; invoke inside tmux for disconnect safety.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="${1:-banana_pilot_v1}"
if [[ ! "$EXPERIMENT" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Invalid experiment name" >&2
  exit 2
fi
if [[ -f "$PROJECT_ROOT/env.sh" ]]; then source "$PROJECT_ROOT/env.sh"; fi
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$PROJECT_ROOT/cache/openpi"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache"
export HF_LEROBOT_HOME="$PROJECT_ROOT/data/lerobot/20261023-213415-07fb84"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT/openpi"
set +e
"$PROJECT_ROOT/openpi/.venv/bin/python" -u "$PROJECT_ROOT/server/banana_finetune.py" train \
  --root "$PROJECT_ROOT" --exp-name "$EXPERIMENT" "${@:2}"
RESULT=$?
printf '%s\n' "$RESULT" > "$PROJECT_ROOT/logs/$EXPERIMENT.exit-code"
exit "$RESULT"
