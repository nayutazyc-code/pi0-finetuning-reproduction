#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:?training TOML required}"
CHECKPOINT="${2:?checkpoint directory required}"
DENOISE_STEPS="${3:-10}"
RTC_ARGS=()
if [[ "${PI0_RTC:-0}" == "1" ]]; then RTC_ARGS+=(--rtc); fi
if [[ -f "$PROJECT_ROOT/env.sh" ]]; then source "$PROJECT_ROOT/env.sh"; fi
export HF_HOME="$PROJECT_ROOT/cache/huggingface"
export OPENPI_DATA_HOME="$PROJECT_ROOT/cache/openpi"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache"
export JAX_PLATFORMS=cuda
export JAX_COMPILATION_CACHE_DIR="$PROJECT_ROOT/cache/jax-serve"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT_ROOT/openpi"
exec "$PROJECT_ROOT/openpi/.venv/bin/python" -u "$PROJECT_ROOT/server/serve_pi0.py" \
  --profile banana --training-config "$CONFIG_FILE" --checkpoint "$CHECKPOINT" --denoise-steps "$DENOISE_STEPS" "${RTC_ARGS[@]}"
