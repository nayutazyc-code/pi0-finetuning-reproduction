#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
LEROBOT_SOURCE="$PROJECT_DIR/.vendor/lerobot-0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
if [[ ! -x "$PROJECT_DIR/.venv-data/bin/python" || ! -d "$LEROBOT_SOURCE/lerobot" ]]; then
  printf '%s\n' '缺少独立数据环境或固定版本 LeRobot 源码；见 docs/LeRobot数据转换.md。' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_DIR/src:$LEROBOT_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PROJECT_DIR/.cache/huggingface-data"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache/huggingface-data/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
exec "$PROJECT_DIR/.venv-data/bin/python" -m pi0_repro.lerobot_export "$@"
