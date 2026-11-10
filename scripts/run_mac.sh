#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  printf '%s\n' '缺少 .venv；请先运行 scripts/setup_mac.sh。' >&2
  exit 1
fi
# Python skips .pth files with the macOS UF_HIDDEN flag. Resolve this
# checkout's source paths directly so the launcher works with hidden venvs.
if [[ ! -f "$PROJECT_DIR/src/pi0_repro/__main__.py" ]]; then
  printf '%s\n' '缺少 src/pi0_repro/__main__.py；请使用完整项目目录。' >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR/repos/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.mplconfig}"
exec "$PROJECT_DIR/.venv/bin/python" -m pi0_repro "$@"
