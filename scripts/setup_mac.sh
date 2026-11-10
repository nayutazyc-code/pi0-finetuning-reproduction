#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  printf '%s\n' '本安装入口针对 Apple Silicon macOS；云端使用 server/README.md。' >&2
  exit 1
fi
git submodule update --init repos/openpi
if [[ ! -x "$PROJECT_DIR/.runtime/python311/bin/python" ]]; then
  CONDA_BIN="${PI0_CONDA:-conda}"
  "$CONDA_BIN" create --no-default-packages --prefix "$PROJECT_DIR/.runtime/python311" \
    --file "$PROJECT_DIR/requirements/python-runtime.osx-arm64.explicit.txt" -y
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "$PROJECT_DIR/.runtime/python311/bin/python" -m venv "$PROJECT_DIR/.venv"
fi
"$PROJECT_DIR/.venv/bin/python" -m pip install 'pip==25.3' 'setuptools==79.0.1' 'wheel==0.45.1'
"$PROJECT_DIR/.venv/bin/python" -m pip install -r requirements/macos-arm64.lock.txt
"$PROJECT_DIR/.venv/bin/python" -m pip install --no-deps --no-build-isolation -e repos/openpi/packages/openpi-client
"$PROJECT_DIR/.venv/bin/python" -m pip install --no-deps --no-build-isolation -e .
printf '%s\n' '安装完成；此脚本只安装依赖，不启动仿真或模型推理。'
