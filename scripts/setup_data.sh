#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
RUNTIME="$PROJECT_DIR/.runtime/python311/bin/python"
if [[ ! -x "$RUNTIME" ]]; then
  printf '%s\n' '请先按 README 完成 Mac Python 3.11 运行时安装。' >&2
  exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv-data/bin/python" ]]; then
  "$RUNTIME" -m venv "$PROJECT_DIR/.venv-data"
fi
"$PROJECT_DIR/.venv-data/bin/python" scripts/fetch_lerobot_data_source.py
"$PROJECT_DIR/.venv-data/bin/python" -m pip install --only-binary=:all: \
  --index-url "${PI0_DATA_PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --find-links "$PROJECT_DIR/.cache/lerobot-wheels" \
  -r requirements/lerobot-data.lock.txt
printf '%s\n' '独立数据转换环境安装完成；未转换或上传数据，未启动训练。'
