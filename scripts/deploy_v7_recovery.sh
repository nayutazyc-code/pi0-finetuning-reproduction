#!/usr/bin/env bash
# Authenticate interactively once, then reuse that SSH session for this deployment.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
REMOTE_TARGET="${1:?Pass the existing SSH alias for the new GPU instance}"
[[ "$REMOTE_TARGET" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo 'Invalid SSH alias'; exit 2; }
REMOTE_ROOT=/root/pi0-repro
LOCAL_OUTPUT="$PROJECT_ROOT/outputs/v6_recovery_20261026"
DATASET_REL="$(.venv/bin/python -c 'import json;print(json.load(open("outputs/v6_recovery_20261026/training_preparation.json"))["dataset"])')"
MANIFEST_SHA="$(.venv/bin/python -c 'import json;print(json.load(open("outputs/v6_recovery_20261026/training_preparation.json"))["dataset_manifest_sha256"])')"
BASE_REL=data/lerobot/20261024-185715-merged-ebdbe3
[[ "$DATASET_REL" =~ ^data/lerobot/[A-Za-z0-9_-]+$ ]] || exit 2
TARGET_DIGEST="$(.venv/bin/python -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].encode()).hexdigest())' "$REMOTE_TARGET")"
CONTROL_SOCKET=""
if [[ -f "$LOCAL_OUTPUT/ssh_control_socket.txt" && -f "$LOCAL_OUTPUT/ssh_target_digest.txt" ]] && \
   [[ "$(cat "$LOCAL_OUTPUT/ssh_target_digest.txt")" == "$TARGET_DIGEST" ]]; then
  CANDIDATE_SOCKET="$(cat "$LOCAL_OUTPUT/ssh_control_socket.txt")"
  if ssh -S "$CANDIDATE_SOCKET" -O check "$REMOTE_TARGET" >/dev/null 2>&1; then CONTROL_SOCKET="$CANDIDATE_SOCKET"; fi
fi
if [[ -z "$CONTROL_SOCKET" ]]; then
  CONTROL_DIR="$(mktemp -d /private/tmp/pi0-v7-XXXXXX)"
  chmod 700 "$CONTROL_DIR"
  CONTROL_SOCKET="$CONTROL_DIR/ssh"
  ssh -M -S "$CONTROL_SOCKET" -o ControlPersist=3600 -fnN "$REMOTE_TARGET"
fi
printf '%s\n' "$CONTROL_SOCKET" > "$LOCAL_OUTPUT/ssh_control_socket.txt"
printf '%s\n' "$TARGET_DIGEST" > "$LOCAL_OUTPUT/ssh_target_digest.txt"
if [[ "${2:-}" == "--auth-only" ]]; then
  echo 'SSH session ready for this task. No data uploaded or training started.'
  exit 0
fi
echo 'Authenticated. Uploading reviewed training data and the separate v7 configuration.'
ssh -S "$CONTROL_SOCKET" "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT' '$DATASET_REL' '$BASE_REL' '$MANIFEST_SHA'" <<'PY'
from pathlib import Path
import hashlib, shutil, sys
root, relative, base = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
assert (root/'openpi/.venv/bin/python').exists()
assert (root/'checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999/params/_METADATA').exists()
norm=root/'checkpoints/pi0_banana_lora/banana_v6_long100_accum8000/7999/assets/local/pi0_banana_train/norm_stats.json'
assert hashlib.sha256(norm.read_bytes()).hexdigest()=='6c6e7a92643c76996bf15ba9fc3d3e346dd6bc3d8697712118d9fdb49dedf269'
manifest=root/relative/'dataset_manifest.json'
if manifest.exists() and hashlib.sha256(manifest.read_bytes()).hexdigest()!=sys.argv[4]:
    raise ValueError('Existing destination manifest differs; do not overwrite another candidate')
target=root/relative/'local/pi0_banana_train'
source=root/base/'local/pi0_banana_train'
if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    if source.exists(): shutil.copytree(source,target)
    else: target.mkdir()
for directory in ('configs','assets_v7_recovery','server','outputs/v7_recovery_20261026'):
    (root/directory).mkdir(parents=True,exist_ok=True)
PY
.venv/bin/python - "$DATASET_REL" "$LOCAL_OUTPUT/train_files.txt" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])/'local/pi0_banana_train'
files=[str(p.relative_to(root)) for p in sorted(root.rglob('*')) if p.is_file() and
       (p.relative_to(root).parts[0] in ('data','meta') or p.name in ('source_episodes.json','conversion_audit.json'))]
Path(sys.argv[2]).write_text('\n'.join(files)+'\n')
PY
rsync -rlpt --checksum --partial --files-from="$LOCAL_OUTPUT/train_files.txt" \
  -e "ssh -S $CONTROL_SOCKET" "$PROJECT_ROOT/$DATASET_REL/local/pi0_banana_train/" \
  "$REMOTE_TARGET:$REMOTE_ROOT/$DATASET_REL/local/pi0_banana_train/"
rsync -rlpt --checksum -e "ssh -S $CONTROL_SOCKET" "$PROJECT_ROOT/$DATASET_REL/dataset_manifest.json" "$REMOTE_TARGET:$REMOTE_ROOT/$DATASET_REL/"
rsync -rlpt --checksum -e "ssh -S $CONTROL_SOCKET" configs/banana_finetune_v7_recovery.toml "$REMOTE_TARGET:$REMOTE_ROOT/configs/"
rsync -rlpt --checksum -e "ssh -S $CONTROL_SOCKET" assets_v7_recovery/banana_sampling.json "$REMOTE_TARGET:$REMOTE_ROOT/assets_v7_recovery/"
rsync -rlpt --checksum -e "ssh -S $CONTROL_SOCKET" server/recovery_train_job.py server/banana_config.py server/banana_finetune.py \
  server/banana_accumulation.py server/banana_probe.py server/banana_sampling.py server/banana_model.py server/banana_roi.py \
  "$REMOTE_TARGET:$REMOTE_ROOT/server/"
ssh -S "$CONTROL_SOCKET" "$REMOTE_TARGET" "'$REMOTE_ROOT/openpi/.venv/bin/python' - '$REMOTE_ROOT'" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1])
with (root/'outputs/v7_recovery_20261026/supervisor.log').open('ab') as log:
    process=subprocess.Popen([str(root/'openpi/.venv/bin/python'),'-u',str(root/'server/recovery_train_job.py'),'--root',str(root)],
        cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('Supervisor process:',process.pid)
PY
echo 'V7 probe/training supervisor dispatched. Its status still needs to be checked.'
