#!/usr/bin/env bash
# Re-score the extended tier's depth sweep on cached states (rubric consistency), VM.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"; INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
PYTHONUNBUFFERED=1 uv run manifolds depth --tier extended --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_ext2.log
echo DEPTH_EXT_DONE
