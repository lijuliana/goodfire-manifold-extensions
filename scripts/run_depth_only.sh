#!/usr/bin/env bash
# Depth sweep + logit/tuned/Jacobian lenses + steering + PC decoding for the core tier (VM).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"; INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
uv run manifolds depth --tier core --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_core.log
echo DEPTH_DONE
