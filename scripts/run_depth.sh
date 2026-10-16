#!/usr/bin/env bash
# Phase 1b on the VM: augmented political bias (extract + analyze + SAE), then the full-model
# depth sweep + lenses + steering for the whole core tier.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"
INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
uv run manifolds prompts --manifold political_bias_aug 2>&1 | tee logs/prompts_aug.log
uv run manifolds extract --manifold political_bias_aug --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/extract_aug.log
uv run manifolds analyze --manifold political_bias_aug 2>&1 | tee logs/analyze_aug.log
uv run manifolds sae --manifold political_bias_aug --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/sae_aug.log
uv run manifolds depth --tier core --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_core.log
echo DEPTH_DONE
