#!/usr/bin/env bash
# Phases 2+3 on the VM: extended + cyclic tiers (prompts, extraction, analysis, SAE where block 19,
# full-model depth sweep + lenses + axis decoding).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"; INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
uv run manifolds prompts --tier extended 2>&1 | tee logs/prompts_ext.log
uv run manifolds prompts --tier cyclic 2>&1 | tee logs/prompts_cyc.log
uv run manifolds extract --tier extended --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/extract_ext.log
uv run manifolds extract --tier cyclic --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/extract_cyc.log
uv run manifolds analyze --tier extended 2>&1 | tee logs/analyze_ext.log
uv run manifolds analyze --tier cyclic 2>&1 | tee logs/analyze_cyc.log
uv run manifolds sae --tier extended --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/sae_ext.log
uv run manifolds sae --tier cyclic --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/sae_cyc.log
PYTHONUNBUFFERED=1 uv run manifolds depth --tier extended --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_ext.log
PYTHONUNBUFFERED=1 uv run manifolds depth --tier cyclic --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_cyc.log
echo PHASE2_DONE
