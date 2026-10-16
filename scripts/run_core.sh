#!/usr/bin/env bash
# Phase 1: the nine core manifolds. Run on the GPU VM from the repo root.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"
INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
uv run manifolds prompts --tier core 2>&1 | tee logs/prompts_core.log
uv run manifolds extract --tier core --provider "$PROVIDER" --instance "$INSTANCE" --precision-check 2>&1 | tee logs/extract_core.log
uv run manifolds analyze --tier core 2>&1 | tee logs/analyze_core.log
uv run manifolds site 2>&1 | tee logs/site.log
uv run manifolds ledger summary
