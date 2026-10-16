#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"; INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
mkdir -p logs
uv run manifolds prompts --manifold weekdays_last months_last hours_last 2>&1 | tee logs/prompts_last.log
uv run manifolds extract --manifold weekdays_last months_last hours_last --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/extract_last.log
uv run manifolds analyze --manifold weekdays_last months_last hours_last 2>&1 | tee logs/analyze_last.log
PYTHONUNBUFFERED=1 uv run manifolds depth --manifold weekdays_last months_last hours_last --provider "$PROVIDER" --instance "$INSTANCE" --batch-size 8 --gpu-gib 11 2>&1 | tee logs/depth_last.log
echo LAST_DONE
