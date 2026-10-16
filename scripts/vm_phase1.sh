#!/usr/bin/env bash
# Phase 1 on the VM: download the model (ungated mirror unless HF_TOKEN grants the gated repo), then run the core tier.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
if [ -n "${HF_TOKEN:-}" ]; then
  uv run hf download meta-llama/Llama-3.1-8B --exclude "*.pth" --exclude "original/*" || true
fi
uv run hf download unsloth/Llama-3.1-8B --exclude "*.pth"
echo DOWNLOAD_DONE
bash scripts/run_core.sh
echo RUN_DONE
