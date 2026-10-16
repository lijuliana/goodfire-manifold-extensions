#!/usr/bin/env bash
# SAE tiling on the core tier (needs results/<m>/activations.npy). Run on the GPU VM.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
PROVIDER="${PROVIDER:-azure}"
INSTANCE="${INSTANCE:-ditloracle-t4-caption}"
uv run manifolds sae --tier core --provider "$PROVIDER" --instance "$INSTANCE" 2>&1 | tee logs/sae_core.log
echo SAE_DONE
