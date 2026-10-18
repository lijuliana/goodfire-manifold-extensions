#!/usr/bin/env bash
# On the instance: uv + project env (python 3.12), HF cache on the big root volume.
set -euo pipefail
cd ~/goodfire-manifold-replication
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --frozen --extra dev >/dev/null 2>&1 || uv sync --extra dev
uv run python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo SETUP_DONE
