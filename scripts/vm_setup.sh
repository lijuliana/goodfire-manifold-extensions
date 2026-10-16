#!/usr/bin/env bash
# One-time setup on a fresh GPU VM (Ubuntu). Idempotent.
set -euo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
uv python install 3.12
uv sync --frozen --extra dev
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
