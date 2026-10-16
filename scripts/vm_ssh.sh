#!/usr/bin/env bash
# ssh/rsync helper for the Azure T4 VM used in this project.
# usage: scripts/vm_ssh.sh 'command'      |  scripts/vm_ssh.sh --sync   (push repo to VM)
set -euo pipefail
KEY="$HOME/.ssh/azure_manifolds_ed25519"
HOST="juliana@20.240.250.7"
OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--sync" ]; then
  rsync -az --delete -e "ssh ${OPTS[*]}" \
    --exclude /.venv --exclude /.git --exclude /results --exclude /site --exclude /logs \
    --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache \
    ./ "$HOST:~/goodfire-manifold-replication/"
  echo "synced"
elif [ "${1:-}" = "--pull" ]; then
  rsync -az -e "ssh ${OPTS[*]}" --exclude 'depth_states.npy' \
    "$HOST:~/goodfire-manifold-replication/results/" results/
  mkdir -p logs/vm
  rsync -az -e "ssh ${OPTS[*]}" "$HOST:~/goodfire-manifold-replication/logs/" logs/vm/
  bash scripts/rename_political.sh
  uv run manifolds ledger merge --from logs/vm/gpu_ledger.jsonl
  for t in core extended cyclic; do uv run manifolds analyze --tier $t >/dev/null 2>&1 || true; done   # reports always under the local rubric
  echo "pulled results + logs (VM logs under logs/vm/, ledger merged, reports re-analysed)"
else
  ssh "${OPTS[@]}" "$HOST" "$@"
fi
