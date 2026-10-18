#!/usr/bin/env bash
# ssh/rsync helper for the AWS L40S instance (state in logs/aws_instance.json, written by aws_launch.sh).
# usage: scripts/aws_ssh.sh 'command' | --sync (push repo + activations) | --pull (fetch extension results + logs)
set -euo pipefail
cd "$(dirname "$0")/.."
KEY="$HOME/.ssh/cs2881r.pem"
IP=$(python3 -c "import json;print(json.load(open('logs/aws_instance.json'))['ip'])")
HOST="ubuntu@$IP"
OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
if [ "${1:-}" = "--sync" ]; then
  rsync -az --delete -e "ssh ${OPTS[*]}" \
    --exclude /.venv --exclude /.git --exclude /results --exclude /site --exclude /logs \
    --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache \
    ./ "$HOST:~/goodfire-manifold-replication/"
  # only the inputs the extensions need: cached activations, prompts, meta, reports
  rsync -az -e "ssh ${OPTS[*]}" --include '*/' --include 'activations.npy' --include 'prompts.json' \
    --include 'meta.json' --include 'report.json' --include 'status.json' --exclude '*' \
    results/ "$HOST:~/goodfire-manifold-replication/results/"
  echo "synced"
elif [ "${1:-}" = "--pull" ]; then
  rsync -az -e "ssh ${OPTS[*]}" --include '*/' \
    --include 'ablation*.json' --include 'neurons*.json' --include 'crosstalk*.json' \
    --include 'woven*.json' --include 'checkpoints*.json' \
    --include 'sae*.json' --include 'steer_figure.json' --include 'depth.json' \
    --include '_ablation/global.json' --include '_ablation/calibration_amax.pt' \
    --include '_interference/*.json' --include '_interference/null_bases.npy' \
    --include '_checkpoints/summary.json' --include '_checkpoints/raw/*.json' --exclude '*' \
    "$HOST:~/goodfire-manifold-replication/results/" results/
  mkdir -p logs/aws
  rsync -az -e "ssh ${OPTS[*]}" "$HOST:~/goodfire-manifold-replication/logs/" logs/aws/
  uv run manifolds ledger merge --from logs/aws/gpu_ledger.jsonl
  echo "pulled extension results + logs (logs under logs/aws/, ledger merged)"
else
  ssh "${OPTS[@]}" "$HOST" "$@"
fi
