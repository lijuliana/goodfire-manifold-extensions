#!/usr/bin/env bash
# Debate extensions on the pilot manifolds (runs on the GPU instance, detached; see aws_cycle.sh).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/goodfire-manifold-replication
mkdir -p logs
PROVIDER="${PROVIDER:-aws}"; INSTANCE="${INSTANCE:-g6e.xlarge}"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MIB:-0}" -lt 30000 ]; then LMB=2; BS_DEF=16; else LMB=4; BS_DEF=32; fi
P="--provider $PROVIDER --instance $INSTANCE --batch-size ${BS:-$BS_DEF}"
PILOT="${PILOT:-addition_sum_fourier colors age weekdays_circle geography months_last political_bias}"
run() { echo "== $1 $(date -u +%H:%M:%S)"; shift; "$@" 2>&1 | tee -a logs/ext_$(date -u +%Y%m%d).log | grep -vE 'it/s\]|%\|' ; }
run interference uv run manifolds interference --manifold $PILOT
run ablate      uv run manifolds ablate --manifold $PILOT $P --lm-batch $LMB
run neurons     uv run manifolds neurons --manifold $PILOT $P
run crosstalk   uv run manifolds crosstalk --manifold $PILOT $P
run checkpoints uv run manifolds checkpoints --manifold $PILOT $P
echo EXT_DONE
