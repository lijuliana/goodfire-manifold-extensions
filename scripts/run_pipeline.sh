#!/usr/bin/env bash
# The canonical per-page pipeline (2026-10-18). Runs every method on every non-archived page in one
# session, on one GPU, with the same parameters, so pages differ only where the method says so
# (probe view on Fourier-probe manifolds, SAE at the read layer, steering where a task is defined).
#
# Inputs reused, identical for every page: the block-19/18 activations (Azure T4 extraction), the
# depth sweep's per-layer scores and lens readouts, the LLM axis readings, and the shared
# weak-activation baseline (results/_ablation: calibration, LM loss, PopQA).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/goodfire-manifold-replication
mkdir -p logs
INSTANCE="${INSTANCE:-unknown}"
export CODE_COMMIT="${CODE_COMMIT:-unknown}"
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MIB:-0}" -lt 30000 ]; then BS=12; GIB=21; else BS=32; GIB=44; fi
P="--provider aws --instance $INSTANCE --batch-size $BS --gpu-gib $GIB"
W="--workers $(( $(nproc) > 2 ? $(nproc) - 1 : 1 ))"
run() { echo "== $1 $(date -u +%H:%M:%S)"; shift; "$@" 2>&1 | tee -a logs/pipeline_$(date -u +%Y%m%d).log | grep -vE 'it/s\]|%\|' ; }
# no --manifold: every non-archived page (manifest order)
run interference uv run manifolds interference
run sae          uv run manifolds sae --provider aws --instance "$INSTANCE"
run steer        uv run manifolds steer $P
run layers       uv run manifolds depth --figure-only --recapture --dtype bfloat16 $P
run ablate       uv run manifolds ablate $P $W --force
run neurons      uv run manifolds neurons $P --force
run crosstalk    uv run manifolds crosstalk $P --force
run checkpoints  uv run manifolds checkpoints $P $W --recompute-metrics
echo PIPELINE_DONE
