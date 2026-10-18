#!/usr/bin/env bash
# Probe-view re-run (2026-10-18): layer sliders for the five Fourier-probe manifolds, ablation +
# neuron sliders for the two probe pilots, and the OLMo checkpoint sweep for all pilots (states
# must be recaptured on a fresh instance; cached per-checkpoint metrics are reused).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/goodfire-manifold-replication
mkdir -p logs
P="--provider aws --instance g6e.2xlarge --batch-size 32"
run() { echo "== $1 $(date -u +%H:%M:%S)"; shift; "$@" 2>&1 | tee -a logs/probe_$(date -u +%Y%m%d).log | grep -vE 'it/s\]|%\|' ; }
run depth-figures uv run manifolds depth --figure-only --dtype bfloat16 --gpu-gib 44 --manifold addition_sum_fourier months_last weekdays_last hours_last numbers_1_100 $P
run ablate uv run manifolds ablate --manifold addition_sum_fourier months_last $P --force
run neurons uv run manifolds neurons --manifold addition_sum_fourier months_last $P --force
run checkpoints uv run manifolds checkpoints --manifold addition_sum_fourier colors age weekdays_circle geography months_last political_bias $P
echo PROBE_DONE
