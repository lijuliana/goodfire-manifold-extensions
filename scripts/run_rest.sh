#!/usr/bin/env bash
# 2026-10-18: bring every non-archived page up to the same set of methods. Layer-18 SAE tiling,
# steering (years fix + political bias, weekdays and months circles), then the four debate
# extensions on the 14 pages that lacked them.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/goodfire-manifold-replication
mkdir -p logs
NEW="temperature days years sent_length ordinals classical_composers statistical_uncertainty slant_rhymes_ore numbers_1_100 emotions_valence_arousal months_circle hours_circle weekdays_last hours_last"
P="--provider aws --instance g6e.2xlarge --batch-size 32"
W="--workers 7"
run() { echo "== $1 $(date -u +%H:%M:%S)"; shift; "$@" 2>&1 | tee -a logs/rest_$(date -u +%Y%m%d).log | grep -vE 'it/s\]|%\|' ; }
run sae          uv run manifolds sae --manifold weekdays_last months_last hours_last addition_sum_fourier --provider aws --instance g6e.2xlarge
run steer        uv run manifolds steer --manifold years political_bias weekdays_circle months_circle $P
run ablate       uv run manifolds ablate --manifold $NEW $P $W
run neurons      uv run manifolds neurons --manifold $NEW $P
run crosstalk    uv run manifolds crosstalk --manifold $NEW $P
run checkpoints  uv run manifolds checkpoints --manifold $NEW $P $W
echo REST_DONE
