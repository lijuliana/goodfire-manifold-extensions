#!/usr/bin/env bash
# One-off, idempotent: the Claude-augmented set becomes `political_bias`; the raw-dataset results are dropped.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d results/political_bias_aug ]; then
  rm -rf results/political_bias
  mv results/political_bias_aug results/political_bias
  for f in results/political_bias/*.json; do
    python3 - "$f" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
if isinstance(d, dict) and d.get("name") == "political_bias_aug":
    d["name"] = "political_bias"; p.write_text(json.dumps(d))
PY
  done
  echo "renamed results/political_bias_aug -> results/political_bias"
fi
