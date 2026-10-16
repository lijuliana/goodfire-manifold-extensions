# Goodfire concept-manifold replication

Replication of the concept manifolds reported in Goodfire's Neural Geometry series and the papers
behind it (Bhalla et al. 2026, Feucht et al. 2026, Fel et al. 2026, Fel et al. 2025), published as a
static site of interactive 3-D PCA plots with a replication-status badge per manifold.

- `manifest.yaml` — Step 0: every manifold the sources report (name, model, layer, expected geometry,
  template/dataset, source, replication mode). Status fields are never hand-edited.
- `docs/sources.md` — the facts extracted from each source, with links.
- `src/manifolds/` — typed pipeline: prompt builders (ported from the released `data.py`),
  activation extraction (truncated Llama-3.1-8B, forward hook on block 19, last token), PCA,
  quantitative shape checks, Goodfire-style Plotly figures, static-site builder, GPU-hour ledger.
- `tests/` — unit tests on synthetic geometry and a random tiny Llama (no downloads).
- `results/<manifold>/` — prompts, activations (git-ignored), PCA coords, `report.json`, figures.
- `site/` — the built site (`index.html` + one page per manifold).
- `logs/gpu_ledger.jsonl` — GPU hours.

## Run

```bash
uv sync --extra dev
uv run pytest -q
uv run manifolds prompts --tier core
uv run manifolds extract --tier core --provider azure --instance <vm> --precision-check
uv run manifolds analyze --tier core
uv run manifolds site
uv run manifolds status
uv run manifolds ledger summary
```

Model access: the pipeline tries `meta-llama/Llama-3.1-8B` (gated; set `HF_TOKEN`) and falls back to
the ungated mirror `unsloth/Llama-3.1-8B` of the same weights. The mirror actually used is recorded
in every `results/<manifold>/meta.json` and on the site.

## Reproducibility

Seeds, the resolved config, model commit hash, dtype, device and software versions are written to
`results/<manifold>/meta.json`. PCA uses the deterministic full SVD solver. fp16 extraction on T4 is
checked against a float32 CPU reference (`precision_check.json`).
