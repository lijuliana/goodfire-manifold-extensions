# Replication report: Goodfire concept manifolds in Llama-3.1-8B

Status as of 2026-10-16 (phase 1 + extensions). Site: `site/index.html` (published as a private
artifact). Every number below is read from `results/<manifold>/*.json`; nothing is hand-edited.

## Setup

- Model: `meta-llama/Llama-3.1-8B` (base), served from the ungated mirror `unsloth/Llama-3.1-8B`
  (identical weights; the gated repo was not accessible without a token). Residual stream after
  block 19, last token, as in Bhalla et al. 2026 App. B and the released `data.py`.
- Hardware: Azure `Standard_NC16as_T4_v3` (one T4, 16 GB). Extraction used blocks 0–19 in fp16,
  checked against a float32 reference on 16 prompts (cosine ≥ 0.99999, max relative error 0.3%).
  The full-model runs (all 32 blocks + output head) kept 20 blocks on the GPU and streamed the
  rest from CPU (exact fp16).
- Prompts: verbatim ports of the released builders. Deviations recorded per manifold on the site.

## What was replicated (core tier, Bhalla et al. 2026 Table 1)

| manifold | paper's shape | n | verdict | key numbers |
|---|---|---|---|---|
| temperature | line | 150 | matches | Spearman 0.99 along best direction, kNN R² 0.99 |
| age | line | 99 | matches | Spearman 0.99, kNN R² 0.97 |
| years | helix | 199 | matches | linear R² 0.94; ones-digit circle after detrending, circular corr 0.72 |
| days | circle | 56 | matches | 7 day centroids in cyclic order; circular corr 0.98 |
| geography | hierarchical tree | 3007 | matches | continent 5-NN accuracy 0.61 above the majority baseline |
| sent_length | line | 4982 | matches | Spearman 0.83, kNN R² 0.74 |
| colors | paraboloid | 1848 | partial | hue circle 0.96 and lightness axis 0.83 clear; paraboloid curvature (height ~ r²) weak, R² 0.14 |
| formality | line | 1000 | partial | Spearman 0.70, kNN R² 0.49 |
| political_bias | 1-D continuous | 799 | matches | our Claude-generated, level-balanced statements + lean template (the paper's augmentation step, re-done): Spearman 0.93, kNN R² 0.91, centroids in order. The raw 657-statement dataset alone, without a template, gave only a partial match (Spearman 0.44, kNN R² 0.33; the neutral class forms its own cluster); that run is not on the site. |

Rubric: v2 (`src/manifolds/analysis/shapes.py`, documented there). v1 required the label to explain
90% of the top-3 variance and judged formality, sent_length and political_bias "failed"; it was
replaced after the first run because it conflated manifold presence with variance share. Both
verdicts are stored in every report (`status`, `status_v1`).

Discrepancies with the paper that are not ours to fix:
- `days`: Table 1 reports n = 420, the released code produces 56 prompts (7 days × 8 times). We ran the code.
- `colors` / `geography`: the public datasets have grown since the paper (1848 vs ~900 and 3007 vs ~4000 prompts after the released filters).
- `political_bias`: the paper used GPT-5 augmentations that were never released; we re-did the augmentation with Claude (`scripts/augment_political_bias.py`, provenance in `data/political_bias.jsonl`).

## SAE tiling (paper Sec. 5), with a public SAE

The paper's SAEs were not released. We used Llama Scope `L19R-8x` (JumpReLU, 32k features, same base
model and hook point), loaded with the OpenMOSS v1.0.0 inference recipe. On the manifold prompts it
fires 27–61 features per token and rebuilds 53–86% of the activation energy. Every manifold shows the
paper's dilution signature: the greedy span of SAE atoms explains 8–44% of what PCA explains with
`ambient_dim` components, needs 6–29 atoms (or > 64 for temperature and colors) to catch up, and
beats random atoms by 11–28×. Reconstruction error always exceeds the within-manifold spread
(`ve_within` < 0), which is why SAE codes alone render these manifolds poorly.

## Extensions (not in the sources)

Methods from other papers applied to every manifold page (sources in `docs/sources.md`).

**Layer slider.** The same prompts read out at all 33 residual positions (0 = embeddings, i =
output of block i−1; the paper's layer 19 is position 20), with the rubric's critical metrics
plotted against depth. Shapes emerge at different depths: temperature and age are lines from
position 1 on; the days-of-week circle is present in the raw embeddings, disappears over
positions 4–13 and returns from 14 to 30; the colour hue circle only passes from position 21;
the years helix is a narrow mid-depth band (positions 20–23); geography's continent tree appears
from position 15; formality never passes.

**Three lenses.** Logit lens, tuned lens (`kdhole/Llama-3.1-8B-tuned-lens`) and the Jacobian
lens (Gurnee et al. 2026; pre-fitted base Llama-3.1-8B lens from Neuronpedia) decode what each
point is disposed to make the model say, shown on hover at every depth. On geography, where the
template's real next token is a country name, the Jacobian lens' top-1 at position 20 agrees with
the model's real next token on 76% of prompts, against 41% for the logit lens and 7% for the tuned
lens. On the other templates the real next token is punctuation, so top-1 agreement is near zero
for all lenses and the comparison is uninformative.

**What the axes mean.** Each principal axis is decoded four ways (three lenses on the direction,
plus the lens-free correlation between the coordinate and the model's real next-token
log-probabilities over word-like tokens). The lenses agree with each other and with the
plots: temperature PC1 runs from snow/frost/freezing to sun/sunshine/tropical; colours PC1 from
purple to yellow/golden and PC2 from blue to red/brick (the hue circle's two axes), PC3 from pink
to black (lightness); days PC1 from Monday/Sunday to Friday and PC2 from the weekend to
Tuesday/Wednesday (the circle's two axes); years PC1 from Soviet/197x/196x to Victorian/184x, while
PC2/PC3 (the ones-digit circle) read out ordinal suffixes (s, st, nd, rd, th); age PC1 from
girls/boys to retirement/retired, PC2 married/marriage to puppies/youngsters; geography PC1 from
Africa/Caribbean to Europe/Baltics, PC2 Central America to Africa/Madagascar, PC3 Kurdistan/Iraq
to Brazil/Uruguay; formality PC1 from you/my/really (and emoticons in the empirical readout) to
Reuters/spokesman. Political bias separates into a lean axis (PC1: socialist/feminist/progressive
vs conservative/GOP/Republican, identical under the Jacobian lens and the lens-free readout) and
an intensity axis (PC3: extremist/ultra/very vs neutral/balanced/bipartisan), which is why the
raw-dataset run looked like two arms with a separate neutral cluster. Sentence length's PC1 reads
question words (what/why/how) against web boilerplate (Contains/Download/Page), i.e. length is
entangled with genre in WikiText. For template manifolds (temperature, years, days) the lens-free
readout is uninformative because the template fixes the next-token distribution; the lenses on the
direction are what carry the meaning there.

**Steering (App. B.1, Table 2).** Interpolating between bin centroids in PCA space and adding
the delta at block 19: temperature moves the answer from "cold" to "warm" along the path; colours
walk the hue circle purple → yellow → green → blue → purple; days moves Monday → Thursday → Friday
→ Sunday but stays on the base day for most of the path; age flips between "old" and "young"
non-monotonically; years never leaves "17" (the Table 2 target tokens are century prefixes and the
base prompt's own answer dominates). We report these as obtained; the paper's Fig. 2 shows smooth
transitions for days and temperature.

## Skipped (and why)

- Extended text tier (10) and cyclic tier (4): scheduled after review of the core tier.
- Vision tier (DINOv3 lighting/shadow, InceptionV1 curves, SDXL pretzel, DINOv2 position sheet): listed in `manifest.yaml`; not run (user decision, 2026-10-15).
- Unsupervised Ising discovery pipeline: needs the paper's SAEs and a 500M-token background corpus; the blog's discovered manifolds are approximated with supervised prompt sets instead.
- Template lens / activation oracles: no pre-trained oracle for base Llama-3.1-8B; single-token readouts suffice here.

## Compute

GPU ledger: `logs/gpu_ledger.jsonl` (`uv run manifolds ledger summary`). Billed Azure T4 uptime,
reconciled against the Azure activity log: 3.59 GPU-hours over three sessions (one of them
ran ~2 h longer than needed because a cycle script died after a crash without deallocating). Actual
model computation inside those sessions was well under an hour. No AWS or GCP GPU time was used.
