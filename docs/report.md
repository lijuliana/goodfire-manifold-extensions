# Replication report: Goodfire concept manifolds in Llama-3.1-8B

Status as of 2026-10-17 (phase 1, extensions, and the debate extensions on seven pilot manifolds). Site: `site/index.html` (published as a private
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

**What the axes mean.** Each principal axis is decoded four ways, and Claude Opus 5.5 writes a
one-line reading of the four token lists with a confidence flag (`results/<m>/axis_labels.json`,
shown above the tables; a guide, not a measurement). Each principal axis is decoded four ways (three lenses on the direction,
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

## Extended tier (blog manifolds, supervised approximations; rubric v2.1)

The blog's unsupervised figure names eight manifolds without templates; two more come from other
posts. All templates and item lists are ours (recorded on each page), so these test whether the
*same variables* organise block-19 activations under a reasonable template, not the blog's exact
objects.

| manifold | ordered by | n | verdict | notes |
|---|---|---|---|---|
| ordinals | rank first..hundredth | 100 | matches | clean line along PC1 (49% of variance) |
| classical_composers | birth year | 84 | matches | chronological line |
| scotus_cases | decision year | 57 | failed | year ordering is present along the best direction (Spearman ≥ 0.8) but the nonlinear readability check fails at this sample size |
| chemical_bonds | bond length (pm) | 47 | failed | no ordering by length at the answer position (Spearman 0.49); the 8B base model may simply not know these values |
| adverbs_pace | our speed ranking of 18 adverbs × 5 frames | 90 | partial | after centring per sentence frame the ranking is readable (partial band); centroid path broken |
| statistical_uncertainty | ln(std/mean) in "mean +/- std" strings | 300 | partial | Spearman 0.62, kNN R² 0.47 after frame centring; the paper found this manifold unsupervised in scientific text |
| bible_verses | book order × chapter | 180 | matches | books trace a line in canonical order |
| slant_rhymes_ore | rhyme grade with "-ore" (3 levels) | 75 | matches | perfect / slant / none separate and order |
| numbers_1_100 | n | 100 | matches | tested as the source claims it: Fourier probes for periods 10, 5, 2 read n well; the raw PCA is a magnitude line, not a visible helix |
| emotions_valence_arousal | six emotions (Claude-generated sentences) | 240 | partial | emotions cluster and are separable, but centroid distances do not track Russell's valence-arousal circumplex in PCA-3 |

## Cyclic tier (Feucht et al. 2026 templates)

| manifold | read at | verdict | notes |
|---|---|---|---|
| weekdays_circle | concept token, block 19 | matches | circle with all 7 days in order (98 prompts) |
| months_circle | concept token, block 19 | partial | circle (circular corr passes); 10 of 12 months in cyclic order |
| hours_circle | concept token, block 19 | partial | circle; 22 of 24 hours in cyclic order (1152 prompts) |
| addition_sum_fourier | last token "=", block 18 | matches | ridge Fourier probes read the sum with R² 0.91 (period 10), 0.93 (period 5), 0.98 (period 2); a control probe for period 7 scores near zero. The mod-10 circle is not visible in the raw 3-D PCA, as the paper says. |
| weekdays_last | last token, block 18 | matches | the paper's negative claim (Fig. 3): the weekday circle is absent here (circular corr 0.00), while probes read the pre-modulo sum day + offset with R² 0.95 (period 10) |
| months_last | last token, block 18 | matches | month circle absent (0.08); month + offset probes R² 0.94 (period 10), control period 7 ≈ 0 |
| hours_last | last token, block 18 | matches | hour circle absent (0.09); hour + offset probes R² 0.84 (period 10), 0.96 (period 2) |

For the probe-based manifolds the site shows a "Fourier-probe view" as the main plot: points at
out-of-fold ridge-probe predictions (cos, sin of the ones digit; magnitude), which renders the
circle-plus-magnitude helix the raw PCA cannot show (addition: R² 0.91 circle, 0.997 magnitude).

Rubric v2.1 changes (made after the first pass on these tiers, documented in `shapes.py`): k-NN
neighbour count scales with n; cyclic order is graded (fraction of expected neighbours adjacent,
partial ≥ 0.75) instead of all-or-nothing for 12- and 24-class circles; informative checks never
gate; multi-frame prompt sets are centred per frame before PCA. Under the first pass months and
hours were "failed" and addition "partial"; those verdicts changed only through these mechanics.

## Extensions II: the "already clean" debate on seven pilot manifolds (2026-10-17)

Prompted by Cammarata's *Neural Networks Are Already Clean*, Gao's reply, and the research plan in
`concept_manifold_research_plan.md`. Pilots: addition_sum_fourier, colors, age, weekdays_circle,
geography, months_last, political_bias. Hardware: one AWS g6e.2xlarge (L40S 48 GB), Llama-3.1-8B
in bf16 with no offload (block-19 activations agree with the cached fp16 extraction at cosine
≥ 0.9995). All numbers are in `results/<manifold>/{ablation,neurons,crosstalk,checkpoints}.json`
and `results/_ablation`, `results/_interference`, `results/_checkpoints`. Sources: `docs/sources.md`.

### 1. Weak-activation ablation vs geometry (`manifolds ablate`)

Method ported from Li 2026 (`weak-activation-ablation`): zero every SwiGLU activation in every
MLP block with |a| < X% of that neuron's max over 512 × 1024 FineWeb-edu tokens (position 0
excluded). Controls: the same count zeroed at random per (layer, token); the weak tail scaled by
s = 0.5 or 2 instead of zeroed. The manifold is re-read at the paper's layer under the same
perturbation that the behaviour is scored under.

Model-wide (64 × 1024 held-out FineWeb-edu tokens; 300 PopQA questions, 8-shot, clean 28.7%):

| condition | share of activations zeroed | Δ held-out loss (nats) | PopQA |
|---|---|---|---|
| weak X = 0.5% | 33% | +0.017 | 30.0% |
| weak X = 1% | 49% | +0.074 | 25.3% |
| weak X = 2% | 68% | +0.264 | 20.0% |
| weak X = 3% | 78% | +0.556 | 16.0% |
| weak X = 5% | 89% | +1.513 | 7.7% |
| random, count matched to X = 0.1% | 10% | +0.474 | 8.0% |
| random, count matched to X = 0.5% | 38% | +3.49 | 0.7% |
| tail × 0.5 below 1.5% | (61% affected) | +0.047 | 26.0% |
| tail × 2 below 1.5% | (63% affected) | +0.204 | 18.0% |

This reproduces the Qwen3-1.7B picture on Llama-3.1-8B: the weak half of all MLP activations is
close to dead weight, the same count at random is catastrophic, and amplifying the tail costs
about as much as deleting it (the readout is linear in the tail).

Per manifold, the first X at which each readout has lost half its clean value (shape = mean
critical shape metric on a fresh PCA; k-NN = share of each prompt's 10 nearest neighbours kept;
task = the manifold's own next-token task; "–" = never within X ≤ 10%):

| manifold | task (clean acc.) | shape halves | k-NN < ½ | task halves | PopQA halves | loss +0.1 |
|---|---|---|---|---|---|---|
| age | – | – (0.96 at X = 5) | 10 | – | 5 | 1.5 |
| weekdays_circle | weekday arithmetic (70%) | 7.5 | 10 | 3 | 5 | 1.5 |
| geography | country recall, a fact (81%) | 5 | 5 | 3 | 5 | 1.5 |
| addition_sum_fourier | exact sum (95%) | 5 | 5 | 5 | 5 | 1.5 |
| months_last | month arithmetic (65%) | 3 | 5 | 3 | 5 | 1.5 |
| political_bias | lean, 2-way (96%) | 3 | 1 | – (92% at X = 5 is a 2-way guess rate) | 5 | 1.5 |
| colors | hue-family naming (60%) | 1 | 1 | 3 | 5 | 1.5 |

Reading. The plan's hypothesis ("continuous quantities break before facts") holds for some
manifolds and fails for others. The age line and the weekday circle are *more* robust than
facts: at X = 5% (89% of activations zeroed, PopQA at a quarter of baseline) their refit shape
metrics are still 0.96 and 0.93. Colours, the last-token month features and the political-bias
neighbourhoods degrade earlier than facts: colours lose half their shape metric and two thirds
of their neighbourhoods at X = 1% (loss +0.07, PopQA 25%), the point where the model's own
top-1 continuation also changes on two thirds of colour prompts. The tasks that sit on a
manifold (weekday / month arithmetic, addition) fall at the same X as general top-1 agreement
(2–3%), not earlier. Across all seven, the count-matched random control at X = 0.1% (10% of
activations) destroys the geometry (shape ≤ 0.5, k-NN ≤ 0.4) and the tasks, so what the sweep
measures is specific to weak activations and not to perturbation size. Scaling the tail by 0.5
leaves every manifold intact; scaling by 2 damages months, colours and political bias about as
much as zeroing. Caveat: "displacement / spread" in the per-condition tables includes a common
shift of the whole cloud; the k-NN overlap and the shape metric are translation-invariant.

### 2. Neurons that write and read each manifold (`manifolds neurons`)

Writers: a neuron's own contribution to the variance inside the 3-PC subspace, Var(a)·‖C w‖²
(w = its `down_proj` column), on ≤ 1000 prompts. Readers: neurons in the four blocks after the
read layer whose activation is predicted (5-fold CV ridge on the three coordinates, their
squares and products) from the manifold position. Causal: zero the top-k writers in all
positions vs k random neurons of the same blocks.

| manifold | Σ individual writer shares | Σ per-block MLP / attention projected variance | top writers' tuning η² (random) | top-256 writers zeroed: shape, k-NN, task (random 256) |
|---|---|---|---|---|
| addition_sum_fourier | 12% | 19% / 4% | 0.27 (0.15) | 0.38, 0.49, 3% (0.91, 0.96, 94%) |
| months_last | 5% | 14% / 4% | 0.34 (0.29) | 0.33, 0.49, 11% (0.88, 0.97, 62%) |
| weekdays_circle | 2% | 9% / 0% | 0.99 (0.73) | 0.62, 0.86, 26% (0.99, 0.98, 73%) |
| political_bias | 2% | 11% / 3% | 0.62 (0.07) | 0.69, 0.59, 92% (0.92, 0.84, 96%) |
| geography | 4% | 10% / 8% | 0.17 (0.26) | 0.61, 0.85, 59% (0.59, 0.97, 81%) |
| colors | 2% | 12% / 4% | 0.60 (0.17) | 0.94, 0.68, 61% (0.93, 0.87, 60%) |
| age | 1% | 5% / 4% | 0.84 (0.69) | 0.97, 0.92, – (0.98, 0.97, –) |

Reading. No manifold is written by a handful of neurons: the individual-neuron variances sum to
1–12% of the subspace variance, and even whole-block MLP outputs sum to 5–19%; the rest is
covariance between many neurons and blocks. Yet the top writers are causally specific: zeroing
16 of them (block 18 for addition, e.g. neuron 1712) halves addition accuracy and 64 of them
bring it to 5%, while 256 random neurons change nothing; the same holds for month and weekday
arithmetic. For the weekday circle the top writers are the tiling the plan predicted: η² ≈ 1.0,
each peaked on one day (Wednesday, Friday, Thursday, …), and for colours η² ≈ 0.9 with peaks
spread around the hue circle (hue 0.03, 0.78, 0.38, 0.70, 0.95); political-bias writers peak at
the two poles. Two manifolds resist the causal test: the age line and the colour paraboloid keep
their shape with the top 256 writers zeroed (0.97 and 0.94), so their geometry is carried
collectively or by attention rather than by the top-ranked neurons. Readers exist for every
manifold: in blocks L+1…L+4 the best neurons are predicted from the three coordinates with CV
R² 0.8–0.99, and the fraction of their gate weight inside the subspace is 2–70 × the isotropic
expectation (3/4096).

### 3. Interference between manifolds (`manifolds interference`, `manifolds crosstalk`)

Subspace overlap (mean squared cosine of the principal angles between 3-PC subspaces) for the
237 pairs of manifolds read at the same layer (22 at block 19, 4 at block 18), against random
3-D subspaces drawn from the pooled activation covariance (mean 0.015, 95th percentile 0.028;
isotropic expectation 0.0007):

- median pair overlap 0.003; 88% of pairs are below the null mean and 94% below its 95th
  percentile; 10 pairs exceed 0.05 and 4 exceed 0.1.
- the four large overlaps are shared-concept pairs: days–weekdays_circle 0.52 (the same weekday
  concept in two templates), weekdays_last–months_last 0.30 and months_last–hours_last 0.12 (all
  last-token Fourier features of concept + offset), formality–sent_length 0.15 (both read from
  wikitext sentences).
- split-half reliability of a manifold's own subspace ranges from 1.0 (colors) to 0.2
  (adverbs_pace), so low overlaps for the unstable extended-tier manifolds are also low ceilings.

So at the level of their top-3 subspaces the manifolds of different concepts are not "woven
through each other"; they are as orthogonal as random directions of the residual stream allow.
Crosstalk (`manifolds crosstalk`): each prompt of a pilot manifold was prefixed with one of six
sentences from another pilot's domain (e.g. "The hex code #e63946 is for the color red. ") or one
of six neutral sentences, and the manifold was re-read at its layer. Because a prefix sentence
adds its own offset to the read token, prefixed activations are centred per prefix sentence
before the comparison (a first run without this centring showed prefix-identity clusters in the
top PCs and was discarded). k-NN overlap with the clean cloud, neutral prefix vs the most
disruptive domain, and where the displacement went:

| manifold | neutral: k-NN, shape | worst domain: k-NN, shape | task clean → neutral | move inside own subspace | inside prefix-domain subspace (random level) |
|---|---|---|---|---|---|
| weekdays_circle (concept token) | 0.88, 0.98 | colors 0.86, 0.98 | 70% → 27% | 14% | ≤ 1.8% (0.7%) |
| geography | 0.91, 0.58 | politics 0.88, 0.54 | 84% → 81% | 3% | ≤ 0.4% (0.3%) |
| addition_sum_fourier | 0.70, 0.90 | age 0.68, 0.91 | 95% → 100% | 7% | ≤ 1.2% (0.5%) |
| colors | 0.33, 0.90 | weekdays 0.35, 0.88 | 58% → 60% | 28% | ≤ 1.1% (0.5%) |
| age | 0.39, 0.79 | politics 0.33, 0.75 | – | 32% | ≤ 0.8% (0.7%) |
| months_last (last token) | 0.43, 0.31 | politics 0.35, 0.28 | 65% → 16% | 25% | ≤ 1.6% (1.1%) |
| political_bias | 0.12, 0.62 | weekdays 0.14, 0.79 | 96% → 92% | 22% | ≤ 1.1% (0.8%) |

Reading. There is no concept-specific crosstalk to measure: a colour sentence does not disturb
the weekday circle more than a neutral sentence does, and for every manifold the most disruptive
domain is within a few points of the neutral control. What a prefix does is generic context
sensitivity, and it is large for some read positions: the last-token manifolds (age, colours,
political bias, the last-token month features) lose most of their neighbourhoods under *any*
prefix, while the concept-token weekday circle and the "=" and "of" tokens of addition and
geography barely move. The displacement lies inside the manifold's own 3-PC subspace at 3–32%,
far above the random level, and inside the prefix domain's subspace at the random level
(≤ 1.8%): prefixes slide or stretch a manifold along its own axes rather than pulling it toward
the other concept's directions, which is what the orthogonal subspaces above predict. The
Q/A-format tasks are the prefix-sensitive ones (weekday arithmetic 70% → 27%, month arithmetic
65% → 16% with a neutral sentence; the model's unrestricted top-1 changes on 70–80% of those
prompts), whereas addition, country recall and the lean classification are unaffected.

### 4. Forming during pre-training (`manifolds checkpoints`, OLMo-2 1B)

`allenai/OLMo-2-0425-1B` (16 blocks), the same prompt sets (≤ 600 per manifold), twelve log-spaced
stage-1 checkpoints (0, 1B, 21B, 42B, 105B, 210B, 399B, 797B, 1.2T, 2.0T, 3.0T, 4.0T tokens) and
`main` (after the 50B-token stage-2 anneal). Every residual position is shape-checked at the final
checkpoint and every 4th position plus the display layer elsewhere (the Fourier-probe check is
CPU-heavy); the display layer is the best cell over all checkpoints among positions checked
everywhere. Slider frames are Procrustes-aligned to the final checkpoint. Shape metric at the
display layer (position of 16; "–" = below 0.1), with the manifold's task accuracy in the final model:

| manifold | display pos. | 0 | 1B | 21B | 42B | 105B | 399B | 797B | 1.2T | 2T | 4T | final | task (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| age | 16 | – | 0.23 | 0.96 | 0.97 | 0.98 | 0.98 | 0.99 | 0.98 | 0.98 | 0.99 | 0.96 | – |
| weekdays_circle | 12 | 0.36 | 0.10 | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.99 | 14% (chance 14%) |
| colors | 12 | – | 0.22 | 0.40 | 0.47 | 0.41 | 0.38 | 0.46 | 0.26 | 0.37 | 0.38 | 0.46 | 9% |
| months_last | 16 | – | – | 0.26 | 0.28 | 0.32 | 0.37 | 0.38 | 0.68 | 0.61 | 0.73 | 0.14 | 8% (chance 8%) |
| political_bias | 16 | – | – | – | – | – | 0.36 | 0.43 | 0.43 | 0.50 | 0.53 | – | 73% (2-way) |
| addition_sum_fourier | 15 | – | – | – | – | – | – | 0.18 | 0.12 | – | 0.15 | 0.55 | 16% |
| geography | 16 | 0.13 | 0.10 | 0.12 | 0.24 | 0.11 | 0.23 | 0.24 | 0.20 | 0.25 | 0.27 | 0.12 | 1% |

Reading. Two manifolds form early and completely: the age line and the weekday circle are at
their final quality by the 21B-token checkpoint (the first one after 1B; OLMo-2 publishes nothing
in between, so the formation window itself is not resolved) and stay there for the remaining 4T
tokens, while OLMo-2 1B never learns weekday arithmetic (14%, chance). Geometry precedes
behaviour here, or behaviour never arrives at 1B. Three structures form late: the last-token
month features reach 0.68 only at 1.2T tokens, political lean at position 16 rises from nothing at
210B to 0.36 at 399B and 0.53 at 4T (its lean accuracy from 49% to 87% over the same span), and
addition's base-10 Fourier features appear only in the final model (0.55 after the stage-2 anneal,
≤ 0.18 during stage 1), with addition accuracy at 16%. The anneal also removes the last-position
month and political structure of the stage-1 checkpoints (0.73 → 0.14 and 0.53 → 0.04) while the
lean accuracy stays at 73%, so the final 1B model keeps the behaviour in another form. Colours and
geography never pass the checks at 1B (0.46 and 0.31 at best). Coarse-to-fine was not observed at
this checkpoint spacing: the centroid dimensionality of the age and weekday manifolds is already
at its final value (≈ 2.1–2.4 and ≈ 2.7) at 21B tokens, with no intermediate axis-like stage
between the 1B and 21B checkpoints.

### Views and interaction (2026-10-18 revision)

For the five Fourier-probe manifolds (addition_sum_fourier, months_last, weekdays_last, hours_last,
numbers_1_100) every slider now shows the probe view rather than a PCA: the layer slider fits a
probe per depth (held-out predictions), the ablation and neuron sliders apply one probe fitted on
the clean model to every frame (the probe analogue of a fixed PCA basis), and the checkpoint
slider fits a probe per checkpoint. Slider frames carry a caption saying what the step did (for
the ablation: the condition and the share of MLP activations removed). Wheel zoom is disabled on
all plots and the camera cannot be moved farther out than the starting view (the toolbar's zoom
tool still zooms in). `src/manifolds/analysis/probe.py`, `site/build.py`.

### What this says about the debate

- Weak activations are cheap to remove on Llama-3.1-8B exactly as on Qwen3-1.7B, and the cost
  that appears is not specific to graded structure: the two cleanest low-dimensional manifolds
  (a line and a circle) survive longer than fact recall, while colours and the arithmetic Fourier
  features do not. "Weak activations carry the manifolds" is therefore false in general and true
  for particular manifolds; which ones is now a measurable property.
- No manifold is a few neurons, but the top writers of the ordered manifolds have the receptive
  fields the "clean" picture predicts (one weekday each; hue arcs; the two political poles), and
  zeroing a few dozen of them is enough to break the manifold's task where the task is
  carried by the manifold (addition, months, weekdays). Colours and age have no such handle.
- Different concepts occupy near-orthogonal 3-D subspaces, and prefixing another concept's
  sentence perturbs a manifold along its own axes, not toward the other concept: the crosstalk
  Cammarata allowed for is below the random level at this resolution, which is the measurement
  McGrath's position needed.
- In OLMo-2 1B the simplest manifolds exist long before, and independently of, the behaviour
  they would support; the arithmetic features arrive with the math-heavy anneal.

## Skipped (and why)

- Vision tier (DINOv3 lighting/shadow, InceptionV1 curves, SDXL pretzel, DINOv2 position sheet): listed in `manifest.yaml`; not run (user decision, 2026-10-15).
- Last-token variants of the weekday/month/hour circles (the paper's negative claim at block 18): not run; the concept-token reads above are the positive claim.
- Unsupervised Ising discovery pipeline: needs the paper's SAEs and a 500M-token background corpus; the blog's discovered manifolds are approximated with supervised prompt sets instead.
- Template lens / activation oracles: no pre-trained oracle for base Llama-3.1-8B; single-token readouts suffice here.

## Compute

GPU ledger: `logs/gpu_ledger.jsonl` (`uv run manifolds ledger summary`). Billed Azure T4 uptime,
reconciled against the Azure activity log: 4.91 GPU-hours over seven sessions (one of them
ran ~2 h longer than needed because a cycle script died after a crash without deallocating). Actual
model computation inside those sessions was well under an hour.

Debate extensions (2026-10-17/18): one AWS `g6e.2xlarge` (L40S) on-demand session, launched
02:14 UTC and terminated (verified) 03:40 UTC: 1.45 billed hours (≈ $3.25 at $2.24/h), of which
the pipeline logged 1.06 GPU-hours of model computation (ablation 0.47 h, neurons / crosstalk
/ checkpoints the rest; the OLMo sweep was CPU-bound on the Fourier-probe shape checks and was
restarted once with a reduced layer set). `g6e.xlarge` had no capacity in any us-east-1 zone at
launch time. A second `g6e.2xlarge` session on 2026-10-18 (04:37–05:20 UTC, ≈ 0.7 billed hours, ≈ $1.6)
rebuilt the sliders of the Fourier-probe manifolds as probe views and recaptured the OLMo
checkpoint states. No GCP GPU time was used; the Azure VM stayed deallocated.
