# Source notes

Facts extracted from the sources on 2026-10-15. Each bullet names where it came from. Scratch copies of the papers and posts live outside the repo; links are canonical.

## Papers

- **Bhalla et al. 2026, "Do Sparse Autoencoders Capture Concept Manifolds?"** arXiv 2604.28119. https://arxiv.org/abs/2604.28119
  - Model: Llama-3.1-8B (base), residual stream layer 19, d=4096, last-token activations (App. B, Table 1).
  - Nine supervised manifolds with templates (Table 1): colors (paraboloid), temperature (line), age (line), geography (hierarchical tree), days (circle), years (helix), formality (line), sent_length (line), political bias (1D continuous).
  - SAEs trained in-house on 500M tokens of The Pile (App. B.2, Table 3); not released. Unsupervised pipeline used BatchTopK x8, k=64 (Sec. 6).
  - Fig. 10: Ising pipeline recovers temperature, colors, political bias; surfaces a novel "epistemic uncertainty" manifold (cluster 357, "mean +/- std" strings).
  - Steering setup (App. B.1, Table 2): PCA to 90% variance, centroid interpolation, task suffixes and target tokens per manifold.
  - Days discrepancy: Table 1 n=420 vs released code 56 prompts.
- **Feucht et al. 2026, "Arithmetic in the Wild"** arXiv 2605.01148. https://arxiv.org/abs/2605.01148
  - Llama-3.1-8B; addition module at layer 18, last token (Sec. 3-5).
  - Task templates (App. A, Table 1): months, weekdays, hours, addition "a+b=".
  - Fig. 3: circular (Engels-style) structure for the input concept is NOT present at layer 18 last token; instead base-10 Fourier features of the sum (periods 2, 5, 10) are (Sec. 4, Fig. 6).
- **Fel et al. 2026, "Structuring Sparsity: Block-Sparse Featurizers"** arXiv 2606.25234. https://arxiv.org/abs/2606.25234
  - InceptionV1 curve manifold: single Grassmannian block b=16; Fourier modes 1/2/3 carry 59/18/12% variance (Sec. 4, App. E Table 1, n_theta=1800).
  - DINOv3 ViT-B/16 final-layer patches, Grassmannian BSF b=3 on ImageNet-1k; lighting and shadow blocks found with twelve Blender models under a sun sweep (Sec. 5.2, Fig. 13).
  - SDXL layer down.2.1, Grassmannian BSF b=16; pretzel manifold steering, hat/man/coffee Kohonen grids (Sec. 6, Fig. 14-15).
- **Fel et al. 2025, "Into the Rabbit Hull"** arXiv 2510.08638. https://arxiv.org/abs/2510.08638
  - DINOv2-B. Positional subspace collapses to a 2D sheet in final layers (Sec. 6, Fig. 14-15). Per-image PCA RGB maps smooth, survive removing position (Fig. 12-13, 27).
  - Register tokens carry lighting / motion-blur concepts (Sec. 4; cited by BSF paper footnote 5).
  - Minkowski Representation Hypothesis (Sec. 7).

## Goodfire Neural Geometry Series (goodfire.ai/research)

- "The World Inside Neural Networks" (May 7 2026) — slant-rhyme "-ore" subspace found by the unsupervised pipeline; SAE features on it listed. https://www.goodfire.ai/research/the-world-inside-neural-networks
- "Steering Along Manifolds" (May 7 2026, Wurgaft et al., arXiv 2605.05115) — days-of-week circle in behavior and activation space; also months, letters, ages. https://www.goodfire.ai/research/manifold-steering
- "A Geometric Calculator" (May 14 2026) — companion to Feucht et al.; numbers as mod-2/5/10/100 circles. https://www.goodfire.ai/research/a-geometric-calculator
- "Can SAEs Capture Neural Geometry?" (May 21 2026) — companion to Bhalla et al.; figure of eight unsupervised manifolds: ordinals by order, classical composers by year, SCOTUS cases by year, chemical bonds by length, adverbs indicating pace by speed, political bias by lean, statistical uncertainty by ln(std/mean), Bible verses by book & chapter. https://www.goodfire.ai/research/can-saes-capture-neural-geometry (figure: https://static.goodfire.ai/capturing-manifolds/unsupervised-manifolds.webp)
- "Meandering on Manifolds" (Jun 23 2026, Bigelow et al., arXiv 2605.12412; site slug `stories-in-space`) — six basic emotions in Llama 3.1 8B arranged like valence-arousal; story trajectories on the emotion manifold. https://www.goodfire.ai/research/stories-in-space
- "Uncovering Neural Geometry in Vision Models With BSFs" (Jul 7 2026) — companion to Fel et al. 2026. https://www.goodfire.ai/research/bsf-vision

## Code

- https://github.com/goodfire-ai/sae-manifold — `data.py` has the exact prompt builders and extraction (nnsight, `model.model.layers[19].output[0]`, last token). Norm-outlier filter mean+3std.
- https://github.com/goodfire-ai/block-sparse-featurizer — `bsf/` (vanilla, Grassmannian, group-lasso BSFs), `rabbit.npz` (300 images), `pos_mean.npy`, `viz.py` (3D manifold plots with floor + shadow + direction-hue coloring).

## Pretrained checkpoints (verified to exist 2026-10-15)

- `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` — public, Instruct model, L0=91.
- `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` (Llama Scope; moved from `fnlp`) — includes `Llama3_1-8B-Base-L19R-8x`: hook `blocks.19.hook_resid_post`, JumpReLU threshold 0.527, d_sae=32768, base model.
- `Goodfire/SDXL-Turbo-SAE-ldown.attns.2.1` — SDXL-Turbo SAE at the same layer as the BSF pretzel experiment.
- Gated (need HF token + license): `meta-llama/Llama-3.1-8B`, `facebook/dinov3-vitb16-pretrain-lvd1689m`. Not gated: `stabilityai/stable-diffusion-xl-base-1.0`, `facebook/dinov2-base`.

- Llama Scope inference semantics reproduced from `OpenMOSS/Language-Model-SAEs` v1.0.0 `src/lm_saes/sae.py` (`from_pretrained`: dataset-wise standardisation, unit decoder norm, JumpReLU). https://github.com/OpenMOSS/Language-Model-SAEs/blob/v1.0.0/src/lm_saes/sae.py ; Llama Scope paper: He et al. 2024, arXiv 2410.20526.

## Datasets (verified public)

- `burkelibbey/colors` (colors.jsonl: hex + description), `jamescalam/world-cities-geo` (train.jsonl), `osyvokon/pavlick-formality-scores` (train.csv: domain, avg_score, sentence), `cajcodes/political-bias` (political_bias.csv: text, label 0-4; 658 rows), `wikitext` wikitext-103-raw-v1.

## Visual style reference (Goodfire figures)

- 3D scatter, no axes, light-gray square floor plane below the cloud with a blurred gray shadow/density projection of the points; points colored by the ordering variable (viridis / plasma / categorical); uppercase IBM Plex Mono title with italic "by <variable>" subtitle; site text color #1D272A, accent #FFEBAA, fonts Suisse Intl + IBM Plex Mono.
