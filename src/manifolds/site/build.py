# ruff: noqa: E501
"""Build the static site: an index table (manifold | model | layer | geometry | paper | status)
with lazy-rendered previews, and one page per analysed manifold with the full interactive plot,
the paper's claim, the prompt template, shape-check results and extraction metadata.

Everything is inlined (no fetch), so the site works from file:// and from GitHub Pages.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import DictLoader, Environment, select_autoescape

from manifolds.manifest import load_manifest
from manifolds.steer import STEER_TASKS

PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/3.0.1/plotly.min.js"
FONTS = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&display=swap"

SOURCE_LINKS: dict[str, tuple[str, str]] = {
    "bhalla2026": ("Bhalla et al. 2026 (arXiv 2604.28119)", "https://arxiv.org/abs/2604.28119"),
    "sae-manifold": ("goodfire-ai/sae-manifold", "https://github.com/goodfire-ai/sae-manifold"),
    "blog-saes": ("Can SAEs Capture Neural Geometry?", "https://www.goodfire.ai/research/can-saes-capture-neural-geometry"),
    "blog-world": ("The World Inside Neural Networks", "https://www.goodfire.ai/research/the-world-inside-neural-networks"),
    "blog-calc": ("A Geometric Calculator", "https://www.goodfire.ai/research/a-geometric-calculator"),
    "blog-steer": ("Steering Along Manifolds", "https://www.goodfire.ai/research/manifold-steering"),
    "blog-stories": ("Meandering on Manifolds", "https://www.goodfire.ai/research/stories-in-space"),
    "blog-bsf": ("Neural Geometry in Vision Models (BSF)", "https://www.goodfire.ai/research/bsf-vision"),
    "feucht2026": ("Feucht et al. 2026 (arXiv 2605.01148)", "https://arxiv.org/abs/2605.01148"),
    "fel2026bsf": ("Fel et al. 2026 (arXiv 2606.25234)", "https://arxiv.org/abs/2606.25234"),
    "bsf-code": ("goodfire-ai/block-sparse-featurizer", "https://github.com/goodfire-ai/block-sparse-featurizer"),
    "fel2025hull": ("Fel et al. 2025 (arXiv 2510.08638)", "https://arxiv.org/abs/2510.08638"),
}

SITE_TITLE = "Llama Concept Manifolds"

BADGE_TEXT = {"matches": "matches paper", "partial": "partial", "failed": "failed", "pending": "pending"}

CSS = r"""
/* layout: single 1180px column; index = status table + lazy gallery; detail = plot + two-column notes */
:root{--bg:#ffffff;--fg:#1D272A;--muted:#5f6b6e;--line:#e3e6e4;--soft:#f4f6f5;--panel:#ffffff;--accent:#FFEBAA;
--ok:#1e7f4f;--ok-bg:#e3f5ea;--warn:#9a6700;--warn-bg:#fff4d6;--bad:#b3261e;--bad-bg:#fde7e5;--pend:#5b5b5b;--pend-bg:#ececec;
--font-body:"Suisse Intl",-apple-system,"Helvetica Neue",Arial,sans-serif;--font-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15191b;--fg:#e8ecec;--muted:#9aa5a8;--line:#2b3336;--soft:#1d2326;--panel:#ffffff;--accent:#5a4b16;
--ok:#7fd9a6;--ok-bg:#15372a;--warn:#f0c66b;--warn-bg:#3a2e10;--bad:#f3a19b;--bad-bg:#43201d;--pend:#b9c0c2;--pend-bg:#2b3336;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15191b;--fg:#e8ecec;--muted:#9aa5a8;--line:#2b3336;--soft:#1d2326;--panel:#ffffff;--accent:#5a4b16;
--ok:#7fd9a6;--ok-bg:#15372a;--warn:#f0c66b;--warn-bg:#3a2e10;--bad:#f3a19b;--bad-bg:#43201d;--pend:#b9c0c2;--pend-bg:#2b3336;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font-body);font-size:16px;line-height:1.55}
a{color:inherit}
.wrap{max-width:1180px;margin:0 auto;padding-block:40px 80px;padding-inline:16px}
.mono{font-family:var(--font-mono)}
header h1{font-size:34px;font-weight:500;letter-spacing:-0.01em;margin:0 0 6px;text-wrap:balance}
header p.lede{color:var(--muted);max-width:820px;margin:0 0 28px}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 30px}
.kpi{border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px;background:var(--soft)}
.kpi b{display:block;font-size:22px;font-weight:500}
.kpi span{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tablewrap{overflow-x:auto;margin:0 0 40px}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:500;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
tr.tier-row td{background:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;white-space:nowrap}
.badge-matches{color:var(--ok);background:var(--ok-bg)}
.badge-partial{color:var(--warn);background:var(--warn-bg)}
.badge-failed{color:var(--bad);background:var(--bad-bg)}
.badge-pending{color:var(--pend);background:var(--pend-bg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px}
.card{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:var(--panel);text-decoration:none;display:block;color:var(--fg)}
.card .fig{height:300px;width:100%;background:var(--panel)}
.card .cap{background:var(--bg)}
.card .cap{padding:10px 14px 14px;border-top:1px solid var(--line)}
.cap .name{text-transform:uppercase;letter-spacing:.1em;font-size:13px}
.cap .by{font-style:italic;color:var(--muted);font-size:13px}
.detail .fig{height:640px;width:100%;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.fig2d{height:300px;width:100%;background:var(--panel);border-radius:8px}
.detail h1{font-size:26px;font-weight:500;margin:0;text-transform:uppercase;letter-spacing:.12em}
.detail .by{font-style:italic;color:var(--muted);margin:2px 0 16px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:28px}
.two>div{min-width:0}
@media (max-width:760px){.two{grid-template-columns:1fr}.detail .fig{height:460px}.card .fig{height:260px}}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);font-weight:500;margin:26px 0 8px}
.box{border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:14px}
code,pre{font-family:var(--font-mono);font-size:13px}
pre{white-space:pre-wrap;background:var(--soft);padding:10px 12px;border-radius:8px;margin:6px 0}
.check-pass{color:var(--ok)}.check-fail{color:var(--bad)}
.small{font-size:13px;color:var(--muted)}
.back{display:inline-block;margin-bottom:18px;font-size:14px;color:var(--muted);text-decoration:none}
footer{margin-top:50px;font-size:13px;color:var(--muted)}
.axes-summary{display:grid;gap:10px;margin:6px 0 12px}
.axis-line{border-left:3px solid var(--line);padding:4px 10px}
details>summary{cursor:pointer;margin:6px 0}
details.box{margin:10px 0;padding:10px 14px}
details.box>summary{list-style:none;display:flex;align-items:center;gap:8px}
details.box>summary::before{content:"▸";font-family:var(--font-mono);color:var(--muted)}
details.box[open]>summary::before{content:"▾"}
details.box>summary::-webkit-details-marker{display:none}
a:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
"""

JS = r"""
const ZOOM_IN_LIMIT = 0.35;   // closest camera distance, as a fraction of the starting distance
function clampZoom(el){
  // 3-D scenes: the wheel zooms between the starting view (max) and ZOOM_IN_LIMIT of it (min);
  // at either limit the wheel is handed back to the page, so scrolling never gets stuck on a plot
  const sc = el.layout && el.layout.scene; if (!sc) return;
  const e0 = (sc.camera && sc.camera.eye) || {x:1.55, y:1.25, z:0.85};
  const r0 = Math.hypot(e0.x, e0.y, e0.z), rmin = r0 * ZOOM_IN_LIMIT;
  const dist = () => { const c = el.layout.scene && el.layout.scene.camera && el.layout.scene.camera.eye;
                       return c ? Math.hypot(c.x, c.y, c.z) : r0; };
  el.addEventListener('wheel', ev => {
    const r = dist(), out = ev.deltaY > 0;
    if ((out && r >= r0 * 0.995) || (!out && r <= rmin * 1.005)) ev.stopImmediatePropagation();  // page scrolls natively
  }, {capture: true, passive: true});
  el.on('plotly_relayout', ev => {
    const cam = ev['scene.camera']; if (!cam || !cam.eye) return;
    const r = Math.hypot(cam.eye.x, cam.eye.y, cam.eye.z);
    const k = r > r0 * 1.001 ? r0 / r : (r < rmin * 0.999 ? rmin / r : 1);
    if (k !== 1) Plotly.relayout(el, {'scene.camera.eye': {x: cam.eye.x * k, y: cam.eye.y * k, z: cam.eye.z * k}});
  });
}
function renderFig(el, fig){
  // pass the whole figure so slider frames are registered (data/layout alone drops them)
  return Plotly.newPlot(el, {data: fig.data, layout: fig.layout, frames: fig.frames || []},
    {responsive:true, displaylogo:false, scrollZoom:true,
     modeBarButtonsToRemove:['toImage','resetCameraLastSave3d','hoverClosest3d']}).then(() => { clampZoom(el); return el; });
}
function figFrom(id){ return JSON.parse(document.getElementById(id).textContent); }
if ('scrollRestoration' in history) { history.scrollRestoration = 'manual'; }
document.addEventListener('DOMContentLoaded', () => {
  window.scrollTo(0, 0);
  const visible = el => !el.closest('details') || el.closest('details').open;
  const eager = Array.from(document.querySelectorAll('.eager-fig')).filter(visible).map(el => renderFig(el, figFrom(el.dataset.fig)));
  Promise.all(eager).then(() => { window.scrollTo(0, 0); });
  document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => {
    if (!d.open) return;
    d.querySelectorAll('.eager-fig').forEach(el => { if (!el.dataset.rendered) { el.dataset.rendered = '1'; renderFig(el, figFrom(el.dataset.fig)); } else { Plotly.Plots.resize(el); } });
  }));
  const lazy = document.querySelectorAll('.lazy-fig');
  if (!('IntersectionObserver' in window)) { lazy.forEach(el => renderFig(el, figFrom(el.dataset.fig))); return; }
  const io = new IntersectionObserver(entries => entries.forEach(e => {
    if (e.isIntersecting) { renderFig(e.target, figFrom(e.target.dataset.fig)); io.unobserve(e.target); }
  }), {rootMargin: '250px'});
  lazy.forEach(el => io.observe(el));
});
"""

BASE = r"""{% if not artifact %}<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
{% endif %}<title>{{ title }}</title>
<link rel="stylesheet" href="{{ fonts }}">
<style>{{ css|safe }}</style>
<script src="{{ plotly }}"></script>
{% if not artifact %}</head><body>{% endif %}<div class="wrap">{% block body %}{% endblock %}
<footer class="mono">Built {{ built }} · <a href="https://github.com/goodfire-ai/sae-manifold">Goodfire code</a> · every claim on these pages cites its source paper or post.</footer>
</div><script>{{ js|safe }}</script>{% if not artifact %}</body></html>{% endif %}"""

INDEX = r"""{% extends "base" %}{% block body %}
<header>
<h1>Concept manifolds in Llama-3.1-8B</h1>
<p class="lede">A replication of the concept manifolds reported in Goodfire's Neural Geometry series and the papers behind it, on the base Llama-3.1-8B.
Each entry shows a 3-D PCA of residual-stream activations (last token, block 19), the shape the source reports, a quantitative shape check, and a replication badge. Nothing here is hand-edited: badges come from the checks.
Beyond the sources, every page adds a layer slider (the manifold at all 33 residual positions), what the model would say from each point under the logit, tuned and Jacobian lenses, a decoding of what each principal axis means, the paper's steering experiment, and the SAE tiling analysis.</p>
{% if ext %}<p class="lede">Seven pilot pages ({{ ext.pilots }}) also carry four experiments prompted by the <a href="https://already-clean.goodfire.com/">Neural Networks Are Already Clean</a> debate: a weak-activation ablation slider (does the manifold survive zeroing the weak tail of every MLP neuron, and how does that compare with facts and LM loss), the MLP neurons that write and read each manifold, subspace overlap and crosstalk between manifolds, and the manifold forming across OLMo-2 1B training checkpoints.</p>{% endif %}
</header>
{% if cards %}
<h2>Manifolds</h2>
<div class="grid">
{% for c in cards %}
<a class="card" href="manifolds/{{ c.name }}.html">
<div class="fig lazy-fig" id="prev-{{ c.name }}" data-fig="json-{{ c.name }}"></div>
<script type="application/json" id="json-{{ c.name }}">{{ c.preview_json|safe }}</script>
<div class="cap mono"><div class="name">{{ c.title }}</div><div class="by">by {{ c.by }} · <span class="badge badge-{{ c.status }}">{{ c.badge }}</span></div></div>
</a>
{% endfor %}
</div>
{% endif %}
{% if ext %}
<h2>Across manifolds</h2>
<div class="two" style="margin-top:8px">
{% if ext.global_json %}<div><div class="small mono" style="margin-bottom:4px">model-wide cost of zeroing weak activations (Llama-3.1-8B): facts and loss</div><div class="fig2d eager-fig" id="fig-ext-global" data-fig="json-ext-global" style="height:340px"></div>
<script type="application/json" id="json-ext-global">{{ ext.global_json|safe }}</script>
<p class="small">{{ ext.global_text }}</p></div>{% endif %}
{% if ext.heatmap_json %}<div><div class="small mono" style="margin-bottom:4px">subspace overlap between manifolds (3 PCs, same layer; random ≈ {{ ext.null }})</div><div class="fig2d eager-fig" id="fig-ext-heat" data-fig="json-ext-heat" style="height:520px"></div>
<script type="application/json" id="json-ext-heat">{{ ext.heatmap_json|safe }}</script></div>{% endif %}
</div>
{% if ext.breaking %}
<div class="tablewrap" style="margin-top:14px"><table><thead><tr><th>manifold</th><th>task</th><th>clean acc.</th><th>X where shape metric halves</th><th>X where k-NN overlap &lt; ½</th><th>X where task acc. halves</th><th>X where PopQA halves</th><th>X where LM loss +0.1</th></tr></thead><tbody>
{% for b in ext.breaking %}<tr><td class="mono"><a href="manifolds/{{ b.name }}.html">{{ b.name }}</a></td><td class="small">{{ b.task }}</td><td class="mono">{{ b.clean_acc }}</td><td class="mono">{{ b.shape }}</td><td class="mono">{{ b.knn }}</td><td class="mono">{{ b.task_x }}</td><td class="mono">{{ b.popqa }}</td><td class="mono">{{ b.loss }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="small">X is the weak-activation threshold in % of each neuron's calibrated max, swept over 0.1 … 10 (the realised share of activations zeroed is on each page). "–" means the readout never crossed that line within the sweep.</p>
{% endif %}
{% endif %}
<h2>Replication status</h2>
<div class="kpis">
{% for k,v in counts.items() %}<div class="kpi"><b>{{ v }}</b><span>{{ k }}</span></div>{% endfor %}
</div>
<div class="tablewrap"><table>
<thead><tr><th>manifold</th><th>model</th><th>layer</th><th>expected geometry</th><th>paper / post</th><th>status</th><th>SAE tiling</th><th>extensions</th></tr></thead>
<tbody>
{% for tier, rows in tiers %}
<tr class="tier-row"><td colspan="8">{{ tier }}</td></tr>
{% for r in rows %}
<tr>
<td class="mono">{% if r.has_page %}<a href="manifolds/{{ r.name }}.html">{{ r.name }}</a>{% else %}{{ r.name }}{% endif %}</td>
<td class="small">{{ r.model }}</td><td class="mono">{{ r.layer }}</td><td>{{ r.geometry }}</td>
<td class="small">{% for s in r.sources %}<a href="{{ s[1] }}">{{ s[0] }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</td>
<td><span class="badge badge-{{ r.status }}" title="{{ r.reason }}">{{ r.badge }}</span></td>
<td class="small">{{ r.sae_note }}</td>
<td class="small">{{ r.ext_note }}</td>
</tr>
{% endfor %}{% endfor %}
</tbody></table></div>
{% endblock %}"""

DETAIL = r"""{% extends "base" %}{% block body %}
<a class="back mono" href="../index.html">← all manifolds</a>
<div class="detail">
<h1 class="mono">{{ r.title }}</h1>
<div class="by mono">by {{ r.by }} · <span class="badge badge-{{ r.status }}">{{ r.badge }}</span></div>
{% if r.probe %}
<div class="fig eager-fig" id="fig-probe" data-fig="json-probe"></div>
<script type="application/json" id="json-probe">{{ r.probe.figure_json|safe }}</script>
<p class="small mono">Fourier-probe view · drag to rotate · scroll to zoom (between this view and 3× closer; at either limit the page scrolls instead) · hover for the value</p>
<p class="small">Why not PCA here? PCA shows the three directions the points spread along <em>most</em>, and for these prompts that is the size of the numbers and which tokens they use, so the pattern the paper is about (the last digit going round in a circle) is buried under it. A probe is a direction found by training instead: we fit a linear readout on the block activations to predict where each prompt's <code>{{ r.probe.label }}</code> sits on a ones-digit clock (cos and sin of 2π·value/10) and how big it is, then plot each prompt where that readout puts it, using held-out predictions so nothing is fitted to itself. If the circle really is in the activations the readout finds it (circle R² {{ '%.2f'|format(r.probe.r2_circle) }}, magnitude R² {{ '%.2f'|format(r.probe.r2_linear) }}); if it is not, the points stay a blob. Every slider on this page uses the same probe view (fitted per layer, per checkpoint, or on the clean model and reused), so the frames stay readable. The raw PCA, which is a blob as the paper reports, is in the details at the bottom.</p>
{% else %}
<div class="fig eager-fig" id="fig-main" data-fig="json-main"></div>
<script type="application/json" id="json-main">{{ r.figure_json|safe }}</script>
<p class="small mono">drag to rotate · scroll to zoom (between this view and 3× closer; at either limit the page scrolls instead) · hover for the prompt</p>
{% endif %}

{% if r.depth and r.depth.pcs %}
<h2>What the axes mean</h2>
<div class="box">
{% if r.depth.label_model %}
<div class="axes-summary">
{% for pc in r.depth.pcs %}{% if pc.label %}<div class="axis-line"><span class="mono">PC{{ pc.pc }} · {{ pc.var }} of variance</span><br><b>{{ pc.label }}</b> <span class="badge badge-{{ 'matches' if pc.confidence == 'high' else ('partial' if pc.confidence == 'medium' else 'pending') }}">{{ pc.confidence }} confidence</span><div class="small">{{ pc.rationale }}</div></div>{% endif %}{% endfor %}
</div>
<p class="small">Readings by {{ r.depth.label_model }} of four token readouts per axis: the Jacobian, logit and tuned lenses applied to the axis direction (what moving along it makes the model say), plus a lens-free correlation with the model's real next-token log-probabilities. A guide, not a measurement; the tables are the evidence.</p>
{% endif %}
<details><summary class="small mono">token tables</summary>
<div class="tablewrap"><table><thead><tr><th>axis</th><th>readout</th><th>↑ promoted</th><th>↓ suppressed</th></tr></thead><tbody>
{% for pc in r.depth.pcs %}{% for row in pc.rows %}<tr>{% if loop.first %}<td class="mono" rowspan="{{ pc.rows|length }}">PC{{ pc.pc }}</td>{% endif %}<td class="small">{{ row.name }}</td><td class="mono small">{{ row.up }}</td><td class="mono small">{{ row.down }}</td></tr>{% endfor %}{% endfor %}
</tbody></table></div></details>
</div>
{% endif %}

{% if r.depth %}
<h2>Across the layers</h2>
<p class="small">The same prompts read out at every residual-stream position: 0 is the token embedding, position i is the output of block i−1 (the paper's layer 19 is position 20). Drag the slider to watch the manifold form.{% if r.probe %} Each frame is the Fourier-probe view fitted at that depth (held-out predictions), so the slider shows at which depth the digit circle becomes readable.{% endif %} Hover shows what the model would say next from that point at that depth under the logit (L), tuned (T) and Jacobian (J) lenses.</p>
<div class="fig eager-fig" id="fig-depth" data-fig="json-depth" style="height:560px"></div>
<script type="application/json" id="json-depth">{{ r.depth.depth_json|safe }}</script>
{% endif %}

{% if r.ablation %}
<h2>Weak activations: does the manifold survive?</h2>
<p class="small">Cammarata's <a href="https://already-clean.goodfire.com/">Neural Networks Are Already Clean</a> argues that weak neuron activations can be removed at little cost; <a href="https://www.lesswrong.com/posts/neural-networks-are-not-already-clean">Gao's reply</a> argues that graded, weak activations carry essential information. A concept manifold is a graded structure, so it is a direct test.</p>
<p class="small"><b>What the slider does.</b> Every MLP neuron in the model has a typical maximum activation, measured once on ordinary text. A step labelled <code>1</code> sets to zero every activation that is smaller than 1% of its own neuron's maximum, in every block, on every token, and then re-reads the manifold at block {{ r.ablation.layer }} and re-scores the model. Because most activations are tiny, the thresholds bite fast: 1% already removes about half of all MLP activations and 10% removes 97%, which is why the sweep stops there (past 10% almost nothing is left and the model is destroyed). The steps after <code>10</code> are controls, not larger thresholds: <code>r0.1</code>…<code>r1</code> zero the <em>same number</em> of activations chosen at random, and <code>×0.5@1.5</code> / <code>×2@1.5</code> halve or double the weak tail below 1.5% instead of deleting it. The caption above the plot says what each step did and how much it removed. Points are the same prompts drawn in a fixed frame (the clean model's axes, or its probe), so movement is real displacement.</p>
<div class="fig eager-fig" id="fig-ablation" data-fig="json-ablation" style="height:560px"></div>
<script type="application/json" id="json-ablation">{{ r.ablation.figure_json|safe }}</script>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">geometry vs threshold</div><div class="fig2d eager-fig" id="fig-abl-geo" data-fig="json-abl-geo" style="height:340px"></div>
<script type="application/json" id="json-abl-geo">{{ r.ablation.geometry_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">behaviour vs threshold{% if r.ablation.task %} · task: {{ r.ablation.task }}{% endif %}</div><div class="fig2d eager-fig" id="fig-abl-beh" data-fig="json-abl-beh" style="height:340px"></div>
<script type="application/json" id="json-abl-beh">{{ r.ablation.behaviour_json|safe }}</script></div>
</div>
<div class="box" style="margin-top:12px"><b>Reading.</b> {{ r.ablation.summary }}</div>
<details><summary class="small mono">per-condition table</summary>
<div class="tablewrap"><table><thead><tr><th>condition</th><th>zeroed</th><th>status</th><th>shape</th><th>k-NN overlap</th><th>move / spread</th><th>top-1 agreement</th><th>KL</th>{% if r.ablation.task %}<th>task acc.</th>{% endif %}<th>LM Δloss</th><th>PopQA</th></tr></thead><tbody>
{% for row in r.ablation.rows %}<tr><td class="mono" title="{{ row.description }}">{{ row.label }}</td><td class="mono">{{ row.zeroed }}</td><td><span class="badge badge-{{ row.status }}">{{ row.status }}</span></td><td class="mono">{{ row.shape }}</td><td class="mono">{{ row.knn }}</td><td class="mono">{{ row.move }}</td><td class="mono">{{ row.agree }}</td><td class="mono">{{ row.kl }}</td>{% if r.ablation.task %}<td class="mono">{{ row.task }}</td>{% endif %}<td class="mono">{{ row.dloss }}</td><td class="mono">{{ row.popqa }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="small">shape = mean of the critical shape metrics on a fresh PCA of the ablated activations; k-NN overlap = share of each prompt's 10 nearest neighbours (full activation space) kept; KL = KL(clean ‖ ablated) of the next-token distribution over {{ r.ablation.n_kl }} prompts; LM Δloss and PopQA are model-wide (held-out FineWeb-edu; 300 PopQA questions, 8-shot).</p>
</details>
{% endif %}

{% if r.neurons %}
<h2>Which neurons write and read it</h2>
<p class="small">The residual stream at block {{ r.neurons.layer }} is a sum of MLP neuron writes a·w (activation times the neuron's <code>down_proj</code> column). Each neuron's own contribution to the variance inside the manifold's 3-PC subspace is Var(a)·‖Cw‖², with C the PCA axes. Summed over all {{ r.neurons.n_total }} neurons of blocks 0–{{ r.neurons.layer }} these individual contributions come to {{ r.neurons.total_share }} of the subspace variance (the rest is covariance between neurons and attention: summing each block's projected MLP output gives {{ r.neurons.mlp_total }}, each block's attention output {{ r.neurons.attn_total }}, cross-block terms excluded). Half of the neuron-attributable part sits in <b>{{ r.neurons.n_half }}</b> neurons ({{ r.neurons.n_90 }} for 90%). Tiling, in the sense of Anthropic's <a href="https://transformer-circuits.pub/2025/linebreaks/index.html">counting-manifold paper</a>, would show up as top writers with narrow, ordered tuning curves over <code>{{ r.neurons.tuning_label }}</code>; the mean tuning strength (η²) of the top writers is {{ r.neurons.eta_top }} against {{ r.neurons.eta_rand }} for random neurons of the same blocks.</p>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">who builds it: variance written per block</div><div class="fig2d eager-fig" id="fig-neu-blocks" data-fig="json-neu-blocks"></div>
<script type="application/json" id="json-neu-blocks">{{ r.neurons.blocks_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">top writers: tuning curves vs {{ r.neurons.tuning_label }}</div><div class="fig2d eager-fig" id="fig-neu-tuning" data-fig="json-neu-tuning"></div>
<script type="application/json" id="json-neu-tuning">{{ r.neurons.tuning_json|safe }}</script></div>
</div>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">top readers (blocks {{ r.neurons.reader_blocks }}): activation predicted from the 3 coordinates</div><div class="fig2d eager-fig" id="fig-neu-readers" data-fig="json-neu-readers"></div>
<script type="application/json" id="json-neu-readers">{{ r.neurons.readers_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">causal: zero the top-k writers vs k random neurons</div><div class="fig2d eager-fig" id="fig-neu-causal" data-fig="json-neu-causal"></div>
<script type="application/json" id="json-neu-causal">{{ r.neurons.causal_json|safe }}</script></div>
</div>
<div class="small mono" style="margin:14px 0 4px">the manifold with the top writers zeroed ({{ 'probe fitted on the clean model' if r.probe else 'clean PCA axes' }}; drag the slider)</div>
<div class="fig eager-fig" id="fig-neu-3d" data-fig="json-neu-3d" style="height:480px"></div>
<script type="application/json" id="json-neu-3d">{{ r.neurons.figure_json|safe }}</script>
<details><summary class="small mono">top writers and readers</summary>
<div class="tablewrap"><table><thead><tr><th>role</th><th>block</th><th>neuron</th><th>share of subspace var.</th><th>tuning η²</th><th>peak {{ r.neurons.tuning_label }}</th><th>CV R² from coords</th></tr></thead><tbody>
{% for w in r.neurons.writers %}<tr><td>writer</td><td class="mono">{{ w.block }}</td><td class="mono">{{ w.neuron }}</td><td class="mono">{{ '%.3f'|format(w.share) }}</td><td class="mono">{{ '%.2f'|format(w.eta2) }}</td><td class="mono">{{ '%.3g'|format(w.peak_label) }}</td><td class="mono">–</td></tr>{% endfor %}
{% for w in r.neurons.readers %}<tr><td>reader</td><td class="mono">{{ w.block }}</td><td class="mono">{{ w.neuron }}</td><td class="mono">–</td><td class="mono">{{ '%.2f'|format(w.eta2) }}</td><td class="mono">{{ '%.3g'|format(w.peak_label) }}</td><td class="mono">{{ '%.2f'|format(w.cv_r2) }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="small">{{ r.neurons.causal_text }}{% if r.sae %} The SAE tiling section below shows the same kind of tuning curves for Llama Scope features, for a features-vs-neurons comparison on the same manifold.{% endif %}</p>
</details>
{% endif %}

{% if r.interference %}
<h2>Interference with other manifolds</h2>
<p class="small">Each manifold lives in a 3-D slice of the model's 4096-dimensional activation space (its three PCA axes). Do different concepts share a slice, so that one could disturb another, or does each get its own? Cammarata allows that the residual stream may hold "manifolds woven through each other"; this measures it. <b>Left:</b> how much this manifold's slice overlaps every other manifold's slice at the same layer, on a 0–1 scale (1 = the same three directions, 0 = at right angles). Two unrelated 3-D slices of this space score about {{ r.interference.null95 }} or less (the dotted line is the typical random value), and the dashed line is the ceiling you get by comparing two halves of this manifold with itself. <b>Right:</b> this manifold and the one it overlaps most, <b>{{ r.interference.partner }}</b> (overlap {{ '%.2f'|format(r.interference.partner_overlap) }},{{ r.interference.partner_note }}), drawn together inside their two slices.</p>
<div class="two">
<div><div class="fig2d eager-fig" id="fig-int-bars" data-fig="json-int-bars" style="height:380px"></div>
<script type="application/json" id="json-int-bars">{{ r.interference.bars_json|safe }}</script></div>
<div><div class="fig eager-fig" id="fig-int-woven" data-fig="json-int-woven" style="height:380px"></div>
<script type="application/json" id="json-int-woven">{{ r.interference.woven_json|safe }}</script></div>
</div>
{% if r.interference.crosstalk %}
<p class="small" style="margin-top:14px"><b>Does it matter in practice?</b> We put a sentence about another concept in front of every prompt (for example a colour sentence before a weekday question) and re-read the manifold, comparing with a neutral sentence of similar length. If concepts interfered, the colour sentence should warp the weekday manifold more than a neutral one. Bars: how much of the manifold's neighbourhood structure survives, its shape score, and its task accuracy, per prefix domain. (The prefixed activations are centred per prefix sentence first, so the bars measure distortion of the manifold rather than the offset any sentence adds.) {{ r.interference.crosstalk_text }}</p>
<div class="fig2d eager-fig" id="fig-int-cross" data-fig="json-int-cross" style="height:340px"></div>
<script type="application/json" id="json-int-cross">{{ r.interference.crosstalk_json|safe }}</script>
<details><summary class="small mono">where the points moved</summary>
<div class="tablewrap"><table><thead><tr><th>prefix domain</th><th>example</th><th>k-NN overlap</th><th>shape</th><th>top-1 agreement</th>{% if r.interference.task %}<th>task acc.</th>{% endif %}<th>move in own subspace</th><th>move in prefix-domain subspace</th><th>random subspace</th></tr></thead><tbody>
{% for row in r.interference.cross_rows %}<tr><td class="mono">{{ row.domain }}</td><td class="small">{{ row.example }}</td><td class="mono">{{ row.knn }}</td><td class="mono">{{ row.shape }}</td><td class="mono">{{ row.agree }}</td>{% if r.interference.task %}<td class="mono">{{ row.task }}</td>{% endif %}<td class="mono">{{ row.own }}</td><td class="mono">{{ row.other }}</td><td class="mono">{{ row.null }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="small">"move in … subspace" is the mean fraction of each point's displacement that lies inside that 3-D subspace; the random column is the same for pooled-covariance random subspaces.</p>
{% if r.interference.cross_examples %}<div class="tablewrap"><table><thead><tr><th>prefix domain</th><th>prompt</th><th>clean →</th><th>prefixed →</th><th>answer</th></tr></thead><tbody>
{% for e in r.interference.cross_examples %}<tr><td class="mono">{{ e.domain }}</td><td class="small">{{ e.prompt }}</td><td class="mono">{{ e.clean }}</td><td class="mono">{{ e.prefixed }}</td><td class="mono">{{ e.answer }}</td></tr>{% endfor %}
</tbody></table></div>{% endif %}
</details>
{% endif %}
{% endif %}

{% if r.checkpoints %}
<h2>Forming during pre-training (OLMo-2 1B)</h2>
<p class="small">Llama 3.1 has no public training checkpoints, so this is <code>{{ r.checkpoints.model_id }}</code> read with the same prompts at residual position {{ r.checkpoints.layer }} of {{ r.checkpoints.n_layers }} (the position with the best shape metric at the final checkpoint), across 12 log-spaced stage-1 checkpoints and the final model. {% if r.probe %}Frames are the Fourier-probe view fitted per checkpoint (held-out predictions).{% else %}Frames are aligned to the final checkpoint by orthogonal Procrustes, so rotations between independently fitted PCAs do not show.{% endif %} This is a 1B model with 16 blocks, read on 600 prompts, so even a formed manifold looks rougher than the Llama-3.1-8B views above; several manifolds never form at this size. "Final" is the released model after the stage-2 anneal (a different data mix, and a merge of annealing runs), which can rearrange the last positions relative to the last stage-1 checkpoint. {{ r.checkpoints.summary }}</p>
<div class="fig eager-fig" id="fig-ckpt" data-fig="json-ckpt" style="height:520px"></div>
<script type="application/json" id="json-ckpt">{{ r.checkpoints.figure_json|safe }}</script>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">shape metric, task accuracy and centroid dimensionality vs tokens</div><div class="fig2d eager-fig" id="fig-ckpt-curve" data-fig="json-ckpt-curve" style="height:340px"></div>
<script type="application/json" id="json-ckpt-curve">{{ r.checkpoints.curve_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">shape metric by checkpoint and residual position</div><div class="fig2d eager-fig" id="fig-ckpt-heat" data-fig="json-ckpt-heat" style="height:340px"></div>
<script type="application/json" id="json-ckpt-heat">{{ r.checkpoints.heatmap_json|safe }}</script></div>
</div>
{% endif %}

<h2>Details</h2>
{% if r.steer %}
<details class="box"><summary><b>Steering along the manifold</b> <span class="small">· does the geometry drive behaviour?</span></summary>
<p class="small">The paper's causal test (App. B.1, Fig. 2 right). Take one base prompt from the middle of the manifold. During a forward pass, replace its block-19 vector with points that walk along the manifold (straight lines between the centroids of successive label bins, mapped back from PCA space), let the remaining blocks and the output head run, and read the probability of the paper's diagnostic answer tokens. If the shape is causal, sliding along it slides the model's answer: for temperature the answer should move from "cold" to "warm", for days from one weekday to the next.</p>
<p class="small">Base prompt: <code>{{ r.steer.base_prompt }}</code> (label {{ r.steer.base_label }}); PCA with {{ r.steer.k }} components, {{ r.steer.n_bins }} bins of <code>{{ r.steer.label }}</code>.</p>
<div class="fig2d eager-fig" id="fig-steer" data-fig="json-steer" style="height:340px"></div>
<script type="application/json" id="json-steer">{{ r.steer.figure_json|safe }}</script>
</details>
{% endif %}
{% if r.sae %}
<details class="box"><summary><b>SAE tiling</b> <span class="small">· {{ r.sae.repo }} · {{ r.sae.verdict }}</span></summary>
<p class="small">Bhalla et al. 2026 Sec. 5 ask whether a small, fixed group of SAE atoms spans the manifold (<em>compact capture</em>) or whether it is spread over many partially redundant atoms (<em>dilution</em>). The paper's SAEs were not released; this uses the public Llama Scope layer-19 SAE for the same base model (JumpReLU, 32k features). On these {{ r.sae.n_points }} prompts the SAE fires {{ '%.1f'|format(r.sae.l0) }} features per token on average and rebuilds {{ '%.0f'|format(100*r.sae.ve_total) }}% of the activation energy (cosine {{ '%.2f'|format(r.sae.cos) }}); relative to the within-manifold spread its reconstruction error is {{ r.sae.ve_within_text }}.</p>
<div class="tablewrap"><table style="margin:6px 0 12px"><thead><tr><th>method</th>{% for k in r.sae.ks %}<th>k = {{ k }}</th>{% endfor %}</tr></thead><tbody>
{% for row in r.sae.table %}<tr><td class="mono">{{ row.name }}</td>{% for v in row.vals %}<td class="mono">{{ v }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table></div>
<p class="small">Variance of the centered manifold activations explained by the span of k atoms (greedy over decoder directions), by the SAE's codes on k greedily chosen features, by k principal components, and by k random atoms. Ambient dimension of the expected geometry: {{ r.sae.ambient_dim }}; atoms needed to match PCA at that dimension: {{ r.sae.k_match }}.</p>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">variance explained vs number of features</div><div class="fig2d eager-fig" id="fig-sae-curves" data-fig="json-sae-curves"></div>
<script type="application/json" id="json-sae-curves">{{ r.sae.curves_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">tuning curves of the top features vs {{ r.sae.tuning_label }}</div><div class="fig2d eager-fig" id="fig-sae-tuning" data-fig="json-sae-tuning"></div>
<script type="application/json" id="json-sae-tuning">{{ r.sae.tuning_json|safe }}</script></div>
</div>
<div class="small mono" style="margin:14px 0 4px">manifold rebuilt from the first k selected features (grey = raw activations in the same PCA basis; use the dropdown)</div>
<div class="fig eager-fig" id="fig-sae-partial" data-fig="json-sae-partial" style="height:520px"></div>
<script type="application/json" id="json-sae-partial">{{ r.sae.partial_json|safe }}</script>
</details>
{% endif %}
<details class="box"><summary><b>What the source reports</b></summary>
<p><b>{{ r.geometry }}</b><br>
<span class="small">{{ r.model }} · layer {{ r.layer }} · last token · {{ r.replication_mode }}</span><br>
<span class="small">{% for s in r.sources %}<a href="{{ s[1] }}">{{ s[0] }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</span></p>
</details>
<details class="box"><summary><b>Replication status</b> <span class="small">· <span class="badge badge-{{ r.status }}">{{ r.badge }}</span></span></summary>
<p class="small">{{ r.reason }}</p>
<div class="tablewrap"><table style="margin:10px 0 0"><thead><tr><th>check</th><th>value</th><th>threshold</th><th>critical</th><th>result</th></tr></thead><tbody>
{% for c in r.checks %}<tr><td class="mono" title="{{ c.note }}">{{ c.name }}</td><td class="mono">{{ '%.3f'|format(c.value) }}</td><td class="mono">{{ c.op }} {{ c.threshold }}</td><td>{{ 'info only' if c.informative else ('yes' if c.critical else 'no') }}</td><td class="{{ 'check-pass' if c.passed else ('small' if c.informative else 'check-fail') }}">{{ 'pass' if c.passed else 'fail' }}</td></tr>{% endfor %}
</tbody></table></div>
<p class="small">Thresholds are ours; the sources describe shapes qualitatively. PCA explained variance: {{ r.evr }}.{% if r.preprocessing and r.preprocessing != 'none' %} Preprocessing: {{ r.preprocessing }}.{% endif %}{% if r.depth and r.depth.agreement %} Lens top-1 agreement with the model's real next token at the paper's layer: {{ r.depth.agreement }}.{% endif %}{% if r.depth %} Status by layer: {{ r.depth.status_line }}.{% endif %}</p>
</details>
<details class="box"><summary><b>Prompts</b> <span class="small">· {{ r.n_kept }} of {{ r.n_total }} kept</span></summary>
<div>template <pre>{{ r.template }}</pre>
{{ r.n_kept }} prompts kept of {{ r.n_total }} ({{ r.n_outliers }} norm outliers removed).<br>
examples: {% for e in r.examples %}<pre>{{ e }}</pre>{% endfor %}
{% if r.deviations %}<b>Deviations from the source</b><ul>{% for d in r.deviations %}<li class="small">{{ d }}</li>{% endfor %}</ul>{% endif %}
</div>
</details>
<details class="box"><summary><b>Extraction</b></summary>
<div class="small mono">
{% for k,v in r.meta.items() %}{{ k }}: {{ v }}<br>{% endfor %}
{% if r.precision %}<br>precision check vs float32: cosine min {{ '%.6f'|format(r.precision.cosine_min) }}, mean {{ '%.6f'|format(r.precision.cosine_mean) }}, max rel. err {{ '%.4f'|format(r.precision.rel_err_max) }} ({{ 'pass' if r.precision.passed else 'FAIL' }}){% endif %}
{% if r.depth %}<br>full-model run: {{ r.depth.model_id }}; block-{{ r.layer }} output vs truncated-model extraction cosine ≥ {{ '%.5f'|format(r.depth.cos_min) }}; tuned lens {{ r.depth.tuned_repo }}{% endif %}
</div>
</details>
{% if r.probe %}
<details class="box"><summary><b>Raw PCA (reference)</b></summary>
<p class="small">Top three principal components of the raw activations, the view used on every other page. For this manifold it is uninformative, as the paper reports.</p>
<div class="fig eager-fig" id="fig-main" data-fig="json-main" style="height:460px"></div>
<script type="application/json" id="json-main">{{ r.figure_json|safe }}</script>
</details>
{% endif %}
</div>
{% endblock %}"""


def _env() -> Environment:
    return Environment(
        loader=DictLoader({"base": BASE, "index": INDEX, "detail": DETAIL}),
        autoescape=select_autoescape(["html"]),
    )


def _safe_json(s: str) -> str:
    # "</" must not appear inside a <script> block; U+FFFD (undecodable byte tokens) is written as
    # a JSON escape so the page text never contains the literal replacement character.
    return s.replace("</", "<\\/").replace("\ufffd", "\\ufffd")


def _clean_text(s: str) -> str:
    """Visible placeholder for undecodable byte tokens in rendered table text."""
    return s.replace("\ufffd", "\u25a1")


def _round_list(v: Any, nd: int = 3) -> Any:
    if isinstance(v, list):
        return [round(x, nd) if isinstance(x, float) else x for x in v]
    return v


def _compact_depth_figure(raw: str, max_prompt: int = 40) -> str:
    """Shrink the layer-slider figure: drop shadow traces from frames, round coordinates, and
    shorten hover text (prompt truncated; lens readouts abbreviated to J:/L:/T:)."""
    import re

    fig = json.loads(raw)

    def shrink_text(texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            parts = t.split("<br>")
            head = parts[0]
            if len(head) > max_prompt:
                head = head[: max_prompt - 1] + "…"
            lens = [re.sub(r"^(logit lens|tuned lens|J-lens) → (.*?) \((\d\.\d\d)\)$",
                           lambda m: {"logit lens": "L", "tuned lens": "T", "J-lens": "J"}[m.group(1)]
                           + ":" + m.group(2) + " " + m.group(3), x)
                    for x in parts[2:] if "→" in x]
            label = parts[1] if len(parts) > 1 else ""
            out.append("<br>".join([head, label] + ([" · ".join(lens)] if lens else [])))
        return out

    def compact_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        for tr in traces:
            if tr.get("name") == "shadow":
                continue
            for k in ("x", "y", "z"):
                if k in tr:
                    tr[k] = _round_list(tr[k])
            if isinstance(tr.get("text"), list):
                tr["text"] = shrink_text(tr["text"])
            kept.append(tr)
        return kept

    fig["data"] = compact_traces(fig.get("data", []))
    for fr in fig.get("frames", []):
        fr["data"] = compact_traces(fr.get("data", []))
    return json.dumps(fig, separators=(",", ":"))


def _compact_sae_partial(raw: str, keep_ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
                         max_prompt: int = 40) -> str:
    """Keep a subset of the k-values in the partial-reconstruction figure and shorten hover."""
    fig = json.loads(raw)
    data = fig.get("data", [])
    keep_names = {f"top-{k} features" for k in keep_ks}
    new = []
    for tr in data:
        name = tr.get("name", "")
        if name.startswith("top-") and name not in keep_names:
            continue
        for k in ("x", "y", "z"):
            if k in tr:
                tr[k] = _round_list(tr[k])
        if isinstance(tr.get("text"), list):
            tr["text"] = [t.split("<br>")[0][:max_prompt] for t in tr["text"]]
        new.append(tr)
    fig["data"] = new
    menus: list[dict[str, Any]] = fig.get("layout", {}).get("updatemenus", [])
    if menus:
        buttons: list[dict[str, Any]] = []
        names = [tr.get("name") for tr in new]
        for kk in keep_ks:
            nm = f"top-{kk} features"
            if nm in names:
                vis = [n == nm or not str(n).startswith("top-") for n in names]
                buttons.append({"label": f"k = {kk}", "method": "update",
                                "args": [{"visible": vis}]})
        menus[0]["buttons"] = buttons
        menus[0]["active"] = len(buttons) - 1
        for tr in new:
            tr["visible"] = (not str(tr.get("name")).startswith("top-")) or \
                tr.get("name") == f"top-{keep_ks[-1]} features"
    return json.dumps(fig, separators=(",", ":"))


def _compact_figure(raw: str) -> str:
    """Round coordinates of a plain figure (main plot / previews)."""
    fig = json.loads(raw)
    for tr in fig.get("data", []):
        for k in ("x", "y", "z"):
            if k in tr:
                tr[k] = _round_list(tr[k])
    return json.dumps(fig, separators=(",", ":"))


def _source_links(keys: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k in keys:
        short = k.split()[0]
        if short in SOURCE_LINKS:
            label, url = SOURCE_LINKS[short]
            out.append((label, url))
        else:
            out.append((html.escape(k), "#"))
    return out


def _sae_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "sae.json").exists():
        return None
    sj = json.loads((d / "sae.json").read_text(encoding="utf-8"))
    sm, st = sj["summary"], sj["stats"]
    if "ve_at" not in sm or "ve_total" not in st:  # report written by an older pipeline version
        return None
    ks = [1, sm["ambient_dim"], 8, 16, 32, 64]
    ks = list(dict.fromkeys(ks))

    def fmt(v: float | None) -> str:
        return "–" if v is None else f"{v:.2f}"

    table = [{"name": label, "vals": [fmt(sm["ve_at"][str(k)][key] if str(k) in sm["ve_at"]
                                      else sm["ve_at"][k][key]) for k in ks]}
             for key, label in (("pca", "PCA components"), ("geometric", "greedy atoms (span)"),
                                ("statistical", "greedy atoms (codes)"),
                                ("random_atoms", "random atoms"))]
    verdict = ("consistent with the paper's dilution regime"
               if sm.get("consistent_with_dilution_claim") else "no dilution signature")
    vw = st["ve_within"]
    ve_within_text = (f"{(1 - vw):.1f}x that spread (variance explained {vw:.2f})"
                      if vw < 0 else f"{(1 - vw):.2f}x that spread (variance explained {vw:.2f})")
    return {
        "repo": sj["sae"].get("repo", ""), "n_points": sj["n_points"], "l0": st["l0_mean"],
        "ve_total": st["ve_total"], "cos": st.get("cosine_mean", float("nan")),
        "ve_within_text": ve_within_text, "ks": ks, "table": table,
        "ambient_dim": sm["ambient_dim"],
        "k_match": sm["k_geometric_to_match_pca_at_ambient_dim"] or "> 64", "verdict": verdict,
        "tuning_label": sj["tuning"]["label"].replace("_", " "),
        "curves_json": _safe_json((d / "sae_curves.json").read_text(encoding="utf-8")),
        "tuning_json": _safe_json((d / "sae_tuning.json").read_text(encoding="utf-8")),
        "partial_json": _safe_json(_compact_sae_partial((d / "sae_partial.json").read_text(encoding="utf-8"))),
    }


def _probe_payload(d: Path, rep: dict[str, Any]) -> dict[str, Any] | None:
    pv = rep.get("probe_view")
    if not pv or not (d / "probe_figure.json").exists():
        return None
    return {
        "label": pv["label"], "r2_circle": pv["r2_circle"], "r2_linear": pv["r2_linear"],
        "figure_json": _safe_json(_compact_figure((d / "probe_figure.json").read_text(encoding="utf-8"))),
    }


def _depth_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "depth.json").exists() or not (d / "depth_figure.json").exists():
        return None
    dj = json.loads((d / "depth.json").read_text(encoding="utf-8"))
    sc = dj["scores"]
    layers = sorted(int(k) for k in sc)
    status_line = ", ".join(f"{li}: {sc[str(li)]['status']}" for li in layers)
    lens_json = ((d / "lens_figure.json").read_text(encoding="utf-8")
                 if (d / "lens_figure.json").exists() else '{"data":[],"layout":{}}')
    labels_by_pc: dict[int, dict[str, Any]] = {}
    lab_path = d / "axis_labels.json"
    label_model = ""
    if lab_path.exists():
        lj = json.loads(lab_path.read_text(encoding="utf-8"))
        label_model = lj.get("_provenance", {}).get("model", "")
        labels_by_pc = {int(x["pc"]): x for x in lj.get("axes", [])}
    pcs = []
    for e in dj.get("pc_readouts", []):
        rows = []
        for key, name in (("jacobian_lens", "Jacobian lens"), ("logit_lens", "logit lens"),
                          ("tuned_lens", "tuned lens")):
            if key in e:
                rows.append({"name": name,
                             "up": _clean_text(" · ".join(repr(t) for t in e[key]["up"])),
                             "down": _clean_text(" · ".join(repr(t) for t in e[key]["down"]))})
        emp = e.get("empirical", {})
        if emp.get("positive"):
            rows.append({"name": "real logits (correlation)",
                         "up": _clean_text(" · ".join(repr(t) for t in emp["positive"])),
                         "down": _clean_text(" · ".join(repr(t) for t in emp["negative"]))})
        lab = labels_by_pc.get(int(e["pc"]))
        pcs.append({"pc": e["pc"], "rows": rows,
                    "var": f"{100 * e.get('explained_variance_ratio', 0):.0f}%",
                    "label": _clean_text(lab["label"]) if lab else "",
                    "rationale": _clean_text(lab.get("rationale", "")) if lab else "",
                    "confidence": lab.get("confidence", "") if lab else ""})
    agr = dj.get("lens_top1_agreement_with_model_at_layer") or {}
    agreement = ", ".join(f"{k} {v:.0%}" for k, v in agr.items())
    return {
        "pcs": pcs, "agreement": agreement, "label_model": label_model,
        "depth_json": _safe_json(_compact_depth_figure((d / "depth_figure.json").read_text(encoding="utf-8"))),
        "scores_json": _safe_json((d / "depth_scores.json").read_text(encoding="utf-8")),
        "lens_json": _safe_json(lens_json),
        "status_line": status_line, "n_plotted": dj["n_points_plotted"],
        "cos_min": dj["consistency_block19_vs_truncated"]["cosine_min"],
        "model_id": dj.get("model_id", ""),
        "tuned_repo": (dj.get("tuned_lens") or {}).get("repo", "none"),
    }


def _steer_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "depth.json").exists() or not (d / "steer_figure.json").exists():
        return None
    dj = json.loads((d / "depth.json").read_text(encoding="utf-8"))
    st = dj.get("steering")
    if not st:
        return None
    return {
        "figure_json": _safe_json((d / "steer_figure.json").read_text(encoding="utf-8")),
        "k": st["pca_components_used"], "n_bins": st["extras"].get("n_bins_used"),
        "label": STEER_TASKS.get(dj.get("name", ""), {}).get("label", ""),
        "base_prompt": st["base_prompt"],
        "base_label": st["extras"].get("base_label"),
    }


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def _x(v: Any) -> str:
    return "–" if v is None else f"{v:g}%"


def _ablation_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "ablation.json").exists() or not (d / "ablation_figure.json").exists():
        return None
    a = json.loads((d / "ablation.json").read_text(encoding="utf-8"))
    rows = []
    for key in a["order"]:
        r = a["conditions"].get(key)
        if r is None:
            continue
        g = a["global"].get(key, {})
        rows.append({"label": r["label"], "description": r["description"], "status": r["status"],
                     "zeroed": _fmt(100 * g.get("fraction_zeroed", 0.0), 0) + "%",
                     "shape": _fmt(r["shape_metric"]), "knn": _fmt(r["knn_overlap"]),
                     "move": _fmt(r["relative_move"]), "agree": _fmt(r["top1_agreement"]),
                     "kl": _fmt(r["kl_mean"], 3), "task": _fmt(r.get("task_accuracy")),
                     "dloss": _fmt(g.get("lm_loss_delta"), 3), "popqa": _fmt(g.get("popqa_accuracy"))})
    bp = a["breaking_points"]
    clean = a["conditions"]["clean"]
    parts = []
    if bp.get("shape_half") is None and bp.get("knn_half") is None:
        parts.append(f"The geometry never lost half its shape metric or half its neighbourhoods within the sweep (shape metric {clean['shape_metric']:.2f} clean, {a['conditions'].get('weak_10', {}).get('shape_metric', float('nan')):.2f} at X = 10%).")
    else:
        parts.append(f"The shape metric halves at X = {_x(bp.get('shape_half'))} and the neighbourhood overlap drops below ½ at X = {_x(bp.get('knn_half'))}.")
    if a.get("task"):
        parts.append(f"The {a['task']} task ({clean.get('task_accuracy', 0):.0%} clean) halves at X = {_x(bp.get('task_half'))}.")
    parts.append(f"Model-wide, PopQA fact recall halves at X = {_x(bp.get('popqa_half'))} and held-out loss is up 0.1 nats at X = {_x(bp.get('lm_loss_plus_0.1'))}.")
    r1 = a["conditions"].get("random_1")
    if r1:
        parts.append(f"The count-matched random control at the X = 1% count keeps {r1['knn_overlap']:.0%} of neighbourhoods and {r1['top1_agreement']:.0%} top-1 agreement, against {a['conditions']['weak_1']['knn_overlap']:.0%} and {a['conditions']['weak_1']['top1_agreement']:.0%} for the weak ablation.")
    return {
        "layer": a["layer"], "task": a.get("task"), "n_kl": a["n_kl"], "rows": rows,
        "summary": " ".join(parts), "breaking": bp, "clean_task": clean.get("task_accuracy"),
        "figure_json": _safe_json(_compact_depth_figure((d / "ablation_figure.json").read_text(encoding="utf-8"))),
        "geometry_json": _safe_json((d / "ablation_geometry.json").read_text(encoding="utf-8")),
        "behaviour_json": _safe_json((d / "ablation_behaviour.json").read_text(encoding="utf-8")),
    }


def _neurons_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "neurons.json").exists():
        return None
    n = json.loads((d / "neurons.json").read_text(encoding="utf-8"))
    c = n["causal"]
    ks = sorted({v["k"] for v in c.values()})
    txt = []
    for k in ks:
        t, r = c.get(f"top_{k}"), c.get(f"random_{k}")
        if t and r:
            txt.append(f"k = {k}: shape {t['shape_metric']:.2f} vs {r['shape_metric']:.2f} random, "
                       f"neighbourhoods kept {t['knn_overlap']:.0%} vs {r['knn_overlap']:.0%}"
                       + (f", task {t['task_accuracy']:.0%} vs {r['task_accuracy']:.0%}" if t.get("task_accuracy") is not None else ""))
    eta_top = float(np.mean([w["eta2"] for w in n["writers"]])) if n["writers"] else float("nan")
    bc = n["block_contributions"]
    sv = n["subspace_variance"] or 1e-9
    mlp_total, attn_total = sum(bc["mlp"]) / sv, sum(bc["attn"]) / sv
    return {
        "total_share": f"{100 * n['total_writer_share']:.0f}%",
        "mlp_total": f"{100 * mlp_total:.0f}%", "attn_total": f"{100 * attn_total:.0f}%",
        "layer": n["layer"], "n_half": n["n_neurons_for_half_share"], "n_90": n["n_neurons_for_90pct_share"],
        "n_total": n["n_neurons_total"], "tuning_label": n["tuning_label"].replace("_", " "),
        "eta_top": f"{eta_top:.2f}", "eta_rand": f"{n['writers_random_eta2_mean']:.2f}",
        "reader_blocks": f"{n['reader_blocks'][0]}–{n['reader_blocks'][-1]}" if n["reader_blocks"] else "–",
        "writers": n["writers"][:10], "readers": n["readers"][:10],
        "causal_text": f"Zeroing the top writers vs the same number of random neurons (blocks 0–{n['layer']}): {'; '.join(txt)}.",
        "figure_json": _safe_json(_compact_depth_figure((d / "neurons_figure.json").read_text(encoding="utf-8"))),
        "tuning_json": _safe_json((d / "neurons_tuning.json").read_text(encoding="utf-8")),
        "readers_json": _safe_json((d / "neurons_readers.json").read_text(encoding="utf-8")),
        "blocks_json": _safe_json((d / "neurons_blocks.json").read_text(encoding="utf-8")),
        "causal_json": _safe_json((d / "neurons_causal.json").read_text(encoding="utf-8")),
    }


def _interference_payload(d: Path, name: str, geometry: dict[str, Any] | None) -> dict[str, Any] | None:
    if geometry is None or name not in geometry.get("names", []) or not (d / "woven_figure.json").exists():
        return None
    from manifolds.interference import overlap_bar_figure

    w = json.loads((d / "woven.json").read_text(encoding="utf-8"))
    null95 = float(geometry.get("null_pooled_p95", 0.0))
    out: dict[str, Any] = {
        "partner": w["partner"].replace("_", " "), "partner_overlap": w["overlap"],
        "partner_note": (" which is within the range of random subspaces, so even the closest other manifold is essentially orthogonal to this one"
                         if w["overlap"] <= null95 else " which is above the random range"),
        "null95": f"{null95:.3f}",
        "bars_json": _safe_json(overlap_bar_figure(geometry, name).to_json()),
        "woven_json": _safe_json(_compact_figure((d / "woven_figure.json").read_text(encoding="utf-8"))),
        "crosstalk": None, "task": None, "cross_rows": [],
    }
    if (d / "crosstalk.json").exists() and (d / "crosstalk_figure.json").exists():
        ct = json.loads((d / "crosstalk.json").read_text(encoding="utf-8"))
        rows = []
        for dom, r in ct["conditions"].items():
            rows.append({"domain": dom, "example": r["example"][:70], "knn": _fmt(r["knn_overlap"]),
                         "shape": _fmt(r["shape_metric"]), "agree": _fmt(r["top1_agreement"]),
                         "task": _fmt(r.get("task_accuracy")), "own": _fmt(r["move_fraction_in_own_subspace"], 3),
                         "other": _fmt(r.get("move_fraction_in_prefix_domain_subspace"), 3),
                         "null": _fmt(r.get("move_fraction_in_null_subspace"), 3)})
        neutral = ct["conditions"].get("neutral")
        others = {k: v for k, v in ct["conditions"].items() if k != "neutral"}
        worst = min(others, key=lambda k: others[k]["knn_overlap"]) if others else None
        txt = ""
        if neutral and worst:
            txt = (f"A neutral prefix keeps {neutral['knn_overlap']:.0%} of neighbourhoods; the most disruptive domain is "
                   f"{worst} at {others[worst]['knn_overlap']:.0%}")
            if ct["clean"].get("task_accuracy") is not None:
                txt += (f" (task accuracy {ct['clean']['task_accuracy']:.0%} clean, {neutral['task_accuracy']:.0%} neutral, "
                        f"{others[worst]['task_accuracy']:.0%} with a {worst} prefix)")
            txt += "."
        examples = []
        for dom, r in ct["conditions"].items():
            for e in r.get("examples", [])[:2]:
                examples.append({"domain": dom, "prompt": _clean_text(e["prompt"][:90]),
                                 "clean": _clean_text(repr(e["clean_top1"])),
                                 "prefixed": _clean_text(repr(e["prefixed_top1"])),
                                 "answer": _clean_text(repr(e["answer"])) if e.get("answer") is not None else "–"})
        out.update({"crosstalk": ct, "task": ct["clean"].get("task"), "cross_rows": rows, "crosstalk_text": txt,
                    "cross_examples": examples,
                    "crosstalk_json": _safe_json((d / "crosstalk_figure.json").read_text(encoding="utf-8"))})
    return out


def _checkpoints_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "checkpoints.json").exists() or not (d / "checkpoints_figure.json").exists():
        return None
    c = json.loads((d / "checkpoints.json").read_text(encoding="utf-8"))
    ck = c["checkpoints"]
    final = ck[-1]
    first_match = next((x for x in ck if x["status"] == "matches"), None)
    first_half = next((x for x in ck if x["shape_metric"] >= 0.5 * final["shape_metric"]), None)
    parts = [f"At the final checkpoint the shape metric is {final['shape_metric']:.2f} ({final['status']})"]
    if c.get("task") and final.get("task_accuracy") is not None:
        parts[-1] += f" and {c['task']} accuracy {final['task_accuracy']:.0%}"
    parts[-1] += "."
    if first_match:
        parts.append(f"The check first passes at {first_match['label']} of pre-training.")
    elif first_half:
        parts.append(f"Half the final shape metric is reached by {first_half['label']}.")
    prs = [x["centroid_participation_ratio"] for x in ck]
    parts.append(f"Centroid dimensionality goes from {prs[0]:.1f} to {prs[-1]:.1f} (1 = an axis, 2 = a plane or circle).")
    return {"model_id": c["model_id"], "layer": c["display_layer"], "n_layers": c["n_layers"],
            "summary": " ".join(parts),
            "figure_json": _safe_json(_compact_depth_figure((d / "checkpoints_figure.json").read_text(encoding="utf-8"))),
            "curve_json": _safe_json((d / "checkpoints_curve.json").read_text(encoding="utf-8")),
            "heatmap_json": _safe_json((d / "checkpoints_heatmap.json").read_text(encoding="utf-8"))}


def _extensions_index(results_dir: Path, reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    pilots = [n for n in reports if (results_dir / n / "ablation.json").exists()
              or (results_dir / n / "neurons.json").exists() or (results_dir / n / "checkpoints.json").exists()]
    if not pilots:
        return None
    out: dict[str, Any] = {"pilots": ", ".join(p.replace("_", " ") for p in pilots), "global_json": None,
                           "heatmap_json": None, "breaking": [], "null": "", "global_text": ""}
    g = results_dir / "_ablation" / "global.json"
    if g.exists():
        from manifolds.ablate import global_figure

        gj = json.loads(g.read_text(encoding="utf-8"))
        out["global_json"] = _safe_json(global_figure(gj).to_json())
        c = gj["conditions"]
        w1, r1 = c.get("weak_1"), c.get("random_1")
        if w1 and r1:
            out["global_text"] = (f"Zeroing every activation below 1% of its neuron's max removes {w1['fraction_zeroed']:.0%} of all MLP activations "
                                  f"for +{w1['lm_loss_delta']:.3f} nats and PopQA {w1['popqa_accuracy']:.1%} (clean {c['clean']['popqa_accuracy']:.1%}); "
                                  f"the same count at random costs +{r1['lm_loss_delta']:.2f} nats and PopQA {r1['popqa_accuracy']:.1%}.")
    h = results_dir / "_interference" / "heatmap.json"
    if h.exists():
        out["heatmap_json"] = _safe_json(h.read_text(encoding="utf-8"))
        geo = json.loads((results_dir / "_interference" / "geometry.json").read_text(encoding="utf-8"))
        out["null"] = f"{geo['null_pooled_mean']:.3f}"
    for n in pilots:
        a = results_dir / n / "ablation.json"
        if not a.exists():
            continue
        aj = json.loads(a.read_text(encoding="utf-8"))
        bp = aj["breaking_points"]
        out["breaking"].append({"name": n, "task": aj.get("task") or "–",
                                "clean_acc": _fmt(aj.get("clean_task_accuracy")),
                                "shape": _x(bp.get("shape_half")), "knn": _x(bp.get("knn_half")),
                                "task_x": _x(bp.get("task_half")) if aj.get("task") else "–",
                                "popqa": _x(bp.get("popqa_half")), "loss": _x(bp.get("lm_loss_plus_0.1"))})
    return out


def _by_label(report: dict[str, Any]) -> str:
    color = report.get("color", {})
    if color.get("color_mode") == "hex":
        return "hue (each point drawn in its own colour)"
    if color.get("color_mode") == "diverging":
        return str(color.get("color_by", "")).replace("_", " ") + " (red = conservative, blue = liberal)"
    return str(color.get("color_by", "")).replace("_", " ")


def build_site(results_dir: Path, out_dir: Path) -> int:
    manifest = load_manifest()
    env = _env()
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    reports: dict[str, dict[str, Any]] = {}
    for d in sorted(results_dir.glob("*/report.json")):
        reports[d.parent.name] = json.loads(d.read_text(encoding="utf-8"))

    (out_dir / "manifolds").mkdir(parents=True, exist_ok=True)
    geometry = None
    if (results_dir / "_interference" / "geometry.json").exists():
        geometry = json.loads((results_dir / "_interference" / "geometry.json").read_text(encoding="utf-8"))
    ext = _extensions_index(results_dir, reports)

    # ---- index rows ------------------------------------------------------------------
    tier_order = ["core", "extended", "cyclic", "vision"]
    tiers: list[tuple[str, list[dict[str, Any]]]] = []
    counts = {"matches": 0, "partial": 0, "failed": 0, "pending": 0}
    for tier in tier_order:
        rows = []
        for m in manifest.by_tier(tier):  # type: ignore[arg-type]
            rep = reports.get(m.name)
            status = rep["status"] if rep else "pending"
            counts[status] = counts.get(status, 0) + 1
            sae_note = "—"
            sae_path = results_dir / m.name / "sae.json"
            if sae_path.exists():
                sj = json.loads(sae_path.read_text(encoding="utf-8"))
                if "ve_at" in sj["summary"]:
                    sae_note = ("dilution" if sj["summary"].get("consistent_with_dilution_claim")
                                else "no dilution signature")
            dd = results_dir / m.name
            ext_parts = [lab for f, lab in (("ablation.json", "ablation"), ("neurons.json", "neurons"),
                                            ("woven.json", "interference"), ("checkpoints.json", "checkpoints"))
                         if (dd / f).exists()]
            rows.append({
                "name": m.name, "model": m.model, "layer": m.layer, "sae_note": sae_note,
                "ext_note": " · ".join(ext_parts) if ext_parts else "—",
                "geometry": m.expected_geometry, "sources": _source_links(m.sources),
                "status": status, "badge": BADGE_TEXT[status],
                "reason": rep["shape"]["reason"] if rep else "not run yet",
                "has_page": rep is not None,
            })
        if rows:
            tiers.append((tier, rows))

    cards = []
    for name, rep in reports.items():
        prev = (results_dir / name / "figure_preview.json")
        if rep.get("probe_view") and (results_dir / name / "probe_figure.json").exists():
            prev = results_dir / name / "probe_figure.json"
        if not prev.exists():
            continue
        by = _by_label(rep)
        if rep.get("probe_view"):
            by = f"{rep['probe_view']['label']} (Fourier-probe view)"
        cards.append({
            "name": name, "title": name.replace("_", " "), "by": by,
            "status": rep["status"], "badge": BADGE_TEXT[rep["status"]],
            "preview_json": _safe_json(_compact_figure(prev.read_text(encoding="utf-8"))),
        })

    common = {"fonts": FONTS, "plotly": PLOTLY_CDN, "css": CSS, "js": JS, "built": built}
    for artifact, fname in ((False, "index.html"), (True, "artifact.html")):
        index_html = env.get_template("index").render(
            title=SITE_TITLE, root="", counts=counts, tiers=tiers, cards=cards, ext=ext,
            artifact=artifact, **common,
        )
        (out_dir / fname).write_text(index_html, encoding="utf-8")

    # ---- detail pages ----------------------------------------------------------------
    n = 0
    for name, rep in reports.items():
        fig_path = results_dir / name / "figure.json"
        if not fig_path.exists():
            continue
        spec = rep["spec"]
        ps = rep["prompt_set"]
        meta = {k: v for k, v in rep.get("meta", {}).items()
                if k in ("model_id", "model_id_resolution", "commit_sha", "layer", "n_layers_loaded",
                         "dtype", "device", "n_prompts", "elapsed_s", "finished", "torch",
                         "transformers", "seed")}
        r = {
            "title": name.replace("_", " "), "by": _by_label(rep), "status": rep["status"],
            "badge": BADGE_TEXT[rep["status"]], "reason": rep["shape"]["reason"],
            "geometry": spec["expected_geometry"], "model": spec["model"], "layer": spec["layer"],
            "replication_mode": spec["replication_mode"], "sources": _source_links(spec["sources"]),
            "checks": rep["shape"]["checks"], "preprocessing": rep.get("preprocessing", "none"),
            "evr": ", ".join(f"{v:.1%}" for v in rep["pca"]["explained_variance_ratio"]),
            "template": ps["template"], "n_kept": ps["n_kept"], "n_total": ps["n_total"],
            "n_outliers": ps["n_outliers_removed"], "examples": ps["examples"],
            "deviations": ps.get("deviations", []), "meta": meta,
            "precision": rep.get("precision_check"),
            "figure_json": _safe_json(_compact_figure(fig_path.read_text(encoding="utf-8"))),
            "sae": _sae_payload(results_dir / name),
            "probe": _probe_payload(results_dir / name, rep),
            "depth": _depth_payload(results_dir / name),
            "steer": _steer_payload(results_dir / name),
            "ablation": _ablation_payload(results_dir / name),
            "neurons": _neurons_payload(results_dir / name),
            "interference": _interference_payload(results_dir / name, name, geometry),
            "checkpoints": _checkpoints_payload(results_dir / name),
        }
        page = env.get_template("detail").render(
            title=f"{name.replace('_', ' ')} manifold", root="../", r=r, artifact=False,
            **common,
        )
        (out_dir / "manifolds" / f"{name}.html").write_text(page, encoding="utf-8")
        n += 1
    return n
