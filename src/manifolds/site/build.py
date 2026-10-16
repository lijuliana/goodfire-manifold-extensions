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
a:focus-visible{outline:2px solid var(--fg);outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}
"""

JS = r"""
function renderFig(el, fig){
  Plotly.newPlot(el, fig.data, fig.layout, {responsive:true, displaylogo:false,
    modeBarButtonsToRemove:['toImage','resetCameraLastSave3d','hoverClosest3d']});
}
function figFrom(id){ return JSON.parse(document.getElementById(id).textContent); }
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.eager-fig').forEach(el => renderFig(el, figFrom(el.dataset.fig)));
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
</header>
<div class="kpis">
{% for k,v in counts.items() %}<div class="kpi"><b>{{ v }}</b><span>{{ k }}</span></div>{% endfor %}
</div>
<div class="tablewrap"><table>
<thead><tr><th>manifold</th><th>model</th><th>layer</th><th>expected geometry</th><th>paper / post</th><th>status</th><th>SAE tiling</th></tr></thead>
<tbody>
{% for tier, rows in tiers %}
<tr class="tier-row"><td colspan="7">{{ tier }}</td></tr>
{% for r in rows %}
<tr>
<td class="mono">{% if r.has_page %}<a href="manifolds/{{ r.name }}.html">{{ r.name }}</a>{% else %}{{ r.name }}{% endif %}</td>
<td class="small">{{ r.model }}</td><td class="mono">{{ r.layer }}</td><td>{{ r.geometry }}</td>
<td class="small">{% for s in r.sources %}<a href="{{ s[1] }}">{{ s[0] }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</td>
<td><span class="badge badge-{{ r.status }}" title="{{ r.reason }}">{{ r.badge }}</span></td>
<td class="small">{{ r.sae_note }}</td>
</tr>
{% endfor %}{% endfor %}
</tbody></table></div>
{% if cards %}
<h2>Gallery</h2>
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
{% endblock %}"""

DETAIL = r"""{% extends "base" %}{% block body %}
<a class="back mono" href="../index.html">← all manifolds</a>
<div class="detail">
<h1 class="mono">{{ r.title }}</h1>
<div class="by mono">by {{ r.by }} · <span class="badge badge-{{ r.status }}">{{ r.badge }}</span></div>
<div class="fig eager-fig" id="fig-main" data-fig="json-main"></div>
<script type="application/json" id="json-main">{{ r.figure_json|safe }}</script>
<p class="small mono">drag to rotate · scroll to zoom · hover for the prompt</p>
<div class="two">
<div>
<h2>What the source reports</h2>
<div class="box"><b>{{ r.geometry }}</b><br>
<span class="small">{{ r.model }} · layer {{ r.layer }} · last token · {{ r.replication_mode }}</span><br>
<span class="small">{% for s in r.sources %}<a href="{{ s[1] }}">{{ s[0] }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</span></div>
<h2>Replication status</h2>
<div class="box"><span class="badge badge-{{ r.status }}">{{ r.badge }}</span> <span class="small">{{ r.reason }}</span>
<table style="margin:10px 0 0"><thead><tr><th>check</th><th>value</th><th>threshold</th><th>critical</th><th>result</th></tr></thead><tbody>
{% for c in r.checks %}<tr><td class="mono" title="{{ c.note }}">{{ c.name }}</td><td class="mono">{{ '%.3f'|format(c.value) }}</td><td class="mono">{{ c.op }} {{ c.threshold }}</td><td>{{ 'yes' if c.critical else 'no' }}</td><td class="{{ 'check-pass' if c.passed else 'check-fail' }}">{{ 'pass' if c.passed else 'fail' }}</td></tr>{% endfor %}
</tbody></table>
<p class="small">Thresholds are ours; the sources describe shapes qualitatively. PCA explained variance: {{ r.evr }}.</p></div>
</div>
<div>
<h2>Prompts</h2>
<div class="box">template <pre>{{ r.template }}</pre>
{{ r.n_kept }} prompts kept of {{ r.n_total }} ({{ r.n_outliers }} norm outliers removed).<br>
examples: {% for e in r.examples %}<pre>{{ e }}</pre>{% endfor %}
{% if r.deviations %}<b>Deviations from the source</b><ul>{% for d in r.deviations %}<li class="small">{{ d }}</li>{% endfor %}</ul>{% endif %}
</div>
<h2>Extraction</h2>
<div class="box small mono">
{% for k,v in r.meta.items() %}{{ k }}: {{ v }}<br>{% endfor %}
{% if r.precision %}<br>precision check vs float32: cosine min {{ '%.6f'|format(r.precision.cosine_min) }}, mean {{ '%.6f'|format(r.precision.cosine_mean) }}, max rel. err {{ '%.4f'|format(r.precision.rel_err_max) }} ({{ 'pass' if r.precision.passed else 'FAIL' }}){% endif %}
</div>
</div>
</div>
{% if r.depth %}
<h2>Across the layers</h2>
<div class="box">
<p class="small">The paper looks at one depth. Here the same prompts are read out at every residual-stream position: 0 is the token embedding, i is the output of block i−1, so the paper's layer 19 is position 20. Drag the slider to watch the manifold form. Hover shows what the model would say next from that point at that depth (logit lens: the output head applied directly; tuned lens: Belrose et al. 2023's learned per-layer translator, {{ r.depth.tuned_repo }}).</p>
<div class="fig eager-fig" id="fig-depth" data-fig="json-depth" style="height:560px"></div>
<script type="application/json" id="json-depth">{{ r.depth.depth_json|safe }}</script>
<div class="two">
<div><div class="small mono" style="margin:12px 0 4px">shape metric by layer (critical checks of the rubric)</div><div class="fig2d eager-fig" id="fig-depth-scores" data-fig="json-depth-scores"></div>
<script type="application/json" id="json-depth-scores">{{ r.depth.scores_json|safe }}</script>
<p class="small">Status by layer: {{ r.depth.status_line }}.</p></div>
<div><div class="small mono" style="margin:12px 0 4px">most common next token at position 20 (paper's layer 19)</div><div class="fig2d eager-fig" id="fig-lens" data-fig="json-lens"></div>
<script type="application/json" id="json-lens">{{ r.depth.lens_json|safe }}</script>
<p class="small">Top-1 token of the logit lens and of the tuned lens over the {{ r.depth.n_plotted }} plotted prompts.</p></div>
</div>
<p class="small">Consistency check: block-19 output of the full model vs the truncated-model extraction used above, cosine ≥ {{ '%.5f'|format(r.depth.cos_min) }}. Full-model run: {{ r.depth.model_id }}.{% if r.depth.agreement %} Share of prompts where each lens' top-1 token at position 20 equals the model's real next token: {{ r.depth.agreement }}.{% endif %}</p>
</div>
{% if r.depth.pcs %}
<h2>What the axes mean</h2>
<div class="box">
<p class="small">Each principal axis of the plot above, decoded four ways. Three lenses project the axis direction through the model's output head (↑ tokens it promotes, ↓ tokens it suppresses): the logit lens (direct), the tuned lens (learned per-layer translator) and the Jacobian lens (Gurnee et al. 2026: transport by the corpus-averaged Jacobian of the final residual with respect to block 19, pre-fitted lens from Neuronpedia). The fourth is lens-free: the real next-token log-probabilities whose variation across these prompts correlates most with the coordinate. Agreement across the four is the evidence; a single lens can be fooled.</p>
<div class="tablewrap"><table><thead><tr><th>axis</th><th>readout</th><th>↑ promoted</th><th>↓ suppressed</th></tr></thead><tbody>
{% for pc in r.depth.pcs %}{% for row in pc.rows %}<tr>{% if loop.first %}<td class="mono" rowspan="{{ pc.rows|length }}">PC{{ pc.pc }}<br><span class="small">{{ pc.var }} of variance</span></td>{% endif %}<td class="small">{{ row.name }}</td><td class="mono small">{{ row.up }}</td><td class="mono small">{{ row.down }}</td></tr>{% endfor %}{% endfor %}
</tbody></table></div>
</div>
{% endif %}
{% endif %}
{% if r.steer %}
<h2>Steering along the manifold</h2>
<div class="box">
<p class="small">Bhalla et al. 2026 App. B.1 / Fig. 2 right. PCA on the layer-19 activations ({{ r.steer.k }} components for 90% variance), centroids of {{ r.steer.n_bins }} bins of <code>{{ r.steer.label }}</code>, straight-line interpolation between consecutive centroids, and each interpolated point added to the base prompt's layer-19 activation during a forward pass. The curves are the next-token probabilities of the paper's diagnostic tokens.</p>
<p class="small">Base prompt: <code>{{ r.steer.base_prompt }}</code> (label {{ r.steer.base_label }})</p>
<div class="fig2d eager-fig" id="fig-steer" data-fig="json-steer" style="height:340px"></div>
<script type="application/json" id="json-steer">{{ r.steer.figure_json|safe }}</script>
</div>
{% endif %}
{% if r.sae %}
<h2>SAE tiling · {{ r.sae.repo }}</h2>
<div class="box">
<p class="small">Bhalla et al. 2026 Sec. 5 ask whether a small, fixed group of SAE atoms spans the manifold (<em>compact capture</em>) or whether it is spread over many partially redundant atoms (<em>dilution</em>). The paper's own SAEs were not released; this uses the public Llama Scope layer-19 SAE for the same base model (JumpReLU, 32k features). On these {{ r.sae.n_points }} prompts the SAE fires {{ '%.1f'|format(r.sae.l0) }} features per token on average and rebuilds {{ '%.0f'|format(100*r.sae.ve_total) }}% of the activation energy (cosine {{ '%.2f'|format(r.sae.cos) }}); relative to the <em>within-manifold</em> spread its reconstruction error is {{ r.sae.ve_within_text }}.</p>
<table style="margin:6px 0 12px"><thead><tr><th>method</th>{% for k in r.sae.ks %}<th>k = {{ k }}</th>{% endfor %}</tr></thead><tbody>
{% for row in r.sae.table %}<tr><td class="mono">{{ row.name }}</td>{% for v in row.vals %}<td class="mono">{{ v }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table>
<p class="small">Variance of the centered manifold activations explained by the span of k atoms (greedy over decoder directions), by the SAE's actual codes on k greedily chosen features, by k principal components, and by k random atoms. Ambient dimension of the expected geometry: {{ r.sae.ambient_dim }}. Atoms needed to match PCA at the ambient dimension: {{ r.sae.k_match }}. Verdict: <b>{{ r.sae.verdict }}</b>.</p>
<div class="two">
<div><div class="small mono" style="margin-bottom:4px">variance explained vs number of features</div><div class="fig2d eager-fig" id="fig-sae-curves" data-fig="json-sae-curves"></div>
<script type="application/json" id="json-sae-curves">{{ r.sae.curves_json|safe }}</script></div>
<div><div class="small mono" style="margin-bottom:4px">tuning curves of the top features vs {{ r.sae.tuning_label }}</div><div class="fig2d eager-fig" id="fig-sae-tuning" data-fig="json-sae-tuning"></div>
<script type="application/json" id="json-sae-tuning">{{ r.sae.tuning_json|safe }}</script></div>
</div>
<div class="small mono" style="margin:14px 0 4px">manifold rebuilt from the first k selected features (grey = raw activations in the same PCA basis; use the dropdown)</div>
<div class="fig eager-fig" id="fig-sae-partial" data-fig="json-sae-partial" style="height:520px"></div>
<script type="application/json" id="json-sae-partial">{{ r.sae.partial_json|safe }}</script>
</div>
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


def _depth_payload(d: Path) -> dict[str, Any] | None:
    if not (d / "depth.json").exists() or not (d / "depth_figure.json").exists():
        return None
    dj = json.loads((d / "depth.json").read_text(encoding="utf-8"))
    sc = dj["scores"]
    layers = sorted(int(k) for k in sc)
    status_line = ", ".join(f"{li}: {sc[str(li)]['status']}" for li in layers)
    lens_json = ((d / "lens_figure.json").read_text(encoding="utf-8")
                 if (d / "lens_figure.json").exists() else '{"data":[],"layout":{}}')
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
        pcs.append({"pc": e["pc"], "rows": rows,
                    "var": f"{100 * e.get('explained_variance_ratio', 0):.0f}%"})
    agr = dj.get("lens_top1_agreement_with_model_at_layer") or {}
    agreement = ", ".join(f"{k} {v:.0%}" for k, v in agr.items())
    return {
        "pcs": pcs, "agreement": agreement,
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


def _by_label(report: dict[str, Any]) -> str:
    color = report.get("color", {})
    if color.get("color_mode") == "hex":
        return "hue (each point drawn in its own colour)"
    return str(color.get("color_by", "")).replace("_", " ")


def build_site(results_dir: Path, out_dir: Path) -> int:
    manifest = load_manifest()
    env = _env()
    built = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    reports: dict[str, dict[str, Any]] = {}
    for d in sorted(results_dir.glob("*/report.json")):
        reports[d.parent.name] = json.loads(d.read_text(encoding="utf-8"))

    (out_dir / "manifolds").mkdir(parents=True, exist_ok=True)

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
            rows.append({
                "name": m.name, "model": m.model, "layer": m.layer, "sae_note": sae_note,
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
        if not prev.exists():
            continue
        cards.append({
            "name": name, "title": name.replace("_", " "), "by": _by_label(rep),
            "status": rep["status"], "badge": BADGE_TEXT[rep["status"]],
            "preview_json": _safe_json(_compact_figure(prev.read_text(encoding="utf-8"))),
        })

    common = {"fonts": FONTS, "plotly": PLOTLY_CDN, "css": CSS, "js": JS, "built": built}
    for artifact, fname in ((False, "index.html"), (True, "artifact.html")):
        index_html = env.get_template("index").render(
            title=SITE_TITLE, root="", counts=counts, tiers=tiers, cards=cards,
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
            "checks": rep["shape"]["checks"],
            "evr": ", ".join(f"{v:.1%}" for v in rep["pca"]["explained_variance_ratio"]),
            "template": ps["template"], "n_kept": ps["n_kept"], "n_total": ps["n_total"],
            "n_outliers": ps["n_outliers_removed"], "examples": ps["examples"],
            "deviations": ps.get("deviations", []), "meta": meta,
            "precision": rep.get("precision_check"),
            "figure_json": _safe_json(_compact_figure(fig_path.read_text(encoding="utf-8"))),
            "sae": _sae_payload(results_dir / name),
            "depth": _depth_payload(results_dir / name),
            "steer": _steer_payload(results_dir / name),
        }
        page = env.get_template("detail").render(
            title=f"{name.replace('_', ' ')} manifold", root="../", r=r, artifact=False,
            **common,
        )
        (out_dir / "manifolds" / f"{name}.html").write_text(page, encoding="utf-8")
        n += 1
    return n
