# ruff: noqa: E501
"""How much do concept manifolds interfere with each other? (research plan, idea 3)

Geometry (no GPU): for every pair of manifolds read at the same layer, the overlap of their
top-k PCA subspaces, mean squared cosine of the principal angles = ||U_a^T U_b||_F^2 / k, in
[0, 1]. Two references: isotropic random k-dim subspaces of R^d (expected k/d) and random
subspaces drawn from the pooled covariance of all manifolds' activations (the residual stream is
anisotropic, so this is the fair null: directions that are big for everything). The ceiling is
split-half reliability (two halves of the same manifold). A variance-weighted version uses the
top-32 components: tr(S_a S_b) / sqrt(tr S_a^2 tr S_b^2).

Crosstalk (GPU): prefix each prompt of manifold A with a sentence from manifold B's domain (or a
neutral sentence of similar length), re-read A at its layer and re-score its task. Reported: k-NN
overlap and shape metric of the prefixed cloud against the clean one, task accuracy, top-1
agreement, and where the displacement went: the fraction of each point's move that lies inside
B's subspace, against the pooled-covariance null.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch

from manifolds import RESULTS_DIR, ledger
from manifolds.analysis.compare import compare, prepare, summary_metric
from manifolds.analysis.pca import filter_norm_outliers, fit_pca
from manifolds.analysis.shapes import ROLES, slider_coloring
from manifolds.config import RunConfig
from manifolds.manifest import Manifest
from manifolds.prompts.types import PromptSet
from manifolds.tasks import first_token_ids, score_task, task_for
from manifolds.viz.plot3d import (
    CATEGORICAL,
    FONT,
    TEXT,
    _floor_traces,
    colorscale_for,
    line_layout,
    subsample,
)

GLOBAL_DIR = RESULTS_DIR / "_interference"

DOMAIN_OF = {"colors": "colors", "weekdays_circle": "weekdays", "months_last": "months",
             "age": "age", "geography": "geography", "political_bias": "politics",
             "addition_sum_fourier": "arithmetic"}
PREFIXES: dict[str, list[str]] = {
    "colors": ["The hex code #e63946 is for the color red. ", "The hex code #2a9d8f is for the color teal. ",
               "The hex code #f4a261 is for the color orange. ", "The hex code #4361ee is for the color blue. ",
               "The hex code #9d4edd is for the color purple. ", "The hex code #80b918 is for the color green. "],
    "weekdays": ["Today is Wednesday. ", "The meeting is on Friday. ", "It happened on a Monday. ",
                 "We leave on Saturday. ", "The deadline is Tuesday. ", "They rest on Sunday. "],
    "months": ["It is the month of July. ", "The trip is in March. ", "Rent is due in November. ",
               "The festival is in September. ", "It snowed in January. ", "The harvest is in October. "],
    "age": ["They are 34 years old. ", "She is 7 years old. ", "He is 82 years old. ",
            "They are 19 years old. ", "She is 56 years old. ", "He is 45 years old. "],
    "geography": ["The city of Lima is in Peru. ", "The city of Osaka is in Japan. ",
                  "The city of Lagos is in Nigeria. ", "The city of Oslo is in Norway. ",
                  "The city of Perth is in Australia. ", "The city of Quito is in Ecuador. "],
    "politics": ["The senator gave a speech about immigration policy. ",
                 "The governor signed a bill on gun regulation. ",
                 "The candidate spoke about tax cuts and welfare. ",
                 "Protesters gathered over abortion rights. ",
                 "The debate covered healthcare and the minimum wage. ",
                 "The mayor announced a climate and energy plan. "],
    "arithmetic": ["17+25=42. ", "63+9=72. ", "48+37=85. ", "91+6=97. ", "22+58=80. ", "35+46=81. "],
    "neutral": ["The report was published this morning. ", "Please see the note below. ",
                "The door at the end of the hall is open. ", "This paragraph continues the text. ",
                "The package arrived without any delay. ", "A short sentence comes first here. "],
}


# ---- geometry --------------------------------------------------------------------------------
def _load_prepared(results_dir: Path, name: str, cfg: RunConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
    d = results_dir / name
    acts = np.load(d / "activations.npy")
    ps = PromptSet.load(d / "prompts.json")
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    labels = [ps.labels[i] for i in idx]
    return prepare(acts[idx], name, labels), labels


def subspace_overlap(ua: np.ndarray, ub: np.ndarray) -> float:
    """Mean squared cosine of principal angles between row-orthonormal bases [k, d]."""
    m = ua @ ub.T
    return float((m ** 2).sum() / min(ua.shape[0], ub.shape[0]))


def weighted_overlap(ua: np.ndarray, la: np.ndarray, ub: np.ndarray, lb: np.ndarray) -> float:
    """tr(S_a S_b) / sqrt(tr S_a^2 tr S_b^2) with S = U^T diag(l) U (top components)."""
    m = (np.sqrt(la)[:, None] * ua) @ (np.sqrt(lb)[:, None] * ub).T
    return float((m ** 2).sum() / np.sqrt((la ** 2).sum() * (lb ** 2).sum()))


def _orth_from_cov(samples: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """A random k-dim subspace whose directions follow the pooled activation covariance:
    orthonormalise k random combinations of centred activation vectors."""
    w = rng.normal(size=(k, samples.shape[0]))
    v = w @ samples
    q, _ = np.linalg.qr(v.T)
    return q.T


def _pairs_sorted(order: list[str], mat: np.ndarray, wmat: np.ndarray) -> list[dict[str, Any]]:
    pairs: list[tuple[float, dict[str, Any]]] = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if np.isfinite(mat[i, j]):
                pairs.append((float(mat[i, j]), {"a": order[i], "b": order[j],
                                                 "overlap": float(mat[i, j]),
                                                 "weighted": float(wmat[i, j])}))
    pairs.sort(key=lambda t: -t[0])
    return [p for _, p in pairs]


def run_geometry(manifest: Manifest, cfg: RunConfig, names: list[str], *,
                 results_dir: Path = RESULTS_DIR, out_dir: Path = GLOBAL_DIR, k: int = 3,
                 n_null: int = 50, seed: int = 0) -> dict[str, Any]:
    """Overlap matrices for the manifolds in `names` that share a read layer."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bases: dict[str, dict[str, Any]] = {}
    pooled: list[np.ndarray] = []
    for name in names:
        if not (results_dir / name / "activations.npy").exists():
            continue
        x, labels = _load_prepared(results_dir, name, cfg)
        spec = manifest.get(name)
        layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
        p = fit_pca(x, 32)
        xc = x - x.mean(0)
        # split-half reliability of the k-subspace
        perm = rng.permutation(len(x))
        h1, h2 = perm[: len(x) // 2], perm[len(x) // 2:]
        u1 = fit_pca(x[h1], k).components
        u2 = fit_pca(x[h2], k).components
        bases[name] = {"layer": layer, "U": p.components[:k], "U32": p.components,
                       "lam32": p.explained_variance_ratio * float((xc ** 2).sum(1).mean()),
                       "split_half": subspace_overlap(u1, u2), "n": int(len(x)),
                       "mean_norm": float(np.linalg.norm(x.mean(0)))}
        pooled.append(xc[subsample(len(xc), 300, seed=seed)])
    pool = np.concatenate(pooled, 0)
    d = pool.shape[1]
    null_bases = [_orth_from_cov(pool, k, rng) for _ in range(n_null)]
    iso = [np.linalg.qr(rng.normal(size=(d, k)))[0].T for _ in range(n_null)]
    null_pairs = [subspace_overlap(null_bases[i], null_bases[j])
                  for i in range(n_null) for j in range(i + 1, n_null)]
    iso_pairs = [subspace_overlap(iso[i], iso[j]) for i in range(n_null) for j in range(i + 1, n_null)]
    np.save(out_dir / "null_bases.npy", np.stack(null_bases).astype(np.float32))
    order = sorted(bases, key=lambda n: (bases[n]["layer"], names.index(n)))
    mat = np.full((len(order), len(order)), np.nan)
    wmat = np.full((len(order), len(order)), np.nan)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if bases[a]["layer"] != bases[b]["layer"]:
                continue
            mat[i, j] = subspace_overlap(bases[a]["U"], bases[b]["U"])
            wmat[i, j] = weighted_overlap(bases[a]["U32"], bases[a]["lam32"],
                                          bases[b]["U32"], bases[b]["lam32"])
    # overlap of each manifold with the null subspaces (is it in the "big" directions?)
    null_overlap = {a: float(np.mean([subspace_overlap(bases[a]["U"], nb) for nb in null_bases]))
                    for a in order}
    res = {
        "k": k, "d": int(d), "names": order, "layers": {a: bases[a]["layer"] for a in order},
        "overlap": mat.tolist(), "weighted_overlap": wmat.tolist(),
        "split_half": {a: bases[a]["split_half"] for a in order},
        "null_pooled_mean": float(np.mean(null_pairs)), "null_pooled_p95": float(np.percentile(null_pairs, 95)),
        "null_isotropic_mean": float(np.mean(iso_pairs)), "null_isotropic_expected": k / d,
        "overlap_with_pooled_null": null_overlap,
        "n_points": {a: bases[a]["n"] for a in order},
        "pairs_sorted": _pairs_sorted(order, mat, wmat),
    }
    (out_dir / "geometry.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out_dir / "heatmap.json").write_text(heatmap_figure(res).to_json(), encoding="utf-8")
    return res


def woven_figure(results_dir: Path, cfg: RunConfig, a: str, b: str, max_points: int = 600,
                 seed: int = 0) -> go.Figure:
    """Both manifolds (each centred on its own mean) in the span of their two 3-D subspaces,
    reduced to 3-D by PCA of the union."""
    xa, la = _load_prepared(results_dir, a, cfg)
    xb, lb = _load_prepared(results_dir, b, cfg)
    ia, ib = subsample(len(xa), max_points, seed), subsample(len(xb), max_points, seed)
    xa, xb = xa[ia] - xa.mean(0), xb[ib] - xb.mean(0)
    ua, ub = fit_pca(xa, 3).components, fit_pca(xb, 3).components
    q, _ = np.linalg.qr(np.r_[ua, ub].T)                       # [d, 6]
    za, zb = xa @ q, xb @ q
    p = fit_pca(np.r_[za, zb], 3)
    ca, cb = p.coords[: len(za)], p.coords[len(za):]
    tla, tlb = ROLES[a]["tuning_label"], ROLES[b]["tuning_label"]
    fig = go.Figure(data=_floor_traces(np.r_[ca, cb], 3.5))
    labs_a = [la[i] for i in ia]
    cvals, cmode = slider_coloring(a, labs_a)
    marker_a: dict[str, Any] = {"size": 3.5, "opacity": 0.9}
    if cmode == "hex":
        marker_a["color"] = [str(v) for v in cvals]
    elif cmode == "categorical":
        cats = list(dict.fromkeys(cvals))
        marker_a["color"] = [CATEGORICAL[cats.index(v) % len(CATEGORICAL)] for v in cvals]
    else:
        marker_a["color"] = [float(v) for v in cvals]
        marker_a["colorscale"] = colorscale_for(cmode)
    # the page's own manifold keeps its colour coding; the partner is drawn in grey
    fig.add_trace(go.Scatter3d(x=cb[:, 0], y=cb[:, 1], z=cb[:, 2], mode="markers",
                               marker={"size": 3.2, "color": "#9a9a9a", "opacity": 0.55},
                               name=b.replace("_", " ") + " (partner, grey)",
                               text=[f"{b}: {tlb} = {lab[tlb]}" for lab in [lb[i] for i in ib]],
                               hoverinfo="text"))
    fig.add_trace(go.Scatter3d(x=ca[:, 0], y=ca[:, 1], z=ca[:, 2], mode="markers", marker=marker_a,
                               name=a.replace("_", " "),
                               text=[f"{a}: {tla} = {lab[tla]}" for lab in labs_a],
                               hoverinfo="text"))
    axis = {"visible": False, "showbackground": False}
    fig.update_layout(scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data"},
                      paper_bgcolor="white", margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      font={"family": FONT, "color": TEXT}, showlegend=True,
                      legend={"font": {"family": FONT, "size": 11}, "orientation": "h", "y": -0.02,
                              "x": 0.5, "xanchor": "center"},
                      hoverlabel={"font": {"family": FONT, "size": 11}, "bgcolor": "white"})
    return fig


def heatmap_figure(res: dict[str, Any]) -> go.Figure:
    names = [n.replace("_", " ") for n in res["names"]]
    z = np.array(res["overlap"], dtype=float)
    text = [[("" if not np.isfinite(z[i, j]) else f"{z[i, j]:.2f}") for j in range(len(names))]
            for i in range(len(names))]
    fig = go.Figure(go.Heatmap(z=z.tolist(), x=names, y=names, colorscale="Viridis", zmin=0, zmax=1,
                               text=text, texttemplate="%{text}", textfont={"size": 9},
                               hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                               colorbar={"title": {"text": "subspace overlap"}, "thickness": 8}))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font={"family": FONT, "color": TEXT, "size": 11},
                      margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      xaxis={"tickangle": 45, "tickfont": {"size": 10}},
                      yaxis={"autorange": "reversed", "tickfont": {"size": 10}})
    return fig


def overlap_bar_figure(res: dict[str, Any], name: str) -> go.Figure:
    i = res["names"].index(name)
    row: list[tuple[str, float]] = [
        (res["names"][j], float(res["overlap"][i][j])) for j in range(len(res["names"]))
        if j != i and res["overlap"][i][j] is not None and np.isfinite(res["overlap"][i][j])]
    row.sort(key=lambda r: -r[1])
    fig = go.Figure(go.Bar(x=[r[0].replace("_", " ") for r in row], y=[r[1] for r in row],
                           marker={"color": CATEGORICAL[0]}, name="overlap"))
    fig.add_hline(y=res["null_pooled_mean"], line={"color": CATEGORICAL[1], "dash": "dot"},
                  annotation={"text": "random (pooled covariance)", "font": {"family": FONT, "size": 10}})
    fig.add_hline(y=res["split_half"][name], line={"color": CATEGORICAL[2], "dash": "dash"},
                  annotation={"text": "split-half ceiling", "font": {"family": FONT, "size": 10}})
    lay = line_layout("other manifold (same layer)", "subspace overlap (3 PCs)", y_range=(0, 1.02))
    lay["xaxis"]["tickangle"] = 45
    lay["margin"]["b"] = 110
    fig.update_layout(**lay)
    return fig


# ---- crosstalk (GPU) -------------------------------------------------------------------------
def run_crosstalk(spec: Any, cfg: RunConfig, fm: Any, geometry: dict[str, Any], *,
                  results_dir: Path = RESULTS_DIR, max_prompts: int = 800,
                  provider: str = "unknown", instance: str = "unknown") -> dict[str, Any]:
    out_dir = results_dir / spec.name
    ps = PromptSet.load(out_dir / "prompts.json")
    acts = np.load(out_dir / "activations.npy")
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    idx = idx[subsample(len(idx), max_prompts, seed=cfg.seed)]
    labels = [ps.labels[i] for i in idx]
    prompts = [ps.prompts[i] for i in idx]
    spans = [ps.spans[i] for i in idx] if ps.spans is not None else None
    layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
    task = task_for(spec.name, labels)
    cand_ids = first_token_ids(fm.tokenizer, task.candidates) if task and task.candidates else None
    rng = np.random.default_rng(cfg.seed)
    start = ledger.now_iso()
    t0 = time.time()

    def cap(pr: list[str], sp: list[tuple[int, int]] | None) -> dict[str, Any]:
        return fm.capture_layer_and_logits(pr, layer + 1, spans=sp, cand_ids=cand_ids, progress=False)

    clean = cap(prompts, spans)
    clean_prep = prepare(clean["hidden"], spec.name, labels)
    basis = fit_pca(clean_prep, 3)
    clean_task = score_task(task, clean["top1"], clean["cand"], fm.tokenizer) if task else None
    null = np.load(GLOBAL_DIR / "null_bases.npy") if (GLOBAL_DIR / "null_bases.npy").exists() else None
    # subspaces of the other manifolds (same layer only), from the geometry run's cached bases
    other_bases: dict[str, np.ndarray] = {}
    for other, dom in DOMAIN_OF.items():
        if other == spec.name or not (results_dir / other / "activations.npy").exists():
            continue
        if geometry["layers"].get(other) != layer:
            continue
        x, lab = _load_prepared(results_dir, other, cfg)
        other_bases[dom] = fit_pca(x, 3).components.astype(np.float32)

    own_domain = DOMAIN_OF.get(spec.name)
    rows: dict[str, dict[str, Any]] = {}
    for dom, sents in PREFIXES.items():
        if dom == own_domain:
            continue
        choice = rng.integers(len(sents), size=len(prompts))
        pre = [sents[c] for c in choice]
        pr = [p + q for p, q in zip(pre, prompts, strict=True)]
        sp = ([(s[0] + len(p), s[1] + len(p)) for s, p in zip(spans, pre, strict=True)]
              if spans is not None else None)
        c2 = cap(pr, sp)
        prep = prepare(c2["hidden"], spec.name, labels)
        # the prefix sentence itself shifts the read token's state; remove each prefix's mean
        # offset (keeping the clean cloud's mean) so the metrics see distortion, not the offset
        prep = prep - np.stack([prep[choice == k].mean(0) for k in choice]) + clean_prep.mean(0)
        cmp = compare(spec.name, labels, clean_prep, prep, basis)
        tk = score_task(task, c2["top1"], c2["cand"], fm.tokenizer) if task else None
        delta = prep - clean_prep
        dn = (delta ** 2).sum(1) + 1e-9
        frac_own = float((((delta @ basis.components.T) ** 2).sum(1) / dn).mean())
        frac_other = (float((((delta @ other_bases[dom].T) ** 2).sum(1) / dn).mean())
                      if dom in other_bases else None)
        frac_null = (float(np.mean([(((delta @ nb.T) ** 2).sum(1) / dn).mean() for nb in null[:20]]))
                     if null is not None else None)
        rows[dom] = {"prefix_domain": dom, "status": cmp["status_refit"],
                     "shape_metric": summary_metric(spec.name, cmp["metrics_refit"]),
                     "metrics": cmp["metrics_refit"], "knn_overlap": cmp["knn_overlap"],
                     "relative_move": cmp["relative_move"],
                     "top1_agreement": float((clean["top1"] == c2["top1"]).mean()),
                     "task_accuracy": tk["accuracy"] if tk else None,
                     "move_fraction_in_own_subspace": frac_own,
                     "move_fraction_in_prefix_domain_subspace": frac_other,
                     "move_fraction_in_null_subspace": frac_null,
                     "example": pr[0],
                     "examples": [{"prompt": pr[i], "clean_top1": fm.tokenizer.decode([int(clean["top1"][i])]),
                                   "prefixed_top1": fm.tokenizer.decode([int(c2["top1"][i])]),
                                   "answer": (task.answers[i] if task and task.kind == "exact" else
                                              (task.candidates[task.answers[i]] if task and task.candidates and not isinstance(task.answers[i], list) and task.answers[i] >= 0 else None))}
                                  for i in range(min(3, len(pr)))]}
        print(f"  {spec.name:20s} prefix={dom:10s} knn={cmp['knn_overlap']:.2f} "
              f"shape={rows[dom]['shape_metric']:.3f} task={rows[dom]['task_accuracy']} "
              f"own={frac_own:.3f} other={frac_other} null={frac_null}")
    elapsed = time.time() - t0
    clean_cmp = compare(spec.name, labels, clean_prep, clean_prep, basis)
    rep = {"name": spec.name, "layer": layer, "n_prompts": int(len(idx)),
           "preprocessing": "prefixed activations centred per prefix sentence (offset removed)",
           "clean": {"status": clean_cmp["status_refit"],
                     "shape_metric": summary_metric(spec.name, clean_cmp["metrics_refit"]),
                     "task_accuracy": clean_task["accuracy"] if clean_task else None,
                     "task": task.name if task else None},
           "conditions": rows, "model_id": fm.model_id, "elapsed_s": round(elapsed, 1)}
    (out_dir / "crosstalk.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out_dir / "crosstalk_figure.json").write_text(crosstalk_figure(rep).to_json(), encoding="utf-8")
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round(elapsed / 3600, 4),
                                         f"crosstalk prefixes {spec.name}"))
    return rep


def crosstalk_figure(rep: dict[str, Any]) -> go.Figure:
    rows = rep["conditions"]
    doms = sorted(rows, key=lambda d: (d == "neutral", -rows[d]["knn_overlap"]))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=doms, y=[rows[d]["knn_overlap"] for d in doms], name="k-NN overlap with clean",
                         marker={"color": CATEGORICAL[1]}))
    fig.add_trace(go.Bar(x=doms, y=[rows[d]["shape_metric"] for d in doms], name="shape metric",
                         marker={"color": CATEGORICAL[0]}))
    if rep["clean"].get("task_accuracy") is not None:
        fig.add_trace(go.Bar(x=doms, y=[rows[d]["task_accuracy"] for d in doms], name="task accuracy",
                             marker={"color": CATEGORICAL[3]}))
        fig.add_hline(y=rep["clean"]["task_accuracy"], line={"color": CATEGORICAL[3], "dash": "dot"},
                      annotation={"text": "clean accuracy", "font": {"family": FONT, "size": 10}})
    lay = line_layout("prefix sentence domain", "value", y_range=(0, 1.05), legend_y=-0.35)
    fig.update_layout(barmode="group", **lay)
    return fig


__all__ = ["run_geometry", "run_crosstalk", "woven_figure", "heatmap_figure", "overlap_bar_figure",
           "crosstalk_figure", "DOMAIN_OF", "PREFIXES", "subspace_overlap"]
