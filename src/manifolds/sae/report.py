"""Per-manifold SAE tiling report (Bhalla et al. 2026 Sec. 5) with the Llama Scope L19R-8x SAE."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch

from manifolds import RESULTS_DIR, ledger
from manifolds.analysis.pca import filter_norm_outliers, fit_pca
from manifolds.analysis.shapes import ROLES
from manifolds.config import RunConfig
from manifolds.manifest import ManifoldSpec
from manifolds.prompts.types import PromptSet
from manifolds.sae.capture import (
    greedy_geometric,
    greedy_statistical,
    partial_reconstructions,
    pca_curve,
    random_atoms_curve,
    summarize,
    tuning_curves,
)
from manifolds.sae.llama_scope import LlamaScopeSAE
from manifolds.viz.plot3d import CATEGORICAL, FONT, TEXT, subsample

PARTIAL_KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]


def _line_layout(title_x: str, title_y: str) -> dict[str, Any]:
    ax = {"showgrid": True, "gridcolor": "#eeeeee", "zeroline": False,
          "tickfont": {"family": FONT, "size": 11}, "title": {"font": {"family": FONT, "size": 12}}}
    return {
        "paper_bgcolor": "white", "plot_bgcolor": "white",
        "font": {"family": FONT, "color": TEXT, "size": 12},
        "margin": {"l": 50, "r": 10, "t": 10, "b": 45},
        "xaxis": ax | {"title": {"text": title_x}}, "yaxis": ax | {"title": {"text": title_y}},
        "legend": {"font": {"family": FONT, "size": 11}, "orientation": "h", "y": -0.25},
        "hoverlabel": {"font": {"family": FONT, "size": 11}},
    }


def curves_figure(curves: dict[str, list[float]], ambient_dim: int) -> go.Figure:
    fig = go.Figure()
    styles = {
        "pca": ("PCA (optimal linear)", {"dash": "dash", "color": "#9a9a9a"}),
        "random_atoms": ("random SAE atoms", {"dash": "dot", "color": "#FFB02E"}),
        "geometric": ("greedy: decoder directions", {"color": "#2E5BFF"}),
        "statistical": ("greedy: SAE codes", {"color": "#D62E8C"}),
    }
    for key, (name, line) in styles.items():
        c = curves.get(key)
        if c:
            fig.add_trace(go.Scatter(x=list(range(1, len(c) + 1)), y=c, mode="lines", name=name,
                                     line=line | {"width": 2}))
    fig.add_vline(x=ambient_dim, line={"color": "#1D272A", "width": 1, "dash": "dot"},
                  annotation={"text": f"ambient dim = {ambient_dim}",
                              "font": {"family": FONT, "size": 11}})
    fig.update_layout(**_line_layout("number of SAE features", "variance explained"))
    fig.update_yaxes(range=[0, 1.02])
    return fig


def tuning_figure(tcs: list[Any], label_name: str) -> go.Figure:
    fig = go.Figure()
    for i, tc in enumerate(tcs):
        col = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_trace(go.Scatter(x=tc.x, y=tc.y, mode="lines", name=f"feature {tc.feature}",
                                 line={"color": col, "width": 1.6}, fill="tozeroy",
                                 fillcolor=col + "22"))
    fig.update_layout(**_line_layout(label_name.replace("_", " "), "activation (max-normalised)"))
    return fig


def partial_figure(gt: np.ndarray, partial: Any, color_values: list[Any], color_mode: str,
                   hover: list[str]) -> go.Figure:
    """Ground-truth PCA cloud (grey) plus the reconstruction from the first k features, with a
    dropdown over k (Fig. 5 right / Fig. 9 of the paper)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=gt[:, 0], y=gt[:, 1], z=gt[:, 2], mode="markers",
                               marker={"size": 2.5, "color": "#c8c8c8", "opacity": 0.5},
                               name="raw activations (PCA)", hoverinfo="skip"))
    marker: dict[str, Any] = {"size": 4, "opacity": 0.95}
    if color_mode == "hex":
        marker["color"] = [str(c) for c in color_values]
    elif color_mode == "categorical":
        cats = list(dict.fromkeys(color_values))
        marker["color"] = [CATEGORICAL[cats.index(c) % len(CATEGORICAL)] for c in color_values]
    else:
        marker["color"] = [float(c) for c in color_values]
        marker["colorscale"] = "Viridis"
    for i, k in enumerate(partial.ks):
        c = partial.coords[k]
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers", marker=marker,
                                   name=f"top-{k} features", text=hover, hoverinfo="text",
                                   visible=(i == len(partial.ks) - 1)))
    buttons = []
    for i, k in enumerate(partial.ks):
        vis = [True] + [j == i for j in range(len(partial.ks))]
        buttons.append({"label": f"k = {k}", "method": "update", "args": [{"visible": vis}]})
    axis = {"visible": False, "showbackground": False}
    fig.update_layout(
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data"},
        updatemenus=[{"type": "dropdown", "buttons": buttons, "x": 0.02, "y": 0.98,
                      "active": len(partial.ks) - 1, "font": {"family": FONT, "size": 11}}],
        paper_bgcolor="white", margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"family": FONT, "color": TEXT}, showlegend=False,
        hoverlabel={"font": {"family": FONT, "size": 11}},
    )
    return fig


def analyze_sae(
    spec: ManifoldSpec,
    cfg: RunConfig,
    sae: LlamaScopeSAE,
    results_dir: Path = RESULTS_DIR,
    *,
    max_k: int = 64,
    n_tuning: int = 10,
    max_points: int = 2000,
    acts: np.ndarray | None = None,
    prompt_set: PromptSet | None = None,
    write: bool = True,
    provider: str = "unknown",
    instance: str = "unknown",
) -> dict[str, Any]:
    out_dir = results_dir / spec.name
    if acts is None:
        acts = np.load(out_dir / "activations.npy")
    if prompt_set is None:
        prompt_set = PromptSet.load(out_dir / "prompts.json")
    roles = ROLES[spec.name]
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    n_before = int(len(idx))
    if len(idx) > max_points:  # deterministic subsample keeps the greedy searches tractable
        idx = idx[subsample(len(idx), max_points, seed=cfg.seed)]
    labels = [prompt_set.labels[i] for i in idx]
    prompts = [prompt_set.prompts[i] for i in idx]
    pca = fit_pca(acts[idx], 3)

    start = ledger.now_iso()
    t0 = time.time()
    x = torch.as_tensor(acts[idx], device=sae.device, dtype=sae.w_enc.dtype)
    z = sae.encode(x)
    stats = sae.stats(x, z)
    dec = sae.w_dec
    sel_geo, cur_geo = greedy_geometric(x, dec, max_k=max_k)
    sel_stat, cur_stat = greedy_statistical(x, z, dec, max_k=max_k)
    curves = {
        "geometric": cur_geo, "statistical": cur_stat, "pca": pca_curve(x, max_k),
        "random_atoms": random_atoms_curve(x, dec, max_k=max_k, seed=cfg.seed),
    }
    tlabel = roles["tuning_label"]
    lab = np.asarray([lb[tlabel] for lb in labels], dtype=float)
    tcs = tuning_curves(z, sel_stat[:n_tuning], lab, sigma=3.0)
    partial = partial_reconstructions(x, z, dec, sel_stat, pca.components, pca.mean, PARTIAL_KS)
    elapsed = time.time() - t0
    end = ledger.now_iso()

    color_key = roles["color_by"]
    color_values = [lb[color_key] for lb in labels]
    hover = [f"{p}<br>{tlabel}: {lb[tlabel]}" for p, lb in zip(prompts, labels, strict=True)]
    fig_curves = curves_figure(curves, roles["ambient_dim"])
    fig_tuning = tuning_figure(tcs, tlabel)
    fig_partial = partial_figure(pca.coords, partial, color_values, roles["color_mode"], hover)

    summary = summarize(curves, roles["ambient_dim"])
    # Dilution claim (paper Sec. 5): no compact atom group spans the manifold -- the greedy
    # decoder-direction subspace needs far more than ambient_dim atoms to match what PCA
    # captures with ambient_dim components, yet the atoms beat random atoms by a wide margin.
    k_match = summary["k_geometric_to_match_pca_at_ambient_dim"]
    ratio = summary["geometric_over_pca_at_ambient_dim"]
    over_random = summary["geometric_over_random_at_64"]
    no_compact = (k_match is None or k_match > 2 * roles["ambient_dim"]) and (
        ratio is None or ratio < 0.5)
    aligned = over_random is not None and over_random > 3.0
    dilution = bool(no_compact and aligned)
    report = {
        "name": spec.name,
        "sae": sae.info,
        "stats": stats.to_dict(),
        "n_points": int(len(idx)),
        "n_points_before_subsample": n_before,
        "curves": curves,
        "selected_statistical": sel_stat,
        "selected_geometric": sel_geo,
        "summary": summary | {"consistent_with_dilution_claim": dilution,
                              "no_compact_capture": bool(no_compact),
                              "atoms_aligned_above_chance": bool(aligned)},
        "tuning": {"label": tlabel, "features": [tc.feature for tc in tcs],
                   "raw_max": [tc.raw_max for tc in tcs]},
        "partial_ks": partial.ks,
        "elapsed_s": round(elapsed, 2),
        "device": str(sae.device),
    }
    if write:
        (out_dir / "sae.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (out_dir / "sae_curves.json").write_text(fig_curves.to_json(), encoding="utf-8")
        (out_dir / "sae_tuning.json").write_text(fig_tuning.to_json(), encoding="utf-8")
        (out_dir / "sae_partial.json").write_text(fig_partial.to_json(), encoding="utf-8")
        if sae.device.type == "cuda":
            ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0),
                                             start, end, round(elapsed / 3600, 4),
                                             f"sae tiling {spec.name}"))
    return report
