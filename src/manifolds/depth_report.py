# ruff: noqa: E501
"""Depth sweep report: per-layer PCA + shape score, lens readouts, steering, consistency check."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch

from manifolds import RESULTS_DIR, ledger
from manifolds.analysis.pca import filter_norm_outliers, fit_pca
from manifolds.analysis.probe import cv_probe_coords
from manifolds.analysis.shapes import ROLES, center_by_group, run_shape_check
from manifolds.config import RunConfig
from manifolds.jlens import JacobianLensFull, pc_readouts
from manifolds.lens import TunedLens, lens_readout
from manifolds.manifest import ManifoldSpec
from manifolds.prompts.types import PromptSet
from manifolds.steer import STEER_TASKS, run_steering
from manifolds.viz.plot3d import CATEGORICAL, FONT, TEXT, _floor_traces, colorscale_for, subsample

LENS_LAYERS_DEFAULT = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20, 22, 24, 26, 28, 30, 32]


def _critical_metrics(report: dict[str, Any]) -> dict[str, float]:
    return {c["name"]: c["value"] for c in report["checks"] if c["critical"]}


def depth_figure(coords_by_layer: dict[int, np.ndarray], color_values: list[Any],
                 color_mode: str, hover: list[str], layer_default: int,
                 hover_by_layer: dict[int, list[str]] | None = None) -> go.Figure:
    """One 3-D scatter per layer with a slider (Goodfire styling, shared floor per frame)."""
    layers = sorted(coords_by_layer)
    hover_by_layer = hover_by_layer or {}
    base_marker: dict[str, Any] = {"size": 3.8, "opacity": 0.95, "line": {"width": 0}}
    if color_mode == "hex":
        base_marker["color"] = [str(c) for c in color_values]
    elif color_mode == "categorical":
        cats = list(dict.fromkeys(color_values))
        base_marker["color"] = [CATEGORICAL[cats.index(c) % len(CATEGORICAL)]
                                for c in color_values]
    else:
        base_marker["color"] = [float(c) for c in color_values]
        base_marker["colorscale"] = colorscale_for(color_mode)

    def traces(li: int) -> list[go.BaseTraceType]:
        c = np.round(coords_by_layer[li], 3)
        txt = hover_by_layer.get(li, hover)
        return _floor_traces(c, 3.8) + [go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="markers", marker=base_marker,
            text=txt, hoverinfo="text", name=f"layer {li}")]

    fig = go.Figure(data=traces(layer_default))
    fig.frames = [go.Frame(data=traces(li), name=str(li)) for li in layers]
    steps = [{"method": "animate", "label": str(li),
              "args": [[str(li)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}]} for li in layers]
    axis = {"visible": False, "showbackground": False}
    fig.update_layout(
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "data",
               "camera": {"eye": {"x": 1.55, "y": 1.25, "z": 0.85}}},
        sliders=[{"active": layers.index(layer_default), "steps": steps, "x": 0.05, "len": 0.9,
                  "y": 0.02, "pad": {"t": 10}, "font": {"family": FONT, "size": 11},
                  "currentvalue": {"prefix": "layer ", "font": {"family": FONT, "size": 12}}}],
        paper_bgcolor="white", margin={"l": 0, "r": 0, "t": 0, "b": 0},
        font={"family": FONT, "color": TEXT}, showlegend=False,
        hoverlabel={"font": {"family": FONT, "size": 11}, "bgcolor": "white"},
    )
    return fig


def scores_figure(layers: list[int], metrics: dict[str, list[float]], layer_marker: int
                  ) -> go.Figure:
    fig = go.Figure()
    for i, (name, vals) in enumerate(metrics.items()):
        fig.add_trace(go.Scatter(x=layers, y=vals, mode="lines+markers", name=name,
                                 line={"color": CATEGORICAL[i % len(CATEGORICAL)], "width": 2},
                                 marker={"size": 5}))
    fig.add_vline(x=layer_marker, line={"color": "#1D272A", "width": 1, "dash": "dot"},
                  annotation={"text": "layer 19", "font": {"family": FONT, "size": 11}})
    ax = {"showgrid": True, "gridcolor": "#eeeeee", "zeroline": False,
          "tickfont": {"family": FONT, "size": 11}}
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font={"family": FONT, "color": TEXT, "size": 12},
                      margin={"l": 45, "r": 10, "t": 10, "b": 40},
                      xaxis=ax | {"title": {"text": "layer (0 = embeddings)"}},
                      yaxis=ax | {"title": {"text": "shape metric"}, "range": [-0.05, 1.05]},
                      legend={"font": {"family": FONT, "size": 11}, "orientation": "h", "y": -0.3})
    return fig


def lens_summary_figure(lens: dict[str, Any], layer: int, top_n: int = 12) -> go.Figure:
    """Most frequent top-1 next tokens over the manifold at one depth: logit lens vs tuned lens."""
    from collections import Counter

    fig = go.Figure()
    for i, kind in enumerate(("logit", "tuned", "jacobian")):
        d = lens.get(kind, {}).get(str(layer))
        if not d:
            continue
        cnt = Counter(t[0] for t in d["tokens"])
        items = cnt.most_common(top_n)
        fig.add_trace(go.Bar(y=[repr(t) for t, _ in items][::-1], x=[n for _, n in items][::-1],
                             orientation="h", name=f"{kind} lens",
                             marker={"color": CATEGORICAL[i]}))
    ax = {"showgrid": True, "gridcolor": "#eeeeee", "zeroline": False,
          "tickfont": {"family": FONT, "size": 11}}
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", barmode="group",
                      font={"family": FONT, "color": TEXT, "size": 12},
                      margin={"l": 110, "r": 10, "t": 10, "b": 40},
                      xaxis=ax | {"title": {"text": "number of prompts"}}, yaxis=ax,
                      legend={"font": {"family": FONT, "size": 11}, "orientation": "h", "y": -0.25})
    return fig


def steer_figure(res: dict[str, Any]) -> go.Figure:
    fig = go.Figure()
    probs = np.array(res["probs"])
    for j, t in enumerate(res["targets"]):
        fig.add_trace(go.Scatter(x=res["positions"], y=probs[:, j], mode="lines", name=t,
                                 line={"color": CATEGORICAL[j % len(CATEGORICAL)], "width": 2}))
    ax = {"showgrid": True, "gridcolor": "#eeeeee", "zeroline": False,
          "tickfont": {"family": FONT, "size": 11}}
    centers = res["bin_centers"]
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font={"family": FONT, "color": TEXT, "size": 12},
                      margin={"l": 45, "r": 10, "t": 10, "b": 40},
                      xaxis=ax | {"title": {"text": "position along the manifold (bin centroids)"},
                                  "tickvals": list(range(len(centers))),
                                  "ticktext": [f"{c:.3g}" for c in centers]},
                      yaxis=ax | {"title": {"text": "next-token probability"}},
                      legend={"font": {"family": FONT, "size": 11}, "orientation": "h", "y": -0.3})
    return fig


def run_depth(
    spec: ManifoldSpec,
    cfg: RunConfig,
    full_model: Any,
    tuned: TunedLens | None,
    results_dir: Path = RESULTS_DIR,
    *,
    jlens: JacobianLensFull | None = None,
    lens_layers: list[int] | None = None,
    max_points: int = 800,
    do_steer: bool = True,
    provider: str = "unknown",
    instance: str = "unknown",
) -> dict[str, Any]:
    out_dir = results_dir / spec.name
    ps = PromptSet.load(out_dir / "prompts.json")
    acts = np.load(out_dir / "activations.npy")
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    labels = [ps.labels[i] for i in idx]
    prompts = [ps.prompts[i] for i in idx]
    roles = ROLES[spec.name]
    layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
    start = ledger.now_iso()
    t0 = time.time()

    states_path = out_dir / "depth_states.npy"
    if states_path.exists():
        states = np.load(states_path)
    else:
        spans = [ps.spans[i] for i in idx] if ps.spans is not None else None
        states = full_model.capture_all_layers(prompts, spans=spans)
        np.save(states_path, states)
    n_layers = states.shape[1] - 1

    # consistency: the paper's block output of the full model vs the truncated-model extraction
    h19 = states[:, layer + 1].astype(np.float32)
    a19 = acts[idx]
    cos = (h19 * a19).sum(1) / (np.linalg.norm(h19, axis=1) * np.linalg.norm(a19, axis=1) + 1e-9)
    consistency = {"cosine_min": float(cos.min()), "cosine_mean": float(cos.mean())}

    # per-layer PCA + shape score
    sub = subsample(len(idx), max_points, seed=cfg.seed)
    coords_by_layer: dict[int, np.ndarray] = {}
    evr: dict[int, list[float]] = {}
    scores: dict[int, dict[str, Any]] = {}
    metric_names: list[str] = []
    nuisance = roles.get("nuisance_group")
    is_probe = roles["kind"] == "fourier_probe"
    value = (np.asarray([lab[roles["value"]] for lab in labels], float) if is_probe else None)
    for li in range(n_layers + 1):
        xl = states[:, li].astype(np.float32)
        if nuisance:
            xl = center_by_group(xl, labels, nuisance)
        pca = fit_pca(xl, 3)
        # frames: PCA, or for Fourier-probe manifolds the probe view fitted at this depth
        coords_by_layer[li] = (cv_probe_coords(xl[sub], cast(np.ndarray, value)[sub], seed=cfg.seed) if is_probe
                               else pca.coords[sub])
        evr[li] = [float(v) for v in pca.explained_variance_ratio]
        rep = run_shape_check(spec.name, pca.coords, labels, acts=xl).to_dict()
        crit = _critical_metrics(rep)
        metric_names = list(crit)
        scores[li] = {"status": rep["status"], "metrics": crit}

    lens_layers = lens_layers or [x for x in LENS_LAYERS_DEFAULT if x <= n_layers]
    lens = lens_readout(full_model, states[sub], lens_layers, tuned, k=3, jlens=jlens)

    # the model's actual next-token logits (final residual through the real unembedding)
    final = torch.as_tensor(states[sub, n_layers].astype(np.float32))
    final_logits = np.concatenate([full_model.unembed(final[i : i + 128]).cpu().numpy()
                                   for i in range(0, len(sub), 128)])
    actual_top1 = final_logits.argmax(1)
    # agreement of each lens' top-1 at the paper's layer with the model's actual top-1
    li19 = str(layer + 1)
    agreement = {}
    for kind in ("logit", "tuned", "jacobian"):
        d = lens.get(kind, {}).get(li19)
        if d:
            tok = full_model.tokenizer
            actual_str = [tok.decode([int(t)]) for t in actual_top1]
            agreement[kind] = float(np.mean([a == b[0] for a, b in zip(actual_str, d["tokens"],
                                                                       strict=True)]))
    # principal axes of the manifold at the paper's layer, decoded four ways
    pca19 = fit_pca(states[:, layer + 1].astype(np.float32), 3)
    pcs = pca19.components.astype(np.float32)
    pc_read = pc_readouts(full_model, pcs, pca19.coords[sub], final_logits,
                          layer + 1, jlens=jlens, tuned=tuned, k=10)
    for e, var in zip(pc_read, pca19.explained_variance_ratio, strict=True):
        e["explained_variance_ratio"] = float(var)

    steer_res = None
    if do_steer and spec.name in STEER_TASKS:
        steer_res = run_steering(full_model, spec.name, a19, labels, prompts,
                                 layer=layer).to_dict()
    elapsed = time.time() - t0
    end = ledger.now_iso()

    color_key = roles["color_by"]
    color_values = [labels[i][color_key] for i in sub]
    tl = roles["tuning_label"]

    def short(p: str) -> str:
        return p if len(p) <= 70 else p[:67] + "…"

    hover = [f"{short(prompts[i])}<br>{tl}: {labels[i][tl]}" for i in sub]
    hover_by_layer: dict[int, list[str]] = {}
    for li in lens_layers:
        lg = lens["logit"].get(str(li))
        tn = lens["tuned"].get(str(li))
        jl = lens["jacobian"].get(str(li))
        rows = []
        for j, base in enumerate(hover):
            parts = [base]
            if lg:
                parts.append(f"logit lens → {lg['tokens'][j][0]!r} ({lg['probs'][j][0]:.2f})")
            if tn:
                parts.append(f"tuned lens → {tn['tokens'][j][0]!r} ({tn['probs'][j][0]:.2f})")
            if jl:
                parts.append(f"J-lens → {jl['tokens'][j][0]!r} ({jl['probs'][j][0]:.2f})")
            rows.append("<br>".join(parts))
        hover_by_layer[li] = rows
    fig_depth = depth_figure(coords_by_layer, color_values, roles["color_mode"], hover,
                             layer_default=layer + 1, hover_by_layer=hover_by_layer)
    fig_lens = lens_summary_figure(lens, layer + 1)
    metrics = {m: [scores[li]["metrics"].get(m, float("nan")) for li in range(n_layers + 1)]
               for m in metric_names}
    fig_scores = scores_figure(list(range(n_layers + 1)), metrics, layer + 1)

    report = {
        "name": spec.name,
        "n_points": int(len(idx)),
        "n_points_plotted": int(len(sub)),
        "subsample_index": sub.tolist(),
        "n_layers": n_layers,
        "layer_convention": "0 = embeddings, i = output of block i-1",
        "paper_layer": layer,
        "frames_view": ("Fourier probe (out-of-fold), fitted per layer" if is_probe else "PCA per layer"),
        "consistency_block19_vs_truncated": consistency,
        "evr": {str(k): v for k, v in evr.items()},
        "scores": {str(k): v for k, v in scores.items()},
        "lens": lens,
        "lens_top1_agreement_with_model_at_layer": agreement,
        "pc_readouts": pc_read,
        "tuned_lens": (tuned.info if tuned is not None else None),
        "jacobian_lens": (jlens.info if jlens is not None else None),
        "steering": steer_res,
        "elapsed_s": round(elapsed, 2),
        "model_id": full_model.model_id,
    }
    (out_dir / "depth.json").write_text(json.dumps(report), encoding="utf-8")
    (out_dir / "depth_figure.json").write_text(fig_depth.to_json(), encoding="utf-8")
    (out_dir / "depth_scores.json").write_text(fig_scores.to_json(), encoding="utf-8")
    (out_dir / "lens_figure.json").write_text(fig_lens.to_json(), encoding="utf-8")
    if steer_res is not None:
        (out_dir / "steer_figure.json").write_text(steer_figure(steer_res).to_json(),
                                                   encoding="utf-8")
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         end, round(elapsed / 3600, 4),
                                         f"depth sweep + lens + steering {spec.name}"))
    return report


def rebuild_depth_figure(spec: ManifoldSpec, cfg: RunConfig, full_model: Any,
                         results_dir: Path = RESULTS_DIR, *, max_points: int = 800,
                         provider: str = "unknown", instance: str = "unknown") -> dict[str, Any]:
    """Rebuild only the layer-slider figure from an existing depth.json (lens hover text is
    reused); the residual states are captured again if `depth_states.npy` is absent."""
    out_dir = results_dir / spec.name
    dj = json.loads((out_dir / "depth.json").read_text(encoding="utf-8"))
    ps = PromptSet.load(out_dir / "prompts.json")
    acts = np.load(out_dir / "activations.npy")
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    labels = [ps.labels[i] for i in idx]
    prompts = [ps.prompts[i] for i in idx]
    roles = ROLES[spec.name]
    layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
    start = ledger.now_iso()
    t0 = time.time()
    states_path = out_dir / "depth_states.npy"
    if states_path.exists():
        states = np.load(states_path)
    else:
        spans = [ps.spans[i] for i in idx] if ps.spans is not None else None
        states = full_model.capture_all_layers(prompts, spans=spans)
        np.save(states_path, states)
    n_layers = states.shape[1] - 1
    sub = np.asarray(dj.get("subsample_index") or subsample(len(idx), max_points, seed=cfg.seed))
    nuisance = roles.get("nuisance_group")
    is_probe = roles["kind"] == "fourier_probe"
    value = (np.asarray([lab[roles["value"]] for lab in labels], float) if is_probe else None)
    coords_by_layer: dict[int, np.ndarray] = {}
    for li in range(n_layers + 1):
        xl = states[:, li].astype(np.float32)
        if nuisance:
            xl = center_by_group(xl, labels, nuisance)
        coords_by_layer[li] = (cv_probe_coords(xl[sub], cast(np.ndarray, value)[sub], seed=cfg.seed) if is_probe
                               else fit_pca(xl, 3).coords[sub])
    lens = dj["lens"]
    tl = roles["tuning_label"]

    def short(p: str) -> str:
        return p if len(p) <= 70 else p[:67] + "…"

    hover = [f"{short(prompts[i])}<br>{tl}: {labels[i][tl]}" for i in sub]
    hover_by_layer: dict[int, list[str]] = {}
    for li_s in lens.get("logit", {}):
        li = int(li_s)
        lg, tn, jl = lens["logit"].get(li_s), lens.get("tuned", {}).get(li_s), lens.get("jacobian", {}).get(li_s)
        rows = []
        for j, base in enumerate(hover):
            parts = [base]
            if lg:
                parts.append(f"logit lens → {lg['tokens'][j][0]!r} ({lg['probs'][j][0]:.2f})")
            if tn:
                parts.append(f"tuned lens → {tn['tokens'][j][0]!r} ({tn['probs'][j][0]:.2f})")
            if jl:
                parts.append(f"J-lens → {jl['tokens'][j][0]!r} ({jl['probs'][j][0]:.2f})")
            rows.append("<br>".join(parts))
        hover_by_layer[li] = rows
    color_key = roles["color_by"]
    color_values = [labels[i][color_key] for i in sub]
    fig_depth = depth_figure(coords_by_layer, color_values, roles["color_mode"], hover,
                             layer_default=layer + 1, hover_by_layer=hover_by_layer)
    (out_dir / "depth_figure.json").write_text(fig_depth.to_json(), encoding="utf-8")
    dj["frames_view"] = ("Fourier probe (out-of-fold), fitted per layer" if is_probe
                         else "PCA per layer")
    (out_dir / "depth.json").write_text(json.dumps(dj), encoding="utf-8")
    elapsed = time.time() - t0
    if torch.cuda.is_available() and not states_path.exists():
        pass
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round(elapsed / 3600, 4),
                                         f"depth figure rebuild {spec.name}"))
    return {"name": spec.name, "frames_view": dj["frames_view"], "elapsed_s": round(elapsed, 1)}
