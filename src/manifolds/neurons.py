# ruff: noqa: E501
"""Which MLP neurons write and read a manifold (research plan, idea 2).

Setting: the manifold is the top-3 PCA subspace C (3 x d) of the residual stream at the read
layer L (output of block L) for the manifold's prompts. The residual at the read token is the
sum of the embedding, every attention output and every MLP output of blocks 0..L, so each MLP
neuron (block b <= L, index j) adds a_bj(x) * w_bj to it, with a_bj the SwiGLU activation and
w_bj the j-th column of down_proj. Its contribution inside the subspace is a_bj(x) * (C w_bj).

Writers: the variance a neuron contributes along the subspace over the manifold's prompts,
Var_x(a_bj) * ||C w_bj||^2, as a share of the manifold's subspace variance (sum of the three
PCA eigenvalues). Because the manifold is an ordered structure, each neuron also gets a tuning
curve (mean activation per bin of the manifold's label) and a tuning strength (eta^2: between-bin
share of its activation variance). "Tiling" (Anthropic's counting-manifold paper, Cammarata's
essay) predicts strong writers with narrow, ordered receptive fields over the label.

Readers: neurons in blocks L+1..L+k whose activation at the read token is predicted by the three
manifold coordinates (cross-validated R^2 of a ridge fit on [PC1, PC2, PC3] plus their squares
and products), with the same tuning curves. Weight-based reading (||C W_gate[j]||^2 /
||W_gate[j]||^2) is reported too, with the caveat that it ignores the RMSNorm scale.

Per block, the subspace variance written by attention vs MLP (projection of each block's
attention and MLP outputs at the read token) says where the manifold is built.

Causal check: zero the top-k writers (in all positions) vs k random neurons from the same
blocks, re-read the manifold, and compare the shape metric, k-NN overlap and task accuracy.
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
from manifolds.analysis.probe import apply_probe, fit_probe
from manifolds.analysis.shapes import ROLES, category_order, slider_coloring
from manifolds.config import RunConfig
from manifolds.manifest import ManifoldSpec
from manifolds.prompts.types import PromptSet
from manifolds.tasks import first_token_ids, score_task, task_for
from manifolds.viz.plot3d import CATEGORICAL, frames_figure, line_layout, subsample

TOP_K_ABLATE = [16, 64, 256]
N_READER_BLOCKS = 4


def _binned_tuning(a: np.ndarray, label: np.ndarray, n_bins: int = 12
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean activation per label bin (equal-count bins for continuous labels; one bin per value
    when there are <= n_bins distinct values) and eta^2."""
    vals = np.unique(label)
    if len(vals) <= n_bins:
        edges = None
        bins = np.searchsorted(vals, label)
        centers = vals.astype(float)
    else:
        edges = np.quantile(label, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, label, side="right") - 1, 0, n_bins - 1)
        centers = np.array([(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)])
    means = np.array([a[bins == b].mean() if (bins == b).any() else np.nan
                      for b in range(len(centers))])
    tot = float(((a - a.mean()) ** 2).sum())
    between = float(sum(((bins == b).sum()) * (means[b] - a.mean()) ** 2
                        for b in range(len(centers)) if (bins == b).any()))
    eta2 = between / tot if tot > 0 else 0.0
    return centers, means, eta2


def _tuning_label(labels: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(lab[key]) for lab in labels])


@torch.no_grad()
def _block_contributions(fm: Any, prompts: list[str], spans: list[tuple[int, int]] | None,
                         blocks: list[int], basis_components: np.ndarray, basis_mean: np.ndarray
                         ) -> dict[str, Any]:
    """Per block: attention and MLP outputs at the read token projected onto the subspace."""
    base = fm.model.model if hasattr(fm.model, "model") else fm.model
    captured: dict[tuple[str, int], torch.Tensor] = {}
    handles = []

    def mk(kind: str, bi: int) -> Any:
        def hook(_m: Any, _i: Any, out: Any) -> None:
            captured[(kind, bi)] = out[0] if isinstance(out, tuple) else out
        return hook
    for bi in blocks:
        handles.append(base.layers[bi].self_attn.register_forward_hook(mk("attn", bi)))
        handles.append(base.layers[bi].mlp.register_forward_hook(mk("mlp", bi)))
    n = len(prompts)
    per_block = {k: np.empty((n, len(blocks), 3), dtype=np.float32) for k in ("attn", "mlp")}
    C = torch.as_tensor(basis_components, dtype=torch.float32)
    bs = fm.cfg.batch_size
    try:
        for s in range(0, n, bs):
            batch = prompts[s : s + bs]
            enc = fm.encode(batch)
            fm.model(**enc)
            last = enc["attention_mask"].sum(dim=1) - 1
            read = last.clone()
            if spans is not None:
                for j, p in enumerate(batch):
                    ti = fm.token_index(p, spans[s + j])
                    if ti is not None:
                        read[j] = min(ti, int(last[j]))
            for kind in ("attn", "mlp"):
                for bj, bi in enumerate(blocks):
                    h = captured[(kind, bi)]
                    rows = torch.arange(h.shape[0], device=h.device)
                    sel = h[rows, read.to(h.device)].float().cpu()
                    per_block[kind][s : s + len(batch), bj] = (sel @ C.T).numpy()
            captured.clear()
    finally:
        for hd in handles:
            hd.remove()
    out: dict[str, Any] = {}
    for kind in ("attn", "mlp"):
        v = per_block[kind]
        out[kind] = [float(((v[:, bj] - v[:, bj].mean(0)) ** 2).sum(1).mean())
                     for bj in range(len(blocks))]
    return out


class NeuronMask:
    """Zero chosen (block, neuron) activations at the down_proj input in every position."""

    def __init__(self, fm: Any) -> None:
        base = fm.model.model if hasattr(fm.model, "model") else fm.model
        self.blocks = list(base.layers)
        self.masks: dict[int, torch.Tensor] = {}
        self.handles = [b.mlp.down_proj.register_forward_pre_hook(self._mk(i))
                        for i, b in enumerate(self.blocks)]

    def set(self, pairs: list[tuple[int, int]], d_ff: int, device: Any) -> None:
        self.masks = {}
        for b, j in pairs:
            if b not in self.masks:
                self.masks[b] = torch.ones(d_ff, device=device)
            self.masks[b][j] = 0.0

    def clear(self) -> None:
        self.masks = {}

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def _mk(self, i: int) -> Any:
        def hook(_m: Any, args: Any) -> Any:
            m = self.masks.get(i)
            if m is None:
                return None
            h = args[0]
            return (h * m.to(h.device, h.dtype),)
        return hook


def run_neurons(spec: ManifoldSpec, cfg: RunConfig, fm: Any, *, results_dir: Path = RESULTS_DIR,
                max_prompts: int = 1000, max_points: int = 800, top_n: int = 12,
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
    roles = ROLES[spec.name]
    tl = roles["tuning_label"]
    label = _tuning_label(labels, tl)
    task = task_for(spec.name, labels)
    cand_ids = first_token_ids(fm.tokenizer, task.candidates) if task and task.candidates else None
    start = ledger.now_iso()
    t0 = time.time()
    base = fm.model.model if hasattr(fm.model, "model") else fm.model
    n_blocks = len(base.layers)
    writer_blocks = list(range(0, layer + 1))
    reader_blocks = list(range(layer + 1, min(n_blocks, layer + 1 + N_READER_BLOCKS)))

    # clean pass: residual at L, MLP activations of all relevant blocks, logits
    cap = fm.capture_layer_and_logits(prompts, layer + 1, spans=spans, cand_ids=cand_ids,
                                      mlp_blocks=writer_blocks + reader_blocks, progress=False)
    clean_prep = prepare(cap["hidden"], spec.name, labels)
    basis = fit_pca(clean_prep, 3)
    C = basis.components.astype(np.float32)                       # [3, d]
    coords = basis.coords.astype(np.float32)                      # [N, 3]
    subspace_var = float((coords ** 2).sum(1).mean())             # mean sq. radius = sum eigs
    mlp = cap["mlp"].astype(np.float32)                           # [N, B, d_ff]
    d_ff = mlp.shape[-1]
    clean_task = score_task(task, cap["top1"], cap["cand"], fm.tokenizer) if task else None

    # ---- writers ------------------------------------------------------------------------
    writers: list[dict[str, Any]] = []
    cw_norm2 = {}
    share_by_block = []
    for bj, bi in enumerate(writer_blocks):
        w = base.layers[bi].mlp.down_proj.weight                  # [d, d_ff]
        cw = (torch.as_tensor(C, device=w.device, dtype=torch.float32) @ w.float()).cpu().numpy()
        cw_norm2[bi] = (cw ** 2).sum(0)                           # [d_ff]
        a = mlp[:, bj]                                            # [N, d_ff]
        var = a.var(0)
        share = var * cw_norm2[bi] / (subspace_var + 1e-12)
        share_by_block.append(float(share.sum()))
        top = np.argsort(-share)[:top_n]
        for j in top:
            centers, means, eta2 = _binned_tuning(a[:, j], label)
            peak = float(centers[int(np.nanargmax(np.abs(means)))])
            writers.append({"block": bi, "neuron": int(j), "share": float(share[j]),
                            "eta2": eta2, "peak_label": peak, "mean_act": float(a[:, j].mean()),
                            "direction": (cw[:, j] / (np.linalg.norm(cw[:, j]) + 1e-9)).tolist(),
                            "tuning_x": centers.tolist(), "tuning_y": means.tolist()})
    writers.sort(key=lambda r: -r["share"])
    writers_top = writers[:top_n]
    # matched comparison: how tuned are the top writers vs random neurons of the same blocks
    rng = np.random.default_rng(cfg.seed)
    rand_eta = []
    for _ in range(top_n):
        bj = int(rng.integers(len(writer_blocks)))
        j = int(rng.integers(d_ff))
        rand_eta.append(_binned_tuning(mlp[:, bj, j], label)[2])
    # cumulative share: how many neurons carry half the subspace variance?
    all_share = np.concatenate([mlp[:, bj].var(0) * cw_norm2[bi] / (subspace_var + 1e-12)
                                for bj, bi in enumerate(writer_blocks)])
    srt = np.sort(all_share)[::-1]
    csum = np.cumsum(srt)
    total_share = float(csum[-1])
    n_half = int(np.searchsorted(csum, 0.5 * total_share) + 1)
    n_90 = int(np.searchsorted(csum, 0.9 * total_share) + 1)

    # ---- block contributions (attention vs MLP) -------------------------------------------
    blocks_contrib = _block_contributions(fm, prompts, spans, writer_blocks, C, basis.mean)

    # ---- readers ------------------------------------------------------------------------
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict

    feats = np.c_[coords, coords ** 2, coords[:, [0]] * coords[:, [1]],
                  coords[:, [0]] * coords[:, [2]], coords[:, [1]] * coords[:, [2]]]
    feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)
    cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    readers: list[dict[str, Any]] = []
    for bj, bi in enumerate(reader_blocks, start=len(writer_blocks)):
        a = mlp[:, bj]
        # pre-screen by correlation with the coordinates (cheap), then CV-R^2 on the top 200
        a_c = a - a.mean(0)
        a_c /= (a_c.std(0) + 1e-9)
        corr = np.abs(feats.T @ a_c) / len(a)                     # [9, d_ff]
        cand = np.argsort(-corr.max(0))[:200]
        pred = cross_val_predict(Ridge(alpha=1.0), feats, a[:, cand], cv=cv)
        ss_res = ((a[:, cand] - pred) ** 2).sum(0)
        ss_tot = ((a[:, cand] - a[:, cand].mean(0)) ** 2).sum(0) + 1e-9
        r2 = 1 - ss_res / ss_tot
        wg = base.layers[bi].mlp.gate_proj.weight.float()        # [d_ff, d]
        cg = (torch.as_tensor(C, device=wg.device) @ wg.T).cpu().numpy()   # [3, d_ff]
        wfrac = (cg ** 2).sum(0) / (wg.norm(dim=1).cpu().numpy() ** 2 + 1e-9)
        for k in np.argsort(-r2)[:top_n]:
            j = int(cand[k])
            centers, means, eta2 = _binned_tuning(a[:, j], label)
            readers.append({"block": bi, "neuron": j, "cv_r2": float(r2[k]), "eta2": eta2,
                            "weight_subspace_fraction": float(wfrac[j]),
                            "peak_label": float(centers[int(np.nanargmax(np.abs(means)))]),
                            "tuning_x": centers.tolist(), "tuning_y": means.tolist()})
    readers.sort(key=lambda r: -r["cv_r2"])
    readers_top = readers[:top_n]
    rand_wfrac = float(np.mean([3 / fm.d_model] * 1))             # isotropic expectation

    # ---- causal: ablate top-k writers vs random -------------------------------------------
    masker = NeuronMask(fm)
    causal: dict[str, dict[str, Any]] = {}
    sub = subsample(len(idx), max_points, seed=cfg.seed)
    is_probe = roles["kind"] == "fourier_probe"
    probe = (fit_probe(clean_prep, np.asarray([lab[roles["value"]] for lab in labels], float))
             if is_probe else None)

    def view(prep: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        return apply_probe(probe, prep) if probe is not None else fixed

    steps: list[tuple[str, np.ndarray]] = [("clean", view(clean_prep, coords)[sub])]
    captions: dict[str, str] = {"clean": "no neurons zeroed"}
    order = [(r["block"], r["neuron"]) for r in writers]
    # writers list only holds top_n per block; extend with the global ranking for larger k
    flat = [(bi, int(j)) for bj, bi in enumerate(writer_blocks)
            for j in range(d_ff)]
    rank = np.argsort(-all_share)
    order = [flat[i] for i in rank[: max(TOP_K_ABLATE)]]
    ks = sorted({min(k, len(flat)) for k in TOP_K_ABLATE})
    try:
        for k in ks:
            for cond, pairs in (("top", order[:k]),
                                ("random", [flat[i] for i in rng.choice(len(flat), k, replace=False)])):
                masker.set(pairs, d_ff, fm.input_device)
                c2 = fm.capture_layer_and_logits(prompts, layer + 1, spans=spans,
                                                 cand_ids=cand_ids, progress=False)
                prep = prepare(c2["hidden"], spec.name, labels)
                cmp = compare(spec.name, labels, clean_prep, prep, basis)
                tk = score_task(task, c2["top1"], c2["cand"], fm.tokenizer) if task else None
                key = f"{cond}_{k}"
                causal[key] = {"condition": cond, "k": k, "status": cmp["status_refit"],
                               "shape_metric": summary_metric(spec.name, cmp["metrics_refit"]),
                               "metrics": cmp["metrics_refit"], "knn_overlap": cmp["knn_overlap"],
                               "relative_move": cmp["relative_move"],
                               "top1_agreement": float((cap["top1"] == c2["top1"]).mean()),
                               "task_accuracy": tk["accuracy"] if tk else None,
                               "share_removed": float(all_share[rank[:k]].sum() / (total_share + 1e-12))
                               if cond == "top" else
                               float(sum(all_share[flat.index(p)] for p in pairs) / (total_share + 1e-12))}
                lab = f"{'top' if cond == 'top' else 'random'} {k}"
                steps.append((lab, view(prep, cmp["coords_fixed_basis"])[sub]))
                captions[lab] = (f"top {k} writer neurons zeroed · "
                                 f"{causal[key]['share_removed']:.0%} of the neuron-attributable "
                                 f"subspace variance" if cond == "top" else
                                 f"{k} random neurons of the same blocks zeroed")
                print(f"  {spec.name:20s} ablate {key:11s} shape={causal[key]['shape_metric']:.3f} "
                      f"knn={cmp['knn_overlap']:.2f} task={causal[key]['task_accuracy']}")
    finally:
        masker.clear()
        masker.remove()
    elapsed = time.time() - t0

    clean_shape = compare(spec.name, labels, clean_prep, clean_prep, basis)
    report = {
        "name": spec.name, "layer": layer, "n_prompts": int(len(idx)), "d_ff": int(d_ff),
        "writer_blocks": writer_blocks, "reader_blocks": reader_blocks,
        "subspace_variance": subspace_var,
        "evr": [float(v) for v in basis.explained_variance_ratio],
        "tuning_label": tl,
        "writers": writers_top,
        "writers_random_eta2_mean": float(np.mean(rand_eta)),
        "writer_share_by_block": {str(b): s for b, s in zip(writer_blocks, share_by_block, strict=True)},
        "total_writer_share": total_share,
        "n_neurons_for_half_share": n_half, "n_neurons_for_90pct_share": n_90,
        "n_neurons_total": int(len(all_share)),
        "block_contributions": {"blocks": writer_blocks, **blocks_contrib},
        "readers": readers_top, "reader_weight_fraction_isotropic": rand_wfrac,
        "causal": causal,
        "clean": {"status": clean_shape["status_refit"],
                  "shape_metric": summary_metric(spec.name, clean_shape["metrics_refit"]),
                  "task_accuracy": clean_task["accuracy"] if clean_task else None,
                  "task": task.name if task else None},
        "view": ("Fourier probe fitted on the clean activations, applied to every frame"
                 if is_probe else "clean PCA basis"),
        "model_id": fm.model_id, "elapsed_s": round(elapsed, 1),
    }
    (out_dir / "neurons.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    color_values, color_mode = slider_coloring(spec.name, [labels[i] for i in sub])
    hover = [(prompts[i] if len(prompts[i]) <= 60 else prompts[i][:57] + "…")
             + f"<br>{tl}: {labels[i][tl]}" for i in sub]
    fig = frames_figure(steps, color_values, color_mode=color_mode, hover_text=hover,
                        prefix="neurons zeroed: ", captions=captions,
                        category_order=category_order(spec.name, labels))
    (out_dir / "neurons_figure.json").write_text(fig.to_json(), encoding="utf-8")
    (out_dir / "neurons_tuning.json").write_text(tuning_figure(writers_top, tl, "writer").to_json(),
                                                 encoding="utf-8")
    (out_dir / "neurons_readers.json").write_text(tuning_figure(readers_top, tl, "reader").to_json(),
                                                  encoding="utf-8")
    (out_dir / "neurons_blocks.json").write_text(blocks_figure(report).to_json(), encoding="utf-8")
    (out_dir / "neurons_causal.json").write_text(causal_figure(report).to_json(), encoding="utf-8")
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round(elapsed / 3600, 4),
                                         f"neuron writers/readers + causal ablation {spec.name}"))
    return report


def tuning_figure(rows: list[dict[str, Any]], label: str, kind: str) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(rows[:10]):
        col = CATEGORICAL[i % len(CATEGORICAL)]
        y = np.asarray(r["tuning_y"], dtype=float)
        scale = float(np.nanmax(np.abs(y))) or 1.0
        fig.add_trace(go.Scatter(x=r["tuning_x"], y=(y / scale).tolist(), mode="lines+markers",
                                 name=f"L{r['block']} n{r['neuron']}", line={"color": col, "width": 1.6},
                                 marker={"size": 4}))
    fig.update_layout(**line_layout(label.replace("_", " "),
                                    f"{kind} activation (max-normalised)", legend_y=-0.35))
    return fig


def blocks_figure(rep: dict[str, Any]) -> go.Figure:
    bc = rep["block_contributions"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bc["blocks"], y=bc["mlp"], name="MLP output", marker={"color": CATEGORICAL[0]}))
    fig.add_trace(go.Bar(x=bc["blocks"], y=bc["attn"], name="attention output",
                         marker={"color": CATEGORICAL[1]}))
    lay = line_layout("block", "variance written into the manifold subspace", legend_y=-0.3)
    fig.update_layout(barmode="group", **lay)
    return fig


def causal_figure(rep: dict[str, Any]) -> go.Figure:
    c = rep["causal"]
    fig = go.Figure()
    for cond, dash, sym in (("top", "solid", "circle"), ("random", "dot", "x")):
        ks = sorted({v["k"] for v in c.values() if v["condition"] == cond})
        xs = ["0"] + [str(k) for k in ks]
        for key, name, col in (("shape_metric", "shape metric", CATEGORICAL[0]),
                               ("knn_overlap", "k-NN overlap", CATEGORICAL[1]),
                               ("task_accuracy", "task accuracy", CATEGORICAL[3])):
            base_val = {"shape_metric": rep["clean"]["shape_metric"], "knn_overlap": 1.0,
                        "task_accuracy": rep["clean"]["task_accuracy"]}[key]
            if base_val is None:
                continue
            ys = [base_val] + [c[f"{cond}_{k}"][key] for k in ks]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                                     name=f"{name} · {'top writers' if cond == 'top' else 'random neurons'}",
                                     line={"color": col, "dash": dash, "width": 2},
                                     marker={"size": 6, "symbol": sym}))
    lay = line_layout("neurons zeroed (k)", "value", legend_y=-0.4)
    lay["xaxis"]["categoryorder"] = "array"
    lay["xaxis"]["categoryarray"] = ["0"] + [str(k) for k in sorted({v["k"] for v in c.values()})]
    fig.update_layout(**lay)
    return fig


__all__ = ["run_neurons", "NeuronMask", "tuning_figure", "blocks_figure", "causal_figure",
           "TOP_K_ABLATE"]
