# ruff: noqa: E501
"""Watch a manifold form during pre-training (research plan, idea 4).

Llama 3.1 has no public training checkpoints, so this sweep uses OLMo-2 1B
(allenai/OLMo-2-0425-1B), whose stage-1 checkpoints are published every 1000 steps with token
counts, plus `main` (after the stage-2 anneal). Twelve log-spaced stage-1 checkpoints and `main`
are read with the same prompt sets as the Llama pages; every residual position is captured, so
each checkpoint has a shape metric at every layer. The displayed layer is the one with the best
metric at the final checkpoint. Frames of the slider are PCA-3 of that layer, aligned to the
final checkpoint by orthogonal Procrustes (same points, so the alignment is exact up to the fit).

Coarse-to-fine: the label is binned, bin centroids are taken in the layer's 3-D PCA, and the
participation ratio of the centroid covariance is reported (about 1 when the classes sit on an
axis, about 2 when they form a circle or plane), together with the shape metric and the task
accuracy at each checkpoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import torch

from manifolds import RESULTS_DIR, ledger
from manifolds.analysis.compare import critical_metrics, procrustes_align, summary_metric
from manifolds.analysis.pca import fit_pca
from manifolds.analysis.probe import cv_probe_coords
from manifolds.analysis.shapes import ROLES, center_by_group, run_shape_check
from manifolds.config import RunConfig
from manifolds.depth import DepthConfig, FullModel
from manifolds.manifest import Manifest
from manifolds.prompts import build
from manifolds.tasks import first_token_ids, score_task, task_for
from manifolds.viz.plot3d import CATEGORICAL, FONT, frames_figure, line_layout, subsample

MODEL_ID = "allenai/OLMo-2-0425-1B"
# (revision, billions of tokens); stage-1 checkpoints nearest to a log-spaced token schedule
CHECKPOINTS: list[tuple[str, float]] = [
    ("stage1-step0-tokens0B", 0.0), ("stage1-step300-tokens1B", 1.0),
    ("stage1-step10000-tokens21B", 21.0), ("stage1-step20000-tokens42B", 42.0),
    ("stage1-step50000-tokens105B", 105.0), ("stage1-step100000-tokens210B", 210.0),
    ("stage1-step190000-tokens399B", 399.0), ("stage1-step380000-tokens797B", 797.0),
    ("stage1-step570000-tokens1196B", 1196.0), ("stage1-step950000-tokens1993B", 1993.0),
    ("stage1-step1430000-tokens2999B", 2999.0), ("stage1-step1907359-tokens4001B", 4001.0),
    ("main", 4051.0),   # stage 2 adds ~50B tokens of annealing (OLMo 2 report)
]
GLOBAL_DIR = RESULTS_DIR / "_checkpoints"


def _free_revision(model_id: str, revision: str) -> None:
    """Delete one revision's files from the HF cache (the 1B checkpoints are fp32, ~6 GB each)."""
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir()
        for repo in info.repos:
            if repo.repo_id != model_id:
                continue
            for rev in repo.revisions:
                if revision in rev.refs:
                    info.delete_revisions(rev.commit_hash).execute()
    except Exception as e:  # cache cleanup is best effort
        print(f"cache cleanup skipped: {e}")


def load_checkpoint(revision: str, device: str = "auto", batch_size: int = 32,
                    model_id: str = MODEL_ID, dtype: str = "bfloat16") -> FullModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from manifolds.extract import _DTYPES, pick_device

    dev = pick_device(device)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=_DTYPES[dtype])
    model = cast(Any, model).to(dev)
    tok = AutoTokenizer.from_pretrained(model_id)
    cfg = DepthConfig(model_id=model_id, dtype=dtype, device=device, batch_size=batch_size,
                      max_length=256)
    return FullModel(cfg, model=model, tokenizer=tok)


def _bin_centroids(coords: np.ndarray, label: np.ndarray, n_bins: int = 12) -> dict[str, float]:
    vals = np.unique(label)
    if len(vals) <= n_bins:
        bins = np.searchsorted(vals, label)
        nb = len(vals)
    else:
        edges = np.quantile(label, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, label, side="right") - 1, 0, n_bins - 1)
        nb = n_bins
    cents = np.array([coords[bins == b].mean(0) for b in range(nb) if (bins == b).any()])
    cov = np.cov(cents.T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eig = np.clip(eig, 0, None)
    pr = float(eig.sum() ** 2 / ((eig ** 2).sum() + 1e-12))
    within = float(np.mean([((coords[bins == b] - coords[bins == b].mean(0)) ** 2).sum(1).mean()
                            for b in range(nb) if (bins == b).sum() > 1]))
    between = float(((cents - cents.mean(0)) ** 2).sum(1).mean())
    return {"centroid_participation_ratio": pr,
            "centroid_eig_ratio_2_over_1": float(eig[1] / (eig[0] + 1e-12)) if len(eig) > 1 else 0.0,
            "between_over_within": between / (within + 1e-12)}


def run_checkpoints(manifest: Manifest, cfg: RunConfig, names: list[str], *,
                    checkpoints: list[tuple[str, float]] | None = None, device: str = "auto",
                    batch_size: int = 32, max_prompts: int = 600, max_points: int = 600,
                    out_dir: Path = GLOBAL_DIR, results_dir: Path = RESULTS_DIR,
                    free_cache: bool = True, provider: str = "unknown", instance: str = "unknown",
                    model_id: str = MODEL_ID, layer_stride: int = 4) -> dict[str, Any]:
    checkpoints = checkpoints or CHECKPOINTS
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    sets: dict[str, Any] = {}
    for name in names:
        ps = build(name)
        sub = subsample(len(ps), max_prompts, seed=cfg.seed)
        sets[name] = {"ps": ps, "idx": sub, "prompts": [ps.prompts[i] for i in sub],
                      "labels": [ps.labels[i] for i in sub],
                      "spans": [ps.spans[i] for i in sub] if ps.spans is not None else None}
        sets[name]["task"] = task_for(name, sets[name]["labels"])
    start = ledger.now_iso()
    t0 = time.time()
    per_ckpt: dict[str, dict[str, Any]] = {}
    final_rev = checkpoints[-1][0]
    order = [checkpoints[-1]] + list(checkpoints[:-1])      # final first: it picks the layers
    display: dict[str, int] = {}
    for rev, tokens in order:
        done_path = raw_dir / f"{rev}.json"
        entry: dict[str, Any] | None = None
        if done_path.exists():
            entry = json.loads(done_path.read_text(encoding="utf-8"))
        # states are needed for the frames (and metrics for new manifolds); recapture whatever is
        # missing, reuse the rest
        need_states = [n for n in names if not (raw_dir / f"{rev}__{n}.npy").exists()]
        need_metrics = [n for n in names if entry is None or n not in entry["manifolds"]]
        if entry is not None and not need_states and not need_metrics:
            per_ckpt[rev] = entry
            if rev == final_rev:
                for name in names:
                    fl = entry["manifolds"][name]["layers"]
                    display[name] = int(max(fl, key=lambda k: fl[k]["shape_metric"]))
            continue
        t1 = time.time()
        fm = load_checkpoint(rev, device=device, batch_size=batch_size, model_id=model_id)
        if entry is None:
            entry = {"revision": rev, "tokens_B": tokens, "n_layers": fm.n_layers, "manifolds": {}}
        for name in names:
            s = sets[name]
            if name in need_states:
                states = fm.capture_all_layers(s["prompts"], progress=False, spans=s["spans"])
                np.save(raw_dir / f"{rev}__{name}.npy", states)
            if name not in need_metrics:
                continue
            states = np.load(raw_dir / f"{rev}__{name}.npy")
            task = s["task"]
            cand = first_token_ids(fm.tokenizer, task.candidates) if task and task.candidates else None
            cap = fm.capture_layer_and_logits(s["prompts"], 0, spans=s["spans"], cand_ids=cand,
                                              progress=False)
            tk = score_task(task, cap["top1"], cap["cand"], fm.tokenizer) if task else None
            layers: dict[str, Any] = {}
            nuisance = ROLES[name].get("nuisance_group")
            n_pos = states.shape[1]
            # the shape checks are CPU-heavy (Fourier probes): every position for the final
            # checkpoint, every `layer_stride`-th position plus the display layer otherwise
            if rev == final_rev:
                todo = list(range(n_pos))
            else:
                todo = sorted(set(range(0, n_pos, layer_stride)) | {n_pos - 1} | {display.get(name, 0)})
            for li in todo:
                x = states[:, li].astype(np.float32)
                if nuisance:
                    x = center_by_group(x, s["labels"], nuisance)
                p = fit_pca(x, 3)
                rep = run_shape_check(name, p.coords, s["labels"], acts=x).to_dict()
                crit = critical_metrics(rep)
                layers[str(li)] = {"status": rep["status"], "metrics": crit,
                                   "shape_metric": summary_metric(name, crit),
                                   "evr": [float(v) for v in p.explained_variance_ratio]}
            entry["manifolds"][name] = {"layers": layers,
                                        "task_accuracy": tk["accuracy"] if tk else None,
                                        "task": task.name if task else None}
            best = max(layers, key=lambda k: layers[k]["shape_metric"])
            if rev == final_rev:
                display[name] = int(best)
            print(f"  {rev:34s} {name:20s} best layer {best} metric "
                  f"{layers[best]['shape_metric']:.3f} task={entry['manifolds'][name]['task_accuracy']}")
        entry["elapsed_s"] = round(time.time() - t1, 1)
        if rev == final_rev:
            for name in names:
                fl = entry["manifolds"][name]["layers"]
                display[name] = int(max(fl, key=lambda k: fl[k]["shape_metric"]))
        done_path.write_text(json.dumps(entry), encoding="utf-8")
        per_ckpt[rev] = entry
        del fm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if free_cache and rev != "main":
            _free_revision(model_id, rev)
    elapsed = time.time() - t0

    # ---- assemble per manifold: display layer, aligned frames, curves ------------------------
    reports: dict[str, Any] = {}
    for name in names:
        s = sets[name]
        roles = ROLES[name]
        n_layers = per_ckpt[final_rev]["n_layers"]
        # display layer: the best (checkpoint, layer) cell among layers checked at every checkpoint
        # (the final model can lose structure that earlier checkpoints have, e.g. after stage 2)
        common = set.intersection(*[set(per_ckpt[r]["manifolds"][name]["layers"])
                                    for r, _ in checkpoints])
        best_cell = {li: max(per_ckpt[r]["manifolds"][name]["layers"][li]["shape_metric"]
                             for r, _ in checkpoints) for li in common}
        layer = int(max(best_cell, key=lambda k: best_cell[k]))
        tl = roles["tuning_label"]
        label = np.asarray([float(lab[tl]) for lab in s["labels"]])
        ref_states = np.load(raw_dir / f"{final_rev}__{name}.npy")[:, layer].astype(np.float32)
        nuisance = roles.get("nuisance_group")
        if nuisance:
            ref_states = center_by_group(ref_states, s["labels"], nuisance)
        ref = fit_pca(ref_states, 3).coords
        sub = subsample(len(ref), max_points, seed=cfg.seed)
        steps: list[tuple[str, np.ndarray]] = []
        curve: list[dict[str, Any]] = []
        matrix: list[list[float | None]] = []
        is_probe = roles["kind"] == "fourier_probe"
        value = (np.asarray([lab[roles["value"]] for lab in s["labels"]], float)
                 if is_probe else None)
        captions: dict[str, str] = {}
        for rev, tokens in checkpoints:
            x = np.load(raw_dir / f"{rev}__{name}.npy")[:, layer].astype(np.float32)
            if nuisance:
                x = center_by_group(x, s["labels"], nuisance)
            if is_probe:   # probe axes are label-defined, so frames are comparable without alignment
                c = cv_probe_coords(x[sub], cast(np.ndarray, value)[sub], seed=cfg.seed)
                c_full = np.zeros((len(x), 3), dtype=np.float32)
                c_full[sub] = c
                c = c_full
            else:
                c = procrustes_align(fit_pca(x, 3).coords, ref)
            lab = f"{tokens:g}B" if rev != "main" else "final"
            steps.append((lab, c[sub]))
            m = per_ckpt[rev]["manifolds"][name]
            row = m["layers"][str(layer)]
            captions[lab] = (f"{'after ' + format(tokens, 'g') + 'B tokens' if rev != 'main' else 'final model (after the stage-2 anneal)'}"
                             f" · shape metric {row['shape_metric']:.2f} ({row['status']})"
                             + (f" · {m['task']} {m['task_accuracy']:.0%}" if m.get("task_accuracy") is not None else ""))
            best_layer = max(m["layers"], key=lambda k: m["layers"][k]["shape_metric"])
            computed = sorted(int(k) for k in m["layers"])
            curve.append({"revision": rev, "tokens_B": tokens, "label": lab,
                          "status": row["status"], "shape_metric": row["shape_metric"],
                          "metrics": row["metrics"], "evr": row["evr"],
                          "best_layer": int(best_layer),
                          "best_layer_metric": m["layers"][best_layer]["shape_metric"],
                          "task_accuracy": m["task_accuracy"], **_bin_centroids(c, label)})
            matrix.append([m["layers"][str(li)]["shape_metric"] if str(li) in m["layers"] else None
                           for li in range(n_layers + 1)])
            curve[-1]["layers_checked"] = computed
        rep = {"name": name, "model_id": model_id, "display_layer": layer, "n_layers": n_layers,
               "display_layer_rule": "best shape metric over all checkpoints among positions "
                                     "checked at every checkpoint",
               "layer_convention": "0 = embeddings, i = output of block i-1",
               "n_prompts": int(len(s["prompts"])), "n_plotted": int(len(sub)),
               "task": s["task"].name if s["task"] else None,
               "checkpoints": curve, "metric_by_checkpoint_and_layer": matrix,
               "view": ("Fourier probe, out-of-fold predictions, fitted per checkpoint"
                        if is_probe else "PCA per checkpoint, Procrustes-aligned to the final one"),
               "note": "no norm-outlier filter (fixed point set across checkpoints); shape checks at "
                       "every position for the final checkpoint, at every 4th position plus the "
                       "display layer for the others"}
        reports[name] = rep
        d = results_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "checkpoints.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        color_key = roles["color_by"]
        cv = [s["labels"][i][color_key] for i in sub]
        if roles["color_mode"] == "categorical":
            cv = [str(v) for v in cv]
        hover = [(s["prompts"][i] if len(s["prompts"][i]) <= 60 else s["prompts"][i][:57] + "…")
                 + f"<br>{tl}: {s['labels'][i][tl]}" for i in sub]
        fig = frames_figure(steps, cv, color_mode=roles["color_mode"], hover_text=hover,
                            active=len(steps) - 1, prefix="checkpoint: ", captions=captions)
        (d / "checkpoints_figure.json").write_text(fig.to_json(), encoding="utf-8")
        (d / "checkpoints_curve.json").write_text(curve_figure(rep).to_json(), encoding="utf-8")
        (d / "checkpoints_heatmap.json").write_text(layer_heatmap(rep).to_json(), encoding="utf-8")
    summary = {"model_id": model_id, "checkpoints": checkpoints, "names": names,
               "elapsed_s": round(elapsed, 1),
               "display_layers": {n: reports[n]["display_layer"] for n in names}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if torch.cuda.is_available() and elapsed > 1:
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round(elapsed / 3600, 4),
                                         f"OLMo-2 1B checkpoint sweep ({len(names)} manifolds)"))
    return summary


def curve_figure(rep: dict[str, Any]) -> go.Figure:
    ck = rep["checkpoints"]
    x = [max(c["tokens_B"], 0.3) for c in ck]      # log axis; step 0 drawn at 0.3B
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[c["shape_metric"] for c in ck], mode="lines+markers",
                             name=f"shape metric (layer {rep['display_layer']})",
                             line={"color": CATEGORICAL[0], "width": 2}))
    fig.add_trace(go.Scatter(x=x, y=[c["best_layer_metric"] for c in ck], mode="lines+markers",
                             name="shape metric (best layer)", line={"color": CATEGORICAL[0], "dash": "dot"},
                             text=[f"layer {c['best_layer']}" for c in ck], hoverinfo="text+y"))
    if rep.get("task"):
        fig.add_trace(go.Scatter(x=x, y=[c["task_accuracy"] for c in ck], mode="lines+markers",
                                 name=f"{rep['task']} accuracy", line={"color": CATEGORICAL[3], "width": 2}))
    fig.add_trace(go.Scatter(x=x, y=[c["centroid_participation_ratio"] / 3 for c in ck],
                             mode="lines+markers", name="centroid dimensionality / 3 (axis ≈ 0.33, plane ≈ 0.67)",
                             line={"color": CATEGORICAL[2], "dash": "dash"}))
    lay = line_layout("pre-training tokens (billions, log)", "value", y_range=(-0.02, 1.05),
                      legend_y=-0.4)
    lay["xaxis"]["type"] = "log"
    lay["xaxis"]["tickvals"] = [0.3, 1, 10, 100, 1000, 4000]
    lay["xaxis"]["ticktext"] = ["0", "1B", "10B", "100B", "1T", "4T"]
    fig.update_layout(**lay)
    return fig


def layer_heatmap(rep: dict[str, Any]) -> go.Figure:
    z = np.array([[np.nan if v is None else v for v in row]
                  for row in rep["metric_by_checkpoint_and_layer"]], dtype=float)
    y = [c["label"] for c in rep["checkpoints"]]
    fig = go.Figure(go.Heatmap(z=z.tolist(), x=list(range(z.shape[1])), y=y, colorscale="Viridis",
                               zmin=0, zmax=1, colorbar={"title": {"text": "shape metric"}, "thickness": 8},
                               hovertemplate="%{y}, layer %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font={"family": FONT, "size": 11},
                      margin={"l": 60, "r": 10, "t": 10, "b": 40},
                      xaxis={"title": {"text": "residual position (0 = embeddings)"}},
                      yaxis={"title": {"text": "checkpoint"}})
    return fig


__all__ = ["run_checkpoints", "CHECKPOINTS", "MODEL_ID", "load_checkpoint", "curve_figure",
           "layer_heatmap"]
