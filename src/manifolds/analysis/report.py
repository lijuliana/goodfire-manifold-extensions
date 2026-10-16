"""Assemble a per-manifold report: outlier filter -> PCA -> shape check -> figure -> report.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from manifolds import RESULTS_DIR
from manifolds.analysis.pca import filter_norm_outliers, fit_pca
from manifolds.analysis.shapes import ROLES, center_by_group, run_shape_check
from manifolds.config import RunConfig
from manifolds.manifest import ManifoldSpec
from manifolds.prompts.types import PromptSet
from manifolds.viz.plot3d import build_figure, figure_to_json, subsample


def _hover(prompt: str, label: dict[str, Any]) -> str:
    parts = [prompt]
    for k, v in label.items():
        if isinstance(v, float):
            parts.append(f"{k}: {v:.3g}")
        else:
            parts.append(f"{k}: {v}")
    return "<br>".join(parts)


def _color_config(name: str, labels: list[dict[str, Any]]) -> dict[str, Any]:
    r = ROLES[name]
    key = r["color_by"]
    cfg: dict[str, Any] = {"color_by": key, "color_mode": r["color_mode"]}
    if r["color_mode"] == "categorical":
        if name == "days":
            # keep weekday order
            cfg["category_order"] = list(dict.fromkeys(lab["day"] for lab in labels))
        else:
            cfg["category_order"] = sorted({str(lab[key]) for lab in labels})
    return cfg


_STATUS_V1 = {  # verdicts of the first core run under rubric v1 (2026-10-15), for transparency
    "colors": "partial", "temperature": "matches", "age": "matches", "geography": "matches",
    "days": "matches", "years": "matches", "formality": "failed", "sent_length": "failed",
    # the raw-dataset political_bias run was "failed" under v1 / "partial" under v2; the current
    # political_bias set (Claude-augmented) never ran under v1
}


def _status_v1(name: str) -> str | None:
    return _STATUS_V1.get(name)


def _probe_view(name: str, acts: np.ndarray, labels: list[dict[str, Any]],
                out_dir: Path | None, value_key: str) -> dict[str, Any]:
    """Cross-validated ridge-probe coordinates: (cos, sin)(2 pi v/10) and the magnitude v."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    v = np.asarray([lab[value_key] for lab in labels], dtype=float)
    ph = 2 * np.pi * (v % 10) / 10
    y_circ = np.c_[np.cos(ph), np.sin(ph)]
    y_lin = (v - v.mean()) / (v.std() + 1e-9)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    def model():
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))

    pc = cross_val_predict(model(), acts, y_circ, cv=cv)
    pl = cross_val_predict(model(), acts, y_lin, cv=cv)
    r2_circle = float(1 - ((y_circ - pc) ** 2).sum() / ((y_circ - y_circ.mean(0)) ** 2).sum())
    r2_lin = float(1 - ((y_lin - pl) ** 2).sum() / ((y_lin - y_lin.mean()) ** 2).sum())
    coords = np.c_[pc[:, 0], pc[:, 1], pl]
    out = {"label": value_key, "r2_circle": r2_circle, "r2_linear": r2_lin,
           "axes": ["pred cos(2pi v/10)", "pred sin(2pi v/10)", "pred magnitude"]}
    if out_dir is not None:
        hover = [f"{lab[value_key]:g}" for lab in labels]
        fig = build_figure(coords, [lab[value_key] % 10 for lab in labels], color_mode="cyclic",
                           hover_text=hover, color_title=f"{value_key} mod 10")
        (out_dir / "probe_figure.json").write_text(figure_to_json(fig), encoding="utf-8")
    return out


def analyze_manifold(
    spec: ManifoldSpec,
    cfg: RunConfig,
    results_dir: Path = RESULTS_DIR,
    *,
    acts: np.ndarray | None = None,
    prompt_set: PromptSet | None = None,
    write: bool = True,
) -> dict[str, Any]:
    out_dir = results_dir / spec.name
    if acts is None:
        acts = np.load(out_dir / "activations.npy")
    if prompt_set is None:
        prompt_set = PromptSet.load(out_dir / "prompts.json")
    meta: dict[str, Any] = {}
    if (out_dir / "meta.json").exists():
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    precision: dict[str, Any] | None = None
    if (out_dir / "precision_check.json").exists():
        precision = json.loads((out_dir / "precision_check.json").read_text(encoding="utf-8"))

    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    labels = [prompt_set.labels[i] for i in idx]
    prompts = [prompt_set.prompts[i] for i in idx]
    x = acts[idx]
    nuisance = ROLES[spec.name].get("nuisance_group")
    if nuisance:
        x = center_by_group(x, labels, nuisance)
    pca = fit_pca(x, cfg.analysis.pca_components)
    shape = run_shape_check(spec.name, pca.coords, labels, acts=x)

    probe_view = None
    if ROLES[spec.name]["kind"] == "fourier_probe":
        probe_view = _probe_view(spec.name, x, labels, out_dir if write else None,
                                 ROLES[spec.name]["value"])

    color = _color_config(spec.name, labels)
    color_values = [lab[color["color_by"]] for lab in labels]
    if color["color_mode"] == "categorical":
        color_values = [str(v) for v in color_values]
    hover = [_hover(p, lab) for p, lab in zip(prompts, labels, strict=True)]
    fig = build_figure(
        pca.coords, color_values, color_mode=color["color_mode"], hover_text=hover,
        color_title=color["color_by"], category_order=color.get("category_order"),
    )
    sub = subsample(len(idx), cfg.site.preview_max_points, seed=cfg.seed)
    fig_preview = build_figure(
        pca.coords[sub], [color_values[i] for i in sub], color_mode=color["color_mode"],
        hover_text=[hover[i] for i in sub], color_title=color["color_by"],
        category_order=color.get("category_order"), point_size=3.5,
    )

    report: dict[str, Any] = {
        "name": spec.name,
        "spec": {
            "tier": spec.tier,
            "model": spec.model,
            "layer": spec.layer,
            "expected_geometry": spec.expected_geometry,
            "replication_mode": spec.replication_mode,
            "sources": spec.sources,
            "prompt_template": spec.prompt_template,
            "dataset": spec.dataset,
            "n_paper": spec.n_paper,
        },
        "preprocessing": (f"activations centred per '{nuisance}' before PCA" if nuisance
                          else "none"),
        "prompt_set": {
            "template": prompt_set.template,
            "source": prompt_set.source,
            "deviations": prompt_set.deviations,
            "n_total": len(prompt_set),
            "n_kept": int(len(idx)),
            "n_outliers_removed": int(len(prompt_set) - len(idx)),
            "examples": prompt_set.prompts[:5],
        },
        "pca": pca.to_dict(),
        "shape": shape.to_dict(),
        "status": shape.status,
        "status_v1": _status_v1(spec.name),
        "probe_view": probe_view,
        "color": color,
        "meta": meta,
        "precision_check": precision,
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        (out_dir / "figure.json").write_text(figure_to_json(fig), encoding="utf-8")
        (out_dir / "figure_preview.json").write_text(figure_to_json(fig_preview), encoding="utf-8")
        np.save(out_dir / "pca_coords.npy", pca.coords.astype(np.float32))
    return report
