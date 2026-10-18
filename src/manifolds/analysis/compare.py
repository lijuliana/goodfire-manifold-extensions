"""Comparing a perturbed copy of a manifold with the clean one.

Three views, used by the ablation, neuron and checkpoint sweeps:
  * fixed basis: project the perturbed activations onto the *clean* PCA basis so successive
    frames of a slider are comparable (points move, axes do not);
  * refit: run the shape check on a fresh PCA of the perturbed activations (is it still a circle,
    on its own terms?);
  * neighbourhoods: fraction of each point's k nearest neighbours (full activation space) that
    are still among its k nearest after the perturbation, and how far points moved relative to
    the manifold's own spread.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from manifolds.analysis.pca import PCAResult, fit_pca
from manifolds.analysis.shapes import ROLES, center_by_group, run_shape_check


def prepare(acts: np.ndarray, name: str, labels: list[dict[str, Any]]) -> np.ndarray:
    """The preprocessing the main analysis applies before PCA (nuisance-group centering)."""
    nuisance = ROLES[name].get("nuisance_group")
    x = acts.astype(np.float32)
    return center_by_group(x, labels, nuisance) if nuisance else x


def project(acts: np.ndarray, basis: PCAResult) -> np.ndarray:
    return (acts.astype(np.float64) - basis.mean) @ basis.components.T


def knn_overlap(clean: np.ndarray, other: np.ndarray, k: int = 10) -> float:
    """Mean over points of |kNN_clean ∩ kNN_other| / k (Euclidean, excluding the point itself)."""
    from sklearn.neighbors import NearestNeighbors

    k = int(min(k, len(clean) - 1))
    if k < 1:
        return float("nan")
    a = NearestNeighbors(n_neighbors=k + 1).fit(clean).kneighbors(clean, return_distance=False)
    b = NearestNeighbors(n_neighbors=k + 1).fit(other).kneighbors(other, return_distance=False)
    ov = [len(set(a[i, 1:]) & set(b[i, 1:])) / k for i in range(len(clean))]
    return float(np.mean(ov))


def displacement(clean: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """Mean distance moved, relative to the clean manifold's RMS spread about its mean."""
    c = clean.astype(np.float64)
    spread = float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))
    d = np.linalg.norm(other.astype(np.float64) - c, axis=1)
    return {"mean_move": float(d.mean()), "spread": spread,
            "relative_move": float(d.mean() / (spread + 1e-9))}


def critical_metrics(shape: dict[str, Any]) -> dict[str, float]:
    return {c["name"]: float(c["value"]) for c in shape["checks"] if c["critical"]}


def compare(name: str, labels: list[dict[str, Any]], clean: np.ndarray, other: np.ndarray,
            basis: PCAResult, k: int = 10) -> dict[str, Any]:
    """All three views for one perturbed copy. `clean` and `other` are prepared activations."""
    fixed = project(other, basis)
    refit = fit_pca(other, 3)
    shape_refit = run_shape_check(name, refit.coords, labels, acts=other).to_dict()
    shape_fixed = run_shape_check(name, fixed, labels, acts=other).to_dict()
    return {
        "status_refit": shape_refit["status"],
        "metrics_refit": critical_metrics(shape_refit),
        "status_fixed_basis": shape_fixed["status"],
        "metrics_fixed_basis": critical_metrics(shape_fixed),
        "evr_refit": [float(v) for v in refit.explained_variance_ratio],
        "knn_overlap": knn_overlap(clean, other, k),
        "knn_overlap_pca3": knn_overlap(project(clean, basis), fixed, k),
        **displacement(clean, other),
        "coords_fixed_basis": fixed,
    }


def procrustes_align(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate/reflect `src` (N x 3, centred) onto `ref` (same points) so slider frames of
    independently fitted PCAs do not flip between steps. Orthogonal Procrustes, no scaling."""
    a = src - src.mean(0)
    b = ref - ref.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    return a @ r


def summary_metric(name: str, metrics: dict[str, float]) -> float:
    """One number per manifold for the curves: the mean of the critical shape metrics."""
    vals = [v for v in metrics.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


__all__ = ["prepare", "project", "knn_overlap", "displacement", "compare", "procrustes_align",
           "summary_metric", "critical_metrics"]
