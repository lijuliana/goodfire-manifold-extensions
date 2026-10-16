"""Outlier filtering and deterministic 3-component PCA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


def filter_norm_outliers(acts: np.ndarray, n_std: float = 3.0) -> np.ndarray:
    """Boolean keep-mask: drop rows whose L2 norm exceeds mean + n_std * std.

    Port of sae-manifold data.py `filter_norm_outliers` (attention-sink rows).
    """
    norms = np.linalg.norm(acts.astype(np.float64), axis=1)
    cutoff = norms.mean() + n_std * norms.std()
    return norms <= cutoff


@dataclass(frozen=True)
class PCAResult:
    coords: np.ndarray                 # [N, k]
    explained_variance_ratio: np.ndarray  # [k]
    mean: np.ndarray                   # [d]
    components: np.ndarray             # [k, d]

    def to_dict(self) -> dict:
        return {
            "explained_variance_ratio": [float(v) for v in self.explained_variance_ratio],
            "n_points": int(self.coords.shape[0]),
            "n_components": int(self.coords.shape[1]),
        }


def fit_pca(acts: np.ndarray, n_components: int = 3) -> PCAResult:
    """Centered PCA with the deterministic full SVD solver."""
    x = acts.astype(np.float64)
    k = min(n_components, x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=k, svd_solver="full")
    coords = pca.fit_transform(x)
    return PCAResult(
        coords=coords,
        explained_variance_ratio=pca.explained_variance_ratio_,
        mean=pca.mean_,
        components=pca.components_,
    )
