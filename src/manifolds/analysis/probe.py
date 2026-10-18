"""Fourier-probe coordinates (Feucht et al. 2026, App. G) as a view of a manifold.

PCA shows the three directions along which the activations vary most; for the arithmetic prompts
those directions are dominated by the size of the numbers and the tokens, not by the last-digit
pattern the paper is about. A probe is a linear readout fitted to predict cos and sin of
2π·value/10 (the ones-digit circle) and the value itself (magnitude); plotting every prompt at
the probe's predictions shows the circle that exists in the 4096-dim activations but is not among
the top-3 variance directions.

Two uses: `cv_probe_coords` (out-of-fold predictions on one set of activations; the honest view
of a clean manifold) and `fit_probe` + `apply_probe` (one probe fitted on the clean activations
and applied to perturbed copies, the probe analogue of projecting onto the clean PCA basis).
"""

from __future__ import annotations

from typing import Any

import numpy as np

PROBE_AXES = ["pred cos(2pi v/10)", "pred sin(2pi v/10)", "pred magnitude"]


def _targets(value: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = np.asarray(value, dtype=float)
    ph = 2 * np.pi * (v % 10) / 10
    mu, sd = float(v.mean()), float(v.std() + 1e-9)
    return np.c_[np.cos(ph), np.sin(ph), (v - mu) / sd], mu, sd


def _model() -> Any:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))


def cv_probe_coords(acts: np.ndarray, value: np.ndarray, seed: int = 0) -> np.ndarray:
    """[N, 3] out-of-fold predictions (5-fold) of (cos, sin, standardised magnitude)."""
    from sklearn.model_selection import KFold, cross_val_predict

    y, _, _ = _targets(value)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    out: np.ndarray = cross_val_predict(_model(), np.asarray(acts, dtype=np.float32), y, cv=cv)
    return out


def fit_probe(acts: np.ndarray, value: np.ndarray) -> Any:
    y, _, _ = _targets(value)
    return _model().fit(np.asarray(acts, dtype=np.float32), y)


def apply_probe(probe: Any, acts: np.ndarray) -> np.ndarray:
    out: np.ndarray = probe.predict(np.asarray(acts, dtype=np.float32))
    return out


def probe_r2(coords: np.ndarray, value: np.ndarray) -> dict[str, float]:
    """R² of the circle (cos, sin jointly) and of the magnitude from predicted coordinates."""
    y, _, _ = _targets(value)
    c = coords[:, :2]
    r2c = 1 - ((y[:, :2] - c) ** 2).sum() / ((y[:, :2] - y[:, :2].mean(0)) ** 2).sum()
    r2m = 1 - ((y[:, 2] - coords[:, 2]) ** 2).sum() / ((y[:, 2] - y[:, 2].mean()) ** 2).sum()
    return {"r2_circle": float(r2c), "r2_linear": float(r2m)}


__all__ = ["cv_probe_coords", "fit_probe", "apply_probe", "probe_r2", "PROBE_AXES"]
