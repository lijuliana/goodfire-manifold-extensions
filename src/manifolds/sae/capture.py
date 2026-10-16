"""Subspace-capture curves and feature tuning curves (Bhalla et al. 2026 Sec. 5, Figs 5-7).

Ports of goodfire-ai/sae-manifold `subspace_capture.py` (MIT) to torch so the 32k-atom greedy
searches run on the GPU. All curves report variance explained of the *centered* manifold
activations as a function of the number of SAE features used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _orth_basis(d_sel: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis (rows) for the row span of d_sel [k, d]."""
    q, r = torch.linalg.qr(d_sel.T)                      # d x k
    keep = r.diagonal().abs() > 1e-6
    return q[:, keep].T


@torch.no_grad()
def greedy_geometric(x: torch.Tensor, decoder: torch.Tensor, max_k: int = 64
                     ) -> tuple[list[int], list[float]]:
    """Greedy subspace pursuit over decoder directions (sae-manifold `find_support_greedy`).

    At each step add the atom whose direction captures the most remaining variance; the explained
    variance is measured by projecting onto the span of all selected atoms.
    """
    xc = x - x.mean(0)
    total = float((xc**2).sum())
    d_norm_sq = (decoder**2).sum(1).clamp_min(1e-10)
    residual = xc.clone()
    selected: list[int] = []
    curve: list[float] = []
    for _ in range(max_k):
        scores = ((residual @ decoder.T) ** 2).sum(0) / d_norm_sq
        if selected:
            scores[torch.tensor(selected, device=scores.device)] = -float("inf")
        best = int(torch.argmax(scores))
        if float(scores[best]) <= 0:
            break
        selected.append(best)
        basis = _orth_basis(decoder[selected])
        residual = xc - (xc @ basis.T) @ basis
        curve.append(1.0 - float((residual**2).sum()) / total)
    return selected, curve


@torch.no_grad()
def greedy_statistical(x: torch.Tensor, codes: torch.Tensor, decoder: torch.Tensor,
                       max_k: int = 64) -> tuple[list[int], list[float]]:
    """Greedy selection by variance explained using the SAE's actual codes
    (sae-manifold `find_support_greedy_codes`): feature i contributes (z_i - <z_i>) d_i.
    """
    xc = x - x.mean(0)
    total = float((xc**2).sum())
    cand = torch.nonzero((codes > 0).any(0)).flatten()
    if cand.numel() == 0:
        return [], []
    zc = codes[:, cand] - codes[:, cand].mean(0)
    d_cand = decoder[cand]
    contrib_ss = (zc**2).sum(0) * (d_cand**2).sum(1)
    alive = torch.ones(cand.numel(), dtype=torch.bool, device=x.device)
    residual = xc.clone()
    sel_local: list[int] = []
    curve: list[float] = []
    for _ in range(max_k):
        cross = (residual @ d_cand.T) * zc
        scores = 2 * cross.sum(0) - contrib_ss
        scores[~alive] = -float("inf")
        best = int(torch.argmax(scores))
        if float(scores[best]) <= 0:
            break
        sel_local.append(best)
        alive[best] = False
        idx = torch.tensor(sel_local, device=x.device)
        residual = xc - zc[:, idx] @ d_cand[idx]
        curve.append(1.0 - float((residual**2).sum()) / total)
    return [int(cand[i]) for i in sel_local], curve


@torch.no_grad()
def pca_curve(x: torch.Tensor, max_k: int = 64) -> list[float]:
    xc = x - x.mean(0)
    s = torch.linalg.svdvals(xc)
    ev = s**2 / float((s**2).sum())
    return torch.cumsum(ev[:max_k], 0).tolist()


@torch.no_grad()
def random_atoms_curve(x: torch.Tensor, decoder: torch.Tensor, max_k: int = 64, seed: int = 0,
                       n_rep: int = 3) -> list[float]:
    """Variance explained by the span of k random decoder atoms (averaged over repeats)."""
    xc = x - x.mean(0)
    total = float((xc**2).sum())
    g = torch.Generator(device="cpu").manual_seed(seed)
    curves = []
    for _ in range(n_rep):
        perm = torch.randperm(decoder.shape[0], generator=g)[:max_k].to(x.device)
        c = []
        for k in range(1, max_k + 1):
            basis = _orth_basis(decoder[perm[:k]])
            resid = xc - (xc @ basis.T) @ basis
            c.append(1.0 - float((resid**2).sum()) / total)
        curves.append(c)
    return np.mean(np.array(curves), axis=0).tolist()


def k_for_threshold(curve: list[float], thr: float) -> int | None:
    for i, v in enumerate(curve):
        if v >= thr:
            return i + 1
    return None


# ------------------------------------------------------------------ tuning curves ---------------

@dataclass
class TuningCurve:
    feature: int
    x: list[float]
    y: list[float]           # smoothed, max-normalised activation per unique label value
    raw_max: float


def tuning_curves(codes: torch.Tensor, features: list[int], label: np.ndarray, sigma: float = 3.0
                  ) -> list[TuningCurve]:
    """Mean activation per unique label value for each feature, Gaussian-smoothed over label bins
    (sae-manifold `plot_tuning_curves`, sigma in bins) and max-normalised."""
    from scipy.ndimage import gaussian_filter1d

    lab = np.asarray(label, dtype=float)
    uniq = np.unique(lab)
    z = codes.detach().float().cpu().numpy()
    out = []
    for f in features:
        y = np.array([z[lab == u, f].mean() for u in uniq])
        ys = gaussian_filter1d(y, sigma) if sigma > 0 and len(uniq) > 3 else y
        m = float(ys.max()) if ys.max() > 0 else 1.0
        out.append(TuningCurve(int(f), uniq.tolist(), (ys / m).tolist(), float(y.max())))
    return out


# ------------------------------------------------------------------ partial reconstructions ----

@dataclass
class PartialRecon:
    ks: list[int]
    coords: dict[int, np.ndarray] = field(default_factory=dict)   # k -> [N, 3] in the GT PCA basis


@torch.no_grad()
def partial_reconstructions(x: torch.Tensor, codes: torch.Tensor, decoder: torch.Tensor,
                            order: list[int], pca_components: np.ndarray, pca_mean: np.ndarray,
                            ks: list[int]) -> PartialRecon:
    """Reconstruct from the first k selected features (centered contributions, as in Fig. 5) and
    project into the ground-truth 3-D PCA basis of the raw activations."""
    comps = torch.as_tensor(pca_components, device=x.device, dtype=x.dtype)   # [3, d]
    xmean = x.mean(0)
    out = PartialRecon(ks=[k for k in ks if k <= len(order)])
    for k in out.ks:
        idx = torch.tensor(order[:k], device=x.device)
        zc = codes[:, idx] - codes[:, idx].mean(0)
        recon = zc @ decoder[idx] + xmean              # centered contributions + manifold mean
        out.coords[k] = ((recon - torch.as_tensor(pca_mean, device=x.device, dtype=x.dtype))
                         @ comps.T).float().cpu().numpy()
    return out


def summarize(curves: dict[str, list[float]], ambient_dim: int) -> dict[str, Any]:
    """Headline numbers for the report (all variance-explained values are of the *centered*
    manifold activations, i.e. within-manifold variance)."""
    stat = curves.get("statistical", [])
    geo = curves.get("geometric", [])
    pca = curves.get("pca", [])
    rnd = curves.get("random_atoms", [])

    def at(c: list[float], k: int) -> float | None:
        return c[k - 1] if len(c) >= k else None

    pca_amb = at(pca, ambient_dim)
    k_match = None
    if pca_amb is not None:
        k_match = k_for_threshold(geo, pca_amb)
    geo_amb = at(geo, ambient_dim)
    ratio = (geo_amb / pca_amb) if (pca_amb and geo_amb is not None) else None
    g64, r64 = at(geo, 64), at(rnd, 64)
    return {
        "ambient_dim": ambient_dim,
        "ve_at": {k: {"pca": at(pca, k), "geometric": at(geo, k), "statistical": at(stat, k),
                      "random_atoms": at(rnd, k)} for k in (1, ambient_dim, 8, 16, 32, 64)},
        "pca_ve_at_ambient_dim": pca_amb,
        "k_geometric_to_match_pca_at_ambient_dim": k_match,
        "geometric_over_pca_at_ambient_dim": ratio,
        "geometric_over_random_at_64": (g64 / r64) if (g64 is not None and r64) else None,
        "k90_statistical": k_for_threshold(stat, 0.90),
        "k90_geometric": k_for_threshold(geo, 0.90),
        "k90_pca": k_for_threshold(pca, 0.90),
    }
