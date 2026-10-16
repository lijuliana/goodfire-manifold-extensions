"""Quantitative shape checks: does the 3-D PCA cloud have the geometry the paper reports?

Each check returns a `ShapeReport` with named sub-checks, thresholds and a status:
  matches : every sub-check passes
  partial : every *critical* sub-check passes but a secondary one fails, or some critical pass
  failed  : no critical sub-check passes
Thresholds are our own choices (the papers report shapes qualitatively); they are recorded in
every report so a reader can judge them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    threshold: float
    op: str            # ">=" or "<="
    critical: bool
    passed: bool
    note: str = ""
    partial_threshold: float | None = None   # weaker bound that still counts as "partial"

    @property
    def partial(self) -> bool:
        if self.passed:
            return True
        if self.partial_threshold is None:
            return False
        return (self.value >= self.partial_threshold if self.op == ">="
                else self.value <= self.partial_threshold)


@dataclass
class ShapeReport:
    geometry: str
    checks: list[Check]
    status: str
    reason: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "status": self.status,
            "reason": self.reason,
            "checks": [asdict(c) for c in self.checks],
            "extras": self.extras,
        }


# ---------------------------------------------------------------- primitives ------------------

RUBRIC_VERSION = 2
"""Rubric history (kept for transparency; v1 verdicts are stored in each report as status_v1).

v1 (first core run, 2026-10-15): "line" required R^2 >= 0.9 of the label from the 3 PCs. That
    conflates "the manifold is present" with "the label accounts for most of the top-3 variance",
    which fails for natural sentences (formality, sentence length) where other factors dominate.
v2: ordering (Spearman along the best direction) and nonlinear readability (cross-validated 10-NN
    R^2) are the critical tests for 1-D manifolds; linear R^2 is informative only. Every critical
    check carries a weaker "partial" bound. Status: matches = all checks pass; partial = every
    critical check clears at least its partial bound; failed = otherwise.
"""


def _mk(name: str, value: float, thr: float, op: str = ">=", critical: bool = True,
        note: str = "", partial: float | None = None) -> Check:
    v = float(value) if np.isfinite(value) else (-np.inf if op == ">=" else np.inf)
    passed = bool(v >= thr) if op == ">=" else bool(v <= thr)
    return Check(name, float(value), thr, op, critical, passed, note, partial)


def _finalize(geometry: str, checks: list[Check], extras: dict[str, Any] | None = None
              ) -> ShapeReport:
    crit = [c for c in checks if c.critical]
    if all(c.passed for c in checks):
        status = "matches"
    elif all(c.partial for c in crit):
        status = "partial"
    else:
        status = "failed"
    failed = [f"{c.name}={c.value:.3f} ({c.op}{c.threshold})" for c in checks if not c.passed]
    reason = "all checks passed" if not failed else "failed: " + "; ".join(failed)
    ex = dict(extras or {})
    ex["rubric_version"] = RUBRIC_VERSION
    return ShapeReport(geometry, checks, status, reason, ex)


def knn_r2(x: np.ndarray, y: np.ndarray, k: int = 10, seed: int = 0) -> float:
    """5-fold cross-validated R^2 of a k-NN regression of y from x (nonlinear readability)."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    k = max(1, min(k, len(y) // 5))
    return float(cross_val_score(KNeighborsRegressor(k), x, y, cv=cv, scoring="r2").mean())


def best_direction_spearman(coords: np.ndarray, label: np.ndarray) -> float:
    reg = LinearRegression().fit(coords, label)
    return spearman_abs(coords @ reg.coef_, label)


def ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    reg = LinearRegression().fit(x, y)
    return float(reg.score(x, y))


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    r = stats.spearmanr(a, b).statistic
    return float(abs(r)) if np.isfinite(r) else 0.0


def circ_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def circular_corr(a: np.ndarray, b: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Fisher-Lee (1983) circular correlation in [-1, 1].

    rho_T = sum_{i<j} sin(a_i-a_j) sin(b_i-b_j) / sqrt(sum sin^2(a_i-a_j) * sum sin^2(b_i-b_j)).
    Rotation invariant, so it is well defined for angles spread uniformly around the circle (the
    Jammalamadaka-Sarma form is not). O(n^2): subsampled to `max_n` points deterministically.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) > max_n:
        idx = np.random.default_rng(seed).choice(len(a), size=max_n, replace=False)
        a, b = a[idx], b[idx]
    sa = np.sin(a[:, None] - a[None, :])
    sb = np.sin(b[:, None] - b[None, :])
    num = float((sa * sb).sum())
    den = float(np.sqrt((sa**2).sum() * (sb**2).sum()))
    return num / den if den > 0 else 0.0


def fit_circle_plane(coords: np.ndarray, phase: np.ndarray) -> dict[str, Any]:
    """Least-squares fit coords ~ mu + A [cos(phase), sin(phase)]: the best plane for a circle.

    Returns the fit R^2, the angle of each point in that plane, radii, roundness and the height
    along the plane normal (the third, non-circular coordinate).
    """
    x = np.c_[np.cos(phase), np.sin(phase)]
    reg = LinearRegression().fit(x, coords)
    r2 = float(reg.score(x, coords))
    a = reg.coef_  # [3, 2]
    q, _ = np.linalg.qr(a)
    centered = coords - reg.intercept_
    proj = centered @ q
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    radii = np.linalg.norm(proj, axis=1)
    normal = np.cross(q[:, 0], q[:, 1]) if coords.shape[1] == 3 else None
    height = centered @ normal if normal is not None else np.zeros(len(coords))
    cc = abs(circular_corr(angles, phase))
    return {
        "r2": r2,
        "circ_corr": cc,
        "angles": angles,
        "radii": radii,
        "roundness_cv": float(radii.std() / (radii.mean() + 1e-12)),
        "height": height,
        "basis": q,
    }


def cyclic_order_ok(angles: np.ndarray, classes: np.ndarray, order: Sequence[Any]) -> bool:
    """True if class centroids, sorted by angle, follow `order` cyclically (either direction)."""
    order = list(order)
    cents = {c: circ_mean(angles[classes == c]) for c in order}
    seq = sorted(order, key=lambda c: cents[c])
    k = len(order)
    rev = order[::-1]
    valid = [order[i:] + order[:i] for i in range(k)] + [rev[i:] + rev[:i] for i in range(k)]
    return seq in valid


# ---------------------------------------------------------------- checks ----------------------

def check_line(coords: np.ndarray, label: np.ndarray) -> ShapeReport:
    label = np.asarray(label, dtype=np.float64)
    checks = [
        _mk("spearman_best_direction", best_direction_spearman(coords, label), 0.80,
            critical=True, partial=0.60,
            note="label monotone along the best linear direction in PCA-3"),
        _mk("knn_r2", knn_r2(coords, label), 0.60, critical=True, partial=0.40,
            note="cross-validated 10-NN R^2 of the label from PCA-3 (nonlinear readability)"),
        _mk("linear_r2_from_pcs", ols_r2(coords, label), 0.50, critical=False,
            note="R^2 of label regressed on the 3 PCs (share of top-3 variance)"),
    ]
    return _finalize("line", checks)


def check_ordinal(coords: np.ndarray, label: np.ndarray) -> ShapeReport:
    """1-D continuous / ordinal variable (e.g. political bias 0..4)."""
    label = np.asarray(label, dtype=np.float64)
    classes = np.unique(label)
    cents = np.array([coords[label == c].mean(0) for c in classes])
    # path check: each class centroid's nearest other centroid is an adjacent class
    dist = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.argmin(1)
    path_ok = float(np.mean([abs(int(nearest[i]) - i) == 1 for i in range(len(classes))]))
    checks = [
        _mk("spearman_best_direction", best_direction_spearman(coords, label), 0.60,
            critical=True, partial=0.40,
            note="rank correlation along the best linear direction in PCA-3"),
        _mk("knn_r2", knn_r2(coords, label), 0.50, critical=True, partial=0.30,
            note="cross-validated 10-NN R^2 of the label from PCA-3"),
        _mk("centroid_path_adjacency", path_ok, 1.0, critical=False,
            note="fraction of class centroids whose nearest centroid is an adjacent class"),
    ]
    return _finalize("ordinal", checks, {"class_centroid_distances": np.round(dist, 3).tolist()})


def check_circle(coords: np.ndarray, phase: np.ndarray, classes: np.ndarray | None = None,
                 order: Sequence[Any] | None = None) -> ShapeReport:
    phase = np.asarray(phase, dtype=np.float64)
    f = fit_circle_plane(coords, phase)
    checks = [
        _mk("circular_corr", f["circ_corr"], 0.80, critical=True, partial=0.60,
            note="angle in best-fit plane vs ground-truth phase"),
        _mk("circle_fit_r2", f["r2"], 0.50, critical=False,
            note="variance of PCA-3 explained by cos/sin(phase)"),
        _mk("roundness_cv", f["roundness_cv"], 0.35, op="<=", critical=False,
            note="std/mean of radii in the fitted plane"),
    ]
    extras: dict[str, Any] = {}
    if classes is not None and order is not None:
        ok = cyclic_order_ok(f["angles"], np.asarray(classes), order)
        checks.insert(0, _mk("cyclic_order", float(ok), 1.0, critical=True,
                             note="class centroids appear in cyclic order"))
        extras["centroid_angles"] = {
            str(c): circ_mean(f["angles"][np.asarray(classes) == c]) for c in order
        }
    return _finalize("circle", checks, extras)


def check_helix(coords: np.ndarray, linear_label: np.ndarray, periodic_label: np.ndarray,
                period: float) -> ShapeReport:
    lin = np.asarray(linear_label, dtype=np.float64)
    per = np.asarray(periodic_label, dtype=np.float64)
    r2_lin = ols_r2(coords, lin)
    trend = LinearRegression().fit(lin[:, None], coords)
    resid = coords - trend.predict(lin[:, None])
    phase = 2 * np.pi * (per % period) / period
    f = fit_circle_plane(resid, phase)
    digits = np.round(per % period).astype(int)
    order = list(range(int(period)))
    cyc = float(cyclic_order_ok(f["angles"], digits, order))
    checks = [
        _mk("linear_r2_from_pcs", r2_lin, 0.90, critical=True, partial=0.70,
            note="linear axis (e.g. year)"),
        _mk("periodic_circular_corr", f["circ_corr"], 0.60, critical=True, partial=0.40,
            note="after removing the linear trend, angle vs periodic phase"),
        _mk("periodic_fit_r2", f["r2"], 0.30, critical=False,
            note="residual variance explained by cos/sin(phase)"),
        _mk("periodic_cyclic_order", cyc, 1.0, critical=False),
    ]
    return _finalize("helix", checks)


def check_paraboloid(coords: np.ndarray, hue: np.ndarray, lightness: np.ndarray,
                     saturation: np.ndarray) -> ShapeReport:
    hue = np.asarray(hue, dtype=np.float64)
    phase = 2 * np.pi * hue
    f = fit_circle_plane(coords, phase)
    bins = np.floor(hue * 12).astype(int) % 12
    cyc = float(cyclic_order_ok(f["angles"], bins, list(range(12))))
    h = f["height"]
    sp_l = spearman_abs(h, lightness)
    sp_s = spearman_abs(h, saturation)
    which = "lightness" if sp_l >= sp_s else "saturation"
    r2_quad = ols_r2(f["radii"] ** 2, h)
    checks = [
        _mk("hue_circular_corr", f["circ_corr"], 0.70, critical=True, partial=0.50),
        _mk("hue_cyclic_order", cyc, 1.0, critical=True, note="12 hue bins in cyclic order"),
        _mk("height_vs_lightness_or_saturation", max(sp_l, sp_s), 0.40, critical=True,
            partial=0.25,
            note=f"best = {which}; |spearman| lightness={sp_l:.2f} saturation={sp_s:.2f}"),
        _mk("height_vs_radius2_r2", r2_quad, 0.20, critical=False,
            note="paraboloid signature: height ~ radius^2"),
    ]
    return _finalize("paraboloid", checks, {"height_axis_variable": which})


def within_group_r2(coords: np.ndarray, y: np.ndarray, group: np.ndarray,
                    min_size: int = 20) -> float:
    """Size-weighted mean of per-group OLS R^2 (local linear readout inside each cluster)."""
    num = 0.0
    den = 0
    for g in np.unique(group):
        m = group == g
        if m.sum() < min_size:
            continue
        num += ols_r2(coords[m], y[m]) * int(m.sum())
        den += int(m.sum())
    return num / den if den else 0.0


def check_tree(coords: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               group: np.ndarray) -> ShapeReport:
    """Hierarchical tree: group (continent) clusters, with lat/lon gradients inside each."""
    group = np.asarray(group)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    _, counts = np.unique(group, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    min_count = int(counts.min())
    n_splits = min(5, min_count) if min_count >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    acc = float(cross_val_score(KNeighborsClassifier(5), coords, group, cv=cv).mean())
    sil = float(silhouette_score(coords, group)) if len(counts) > 1 else 0.0
    checks = [
        _mk("knn_group_acc_minus_baseline", acc - baseline, 0.15, critical=True, partial=0.08,
            note=f"5-NN continent accuracy {acc:.2f} vs majority baseline {baseline:.2f}"),
        _mk("silhouette_by_group", sil, 0.05, critical=False),
        _mk("within_group_lat_r2", within_group_r2(coords, lat, group), 0.30, critical=False,
            note="latitude linearly readable inside each continent cluster"),
        _mk("within_group_lon_r2", within_group_r2(coords, lon, group), 0.20, critical=False,
            note="longitude linearly readable inside each continent cluster"),
    ]
    return _finalize("tree", checks, {"knn_acc": acc, "baseline": baseline})


def check_circumplex(coords: np.ndarray, group: np.ndarray, valence: np.ndarray,
                     arousal: np.ndarray) -> ShapeReport:
    """Emotion clusters arranged like the valence-arousal circumplex (Bigelow et al. 2026 figure):
    clusters separable, and distances between cluster centroids track distances in (valence,
    arousal) space."""
    group = np.asarray(group)
    cats = list(dict.fromkeys(group))
    _, counts = np.unique(group, return_counts=True)
    baseline = float(counts.max() / counts.sum())
    n_splits = min(5, int(counts.min())) if counts.min() >= 2 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    acc = float(cross_val_score(KNeighborsClassifier(5), coords, group, cv=cv).mean())
    sil = float(silhouette_score(coords, group)) if len(cats) > 1 else 0.0
    cents = np.array([coords[group == c].mean(0) for c in cats])
    va = np.array([[float(np.asarray(valence)[group == c][0]),
                    float(np.asarray(arousal)[group == c][0])] for c in cats])
    iu = np.triu_indices(len(cats), 1)
    d_c = np.linalg.norm(cents[:, None] - cents[None], axis=-1)[iu]
    d_va = np.linalg.norm(va[:, None] - va[None], axis=-1)[iu]
    rho = float(stats.spearmanr(d_c, d_va).statistic) if len(d_c) > 2 else 0.0
    checks = [
        _mk("knn_group_acc_minus_baseline", acc - baseline, 0.15, critical=True, partial=0.08,
            note=f"5-NN emotion accuracy {acc:.2f} vs majority baseline {baseline:.2f}"),
        _mk("centroid_distance_vs_valence_arousal", rho, 0.50, critical=False, partial=0.30,
            note="Spearman between centroid distances in PCA-3 and distances in the circumplex"),
        _mk("silhouette_by_group", sil, 0.05, critical=False),
    ]
    return _finalize("circumplex", checks, {"groups": cats, "centroid_distances": d_c.tolist()})


def fourier_probe_r2(acts: np.ndarray, value: np.ndarray, period: float, seed: int = 0) -> float:
    """5-fold cross-validated R^2 of a ridge probe from activations to (cos, sin)(2 pi value/T):
    the Fourier probes of Feucht et al. 2026 App. G (linear, one period at a time)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ph = 2 * np.pi * np.asarray(value, dtype=float) / period
    y = np.c_[np.cos(ph), np.sin(ph)]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 11)))
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, te in cv.split(acts):
        model.fit(acts[tr], y[tr])
        pred = model.predict(acts[te])
        ss_res = ((y[te] - pred) ** 2).sum(0)
        ss_tot = ((y[te] - y[te].mean(0)) ** 2).sum(0)
        scores.append(float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12))))
    return float(np.mean(scores))


def check_fourier_probe(acts: np.ndarray, coords: np.ndarray, value: np.ndarray,
                        periods: Sequence[float] = (2, 5, 10), control_period: float = 7
                        ) -> ShapeReport:
    """Base-10 Fourier features of a number (Feucht et al. 2026 Sec. 4): linear probes for the
    circles of period 10 and 5 must read the value well; a control period (7) should not."""
    value = np.asarray(value, dtype=float)
    r2 = {int(T): fourier_probe_r2(acts, value, T) for T in periods}
    r2_ctrl = fourier_probe_r2(acts, value, control_period)
    phase10 = 2 * np.pi * (value % 10) / 10
    f = fit_circle_plane(coords, phase10)
    checks = [
        _mk("probe_r2_period_10", r2[10], 0.50, critical=True, partial=0.30,
            note="cross-validated ridge probe to cos/sin(2 pi sum/10)"),
        _mk("probe_r2_period_5", r2[5], 0.50, critical=True, partial=0.30),
        _mk("probe_r2_period_2", r2[2], 0.50, critical=False, partial=0.30),
        _mk("probe_r2_period_10_minus_control_7", r2[10] - r2_ctrl, 0.20, critical=False,
            note=f"control period 7 probe R^2 = {r2_ctrl:.2f}"),
        _mk("raw_pca_circular_corr_mod10", f["circ_corr"], 0.60, critical=False, partial=0.40,
            note="is the mod-10 circle visible in the raw 3-D PCA (not required by the paper)"),
    ]
    return _finalize("fourier_probe", checks, {"probe_r2": r2, "control_r2": r2_ctrl})


# ---------------------------------------------------------------- dispatch --------------------

# Which label plays which role for each manifold (names from manifest.yaml).
ROLES: dict[str, dict[str, Any]] = {
    "colors": {"kind": "paraboloid", "hue": "hue", "lightness": "lightness",
               "saturation": "saturation", "color_by": "hex", "color_mode": "hex",
               "ambient_dim": 3, "tuning_label": "hue"},
    "temperature": {"kind": "line", "label": "fahrenheit", "color_by": "fahrenheit",
                    "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "fahrenheit"},
    "age": {"kind": "line", "label": "age", "color_by": "age", "color_mode": "continuous",
            "ambient_dim": 1, "tuning_label": "age"},
    "geography": {"kind": "tree", "lat": "latitude", "lon": "longitude", "group": "continent",
                  "color_by": "continent", "color_mode": "categorical", "ambient_dim": 3,
                  "tuning_label": "latitude"},
    "days": {"kind": "circle", "phase": "day_idx", "period": 7, "classes": "day_idx",
             "order": list(range(7)), "color_by": "day", "color_mode": "categorical",
             "ambient_dim": 2, "tuning_label": "day_idx"},
    "years": {"kind": "helix", "linear": "year", "periodic": "year", "period": 10,
              "color_by": "year", "color_mode": "continuous", "ambient_dim": 3,
              "tuning_label": "year"},
    "formality": {"kind": "line", "label": "formality", "color_by": "formality",
                  "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "formality"},
    "sent_length": {"kind": "line", "label": "n_tokens", "color_by": "n_tokens",
                    "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "n_tokens"},
    "political_bias": {"kind": "ordinal", "label": "bias_label", "color_by": "bias_label",
                       "color_mode": "continuous", "ambient_dim": 1,
                       "tuning_label": "bias_label"},
    # ---- extended tier (supervised approximations of the blog's unsupervised manifolds)
    "ordinals": {"kind": "line", "label": "rank", "color_by": "rank", "color_mode": "continuous",
                 "ambient_dim": 1, "tuning_label": "rank"},
    "classical_composers": {"kind": "line", "label": "birth_year", "color_by": "birth_year",
                            "color_mode": "continuous", "ambient_dim": 1,
                            "tuning_label": "birth_year"},
    "scotus_cases": {"kind": "line", "label": "year", "color_by": "year",
                     "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "year"},
    "chemical_bonds": {"kind": "line", "label": "length_pm", "color_by": "length_pm",
                       "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "length_pm"},
    "adverbs_pace": {"kind": "ordinal", "label": "speed_rank", "color_by": "speed_rank",
                     "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "speed_rank"},
    "statistical_uncertainty": {"kind": "line", "label": "ln_std_over_mean",
                                "color_by": "ln_std_over_mean", "color_mode": "continuous",
                                "ambient_dim": 1, "tuning_label": "ln_std_over_mean"},
    "bible_verses": {"kind": "line", "label": "position", "color_by": "book_index",
                     "color_mode": "continuous", "ambient_dim": 1, "tuning_label": "book_index"},
    "slant_rhymes_ore": {"kind": "ordinal", "label": "rhyme_grade", "color_by": "grade_name",
                         "color_mode": "categorical", "ambient_dim": 1,
                         "tuning_label": "rhyme_grade"},
    "numbers_1_100": {"kind": "helix", "linear": "n", "periodic": "n", "period": 10,
                      "color_by": "n", "color_mode": "continuous", "ambient_dim": 3,
                      "tuning_label": "n"},
    "emotions_valence_arousal": {"kind": "circumplex", "group": "emotion", "valence": "valence",
                                 "arousal": "arousal", "color_by": "emotion",
                                 "color_mode": "categorical", "ambient_dim": 2,
                                 "tuning_label": "valence"},
    # ---- cyclic tier (Feucht et al. 2026 templates)
    "weekdays_circle": {"kind": "circle", "phase": "day_idx", "period": 7, "classes": "day_idx",
                        "order": list(range(7)), "color_by": "day", "color_mode": "categorical",
                        "ambient_dim": 2, "tuning_label": "day_idx"},
    "months_circle": {"kind": "circle", "phase": "month_idx", "period": 12,
                      "classes": "month_idx", "order": list(range(12)), "color_by": "month",
                      "color_mode": "categorical", "ambient_dim": 2, "tuning_label": "month_idx"},
    "hours_circle": {"kind": "circle", "phase": "hour", "period": 24, "classes": "hour",
                     "order": [float(h) for h in range(24)], "color_by": "hour",
                     "color_mode": "cyclic", "ambient_dim": 2, "tuning_label": "hour"},
    "addition_sum_fourier": {"kind": "fourier_probe", "value": "sum", "color_by": "sum_mod10",
                             "color_mode": "cyclic", "ambient_dim": 2, "tuning_label": "sum"},
}


def _col(labels: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([lab[key] for lab in labels])


def run_shape_check(name: str, coords: np.ndarray, labels: list[dict[str, Any]],
                    acts: np.ndarray | None = None) -> ShapeReport:
    if name not in ROLES:
        raise KeyError(f"no shape-check roles defined for manifold {name!r}")
    r = ROLES[name]
    kind = r["kind"]
    if kind == "circumplex":
        return check_circumplex(coords, _col(labels, r["group"]),
                                _col(labels, r["valence"]).astype(float),
                                _col(labels, r["arousal"]).astype(float))
    if kind == "fourier_probe":
        if acts is None:
            raise ValueError("fourier_probe check needs the raw activations")
        return check_fourier_probe(acts, coords, _col(labels, r["value"]).astype(float))
    if kind == "line":
        return check_line(coords, _col(labels, r["label"]).astype(float))
    if kind == "ordinal":
        return check_ordinal(coords, _col(labels, r["label"]).astype(float))
    if kind == "circle":
        idx = _col(labels, r["phase"]).astype(float)
        phase = 2 * np.pi * idx / r["period"]
        return check_circle(coords, phase, _col(labels, r["classes"]), r["order"])
    if kind == "helix":
        return check_helix(coords, _col(labels, r["linear"]).astype(float),
                           _col(labels, r["periodic"]).astype(float), r["period"])
    if kind == "paraboloid":
        return check_paraboloid(coords, _col(labels, r["hue"]).astype(float),
                                _col(labels, r["lightness"]).astype(float),
                                _col(labels, r["saturation"]).astype(float))
    if kind == "tree":
        return check_tree(coords, _col(labels, r["lat"]).astype(float),
                          _col(labels, r["lon"]).astype(float), _col(labels, r["group"]))
    raise ValueError(kind)
