"""Synthetic-geometry tests: each checker must pass on its own shape and fail on shuffled labels."""

import numpy as np
import pytest

from manifolds.analysis.shapes import (
    ROLES,
    check_circle,
    check_helix,
    check_line,
    check_ordinal,
    check_paraboloid,
    check_tree,
    cyclic_order_ok,
    run_shape_check,
)


def _embed(local: np.ndarray, rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Rotate a 3-D cloud randomly and add noise (PCA-like coordinates)."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return local @ q.T + noise * rng.normal(size=local.shape)


def test_line(rng):
    t = np.linspace(-1, 1, 150)
    coords = _embed(np.c_[3 * t, 0.3 * t**2, np.zeros_like(t)], rng)
    rep = check_line(coords, t)
    assert rep.status == "matches", rep.reason
    assert rep.extras["rubric_version"] >= 2
    assert check_line(coords, rng.permutation(t)).status == "failed"
    # weak but real ordering -> partial
    noisy = _embed(np.c_[3 * t, np.zeros_like(t), np.zeros_like(t)], rng, noise=2.0)
    assert check_line(noisy, t).status in ("partial", "failed")


def test_ordinal(rng):
    lab = np.repeat(np.arange(5), 40).astype(float)
    coords = _embed(np.c_[lab + 0.3 * rng.normal(size=200), rng.normal(size=200) * 0.2,
                          np.zeros(200)], rng)
    rep = check_ordinal(coords, lab)
    assert rep.status == "matches", rep.reason
    assert check_ordinal(coords, rng.permutation(lab)).status == "failed"
    # V-shaped ordinal manifold: ordering along one direction fails but nearest-centroid path holds
    v = np.c_[np.abs(lab - 2) + 0.2 * rng.normal(size=200), lab + 0.2 * rng.normal(size=200),
              np.zeros(200)]
    rep_v = check_ordinal(_embed(v, rng), lab)
    assert rep_v.checks[2].value == 1.0  # centroid_path_adjacency


def test_circle_with_classes(rng):
    k = 7
    cls = np.repeat(np.arange(k), 8)
    phase = 2 * np.pi * cls / k
    coords = _embed(np.c_[np.cos(phase), np.sin(phase), np.zeros_like(phase)], rng, noise=0.05)
    rep = check_circle(coords, phase, cls, list(range(k)))
    assert rep.status == "matches", rep.reason
    bad = check_circle(coords, phase, rng.permutation(cls), list(range(k)))
    assert bad.status in ("failed", "partial")
    # reversed orientation is still a valid cyclic order
    assert cyclic_order_ok(-phase, cls, list(range(k)))
    from manifolds.analysis.shapes import cyclic_order_score
    swapped = cls.copy()
    swapped[cls == 2], swapped[cls == 3] = 3, 2   # one adjacent swap
    assert 0.5 <= cyclic_order_score(phase, swapped, list(range(k))) < 1.0


def test_helix(rng):
    year = np.arange(1800, 1999).astype(float)
    ang = 2 * np.pi * (year % 10) / 10
    coords = _embed(np.c_[(year - 1900) / 50, np.cos(ang), np.sin(ang)], rng, noise=0.03)
    rep = check_helix(coords, year, year, 10)
    assert rep.status == "matches", rep.reason
    assert check_helix(coords, rng.permutation(year), year, 10).status != "matches"


def test_paraboloid(rng):
    n = 600
    hue = rng.uniform(0, 1, n)
    light = rng.uniform(0.2, 0.8, n)
    sat = rng.uniform(0.2, 0.8, n)
    r = 0.5 + light
    coords = _embed(np.c_[r * np.cos(2 * np.pi * hue), r * np.sin(2 * np.pi * hue), r**2], rng,
                    noise=0.02)
    rep = check_paraboloid(coords, hue, light, sat)
    assert rep.status == "matches", rep.reason
    assert rep.extras["height_axis_variable"] == "lightness"
    assert check_paraboloid(coords, rng.permutation(hue), light, sat).status == "failed"


def test_tree(rng):
    groups = np.repeat(np.array(["A", "B", "C", "D"]), 60)
    centers = rng.normal(size=(4, 3)) * 4
    lat = rng.uniform(-60, 60, 240)
    lon = rng.uniform(-180, 180, 240)
    local = centers[np.repeat(np.arange(4), 60)] + np.c_[lat / 60, lon / 180, np.zeros(240)] * 0.8
    coords = _embed(local, rng, noise=0.05)
    rep = check_tree(coords, lat, lon, groups)
    assert rep.status == "matches", rep.reason
    assert check_tree(coords, lat, lon, rng.permutation(groups)).status != "matches"


def test_run_shape_check_dispatch(rng):
    labels = [{"fahrenheit": float(f), "celsius": 0.0} for f in range(-30, 120)]
    t = np.array([lab["fahrenheit"] for lab in labels])
    coords = _embed(np.c_[t / 50, np.zeros_like(t), np.zeros_like(t)], rng, noise=0.01)
    assert run_shape_check("temperature", coords, labels).status == "matches"
    with pytest.raises(KeyError):
        run_shape_check("nope", coords, labels)
    assert set(ROLES) >= {"colors", "days", "years", "geography", "political_bias"}
