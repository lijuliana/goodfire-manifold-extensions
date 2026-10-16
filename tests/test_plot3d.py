import numpy as np
import pytest

from manifolds.viz.plot3d import build_figure, figure_to_json, subsample


def test_modes_and_trace_counts(rng):
    coords = rng.normal(size=(40, 3))
    hover = [f"p{i}" for i in range(40)]
    fig = build_figure(coords, np.arange(40), color_mode="continuous", hover_text=hover)
    assert len(fig.data) == 3  # floor, shadow, points
    fig = build_figure(coords, ["#ff0000"] * 40, color_mode="hex", hover_text=hover)
    assert len(fig.data) == 3
    cats = ["a", "b", "c", "d"] * 10
    fig = build_figure(coords, cats, color_mode="categorical", hover_text=hover,
                       category_order=["a", "b", "c", "d"])
    assert len(fig.data) == 6
    assert fig.layout.scene.xaxis.visible is False
    assert "data" in figure_to_json(fig)


def test_invalid_inputs(rng):
    with pytest.raises(ValueError):
        build_figure(rng.normal(size=(5, 2)), [0] * 5, color_mode="continuous", hover_text=[""] * 5)
    with pytest.raises(ValueError):
        build_figure(rng.normal(size=(5, 3)), [0] * 4, color_mode="continuous", hover_text=[""] * 5)


def test_subsample_deterministic():
    a = subsample(1000, 100, seed=1)
    b = subsample(1000, 100, seed=1)
    assert np.array_equal(a, b) and len(a) == 100 and np.all(np.diff(a) > 0)
    assert np.array_equal(subsample(50, 100), np.arange(50))
