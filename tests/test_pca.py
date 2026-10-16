import numpy as np

from manifolds.analysis.pca import filter_norm_outliers, fit_pca


def test_pca_is_deterministic_and_sorted(rng):
    x = rng.normal(size=(200, 50)) @ rng.normal(size=(50, 50))
    a = fit_pca(x, 3)
    b = fit_pca(x, 3)
    np.testing.assert_allclose(a.coords, b.coords)
    evr = a.explained_variance_ratio
    assert np.all(np.diff(evr) <= 1e-12)
    assert a.coords.shape == (200, 3)
    np.testing.assert_allclose(a.coords.mean(0), 0, atol=1e-9)


def test_outlier_filter_drops_huge_norm_rows(rng):
    x = rng.normal(size=(100, 8))
    x[3] *= 500.0
    keep = filter_norm_outliers(x, n_std=3.0)
    assert not keep[3]
    assert keep.sum() == 99
