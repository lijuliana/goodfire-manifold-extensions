import json

import numpy as np

from manifolds.analysis.report import analyze_manifold
from manifolds.config import load_config
from manifolds.manifest import load_manifest
from manifolds.prompts.core import build_temperature
from manifolds.site.build import build_site


def test_analyze_and_build_site(tmp_path, rng):
    ps = build_temperature()
    t = np.array([lab["fahrenheit"] for lab in ps.labels])
    acts = np.outer(t, rng.normal(size=64)) + 0.1 * rng.normal(size=(len(t), 64))
    results = tmp_path / "results"
    spec = load_manifest().get("temperature")
    rep = analyze_manifold(spec, load_config(), results, acts=acts.astype(np.float32),
                           prompt_set=ps, write=True)
    assert rep["status"] == "matches", rep["shape"]["reason"]
    assert (results / "temperature" / "figure.json").exists()
    n = build_site(results, tmp_path / "site")
    assert n == 1
    index = (tmp_path / "site" / "index.html").read_text()
    assert "badge-matches" in index and "political_bias" in index and "badge-pending" in index
    page = (tmp_path / "site" / "manifolds" / "temperature.html").read_text()
    assert "Today it&#39;s {f} degrees Fahrenheit outside" in page or "Today it's {f}" in page
    assert json.loads((results / "temperature" / "report.json").read_text())["status"] == "matches"
