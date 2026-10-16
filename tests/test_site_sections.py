"""Build a page with SAE, depth, lens and steering sections from synthetic report files."""

import json

import numpy as np

from manifolds.analysis.report import analyze_manifold
from manifolds.config import load_config
from manifolds.depth_report import depth_figure, lens_summary_figure, scores_figure, steer_figure
from manifolds.manifest import load_manifest
from manifolds.prompts.core import build_temperature
from manifolds.site.build import build_site


def test_page_with_all_sections(tmp_path, rng):
    ps = build_temperature()
    t = np.array([lab["fahrenheit"] for lab in ps.labels])
    acts = np.outer(t, rng.normal(size=64)) + 0.1 * rng.normal(size=(len(t), 64))
    acts = acts.astype(np.float32)
    results = tmp_path / "results"
    spec = load_manifest().get("temperature")
    cfg = load_config()
    analyze_manifold(spec, cfg, results, acts=acts, prompt_set=ps, write=True)
    d = results / "temperature"
    n = len(t)
    # depth files
    coords = {li: rng.normal(size=(n, 3)) for li in range(0, 5)}
    hover = [f"p{i}" for i in range(n)]
    (d / "depth_figure.json").write_text(
        depth_figure(coords, t.tolist(), "continuous", hover, 2, {2: hover}).to_json())
    (d / "depth_scores.json").write_text(
        scores_figure(list(range(5)), {"spearman_best_direction": [0.1, 0.3, 0.8, 0.9, 0.9]}, 2
                      ).to_json())
    lens = {"logit": {"2": {"tokens": [[" hot", " a", " b"]] * n, "probs": [[0.3, 0.2, 0.1]] * n}},
            "tuned": {"2": {"tokens": [[" warm", " a", " b"]] * n, "probs": [[0.5, 0.2, 0.1]] * n}},
            "layers": [2], "k": 3}
    (d / "lens_figure.json").write_text(lens_summary_figure(lens, 2).to_json())
    steer = {"manifold": "temperature", "base_prompt": "Today it's 45 degrees, which is",
             "bin_edges": [0, 1, 2], "bin_centers": [0.5, 1.5], "positions": [0, 0.5, 1],
             "targets": ["cold", "hot"], "probs": [[0.8, 0.1], [0.5, 0.4], [0.2, 0.7]],
             "pca_components_used": 3, "extras": {"base_label": 45.0, "n_bins_used": 2}}
    (d / "steer_figure.json").write_text(steer_figure(steer).to_json())
    (d / "depth.json").write_text(json.dumps({
        "name": "temperature", "n_points": n, "n_points_plotted": n, "n_layers": 4,
        "consistency_block19_vs_truncated": {"cosine_min": 0.9999, "cosine_mean": 1.0},
        "scores": {str(li): {"status": "matches", "metrics": {}} for li in range(5)},
        "lens": lens, "tuned_lens": {"repo": "x/y"}, "steering": steer, "model_id": "m"}))
    # a minimal SAE report in the current format
    (d / "sae.json").write_text(json.dumps({
        "name": "temperature", "sae": {"repo": "r"}, "n_points": n,
        "stats": {"l0_mean": 50.0, "ve_total": 0.8, "ve_within": -2.0, "cosine_mean": 0.9},
        "summary": {"ambient_dim": 1, "ve_at": {str(k): {"pca": 0.5, "geometric": 0.2,
                    "statistical": 0.1, "random_atoms": 0.0} for k in (1, 8, 16, 32, 64)},
                    "k_geometric_to_match_pca_at_ambient_dim": None,
                    "consistent_with_dilution_claim": True},
        "tuning": {"label": "fahrenheit"}}))
    for f in ("sae_curves.json", "sae_tuning.json", "sae_partial.json"):
        (d / f).write_text('{"data":[],"layout":{}}')
    assert build_site(results, tmp_path / "site") == 1
    page = (tmp_path / "site" / "manifolds" / "temperature.html").read_text()
    for marker in ("Across the layers", "json-depth", "Steering along the manifold", "json-steer",
                   "SAE tiling", "json-sae-partial", "tuned lens", "dilution"):
        assert marker in page, marker
    index = (tmp_path / "site" / "index.html").read_text()
    assert "dilution" in index
