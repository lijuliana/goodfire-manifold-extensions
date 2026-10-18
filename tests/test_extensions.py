# ruff: noqa: E501
"""Debate extensions on a random tiny Llama (CPU, no downloads): weak-activation ablation,
neuron writers/readers, interference, checkpoint sweep."""

import json

import numpy as np
import pytest
import torch
from conftest import CharTokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from manifolds import ablate, checkpoints, interference, neurons
from manifolds.analysis.compare import compare, knn_overlap, procrustes_align
from manifolds.analysis.pca import fit_pca
from manifolds.config import load_config
from manifolds.depth import DepthConfig, FullModel
from manifolds.manifest import load_manifest
from manifolds.prompts import build
from manifolds.tasks import kl_and_agreement, score_task, task_for
from manifolds.viz.plot3d import frames_figure


def _tiny_fm(tmp_path, batch_size=8):
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).eval()
    m.save_pretrained(tmp_path / "tiny")
    return FullModel(DepthConfig(model_id=str(tmp_path / "tiny"), dtype="float32", device="cpu",
                                 batch_size=batch_size), tokenizer=CharTokenizer())


def _results_for(tmp_path, name, fm, layer):
    """Synthetic results dir: prompts.json + activations read from the tiny model itself."""
    ps = build(name)
    d = tmp_path / "results" / name
    d.mkdir(parents=True)
    ps.save(d / "prompts.json")
    cap = fm.capture_layer_and_logits(ps.prompts, layer + 1, spans=ps.spans, progress=False)
    np.save(d / "activations.npy", cap["hidden"])
    return ps, d


def test_ablator_modes(tmp_path):
    fm = _tiny_fm(tmp_path)
    ab = ablate.Ablator(fm.model)
    ids = torch.randint(2, 200, (4, 16))
    ab.set("calibrate")
    with torch.no_grad():
        fm.model(input_ids=ids)
    amax = ab.calibration_tensor()
    assert amax.shape == (4, 64) and (amax > 0).all()
    ab.load_calibration(amax, "cpu")
    with torch.no_grad():
        ab.set("off")
        ref = fm.model(input_ids=ids).logits
        fracs = []
        for x in (1.0, 10.0, 50.0):
            ab.set("threshold", x / 100, track=True)
            fm.model(input_ids=ids)
            fracs.append(ab.fraction_zeroed())
        assert fracs == sorted(fracs) and fracs[-1] > fracs[0]
        ab.set("random", 0.1, track=True)
        fm.model(input_ids=ids)
        r = ab.fraction_zeroed()
        ab.set("threshold", 0.1, track=True)
        fm.model(input_ids=ids)
        assert abs(r - ab.fraction_zeroed()) < 0.01   # count-matched (later blocks see perturbed input)
        ab.set("scaleweak", 0.5, scale=1.0)
        same = fm.model(input_ids=ids).logits
        torch.testing.assert_close(same, ref)                 # s = 1 is the identity
    ab.remove()
    c = ablate.conditions()
    assert c[0].group == "clean" and len(c) == 1 + 10 + 4 + 4
    assert c[1].label == "0.1" and c[-1].label == "×2@1.5"


def test_capture_layer_and_logits(tmp_path):
    fm = _tiny_fm(tmp_path)
    prompts = ["ab", "a longer prompt", "xyz"]
    cap = fm.capture_layer_and_logits(prompts, 2, cand_ids=[5, 9], full_index=np.array([1]),
                                      mlp_blocks=[0, 1], progress=False)
    enc = CharTokenizer()([prompts[1]])
    with torch.no_grad():
        out = fm.model(**enc, output_hidden_states=True)
    np.testing.assert_allclose(cap["hidden"][1], out.hidden_states[2][0, -1].numpy(), atol=1e-4)
    assert int(cap["top1"][1]) == int(out.logits[0, -1].argmax())
    np.testing.assert_allclose(cap["full"][0].astype(np.float32), out.logits[0, -1].numpy(),
                               rtol=2e-2, atol=2e-2)
    assert cap["cand"].shape == (3, 2) and cap["mlp"].shape == (3, 2, 64)


def test_tasks_and_compare(rng):
    for name in ("weekdays_circle", "months_last", "addition_sum_fourier", "political_bias", "age"):
        ps = build(name)
        t = task_for(name, ps.labels)
        if name == "age":
            assert t is None
            continue
        assert t is not None and len(t.answers) == len(ps)
    ps = build("weekdays_circle")
    t = task_for("weekdays_circle", ps.labels)
    cand = np.zeros((len(ps), 7))
    cand[np.arange(len(ps)), np.array(t.answers)] = 1.0
    assert score_task(t, np.zeros(len(ps)), cand, CharTokenizer())["accuracy"] == 1.0
    a = rng.normal(size=(50, 16)).astype(np.float32)
    basis = fit_pca(a, 3)
    same = compare("age", [{"age": float(i)} for i in range(50)], a, a, basis)
    assert same["knn_overlap"] == 1.0 and same["relative_move"] == 0.0
    assert knn_overlap(a, a + 5 * rng.normal(size=a.shape)) < 0.9
    beh = kl_and_agreement(a[:, :8], a[:, :8])
    assert beh["kl_mean"] < 1e-9 and beh["top1_agreement"] == 1.0
    c = rng.normal(size=(20, 3))
    r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    np.testing.assert_allclose(procrustes_align(c @ r, c), c - c.mean(0), atol=1e-8)
    fig = json.loads(frames_figure([("a", c), ("b", c * 2)], list(range(20)),
                                   color_mode="continuous", hover_text=["p"] * 20).to_json())
    assert len(fig["frames"]) == 2 and fig["layout"]["sliders"][0]["steps"][1]["label"] == "b"


def test_ablation_sweep_tiny(tmp_path):
    fm = _tiny_fm(tmp_path)
    layer = 2
    spec = load_manifest().get("age").model_copy(update={"layer": layer})
    cfg = load_config()
    ps, d = _results_for(tmp_path, "age", fm, layer)
    ab = ablate.Ablator(fm.model)
    ab.set("calibrate")
    with torch.no_grad():
        fm.model(input_ids=torch.randint(2, 200, (4, 16)))
    ab.load_calibration(ab.calibration_tensor(), "cpu")
    conds = [ablate.Condition("clean", "clean", "off", 0.0),
             ablate.Condition("weak_1", "weak", "threshold", 1.0),
             ablate.Condition("random_1", "random", "random", 1.0),
             ablate.Condition("scale_2_1", "scale", "scaleweak", 1.0, 2.0)]
    glob = {"conditions": {c.key: {"lm_loss": 2.0 + 0.1 * i, "lm_loss_delta": 0.1 * i,
                                   "popqa_accuracy": 0.2 - 0.01 * i, "fraction_zeroed": 0.1 * i}
                           for i, c in enumerate(conds)}}
    rep = ablate.run_manifold(spec, cfg, fm, ab, conds, glob, results_dir=tmp_path / "results",
                              max_points=50, n_kl=20)
    ab.remove()
    assert rep["consistency_with_cached_activations"]["cosine_min"] > 0.999
    assert set(rep["conditions"]) == {c.key for c in conds}
    assert rep["conditions"]["clean"]["knn_overlap"] == 1.0
    assert rep["conditions"]["weak_1"]["top1_agreement"] <= 1.0
    for f in ("ablation.json", "ablation_figure.json", "ablation_geometry.json",
              "ablation_behaviour.json"):
        assert (d / f).exists()
    fig = json.loads((d / "ablation_figure.json").read_text())
    assert len(fig["frames"]) == len(conds)
    ablate.global_figure({"conditions": {k: {**v, "group": c.group, "x": c.x} for (k, v), c in
                                         zip(glob["conditions"].items(), conds, strict=True)}})


def test_neurons_tiny(tmp_path):
    fm = _tiny_fm(tmp_path)
    layer = 2
    spec = load_manifest().get("age").model_copy(update={"layer": layer})
    cfg = load_config()
    _, d = _results_for(tmp_path, "age", fm, layer)
    rep = neurons.run_neurons(spec, cfg, fm, results_dir=tmp_path / "results", max_prompts=60,
                              max_points=40, top_n=5)
    assert rep["writer_blocks"] == [0, 1, 2] and rep["reader_blocks"] == [3]
    assert len(rep["writers"]) == 5 and rep["writers"][0]["share"] >= rep["writers"][1]["share"]
    assert 0 <= rep["writers"][0]["eta2"] <= 1
    assert rep["n_neurons_for_half_share"] <= rep["n_neurons_total"] == 3 * 64
    assert set(rep["causal"]) == {"top_16", "random_16", "top_64", "random_64", "top_192",
                                  "random_192"}
    assert rep["causal"]["top_192"]["share_removed"] > 0.99
    assert len(rep["block_contributions"]["mlp"]) == 3
    for f in ("neurons.json", "neurons_figure.json", "neurons_tuning.json", "neurons_readers.json",
              "neurons_blocks.json", "neurons_causal.json"):
        assert (d / f).exists()


def test_interference_and_crosstalk_tiny(tmp_path, monkeypatch):
    fm = _tiny_fm(tmp_path)
    layer = 2
    cfg = load_config()
    manifest = load_manifest()
    names = ["age", "temperature", "political_bias"]
    for n in names:
        _results_for(tmp_path, n, fm, layer)
    # the manifest says layer 19 for all three; the tiny run reads block 2 everywhere
    specs = {n: manifest.get(n).model_copy(update={"layer": layer}) for n in names}
    monkeypatch.setattr(manifest, "get", lambda n: specs[n])
    monkeypatch.setattr(interference, "GLOBAL_DIR", tmp_path / "_interference")
    res = interference.run_geometry(manifest, cfg, names, results_dir=tmp_path / "results",
                                    out_dir=tmp_path / "_interference", n_null=10)
    m = np.array(res["overlap"])
    assert m.shape == (3, 3) and np.allclose(np.diag(m), 1.0) and np.allclose(m, m.T)
    assert 0 <= res["null_pooled_mean"] <= 1 and (tmp_path / "_interference" / "null_bases.npy").exists()
    assert res["pairs_sorted"][0]["overlap"] >= res["pairs_sorted"][-1]["overlap"]
    fig = interference.woven_figure(tmp_path / "results", cfg, "age", "temperature")
    assert len(fig.data) == 4
    interference.overlap_bar_figure(res, "age")
    rep = interference.run_crosstalk(specs["age"], cfg, fm, res, results_dir=tmp_path / "results",
                                     max_prompts=40)
    assert "neutral" in rep["conditions"] and "age" not in rep["conditions"]
    row = rep["conditions"]["politics"]
    assert row["move_fraction_in_prefix_domain_subspace"] is not None
    assert row["move_fraction_in_null_subspace"] is not None
    assert (tmp_path / "results" / "age" / "crosstalk_figure.json").exists()


def test_checkpoints_tiny(tmp_path, monkeypatch):
    fm = _tiny_fm(tmp_path)
    cfg = load_config()
    manifest = load_manifest()
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda rev, **kw: fm)
    summary = checkpoints.run_checkpoints(
        manifest, cfg, ["age", "weekdays_circle"], checkpoints=[("early", 1.0), ("main", 4.0)],
        max_prompts=50, max_points=30, out_dir=tmp_path / "_ck", results_dir=tmp_path / "results",
        free_cache=False, model_id="tiny")
    assert set(summary["display_layers"]) == {"age", "weekdays_circle"}
    rep = json.loads((tmp_path / "results" / "age" / "checkpoints.json").read_text())
    assert len(rep["checkpoints"]) == 2 and rep["checkpoints"][1]["label"] == "final"
    assert np.array(rep["metric_by_checkpoint_and_layer"], dtype=object).shape == (2, 5)
    assert rep["checkpoints"][1]["layers_checked"] == [0, 1, 2, 3, 4]   # final: every position
    assert rep["checkpoints"][0]["centroid_participation_ratio"] >= 1.0
    fig = json.loads((tmp_path / "results" / "age" / "checkpoints_figure.json").read_text())
    assert len(fig["frames"]) == 2
    wk = json.loads((tmp_path / "results" / "weekdays_circle" / "checkpoints.json").read_text())
    assert wk["task"] == "weekday arithmetic" and wk["checkpoints"][0]["task_accuracy"] is not None


@pytest.mark.parametrize("x", [0.0, 1.0])
def test_popqa_metric(x):
    assert ablate.popqa_correct("The capital is Paris, France.", ["Paris"])
    assert not ablate.popqa_correct("I do not know.", ["Paris"])


def test_site_with_extension_sections(tmp_path, monkeypatch):
    """Tiny-model extension outputs for `age` render into the page and the index."""
    from manifolds.analysis.report import analyze_manifold
    from manifolds.site.build import build_site

    fm = _tiny_fm(tmp_path)
    layer = 2
    cfg = load_config()
    manifest = load_manifest()
    results = tmp_path / "results"
    names = ["age", "temperature"]
    for n in names:
        _results_for(tmp_path, n, fm, layer)
    specs = {n: manifest.get(n).model_copy(update={"layer": layer}) for n in names}
    for n in names:
        analyze_manifold(specs[n], cfg, results)
    ab = ablate.Ablator(fm.model)
    ab.set("calibrate")
    with torch.no_grad():
        fm.model(input_ids=torch.randint(2, 200, (4, 16)))
    ab.load_calibration(ab.calibration_tensor(), "cpu")
    conds = [ablate.Condition("clean", "clean", "off", 0.0),
             ablate.Condition("weak_1", "weak", "threshold", 1.0),
             ablate.Condition("weak_10", "weak", "threshold", 10.0),
             ablate.Condition("random_1", "random", "random", 1.0)]
    glob = {"conditions": {c.key: {**{k: v for k, v in c.__dict__.items()}, "lm_loss": 2.0 + 0.1 * i,
                                   "lm_loss_delta": 0.1 * i, "popqa_accuracy": 0.2 - 0.02 * i,
                                   "fraction_zeroed": 0.2 * i} for i, c in enumerate(conds)}}
    (results / "_ablation").mkdir()
    (results / "_ablation" / "global.json").write_text(json.dumps(glob))
    ablate.run_manifold(specs["age"], cfg, fm, ab, conds, glob, results_dir=results, max_points=40,
                        n_kl=10)
    ab.remove()
    neurons.run_neurons(specs["age"], cfg, fm, results_dir=results, max_prompts=50, max_points=30,
                        top_n=4)
    monkeypatch.setattr(manifest, "get", lambda n: specs[n])
    monkeypatch.setattr(interference, "GLOBAL_DIR", results / "_interference")
    geo = interference.run_geometry(manifest, cfg, names, results_dir=results,
                                    out_dir=results / "_interference", n_null=5)
    (results / "age" / "woven_figure.json").write_text(
        interference.woven_figure(results, cfg, "age", "temperature").to_json())
    (results / "age" / "woven.json").write_text(json.dumps({"partner": "temperature", "overlap": 0.5}))
    interference.run_crosstalk(specs["age"], cfg, fm, geo, results_dir=results, max_prompts=30)
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda rev, **kw: fm)
    checkpoints.run_checkpoints(manifest, cfg, ["age"], checkpoints=[("early", 1.0), ("main", 4.0)],
                                max_prompts=40, max_points=30, out_dir=results / "_checkpoints",
                                results_dir=results, free_cache=False, model_id="tiny")
    out = tmp_path / "site"
    assert build_site(results, out) == 2
    page = (out / "manifolds" / "age.html").read_text()
    for h in ("Weak activations: does the manifold survive?", "Which neurons write and read it",
              "Interference with other manifolds", "Forming during pre-training"):
        assert h in page
    assert "json-ablation" in page and "json-ckpt" in page and "json-int-cross" in page
    other = (out / "manifolds" / "temperature.html").read_text()
    assert "Weak activations" not in other
    index = (out / "index.html").read_text()
    assert "Across manifolds" in index and "json-ext-global" in index and "ablation · neurons" in index


def test_probe_helpers(rng):
    from manifolds.analysis.probe import apply_probe, cv_probe_coords, fit_probe, probe_r2

    v = rng.integers(2, 200, size=300).astype(float)
    ph = 2 * np.pi * (v % 10) / 10
    d = rng.normal(size=(5, 48))
    acts = (np.c_[np.cos(ph), np.sin(ph), v / 100, rng.normal(size=(300, 2))] @ d
            + 0.05 * rng.normal(size=(300, 48))).astype(np.float32)
    c = cv_probe_coords(acts, v)
    r2 = probe_r2(c, v)
    assert c.shape == (300, 3) and r2["r2_circle"] > 0.9 and r2["r2_linear"] > 0.9
    p = fit_probe(acts, v)
    assert np.corrcoef(apply_probe(p, acts)[:, 0], np.cos(ph))[0, 1] > 0.95
    shuffled = cv_probe_coords(acts, rng.permutation(v))
    assert probe_r2(shuffled, v)["r2_circle"] < 0.3


def test_probe_manifold_sliders_tiny(tmp_path, monkeypatch):
    """A Fourier-probe manifold gets probe-view frames in the ablation and checkpoint sliders."""
    fm = _tiny_fm(tmp_path)
    layer = 2
    cfg = load_config()
    manifest = load_manifest()
    spec = manifest.get("months_last").model_copy(update={"layer": layer})
    _, d = _results_for(tmp_path, "months_last", fm, layer)
    ab = ablate.Ablator(fm.model)
    ab.set("calibrate")
    with torch.no_grad():
        fm.model(input_ids=torch.randint(2, 200, (4, 16)))
    ab.load_calibration(ab.calibration_tensor(), "cpu")
    conds = [ablate.Condition("clean", "clean", "off", 0.0),
             ablate.Condition("weak_1", "weak", "threshold", 1.0)]
    glob = {"conditions": {c.key: {"lm_loss": 2.0, "lm_loss_delta": 0.0, "popqa_accuracy": 0.2,
                                   "fraction_zeroed": 0.3 * i} for i, c in enumerate(conds)}}
    rep = ablate.run_manifold(spec, cfg, fm, ab, conds, glob, results_dir=tmp_path / "results",
                              max_points=60, n_kl=10)
    ab.remove()
    assert rep["view"].startswith("Fourier probe")
    fig = json.loads((d / "ablation_figure.json").read_text())
    assert fig["frames"][1]["layout"]["annotations"][0]["text"].endswith("30% of all MLP activations zeroed")
    monkeypatch.setattr(checkpoints, "load_checkpoint", lambda rev, **kw: fm)
    checkpoints.run_checkpoints(manifest, cfg, ["months_last"], checkpoints=[("early", 1.0), ("main", 4.0)],
                                max_prompts=60, max_points=40, out_dir=tmp_path / "_ck",
                                results_dir=tmp_path / "results", free_cache=False, model_id="tiny")
    ck = json.loads((d / "checkpoints.json").read_text())
    assert ck["view"].startswith("Fourier probe")
    # a second call with the state files removed recaptures them without redoing the metrics
    for f in (tmp_path / "_ck" / "raw").glob("*__months_last.npy"):
        f.unlink()
    checkpoints.run_checkpoints(manifest, cfg, ["months_last"], checkpoints=[("early", 1.0), ("main", 4.0)],
                                max_prompts=60, max_points=40, out_dir=tmp_path / "_ck",
                                results_dir=tmp_path / "results", free_cache=False, model_id="tiny")
    assert (tmp_path / "_ck" / "raw" / "main__months_last.npy").exists()


def test_span_steering_and_target_ids(tmp_path, monkeypatch):
    from manifolds import steer

    fm = _tiny_fm(tmp_path)
    ps = build("weekdays_circle")
    cap = fm.capture_layer_and_logits(ps.prompts, 3, spans=ps.spans, progress=False)
    probe_st = steer.Steerer(fm, 2)
    with pytest.raises(ValueError):          # char tokenizer: every " Day" starts with a space
        probe_st.target_ids(["Monday", "Tuesday"])
    probe_st.remove()
    n_hooks = [len(m._forward_hooks) for m in fm.model.model.layers]   # transformers adds its own
    monkeypatch.setattr(steer.Steerer, "target_ids",
                        lambda self, t, digit_chunks=False: [[5 + i] for i in range(len(t))])
    res = steer.run_steering(fm, "weekdays_circle", cap["hidden"], ps.labels, ps.prompts, layer=2,
                             spans=ps.spans).to_dict()
    assert res["extras"]["steered_position"] == "concept token"
    assert np.array(res["probs"]).shape == (len(res["positions"]), 7)
    assert res["extras"]["base_answer"] in steer.STEER_TASKS["weekdays_circle"]["targets"]
    # steering hooks are removed afterwards: the model's output is unchanged
    enc = CharTokenizer()([ps.prompts[0]])
    with torch.no_grad():
        a = fm.model(**enc).logits
        b = fm.model(**enc).logits
    torch.testing.assert_close(a, b)
    assert [len(m._forward_hooks) for m in fm.model.model.layers] == n_hooks


def test_compare_many_parallel_matches_serial(rng):
    from manifolds.analysis.compare import compare_many

    labels = [{"age": float(i)} for i in range(60)]
    clean = (np.outer(np.arange(60), rng.normal(size=24)) + rng.normal(size=(60, 24))).astype(np.float32)
    others = [clean + s * rng.normal(size=clean.shape).astype(np.float32) for s in (0.5, 5.0, 50.0)]
    basis = fit_pca(clean, 3)
    a = compare_many("age", labels, clean, others, basis, workers=1)
    b = compare_many("age", labels, clean, others, basis, workers=2)
    for x, y in zip(a, b, strict=True):
        assert x["status_refit"] == y["status_refit"] and np.isclose(x["knn_overlap"], y["knn_overlap"])
    assert a[0]["knn_overlap"] > a[2]["knn_overlap"]


def test_archived_hidden_from_site(tmp_path):
    from manifolds.analysis.report import analyze_manifold
    from manifolds.site.build import build_site

    manifest = load_manifest()
    assert {m.name for m in manifest.manifolds if m.archived} == {
        "adverbs_pace", "bible_verses", "chemical_bonds", "formality", "scotus_cases"}
    cfg = load_config()
    results = tmp_path / "results"
    rng = np.random.default_rng(0)
    for n in ("age", "formality"):
        ps = build(n) if n == "age" else None
        if ps is None:
            from manifolds.prompts.types import PromptSet
            ps = PromptSet("formality", [f"s{i}" for i in range(40)],
                           [{"formality": float(i), "domain": "x"} for i in range(40)], "{s}", "test")
        acts = (np.outer(np.arange(len(ps)), rng.normal(size=32)) + rng.normal(size=(len(ps), 32))).astype(np.float32)
        analyze_manifold(manifest.get(n), cfg, results, acts=acts, prompt_set=ps, write=True)
    out = tmp_path / "site"
    assert build_site(results, out) == 1
    assert not (out / "manifolds" / "formality.html").exists()
    index = (out / "index.html").read_text()
    assert "Archived for now" in index and 'href="manifolds/formality.html"' not in index
