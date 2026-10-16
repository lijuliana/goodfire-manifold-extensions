"""Full-model capture, lenses and steering on a random tiny Llama (CPU, no downloads)."""

import numpy as np
import torch
from conftest import CharTokenizer
from transformers import LlamaConfig, LlamaForCausalLM

from manifolds.depth import DepthConfig, FullModel
from manifolds.jlens import JacobianLensFull, logit_correlation_readout, pc_readouts
from manifolds.lens import TunedLens, lens_readout, top_tokens
from manifolds.steer import Steerer, plan_interpolation


def _tiny(tmp_path):
    cfg = LlamaConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).eval()
    m.save_pretrained(tmp_path / "tiny")
    return m


def test_capture_all_layers_and_lenses(tmp_path):
    ref = _tiny(tmp_path)
    fm = FullModel(DepthConfig(model_id=str(tmp_path / "tiny"), dtype="float32", device="cpu",
                               batch_size=2), tokenizer=CharTokenizer())
    prompts = ["ab", "a longer prompt"]
    st = fm.capture_all_layers(prompts, progress=False)
    assert st.shape == (2, 5, 32)
    enc = CharTokenizer()([prompts[1]])
    with torch.no_grad():
        hs = ref(**enc, output_hidden_states=True).hidden_states
    for li in range(5):
        np.testing.assert_allclose(st[1, li], hs[li][0, -1].numpy(), rtol=2e-2, atol=2e-2)
    # logit lens on the final state must equal the model's own logits
    with torch.no_grad():
        logits_ref = ref(**enc).logits[0, -1]
    logits = fm.unembed(torch.as_tensor(st[1:2, 4].astype(np.float32)))[0]
    np.testing.assert_allclose(logits.numpy(), logits_ref.numpy(), rtol=2e-2, atol=2e-2)
    ids, probs = top_tokens(fm, torch.as_tensor(st[:, 4].astype(np.float32)), k=3)
    assert ids.shape == (2, 3) and np.all(np.diff(probs, axis=1) <= 1e-6)
    # tuned lens with identity-ish translators (zero weights) reproduces the logit lens
    sd = {f"{i}.weight": torch.zeros(32, 32) for i in range(4)}
    sd |= {f"{i}.bias": torch.zeros(32) for i in range(4)}
    tl = TunedLens.from_state_dict(sd, {"num_hidden_layers": 4}, dtype=torch.float32)
    out = lens_readout(fm, st, [0, 2, 4], tl, k=2)
    assert out["logit"]["2"]["tokens"] == out["tuned"]["2"]["tokens"]
    assert len(out["logit"]["0"]["tokens"]) == 2 and len(out["logit"]["0"]["tokens"][0]) == 2


def test_plan_interpolation_recovers_line(rng):
    t = np.linspace(0, 1, 120)
    acts = np.outer(t, rng.normal(size=16)) + 0.01 * rng.normal(size=(120, 16))
    deltas, pos, edges, centers, k = plan_interpolation(acts, t, n_bins=4, n_between=3)
    assert k >= 1 and len(centers) == 4 and len(edges) == 5
    assert len(deltas) == 3 * 3 + 1 and pos[0] == 0 and pos[-1] == 3
    # deltas move monotonically along the line direction
    proj = deltas @ (acts[-1] - acts[0])
    assert np.all(np.diff(proj) > 0)


def test_steerer_changes_logits(tmp_path):
    _tiny(tmp_path)
    fm = FullModel(DepthConfig(model_id=str(tmp_path / "tiny"), dtype="float32", device="cpu"),
                   tokenizer=CharTokenizer())
    st = Steerer(fm, layer=1)
    targets = ["a", "b"]
    p0 = st.next_token_probs("hello", np.zeros((1, 32), dtype=np.float32), targets)
    p1 = st.next_token_probs("hello", 5.0 * np.ones((1, 32), dtype=np.float32), targets)
    assert p0.shape == (1, 2) and not np.allclose(p0, p1)


def test_jacobian_lens_identity_matches_logit_lens(tmp_path, rng):
    _tiny(tmp_path)
    fm = FullModel(DepthConfig(model_id=str(tmp_path / "tiny"), dtype="float32", device="cpu"),
                   tokenizer=CharTokenizer())
    st = fm.capture_all_layers(["ab", "abc"], progress=False)
    jl = JacobianLensFull({i: torch.eye(32) for i in range(3)}, 7, {"repo": "synthetic"})
    assert jl.block_index(0) is None and jl.block_index(3) == 2 and jl.block_index(4) == 3
    out = lens_readout(fm, st, [0, 2, 4], None, k=2, jlens=jl)
    assert "0" not in out["jacobian"]
    assert out["jacobian"]["2"]["tokens"] == out["logit"]["2"]["tokens"]
    assert out["jacobian"]["4"]["tokens"] == out["logit"]["4"]["tokens"]  # final block: identity
    pcs = rng.normal(size=(3, 32)).astype(np.float32)
    pcs /= np.linalg.norm(pcs, axis=1, keepdims=True)
    coords = rng.normal(size=(2, 3))
    logits = fm.unembed(torch.as_tensor(st[:, 4].astype(np.float32))).numpy()
    reads = pc_readouts(fm, pcs, coords, logits, 2, jlens=jl, tuned=None, k=3)
    assert len(reads) == 3 and reads[0]["jacobian_lens"]["up"] == reads[0]["logit_lens"]["up"]
    emp = logit_correlation_readout(logits, coords[:, 0], lambda i: f"t{i}", k=2, min_prob=0.0)
    assert len(emp["positive"]) == 2 and len(emp["negative"]) == 2
