"""Loader math and greedy subspace capture on a synthetic dictionary (CPU, no downloads)."""

import numpy as np
import torch

from manifolds.sae.capture import (
    greedy_geometric,
    greedy_statistical,
    k_for_threshold,
    partial_reconstructions,
    pca_curve,
    random_atoms_curve,
    tuning_curves,
)
from manifolds.sae.llama_scope import LlamaScopeSAE


def _hyper(d: int) -> dict:
    return {"d_model": d, "norm_activation": "dataset-wise",
            "dataset_average_activation_norm": {"in": 2.0, "out": 2.0},
            "sparsity_include_decoder_norm": True, "act_fn": "jumprelu",
            "jump_relu_threshold": 0.1, "hook_point_in": "blocks.1.hook_resid_post"}


def test_loader_matches_reference_pipeline():
    torch.manual_seed(0)
    d, d_sae = 16, 40
    sd = {"encoder.weight": torch.randn(d_sae, d), "encoder.bias": torch.randn(d_sae) * 0.1,
          "decoder.weight": torch.randn(d, d_sae), "decoder.bias": torch.randn(d) * 0.1}
    hyper = _hyper(d)
    sae = LlamaScopeSAE.from_state_dict({k: v.clone() for k, v in sd.items()}, hyper)
    # reference: apply the two transforms literally as in lm_saes v1.0.0 from_pretrained
    f = (d**0.5) / 2.0
    b_enc = sd["encoder.bias"] / f
    b_dec = sd["decoder.bias"] / f
    w_dec = sd["decoder.weight"] * (f / f)
    dn = torch.norm(w_dec, p=2, dim=0)
    w_dec = w_dec / dn
    w_enc = sd["encoder.weight"] * dn[:, None]
    b_enc = b_enc * dn
    x = torch.randn(7, d) * 2.0
    h = x @ w_enc.T + b_enc
    z_ref = h.where(h > 0.1, torch.zeros_like(h))
    z = sae.encode(x)
    torch.testing.assert_close(z, z_ref)
    torch.testing.assert_close(sae.decode(z), z_ref @ w_dec.T + b_dec)
    torch.testing.assert_close(sae.w_dec.norm(dim=1), torch.ones(d_sae))
    st = sae.stats(x, z)
    assert st.n == 7 and 0 <= st.l0_mean <= d_sae and -1e9 < st.ve_within <= 1.0


def test_greedy_recovers_planted_subspace():
    torch.manual_seed(0)
    d, n = 32, 300
    decoder = torch.randn(200, d)
    decoder = decoder / decoder.norm(dim=1, keepdim=True)
    planted = [5, 17, 99]
    coef = torch.randn(n, 3)
    x = coef @ decoder[planted] + 0.01 * torch.randn(n, d)
    sel, curve = greedy_geometric(x, decoder, max_k=6)
    assert set(sel[:3]) == set(planted)
    assert curve[2] > 0.99 and k_for_threshold(curve, 0.9) == 3
    # statistical greedy with codes equal to the planted coefficients (shifted positive)
    codes = torch.zeros(n, 200)
    codes[:, planted] = coef - coef.min() + 0.1
    sel_s, curve_s = greedy_statistical(x, codes, decoder, max_k=6)
    assert set(sel_s[:3]) == set(planted) and curve_s[2] > 0.99
    pc = pca_curve(x, 6)
    assert pc[2] > 0.99 and len(pc) == 6
    rc = random_atoms_curve(x, decoder, max_k=6, seed=0, n_rep=2)
    assert len(rc) == 6 and rc[-1] <= pc[-1] + 1e-6


def test_tuning_and_partial_recon():
    torch.manual_seed(0)
    n, d = 120, 8
    label = np.repeat(np.arange(12), 10).astype(float)
    codes = torch.zeros(n, 5)
    codes[:, 2] = torch.as_tensor(np.exp(-((label - 4) ** 2) / 4), dtype=torch.float32)
    tc = tuning_curves(codes, [2], label, sigma=1.0)
    assert tc[0].feature == 2 and len(tc[0].x) == 12 and max(tc[0].y) == 1.0
    assert int(np.argmax(tc[0].y)) == 4
    decoder = torch.randn(5, d)
    x = codes @ decoder + 0.01 * torch.randn(n, d)
    xc = (x - x.mean(0)).numpy()
    comps = np.linalg.svd(xc, full_matrices=False)[2][:3]
    pr = partial_reconstructions(x, codes, decoder, [2, 0], comps, x.mean(0).numpy(), [1, 2, 9])
    assert pr.ks == [1, 2] and pr.coords[1].shape == (n, 3)
