"""Manifold-restricted Jacobian lens (Gurnee et al. 2026, transformer-circuits.pub/2026/workspace).

The Jacobian lens transports a layer-l residual vector to the final layer with the corpus-averaged
Jacobian J_l = E[sum_{p' >= p} dh_final[p'] / dh_l[p]] (mean over valid source positions p and
prompts), then decodes with the model's own unembedding: lens_l(h) = unembed(J_l h). Logit lens is
J_l = I. The reference code (anthropics/jacobian-lens, Apache 2.0) estimates full d x d matrices
with one backward pass per output dimension. We only need J_l applied to a manifold's mean and its
top principal directions: J_l is linear, so J_l (mu + sum_k c_k v_k) = J_l mu + sum_k c_k J_l v_k,
which decodes every point of the manifold and every principal axis exactly within that subspace.
Each J_l v is a Jacobian-vector product through blocks l..L, estimated by central finite differences
in float32 with the perturbation applied at every valid source position at once (the same
reduction as the reference estimator: sum over later targets, mean over sources).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

SKIP_FIRST_N_POSITIONS = 16   # reference code: early positions are attention sinks
NEURONPEDIA_REPO = "neuronpedia/jacobian-lens"
NEURONPEDIA_LLAMA31_8B = "llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt"


class JacobianLensFull:
    """A pre-fitted lens in the reference file format: {"J": {block_index: [d, d]}, ...}.

    Lens index i is the *output of block i* (the reference `ActivationRecorder` hooks block
    outputs), i.e. HF hidden_states[i + 1]. The final block's output needs no transport.
    """

    def __init__(self, jacobians: dict[int, torch.Tensor], n_prompts: int, info: dict[str, Any]):
        self.jacobians = jacobians
        self.n_prompts = n_prompts
        self.info = info

    @classmethod
    def load(cls, path: str, info: dict[str, Any] | None = None) -> JacobianLensFull:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in ck:
            raise ValueError(f"{path} is not a JacobianLens file (keys {sorted(ck)})")
        js = {int(k): v for k, v in ck["J"].items()}
        meta = dict(info or {}) | {"source_layers": sorted(js), "d_model": int(ck["d_model"])}
        return cls(js, int(ck.get("n_prompts", 0)), meta)

    @classmethod
    def from_hf(cls, repo: str = NEURONPEDIA_REPO, filename: str = NEURONPEDIA_LLAMA31_8B
                ) -> JacobianLensFull:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, filename)
        lens = cls.load(path, {"repo": f"{repo}/{filename}"})
        lens.info["n_prompts"] = lens.n_prompts
        return lens

    def block_index(self, hidden_index: int) -> int | None:
        """HF hidden_states index -> lens block index (None if not transportable)."""
        if hidden_index <= 0:
            return None
        return hidden_index - 1

    @torch.no_grad()
    def transport(self, h: torch.Tensor, hidden_index: int) -> torch.Tensor:
        """h [N, d] at hidden_states[hidden_index] -> final-layer coordinates J h."""
        b = self.block_index(hidden_index)
        if b is None:
            raise ValueError("no Jacobian for the embedding output")
        if b not in self.jacobians:          # final block output: identity
            return h.float()
        j = self.jacobians[b]
        return (h.float().to(j.device) @ j.float().T)


@dataclass
class ManifoldJLens:
    layer_index: int            # hidden_states index (paper's layer 19 = 20)
    mean: np.ndarray            # [d]
    basis: np.ndarray           # [k, d] orthonormal PCA directions of the manifold
    j_mean: np.ndarray          # [d]  J @ mean
    j_basis: np.ndarray         # [k, d]  J @ v_k
    n_prompts: int
    n_positions: int
    extras: dict[str, Any] = field(default_factory=dict)

    def transport(self, x: np.ndarray) -> np.ndarray:
        """J (mu + V^T c) for activations x [N, d]: project onto the manifold subspace first."""
        c = (x - self.mean) @ self.basis.T            # [N, k]
        return self.j_mean[None, :] + c @ self.j_basis

    def residual_fraction(self, x: np.ndarray) -> float:
        """Share of centered variance of x outside the k-dim subspace (ignored by the lens)."""
        xc = x - self.mean
        proj = (xc @ self.basis.T) @ self.basis
        return float(((xc - proj) ** 2).sum() / ((xc**2).sum() + 1e-12))


class UpperBlocks:
    """Runs blocks l..L-1 of a HF Llama from a given residual h_l (no KV cache), in float32."""

    def __init__(self, full_model: Any, layer_index: int):
        self.fm = full_model
        base = full_model.model.model if hasattr(full_model.model, "model") else full_model.model
        self.base = base
        self.layers = list(base.layers)[layer_index:]
        self.layer_index = layer_index

    @torch.no_grad()
    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """h [B, T, d] residual at hidden_states[layer_index] -> final residual [B, T, d]."""
        base = self.base
        b, t, _ = h.shape
        pos = torch.arange(t, device=h.device)[None, :].expand(b, t)
        cache_position = torch.arange(t, device=h.device)
        pe = base.rotary_emb(h, pos)
        mask = self._causal_mask(attention_mask, h.dtype)
        x = h
        for layer in self.layers:
            out = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=pe,
                        cache_position=cache_position)
            x = out[0] if isinstance(out, tuple) else out
        return x

    @staticmethod
    def _causal_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        b, t = attention_mask.shape
        minv = torch.finfo(dtype).min
        causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=attention_mask.device))
        allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
        return torch.where(allowed, torch.zeros((), dtype=dtype, device=attention_mask.device),
                           torch.full((), minv, dtype=dtype, device=attention_mask.device))


@torch.no_grad()
def jacobian_vector_products(
    upper: UpperBlocks,
    h_corpus: list[torch.Tensor],            # per prompt: [T, d] residual at the lens layer
    masks: list[torch.Tensor],               # per prompt: [T] attention mask
    directions: np.ndarray,                  # [m, d]
    *,
    rel_eps: float = 0.02,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[np.ndarray, int]:
    """Estimate J @ v for every direction v with central differences, following the reference
    reduction (perturb every valid source position, sum over later targets, mean over sources).
    Returns ([m, d], n_valid_positions_total)."""
    m = directions.shape[0]
    acc = torch.zeros(m, directions.shape[1], dtype=torch.float64)
    n_valid_total = 0
    dirs = torch.as_tensor(directions, dtype=torch.float32)
    for h, mask in zip(h_corpus, masks, strict=True):
        t = int(mask.sum())
        valid = torch.zeros(t, dtype=torch.bool)
        valid[skip_first : t - 1] = True
        n_valid = int(valid.sum())
        if n_valid <= 0:
            continue
        dev = upper.fm.input_device
        h32 = h[:t].to(dev, torch.float32)[None]                      # [1, t, d]
        am = mask[:t].to(dev)[None]
        rms = float(h32[0, valid].norm(dim=-1).mean())
        for i in range(m):
            v = dirs[i].to(dev)
            eps = rel_eps * rms / (float(v.norm()) + 1e-12)
            pert = torch.zeros_like(h32)
            pert[0, valid.to(dev)] = eps * v
            plus = upper.forward(h32 + pert, am)[0]
            minus = upper.forward(h32 - pert, am)[0]
            delta = (plus - minus) / (2 * eps)                            # [t, d]
            acc[i] += delta[valid.to(dev)].sum(0).double().cpu() / n_valid
        n_valid_total += n_valid
    return (acc / max(1, len(h_corpus))).numpy().astype(np.float32), n_valid_total


def direction_readout(full_model: Any, transported: np.ndarray, k: int = 12
                      ) -> tuple[list[str], list[str]]:
    """Tokens most promoted / suppressed by moving along a transported direction u
    (linear part of the unembedding, W_U (gamma * u); see FullModel.unembed_direction)."""
    scores = full_model.unembed_direction(torch.as_tensor(transported, dtype=torch.float32)).cpu()
    top = torch.topk(scores, k).indices.tolist()
    bottom = torch.topk(-scores, k).indices.tolist()
    tok = full_model.tokenizer
    return [tok.decode([i]) for i in top], [tok.decode([i]) for i in bottom]


def logit_correlation_readout(logits: np.ndarray, coord: np.ndarray, vocab_decode: Any,
                              k: int = 12, min_prob: float = 1e-4) -> dict[str, list[str]]:
    """Empirical (lens-free) meaning of a coordinate: tokens whose actual next-token log-prob
    correlates most positively / negatively with the coordinate across the manifold's prompts.
    Restricted to tokens with mean probability >= min_prob so rare tokens do not dominate."""
    lp = logits - logits.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    mean_p = np.exp(lp).mean(0)
    cand = np.flatnonzero(mean_p >= min_prob)
    if cand.size == 0:
        return {"positive": [], "negative": []}
    x = (coord - coord.mean()) / (coord.std() + 1e-9)
    y = lp[:, cand]
    y = (y - y.mean(0)) / (y.std(0) + 1e-9)
    r = (y * x[:, None]).mean(0)
    order = np.argsort(r)
    pos = [vocab_decode(int(cand[i])) for i in order[::-1][:k]]
    neg = [vocab_decode(int(cand[i])) for i in order[:k]]
    return {"positive": pos, "negative": neg,
            "r_positive": [float(r[i]) for i in order[::-1][:k]],
            "r_negative": [float(r[i]) for i in order[:k]]}


def pc_readouts(full_model: Any, pcs: np.ndarray, coords: np.ndarray, final_logits: np.ndarray,
                hidden_index: int, *, jlens: JacobianLensFull | None = None,
                tuned: Any | None = None, k: int = 10) -> list[dict[str, Any]]:
    """Decode what each principal axis means.

    pcs [m, d]: unit PCA directions of the manifold at hidden_states[hidden_index];
    coords [N, m]: the prompts' coordinates on them; final_logits [N, vocab]: the model's actual
    next-token logits for the same prompts. For each axis we report tokens promoted/suppressed
    under the logit lens (W_U gamma v), the tuned lens (affine translator, bias dropped), the
    Jacobian lens (W_U gamma J v), and, lens-free, the tokens whose actual log-prob covaries
    most with the coordinate across the manifold.
    """
    tok = full_model.tokenizer
    out = []
    for i in range(pcs.shape[0]):
        v = torch.as_tensor(pcs[i], dtype=torch.float32)
        entry: dict[str, Any] = {"pc": i + 1}
        up, down = direction_readout(full_model, v.numpy(), k)
        entry["logit_lens"] = {"up": up, "down": down}
        if tuned is not None and hidden_index < len(tuned.weights):
            w = tuned.weights[hidden_index]
            vt = v.to(w.device, w.dtype) + v.to(w.device, w.dtype) @ w.T
            up, down = direction_readout(full_model, vt.float().cpu().numpy(), k)
            entry["tuned_lens"] = {"up": up, "down": down}
        if jlens is not None and jlens.block_index(hidden_index) is not None:
            vj = jlens.transport(v[None], hidden_index)[0]
            up, down = direction_readout(full_model, vj.cpu().numpy(), k)
            entry["jacobian_lens"] = {"up": up, "down": down}
        entry["empirical"] = logit_correlation_readout(
            final_logits, coords[:, i], lambda t: tok.decode([t]), k)
        out.append(entry)
    return out
