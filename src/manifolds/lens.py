"""Logit lens and tuned lens readouts of the last-token residual stream at every depth.

Logit lens (nostalgebraist 2020): apply the model's final norm and output head directly to an
intermediate hidden state. Tuned lens (Belrose et al. 2023, arXiv 2303.08112): first apply a
learned affine translator for that depth, h' = h + W_i h + b_i, then the same unembedding.
Translator i is trained on hidden_states[i] in HF numbering (hidden_states[0] = embeddings,
hidden_states[i] = output of block i-1), per tuned-lens `train_loop.py`; the final hidden state
needs no translator. Pretrained lenses for Llama-3.1-8B: kdhole/Llama-3.1-8B-tuned-lens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_TUNED_LENS_REPO = "kdhole/Llama-3.1-8B-tuned-lens"


class TunedLens:
    def __init__(self, weights: list[torch.Tensor], biases: list[torch.Tensor | None],
                 info: dict[str, Any]):
        self.weights = weights      # [L] tensors [d, d]
        self.biases = biases        # [L] tensors [d] or None
        self.info = info

    def __len__(self) -> int:
        return len(self.weights)

    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor], config: dict[str, Any],
                        device: torch.device | str = "cpu", dtype: torch.dtype = torch.float16
                        ) -> TunedLens:
        n = int(config["num_hidden_layers"])
        ws, bs = [], []
        for i in range(n):
            w = sd[f"{i}.weight"].to(device=device, dtype=dtype)
            b = sd.get(f"{i}.bias")
            ws.append(w)
            bs.append(b.to(device=device, dtype=dtype) if b is not None else None)
        return cls(ws, bs, dict(config))

    @classmethod
    def from_hf(cls, repo: str = DEFAULT_TUNED_LENS_REPO, device: torch.device | str = "cpu",
                dtype: torch.dtype = torch.float16) -> TunedLens:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text(encoding="utf-8"))
        sd = torch.load(hf_hub_download(repo, "params.pt"), map_location="cpu")
        lens = cls.from_state_dict(sd, cfg, device=device, dtype=dtype)
        lens.info["repo"] = repo
        return lens

    @torch.no_grad()
    def transform(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        """h [N, d] at hidden_states[idx] -> translated state in final-layer coordinates."""
        if idx >= len(self.weights):   # final hidden state: identity
            return h
        w, b = self.weights[idx], self.biases[idx]
        x = h.to(w.device, w.dtype)
        out = x + x @ w.T
        return out + b if b is not None else out


@torch.no_grad()
def top_tokens(full_model: Any, h: torch.Tensor, k: int = 3, batch_size: int = 256
               ) -> tuple[np.ndarray, np.ndarray]:
    """Apply final norm + output head to [N, d] states; return top-k token ids and probs."""
    ids, probs = [], []
    for s in range(0, h.shape[0], batch_size):
        logits = full_model.unembed(h[s : s + batch_size])
        p = torch.softmax(logits, dim=-1)
        tp, ti = torch.topk(p, k, dim=-1)
        ids.append(ti.cpu().numpy())
        probs.append(tp.cpu().numpy())
    return np.concatenate(ids), np.concatenate(probs)


def decode_tokens(tokenizer: Any, ids: np.ndarray) -> list[list[str]]:
    cache: dict[int, str] = {}

    def one(i: int) -> str:
        if i not in cache:
            cache[i] = tokenizer.decode([int(i)])
        return cache[i]

    return [[one(int(i)) for i in row] for row in ids]


@torch.no_grad()
def lens_readout(full_model: Any, states: np.ndarray, layers: list[int],
                 tuned: TunedLens | None = None, k: int = 3,
                 jlens: Any | None = None) -> dict[str, Any]:
    """states [N, L+1, d] float16 -> {"logit": {layer: {"tokens": [[..k..]], "probs": [[..]]}},
    "tuned": {...}, "jacobian": {...}} for the requested hidden-state indices."""
    out: dict[str, Any] = {"logit": {}, "tuned": {}, "jacobian": {}, "layers": layers, "k": k}
    tok = full_model.tokenizer
    for li in layers:
        h = torch.as_tensor(states[:, li].astype(np.float32))
        ids, pr = top_tokens(full_model, h, k)
        out["logit"][str(li)] = {"tokens": decode_tokens(tok, ids),
                                 "probs": np.round(pr, 4).tolist()}
        if tuned is not None:
            ht = tuned.transform(h, li)
            ids, pr = top_tokens(full_model, ht, k)
            out["tuned"][str(li)] = {"tokens": decode_tokens(tok, ids),
                                     "probs": np.round(pr, 4).tolist()}
        if jlens is not None and jlens.block_index(li) is not None:
            hj = jlens.transport(h, li)
            ids, pr = top_tokens(full_model, hj, k)
            out["jacobian"][str(li)] = {"tokens": decode_tokens(tok, ids),
                                       "probs": np.round(pr, 4).tolist()}
    return out
