"""Inference-only loader for Llama Scope SAEs (OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x).

Reproduces `SparseAutoEncoder.from_pretrained` of Language-Model-SAEs v1.0.0 (the release whose
checkpoint metadata version, 0.1.0, matches these files):
  1. dataset-wise standardisation: b_enc /= f_in, b_dec /= f_out, W_dec *= f_in / f_out with
     f = sqrt(d_model) / dataset_average_activation_norm, after which raw activations are fed in;
  2. unit decoder norm: W_dec columns normalised, encoder rows and bias scaled by the same norms
     (`sparsity_include_decoder_norm`);
  3. JumpReLU: z = h * (h > threshold).
Source: https://github.com/OpenMOSS/Language-Model-SAEs/blob/v1.0.0/src/lm_saes/sae.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_REPO = "OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x"
DEFAULT_SUBFOLDER = "Llama3_1-8B-Base-L19R-8x"


@dataclass
class SAEStats:
    l0_mean: float
    l0_median: float
    ve_total: float        # 1 - |x - x_hat|^2 / |x|^2   (fraction of activation energy rebuilt)
    ve_within: float       # 1 - |x - x_hat|^2 / |x - mean|^2 (relative to within-manifold spread)
    cosine_mean: float
    mse: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {"l0_mean": self.l0_mean, "l0_median": self.l0_median, "ve_total": self.ve_total,
                "ve_within": self.ve_within, "cosine_mean": self.cosine_mean, "mse": self.mse,
                "n": self.n}


class LlamaScopeSAE:
    """Holds W_enc [d_sae, d], b_enc [d_sae], W_dec [d_sae, d] (unit rows), b_dec [d], threshold."""

    def __init__(self, w_enc: torch.Tensor, b_enc: torch.Tensor, w_dec: torch.Tensor,
                 b_dec: torch.Tensor, threshold: float, *, info: dict[str, Any] | None = None):
        if w_enc.shape != w_dec.shape:
            raise ValueError("encoder and decoder must both be [d_sae, d_model]")
        self.w_enc = w_enc
        self.b_enc = b_enc
        self.w_dec = w_dec
        self.b_dec = b_dec
        self.threshold = float(threshold)
        self.info = info or {}

    # ------------------------------------------------------------------ construction ----------
    @classmethod
    def from_state_dict(cls, sd: dict[str, torch.Tensor], hyper: dict[str, Any],
                        device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                        ) -> LlamaScopeSAE:
        d_model = int(hyper["d_model"])
        w_enc = sd["encoder.weight"].to(device=device, dtype=dtype).clone()   # [d_sae, d]
        b_enc = sd["encoder.bias"].to(device=device, dtype=dtype).clone()     # [d_sae]
        w_dec_lin = sd["decoder.weight"].to(device=device, dtype=dtype).clone()  # [d, d_sae]
        b_dec = sd["decoder.bias"].to(device=device, dtype=dtype).clone()     # [d]
        if hyper.get("norm_activation") == "dataset-wise":
            avg = hyper["dataset_average_activation_norm"]
            f_in = math.sqrt(d_model) / float(avg["in"])
            f_out = math.sqrt(d_model) / float(avg["out"])
            b_enc = b_enc / f_in
            b_dec = b_dec / f_out
            w_dec_lin = w_dec_lin * (f_in / f_out)
        if hyper.get("sparsity_include_decoder_norm", False):
            dn = torch.norm(w_dec_lin, p=2, dim=0)          # per-feature decoder norm
            w_dec_lin = w_dec_lin / dn
            w_enc = w_enc * dn[:, None]
            b_enc = b_enc * dn
        act = str(hyper.get("act_fn", "jumprelu")).lower()
        thr = float(hyper.get("jump_relu_threshold", 0.0))
        if act not in ("jumprelu", "topk"):
            raise NotImplementedError(f"act_fn {act!r}")
        if act == "topk" and thr <= 0:
            raise NotImplementedError("TopK without a JumpReLU threshold is not supported")
        info = {"d_model": d_model, "d_sae": int(w_enc.shape[0]), "act_fn": "jumprelu",
                "threshold": thr, "hook_point": hyper.get("hook_point_in"),
                "top_k_trained": hyper.get("top_k")}
        return cls(w_enc, b_enc, w_dec_lin.T.contiguous(), b_dec, thr, info=info)

    @classmethod
    def from_hf(cls, repo: str = DEFAULT_REPO, subfolder: str = DEFAULT_SUBFOLDER,
                device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
                ) -> LlamaScopeSAE:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        hp_path = hf_hub_download(repo, f"{subfolder}/hyperparams.json")
        ck_path = hf_hub_download(repo, f"{subfolder}/checkpoints/final.safetensors")
        hyper = json.loads(Path(hp_path).read_text(encoding="utf-8"))
        sd = load_file(ck_path)
        sae = cls.from_state_dict(sd, hyper, device=device, dtype=dtype)
        sae.info["repo"] = f"{repo}/{subfolder}"
        return sae

    # ------------------------------------------------------------------ inference -------------
    @property
    def device(self) -> torch.device:
        return self.w_enc.device

    @torch.no_grad()
    def encode(self, x: torch.Tensor | np.ndarray, batch_size: int = 1024) -> torch.Tensor:
        """Raw activations [N, d] -> JumpReLU codes [N, d_sae] (dense tensor on self.device)."""
        xt = torch.as_tensor(x).to(self.device, self.w_enc.dtype)
        out = []
        for s in range(0, xt.shape[0], batch_size):
            h = xt[s : s + batch_size] @ self.w_enc.T + self.b_enc
            out.append(h.where(h > self.threshold, torch.zeros_like(h)))
        return torch.cat(out)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.w_dec + self.b_dec

    @torch.no_grad()
    def stats(self, x: torch.Tensor | np.ndarray, z: torch.Tensor | None = None) -> SAEStats:
        xt = torch.as_tensor(x).to(self.device, self.w_enc.dtype)
        if z is None:
            z = self.encode(xt)
        x_hat = self.decode(z)
        l0 = (z > 0).sum(1).float()
        resid = ((xt - x_hat) ** 2).sum()
        energy = (xt**2).sum()
        within = ((xt - xt.mean(0)) ** 2).sum()
        cos = torch.nn.functional.cosine_similarity(xt, x_hat, dim=1).mean()
        return SAEStats(float(l0.mean()), float(l0.median()), float(1 - resid / energy),
                        float(1 - resid / within), float(cos),
                        float(((xt - x_hat) ** 2).mean()), int(xt.shape[0]))

    def decoder_numpy(self) -> np.ndarray:
        return self.w_dec.detach().float().cpu().numpy()
