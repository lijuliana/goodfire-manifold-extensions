"""Last-token residual-stream extraction (Llama-3.1-8B, block 19), with depth truncation.

The model is loaded with `num_hidden_layers = layer + 1`, so only blocks 0..layer are
instantiated. Because decoder blocks run sequentially, the output of block `layer` is identical
to that of the full model (verified in tests/test_extract_tiny.py on a random tiny Llama).
A forward hook on the block captures its output *before* the final RMSNorm, matching
sae-manifold `data.py` (`model.model.layers[LAYER].output[0]`).
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from manifolds import ledger
from manifolds.prompts.types import PromptSet

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


@dataclass(frozen=True)
class ExtractConfig:
    model_id: str
    layer: int = 19
    dtype: str = "float16"
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 1024
    truncate_to_layer: bool = True
    seed: int = 0
    fallback_model_id: str | None = None
    revision: str | None = None


def pick_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_id(primary: str, fallback: str | None) -> tuple[str, str]:
    """Return (model_id, how); use the ungated mirror only if the primary is unreachable."""
    from transformers import AutoConfig

    try:
        AutoConfig.from_pretrained(primary)
        return primary, "primary"
    except Exception as e:  # gated repo without token, offline, etc.
        if fallback is None:
            raise
        AutoConfig.from_pretrained(fallback)
        return fallback, f"fallback:{type(e).__name__}"


class ActivationExtractor:
    """Loads the (truncated) model once and extracts last-token activations for prompt batches."""

    def __init__(
        self,
        cfg: ExtractConfig,
        *,
        model: torch.nn.Module | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model_id, self.resolution = resolve_model_id(cfg.model_id, cfg.fallback_model_id)
        config = AutoConfig.from_pretrained(self.model_id, revision=cfg.revision)
        self.n_layers_full = int(config.num_hidden_layers)
        if cfg.layer >= self.n_layers_full:
            raise ValueError(f"layer {cfg.layer} >= num_hidden_layers {self.n_layers_full}")
        if cfg.truncate_to_layer:
            config.num_hidden_layers = cfg.layer + 1
        self.device = pick_device(cfg.device)
        if model is None:
            major = int(transformers.__version__.split(".")[0])
            kwargs: dict[str, Any] = {"revision": cfg.revision}
            if major >= 5:
                kwargs["dtype"] = _DTYPES[cfg.dtype]
            else:  # transformers 4.x
                kwargs["torch_dtype"] = _DTYPES[cfg.dtype]
                kwargs["low_cpu_mem_usage"] = True
            model = AutoModel.from_pretrained(self.model_id, config=config, **kwargs)
        self.model: Any = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=cfg.revision)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.commit_sha: str | None = getattr(config, "_commit_hash", None)
        self._captured: dict[int, torch.Tensor] = {}
        for i, blk in enumerate(self._blocks()):
            blk.register_forward_hook(self._make_hook(i))
        self.n_span_mismatch = 0

    # -- internals -------------------------------------------------------------------------
    def _blocks(self) -> list[torch.nn.Module]:
        layers = getattr(self.model, "layers", None)
        if layers is None:  # e.g. a CausalLM wrapper
            layers = self.model.model.layers
        return list(layers)

    def _make_hook(self, index: int) -> Any:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            self._captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    def token_index(self, prompt: str, span: tuple[int, int] | None) -> int | None:
        """Index of the last token covering `span` (character offsets), or None for the last token.

        Uses prefix tokenisation: the number of tokens in prompt[:end] minus one. The prefix's
        tokens are checked to be a prefix of the full prompt's tokens; mismatches are counted."""
        if span is None:
            return None
        end = int(span[1])
        full = self.tokenizer(prompt, truncation=True, max_length=self.cfg.max_length)["input_ids"]
        pre = self.tokenizer(prompt[:end], truncation=True, max_length=self.cfg.max_length)["input_ids"]
        if list(full[: len(pre)]) != list(pre):
            self.n_span_mismatch += 1
        return min(len(pre), len(full)) - 1

    # -- public ----------------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, prompts: list[str], progress: bool = True, *,
                spans: list[tuple[int, int]] | None = None,
                layer: int | None = None) -> np.ndarray:
        """float32 [N, d_model]: activations at block `layer` (default cfg.layer) for the last real
        token, or for the token ending each character span when `spans` is given."""
        layer = self.cfg.layer if layer is None else int(layer)
        if layer >= int(self.model.config.num_hidden_layers):
            raise ValueError(f"block {layer} not loaded (loaded {self.model.config.num_hidden_layers})")
        d = int(self.model.config.hidden_size)
        out = np.empty((len(prompts), d), dtype=np.float32)
        bs = self.cfg.batch_size
        it = range(0, len(prompts), bs)
        for s in tqdm(it, disable=not progress, desc="extract"):
            batch = prompts[s : s + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            ).to(self.device)
            self._captured = {}
            self.model(**enc)
            h = self._captured.get(layer)
            if h is None:
                raise RuntimeError("forward hook did not fire")
            idx = enc["attention_mask"].sum(dim=1) - 1
            if spans is not None:
                for j, p in enumerate(batch):
                    ti = self.token_index(p, spans[s + j])
                    if ti is not None:
                        idx[j] = min(ti, int(idx[j]))
            rows = torch.arange(h.shape[0], device=h.device)
            out[s : s + len(batch)] = h[rows, idx].float().cpu().numpy()
        return out

    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return f"{self.device.type} ({platform.processor() or platform.machine()})"

    def metadata(self) -> dict[str, Any]:
        import transformers

        return {
            "model_id": self.model_id,
            "model_id_resolution": self.resolution,
            "commit_sha": self.commit_sha,
            "layer": self.cfg.layer,
            "n_layers_loaded": int(self.model.config.num_hidden_layers),
            "n_layers_full": self.n_layers_full,
            "dtype": self.cfg.dtype,
            "device": self.device_name(),
            "batch_size": self.cfg.batch_size,
            "max_length": self.cfg.max_length,
            "seed": self.cfg.seed,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "token_position": "last",
        }


def extract_and_cache(
    ps: PromptSet,
    extractor: ActivationExtractor,
    out_dir: Path,
    *,
    layer: int | None = None,
    provider: str = "unknown",
    instance: str = "unknown",
    log_ledger: bool = True,
) -> np.ndarray:
    """Run extraction for a prompt set, write activations + meta, log GPU time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = ledger.now_iso()
    t0 = time.time()
    extractor.n_span_mismatch = 0
    acts = extractor.extract(ps.prompts, spans=ps.spans, layer=layer)
    elapsed = time.time() - t0
    end = ledger.now_iso()
    np.save(out_dir / "activations.npy", acts)
    ps.save(out_dir / "prompts.json")
    meta = extractor.metadata() | {
        "manifold": ps.name,
        "layer": extractor.cfg.layer if layer is None else int(layer),
        "token_position": "last" if ps.spans is None else "concept span (last token of span)",
        "n_span_tokenisation_mismatches": int(extractor.n_span_mismatch),
        "n_prompts": len(ps),
        "elapsed_s": round(elapsed, 2),
        "started": start,
        "finished": end,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    if log_ledger and extractor.device.type == "cuda":
        ledger.append(
            ledger.LedgerEntry(
                provider=provider,
                instance=instance,
                gpu=extractor.device_name(),
                start=start,
                end=end,
                hours=round(elapsed / 3600, 4),
                purpose=f"extract {ps.name} ({len(ps)} prompts)",
            )
        )
    return acts


def precision_check(
    gpu_extractor: ActivationExtractor, prompts: list[str], n: int = 16
) -> dict[str, Any]:
    """Compare the GPU/low-precision extractor against a float32 CPU reference on n prompts."""
    sub = prompts[:n]
    ref_cfg = replace(gpu_extractor.cfg, dtype="float32", device="cpu")
    ref = ActivationExtractor(ref_cfg, tokenizer=gpu_extractor.tokenizer)
    a_ref = ref.extract(sub, progress=False)
    a_gpu = gpu_extractor.extract(sub, progress=False)
    num = (a_ref * a_gpu).sum(1)
    den = np.linalg.norm(a_ref, axis=1) * np.linalg.norm(a_gpu, axis=1) + 1e-12
    cos = num / den
    rel = np.linalg.norm(a_ref - a_gpu, axis=1) / (np.linalg.norm(a_ref, axis=1) + 1e-12)
    del ref
    return {
        "n": len(sub),
        "dtype_test": gpu_extractor.cfg.dtype,
        "dtype_ref": "float32",
        "cosine_min": float(cos.min()),
        "cosine_mean": float(cos.mean()),
        "rel_err_max": float(rel.max()),
        "rel_err_mean": float(rel.mean()),
    }


def config_from_run(run_cfg: Any, device: str = "auto") -> ExtractConfig:
    m = run_cfg.model
    return ExtractConfig(
        model_id=m.id,
        fallback_model_id=m.fallback_id,
        layer=m.layer,
        dtype=m.dtype,
        device=device,
        batch_size=m.batch_size,
        max_length=m.max_length,
        truncate_to_layer=m.truncate_to_layer,
        seed=run_cfg.seed,
    )


__all__ = [
    "ActivationExtractor",
    "ExtractConfig",
    "config_from_run",
    "extract_and_cache",
    "precision_check",
    "resolve_model_id",
]
