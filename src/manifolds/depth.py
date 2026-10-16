"""Depth sweep: last-token residual stream at every layer in one forward pass.

Loads the full model (all blocks + final norm + output head). On a 16 GB GPU the fp16 weights do
not fit, so layers that overflow `max_memory` are kept on CPU and streamed to the GPU by accelerate
(exact fp16 arithmetic, slower). Captures hidden_states[0..L] for the last real token of each
prompt: hidden_states[0] is the embedding output and hidden_states[i+1] is the output of block i.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from tqdm import tqdm

from manifolds.extract import _DTYPES, pick_device, resolve_model_id


@dataclass(frozen=True)
class DepthConfig:
    model_id: str
    fallback_model_id: str | None = None
    dtype: str = "float16"
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 256
    gpu_memory_gib: float = 12.0
    seed: int = 0


class FullModel:
    """Causal LM with all blocks, used for the depth sweep, the lenses and steering."""

    def __init__(self, cfg: DepthConfig, *, model: Any | None = None, tokenizer: Any | None = None):
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model_id, self.resolution = resolve_model_id(cfg.model_id, cfg.fallback_model_id)
        self.device = pick_device(cfg.device)
        if model is None:
            major = int(transformers.__version__.split(".")[0])
            kwargs: dict[str, Any] = {}
            if major >= 5:
                kwargs["dtype"] = _DTYPES[cfg.dtype]
            else:
                kwargs["torch_dtype"] = _DTYPES[cfg.dtype]
            if self.device.type == "cuda":
                kwargs["device_map"] = self._device_map(cfg)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if self.device.type != "cuda":
                model = cast(Any, model).to(self.device)
        self.model: Any = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.n_layers = int(self.model.config.num_hidden_layers)
        self.d_model = int(self.model.config.hidden_size)

    # -- helpers ---------------------------------------------------------------------------
    def _device_map(self, cfg: DepthConfig) -> dict[str, Any]:
        """Keep embeddings, the first blocks, the final norm and the output head on the GPU;
        offload the remaining blocks to CPU (accelerate streams their weights to the GPU for
        each forward, exact fp16). Budget: ~1 GB embeddings + ~1 GB head + 0.44 GB per block."""
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(self.model_id)
        n = int(c.num_hidden_layers)
        per_block_gib = (c.hidden_size * (4 * c.hidden_size + 3 * c.intermediate_size)
                         * 2) / 2**30
        fixed_gib = 2 * c.vocab_size * c.hidden_size * 2 / 2**30
        k = int(max(1, min(n, (cfg.gpu_memory_gib - fixed_gib) // per_block_gib)))
        dm: dict[str, Any] = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 0,
                              "lm_head": 0}
        for i in range(n):
            dm[f"model.layers.{i}"] = 0 if i < k else "cpu"
        self.n_blocks_on_gpu = k
        return dm

    @staticmethod
    def exec_device(module: torch.nn.Module) -> torch.device:
        """Device a module computes on (handles accelerate-offloaded modules with meta weights)."""
        hook = getattr(module, "_hf_hook", None)
        if hook is not None and getattr(hook, "execution_device", None) is not None:
            return torch.device(hook.execution_device)
        for p in module.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def input_device(self) -> torch.device:
        return self.exec_device(self.model.get_input_embeddings())

    def encode(self, prompts: list[str]) -> Any:
        return self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                              max_length=self.cfg.max_length).to(self.input_device)

    def token_index(self, prompt: str, span: tuple[int, int] | None) -> int | None:
        if span is None:
            return None
        full = self.tokenizer(prompt, truncation=True, max_length=self.cfg.max_length)["input_ids"]
        pre = self.tokenizer(prompt[: int(span[1])], truncation=True,
                             max_length=self.cfg.max_length)["input_ids"]
        return min(len(pre), len(full)) - 1

    @torch.no_grad()
    def capture_all_layers(self, prompts: list[str], progress: bool = True,
                           spans: list[tuple[int, int]] | None = None) -> np.ndarray:
        """float16 array [N, L+1, d]: residual stream at every depth for the last token (or the
        token ending each character span)."""
        out = np.empty((len(prompts), self.n_layers + 1, self.d_model), dtype=np.float16)
        bs = self.cfg.batch_size
        for s in tqdm(range(0, len(prompts), bs), disable=not progress, desc="depth"):
            batch = prompts[s : s + bs]
            enc = self.encode(batch)
            res = self.model(**enc, output_hidden_states=True)
            last = enc["attention_mask"].sum(dim=1) - 1
            if spans is not None:
                for j, p in enumerate(batch):
                    ti = self.token_index(p, spans[s + j])
                    if ti is not None:
                        last[j] = min(ti, int(last[j]))
            for li, h in enumerate(res.hidden_states):
                rows = torch.arange(h.shape[0], device=h.device)
                sel = h[rows, last.to(h.device)]
                out[s : s + len(batch), li] = sel.to(torch.float16).cpu().numpy()
        return out

    @torch.no_grad()
    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Final norm + output head: [N, d] -> logits [N, vocab] (the logit lens).

        Calls the modules (never reads their weights), so it works with accelerate offload.
        """
        base = self.model.model if hasattr(self.model, "model") else self.model
        norm = base.norm
        head = self.model.get_output_embeddings()
        dev = self.exec_device(norm)
        dtype = next((p.dtype for p in norm.parameters() if p.device.type != "meta"),
                     torch.float16)
        x = norm(h.to(dev, dtype))
        return head(x.to(self.exec_device(head))).float()

    @torch.no_grad()
    def final_norm_gamma(self) -> torch.Tensor:
        """RMSNorm weight gamma, obtained by evaluating the norm on a ones vector (rms = 1)."""
        base = self.model.model if hasattr(self.model, "model") else self.model
        dev = self.exec_device(base.norm)
        return base.norm(torch.ones(1, self.d_model, device=dev, dtype=torch.float32))[0].float()

    @torch.no_grad()
    def unembed_direction(self, u: torch.Tensor) -> torch.Tensor:
        """Linear vocabulary scores of a direction u [d]: W_U (gamma * u). The RMS scale of
        RMSNorm is a positive scalar and does not change the ranking of a direction."""
        head = self.model.get_output_embeddings()
        dev = self.exec_device(head)
        g = self.final_norm_gamma().to(dev)
        dtype = next((p.dtype for p in head.parameters() if p.device.type != "meta"),
                     torch.float16)
        return head((u.to(dev, torch.float32) * g).to(dtype)[None])[0].float()
