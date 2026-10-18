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
    def capture_layer_and_logits(
        self, prompts: list[str], layer_index: int, *, spans: list[tuple[int, int]] | None = None,
        cand_ids: list[int] | None = None, full_index: np.ndarray | None = None,
        mlp_blocks: list[int] | None = None, progress: bool = True,
    ) -> dict[str, Any]:
        """One pass per batch: the residual stream at hidden-state position `layer_index`
        (0 = embeddings, i = output of block i-1) for the read token (last real token, or the
        token ending each character span), plus the next-token readouts at the last token:
        greedy token id, the logits of `cand_ids`, and the full logits (float16) for the prompts
        in `full_index`. With `mlp_blocks`, also the SwiGLU activations (input of down_proj) of
        those blocks at the read token, float16 [N, len(mlp_blocks), d_ff]."""
        n = len(prompts)
        hidden = np.empty((n, self.d_model), dtype=np.float32)
        top1 = np.empty(n, dtype=np.int64)
        cand = np.empty((n, len(cand_ids)), dtype=np.float32) if cand_ids else None
        full_pos = {int(i): j for j, i in enumerate(full_index)} if full_index is not None else {}
        full: np.ndarray | None = None
        mlp_acts: np.ndarray | None = None
        captured: dict[int, torch.Tensor] = {}
        handles = []
        if mlp_blocks:
            base = self.model.model if hasattr(self.model, "model") else self.model

            def mk(bi: int) -> Any:
                def hook(_m: torch.nn.Module, args: Any) -> None:
                    captured[bi] = args[0]
                return hook
            for bi in mlp_blocks:
                handles.append(base.layers[bi].mlp.down_proj.register_forward_pre_hook(mk(bi)))
        bs = self.cfg.batch_size
        try:
            for s in tqdm(range(0, n, bs), disable=not progress, desc="capture"):
                batch = prompts[s : s + bs]
                enc = self.encode(batch)
                res = self.model(**enc, output_hidden_states=True)
                last = enc["attention_mask"].sum(dim=1) - 1
                read = last.clone()
                if spans is not None:
                    for j, p in enumerate(batch):
                        ti = self.token_index(p, spans[s + j])
                        if ti is not None:
                            read[j] = min(ti, int(last[j]))
                h = res.hidden_states[layer_index]
                rows = torch.arange(h.shape[0], device=h.device)
                hidden[s : s + len(batch)] = h[rows, read.to(h.device)].float().cpu().numpy()
                lg = res.logits
                rows_l = torch.arange(lg.shape[0], device=lg.device)
                ll = lg[rows_l, last.to(lg.device)].float()
                top1[s : s + len(batch)] = ll.argmax(1).cpu().numpy()
                if cand is not None:
                    cand[s : s + len(batch)] = ll[:, cand_ids].cpu().numpy()
                if full_pos:
                    for j in range(len(batch)):
                        k = full_pos.get(s + j)
                        if k is not None:
                            if full is None:
                                full = np.empty((len(full_pos), ll.shape[1]), dtype=np.float16)
                            full[k] = ll[j].to(torch.float16).cpu().numpy()
                if mlp_blocks:
                    if mlp_acts is None:
                        d_ff = captured[mlp_blocks[0]].shape[-1]
                        mlp_acts = np.empty((n, len(mlp_blocks), d_ff), dtype=np.float16)
                    for bj, bi in enumerate(mlp_blocks):
                        a = captured[bi]
                        ra = torch.arange(a.shape[0], device=a.device)
                        sel = a[ra, read.to(a.device)]
                        mlp_acts[s : s + len(batch), bj] = sel.to(torch.float16).cpu().numpy()
                    captured.clear()
        finally:
            for hd in handles:
                hd.remove()
        return {"hidden": hidden, "top1": top1, "cand": cand, "full": full, "mlp": mlp_acts}

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
