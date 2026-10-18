"""Manifold steering (Bhalla et al. 2026 App. B.1, Fig. 2 right).

Fit PCA on the cached layer-19 activations of a manifold (enough components for 90% variance),
bin the primary label, compute bin centroids in PCA space, interpolate between consecutive
centroids, map each interpolation point back to activation space as a delta from the manifold
mean, add that delta to the base prompt's layer-19 last-token activation during a forward pass,
and read the next-token probabilities of diagnostic target tokens (Table 2 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

# Table 2 of the paper: task suffix appended to the base prompt and diagnostic target tokens.
STEER_TASKS: dict[str, dict[str, Any]] = {
    "temperature": {"label": "fahrenheit", "suffix": ", which is",
                    "targets": ["freezing", "cold", "chilly", "nice", "warm", "hot", "burning"],
                    "n_bins": 7},
    "age": {"label": "age", "suffix": " They are", "targets": ["young", "middle", "old", "elderly"],
            "n_bins": 6},
    # Llama 3 writes a year as a 3-digit chunk after a separate space token (" 1899" ->
    # " ", "189", "9"), so the paper's century targets "17".."21" are read here as the summed
    # probability of the chunks "170".."179" etc. after "It is " (deviation; the literal recipe
    # gave identical curves for all five targets because their first token is the bare space).
    "years": {"label": "year", "suffix": "-01-01, which year is it? It is ",
              "targets": ["17", "18", "19", "20", "21"], "n_bins": 8, "digit_chunks": True},
    "colors": {"label": "hue", "suffix": "",
               "targets": ["red", "orange", "yellow", "green", "blue", "indigo", "purple"],
               "n_bins": 8},
    "days": {"label": "day_idx", "suffix": ". What day is it?",
             "targets": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                         "Sunday"], "n_bins": 7},
    # ours (not in the paper's Table 2): targets chosen as the natural answer tokens
    "political_bias": {"label": "bias_label", "suffix": "",
                       "targets": ["conservative", "neutral", "liberal"], "n_bins": 5,
                       "ours": True},
    # the circle is read at the concept token, so it is steered there; the answer is read at the
    # last token (concept + offset), as in Feucht et al.'s arithmetic prompts
    "weekdays_circle": {"label": "day_idx", "suffix": "", "at_span": True, "ours": True,
                        "targets": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                    "Saturday", "Sunday"], "n_bins": 7},
    "months_circle": {"label": "month_idx", "suffix": "", "at_span": True, "ours": True,
                      "targets": ["January", "February", "March", "April", "May", "June", "July",
                                  "August", "September", "October", "November", "December"],
                      "n_bins": 12},
}


@dataclass
class SteerResult:
    manifold: str
    base_prompt: str
    bin_edges: list[float]
    bin_centers: list[float]
    positions: list[float]         # interpolation coordinate in bin units (0 .. n_bins-1)
    targets: list[str]
    probs: list[list[float]]       # [n_positions, n_targets]
    pca_components_used: int
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


def plan_interpolation(acts: np.ndarray, label: np.ndarray, n_bins: int, n_between: int = 6,
                       var_threshold: float = 0.90) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                               np.ndarray, int]:
    """Return (deltas [P, d], positions [P], bin_edges, bin_centers, n_components)."""
    x = acts.astype(np.float64)
    mean = x.mean(0)
    pca = PCA(n_components=min(x.shape[0] - 1, x.shape[1]), svd_solver="full").fit(x)
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold) + 1)
    comps = pca.components_[:k]
    z = (x - mean) @ comps.T
    edges = np.linspace(label.min(), label.max(), n_bins + 1)
    idx = np.clip(np.digitize(label, edges[1:-1]), 0, n_bins - 1)
    cents = np.array([z[idx == b].mean(0) if (idx == b).any() else np.full(k, np.nan)
                      for b in range(n_bins)])
    ok = ~np.isnan(cents[:, 0])
    cents = cents[ok]
    centers = (edges[:-1] + edges[1:]) / 2
    centers = centers[ok]
    pts: list[np.ndarray] = []
    pos: list[float] = []
    for b in range(len(cents) - 1):
        for t in np.linspace(0, 1, n_between, endpoint=False):
            pts.append((1 - t) * cents[b] + t * cents[b + 1])
            pos.append(float(b + t))
    pts.append(cents[-1])
    pos.append(float(len(cents) - 1))
    deltas = np.array(pts) @ comps  # back to activation space, relative to the manifold mean
    return deltas.astype(np.float32), np.array(pos), edges, centers, k


class Steerer:
    """Adds a delta to the last-token residual stream at `layer` during a forward pass."""

    def __init__(self, full_model: Any, layer: int):
        self.fm = full_model
        self.layer = layer
        self._delta: torch.Tensor | None = None
        self._last: torch.Tensor | None = None
        base = self.fm.model.model if hasattr(self.fm.model, "model") else self.fm.model
        self._handle: Any = base.layers[layer].register_forward_hook(self._hook)

    def _hook(self, _m: Any, _i: Any, output: Any) -> Any:
        if self._delta is None or self._last is None:
            return None
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        rows = torch.arange(h.shape[0], device=h.device)
        h[rows, self._last.to(h.device)] += self._delta.to(h.device, h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def target_ids(self, targets: list[str], digit_chunks: bool = False) -> list[list[int]]:
        """Token ids per target: the first token of " target", or with `digit_chunks` every
        3-digit token starting with the target (e.g. "18" -> "180".."189")."""
        tok = self.fm.tokenizer
        out: list[list[int]] = []
        for t in targets:
            if digit_chunks:
                ids = []
                for d in range(10):
                    enc = tok.encode(f"{t}{d}", add_special_tokens=False)
                    if len(enc) != 1:
                        raise ValueError(f"{t}{d} is not a single token")
                    ids.append(int(enc[0]))
                out.append(ids)
            else:
                out.append([int(tok.encode(" " + t, add_special_tokens=False)[0])])
        flat = [i[0] for i in out if len(i) == 1]
        if len(set(flat)) != len(flat):
            raise ValueError(f"targets share a first token: {targets}")
        return out

    @torch.no_grad()
    def next_token_probs(self, prompt: str, deltas: np.ndarray, targets: list[str],
                         *, steer_index: int | None = None, digit_chunks: bool = False
                         ) -> np.ndarray:
        """[P, T] next-token probability of each target after adding each delta at
        `steer_index` (default: the last token). The answer is always read at the last token."""
        groups = self.target_ids(targets, digit_chunks)
        enc = self.fm.encode([prompt])
        last = enc["attention_mask"].sum(dim=1) - 1
        pos = last if steer_index is None else torch.tensor([int(steer_index)])
        out = np.zeros((len(deltas), len(targets)), dtype=np.float32)
        for i, d in enumerate(deltas):
            self._delta = torch.as_tensor(d)[None, :]
            self._last = pos
            logits = self.fm.model(**enc).logits[0, int(last[0])].float()
            p = torch.softmax(logits, dim=-1)
            out[i] = [float(p[g].sum()) for g in groups]
        self._delta = None
        return out

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def run_steering(full_model: Any, name: str, acts: np.ndarray, labels: list[dict[str, Any]],
                 prompts: list[str], layer: int = 19, n_between: int = 6,
                 spans: list[tuple[int, int]] | None = None) -> SteerResult:
    task = STEER_TASKS[name]
    lab = np.asarray([lb[task["label"]] for lb in labels], dtype=float)
    deltas, pos, edges, centers, k = plan_interpolation(acts, lab, task["n_bins"], n_between)
    mid = int(np.argmin(np.abs(lab - np.median(lab))))   # base prompt near the manifold midpoint
    base_prompt = prompts[mid] + task["suffix"]
    steer_index = None
    if task.get("at_span"):
        if spans is None:
            raise ValueError(f"{name}: span steering needs the concept spans")
        steer_index = full_model.token_index(base_prompt, spans[mid])
    st = Steerer(full_model, layer)
    # delta relative to the manifold mean, applied on top of the base prompt's own activation:
    # add (target point - base point) so the base prompt is moved *to* the interpolated point.
    base_vec = acts[mid].astype(np.float32) - acts.mean(0).astype(np.float32)
    try:
        probs = st.next_token_probs(base_prompt, deltas - base_vec[None, :], task["targets"],
                                    steer_index=steer_index,
                                    digit_chunks=bool(task.get("digit_chunks")))
    finally:
        st.remove()
    extras = {"base_label": float(lab[mid]), "n_bins_used": int(len(centers)),
              "steered_position": "concept token" if steer_index is not None else "last token",
              "targets_ours": bool(task.get("ours")),
              "readout": ("summed probability of the 3-digit year chunks starting with each "
                          "target" if task.get("digit_chunks") else "first token of each target")}
    if "answer" in labels[mid]:
        extras["base_answer"] = str(labels[mid]["answer"])
    return SteerResult(name, base_prompt, edges.tolist(), centers.tolist(), pos.tolist(),
                       task["targets"], probs.tolist(), k, extras)
