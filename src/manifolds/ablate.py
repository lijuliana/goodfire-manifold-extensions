# ruff: noqa: E501
"""Weak-activation ablation against manifold geometry (research plan, idea 1).

Perturbation (ported from Juliana Li's weak-activation-ablation repo, `weakabl/ablation.py`):
the "neuron activation" is the SwiGLU product act(gate(x)) * up(x), i.e. the input of
`down_proj`, in every MLP block. Each neuron's scale is its max |activation| over 512 x 1024
tokens of FineWeb-edu (position 0 excluded). A condition zeroes every activation with
|a| < X% of its neuron's scale ("weak"), zeroes the same number per (layer, token) at random
("random", count-matched), or multiplies the sub-threshold tail by s instead of zeroing it
("scale"). The ablation is applied in all blocks, so the manifold at the read layer and the
model's behaviour are measured under the same perturbation.

Readouts per manifold and condition: the manifold re-read at the paper's layer (shape check on a
fresh PCA; projection onto the clean PCA basis for the slider; k-NN overlap with the clean
cloud; displacement), the task accuracy where a task is defined (`manifolds.tasks`), and KL /
top-1 agreement with the clean model's next-token distribution. Per condition, model-wide:
held-out LM loss (FineWeb-edu, disjoint from calibration) and 8-shot PopQA accuracy on a fixed
300-question sample (Mallen et al. 2023 metric), so a manifold's breaking point can be placed
against a facts-and-loss baseline.
"""

from __future__ import annotations

import json
import re
import string
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from tqdm import tqdm

from manifolds import RESULTS_DIR, ledger
from manifolds.analysis.compare import compare, prepare, summary_metric
from manifolds.analysis.pca import filter_norm_outliers, fit_pca
from manifolds.analysis.shapes import ROLES
from manifolds.config import RunConfig
from manifolds.manifest import ManifoldSpec
from manifolds.prompts.types import PromptSet
from manifolds.tasks import first_token_ids, kl_and_agreement, score_task, task_for
from manifolds.viz.plot3d import CATEGORICAL, frames_figure, line_layout, subsample

WEAK_X = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]
RANDOM_X = [0.1, 0.25, 0.5, 1.0]
SCALE_CONDITIONS = [(0.5, 0.5), (1.5, 0.5), (0.5, 2.0), (1.5, 2.0)]   # (X, s)
GLOBAL_DIR = RESULTS_DIR / "_ablation"


@dataclass(frozen=True)
class Condition:
    key: str
    group: str        # clean | weak | random | scale
    mode: str         # off | threshold | random | scaleweak
    x: float          # threshold in % of the neuron's max
    scale: float = 0.0
    seed: int = 0

    @property
    def label(self) -> str:
        if self.group == "clean":
            return "clean"
        if self.group == "weak":
            return f"{self.x:g}"
        if self.group == "random":
            return f"r{self.x:g}"
        return f"×{self.scale:g}@{self.x:g}"

    @property
    def description(self) -> str:
        if self.group == "clean":
            return "unmodified model"
        if self.group == "weak":
            return f"weak activations zeroed below {self.x:g}% of each neuron's max"
        if self.group == "random":
            return f"count-matched random activations zeroed (matched to {self.x:g}%)"
        return f"weak tail (below {self.x:g}%) multiplied by {self.scale:g}"


def conditions() -> list[Condition]:
    out = [Condition("clean", "clean", "off", 0.0)]
    out += [Condition(f"weak_{x:g}", "weak", "threshold", x) for x in WEAK_X]
    out += [Condition(f"random_{x:g}", "random", "random", x) for x in RANDOM_X]
    out += [Condition(f"scale_{s:g}_{x:g}", "scale", "scaleweak", x, s) for x, s in SCALE_CONDITIONS]
    return out


# ---- the hook (port of weakabl.ablation.Ablator) ---------------------------------------------
def mlp_down_projs(model: Any) -> list[torch.nn.Module]:
    base = model.model if hasattr(model, "model") else model
    return [layer.mlp.down_proj for layer in base.layers]


class Ablator:
    """Forward pre-hooks on every `down_proj`. Modes: off | calibrate | threshold | random |
    scaleweak (see module docstring). `frac` is a fraction of the neuron's max (X / 100)."""

    def __init__(self, model: Any, seed: int = 0) -> None:
        self.modules = mlp_down_projs(model)
        self.n_layers = len(self.modules)
        self.mode = "off"
        self.frac = 0.0
        self.scale = 0.0
        self.track = False
        self.seed = seed
        self.amax: list[torch.Tensor] | None = None
        self.calib_max: list[torch.Tensor | None] = [None] * self.n_layers
        self.zeroed = np.zeros(self.n_layers)
        self.total = np.zeros(self.n_layers)
        self._gen: torch.Generator | None = None
        self._handles = [m.register_forward_pre_hook(self._make_hook(i))
                         for i, m in enumerate(self.modules)]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def set(self, mode: str, frac: float = 0.0, *, scale: float = 0.0, seed: int | None = None,
            track: bool = False) -> None:
        assert mode in ("off", "calibrate", "threshold", "random", "scaleweak"), mode
        if mode in ("threshold", "random", "scaleweak"):
            assert self.amax is not None, "calibrate first"
        self.mode, self.frac, self.scale, self.track = mode, float(frac), float(scale), track
        if seed is not None:
            self.seed = seed
        self._gen = None
        self.zeroed[:] = 0
        self.total[:] = 0

    def set_condition(self, c: Condition, track: bool = False) -> None:
        self.set(c.mode, c.x / 100.0, scale=c.scale, seed=c.seed, track=track)

    def load_calibration(self, amax: torch.Tensor, device: torch.device | str) -> None:
        assert amax.shape[0] == self.n_layers
        self.amax = [amax[i].to(device=device, dtype=torch.float32) for i in range(self.n_layers)]

    def calibration_tensor(self) -> torch.Tensor:
        assert all(t is not None for t in self.calib_max), "calibration not run"
        return torch.stack([t.float().cpu() for t in self.calib_max])  # type: ignore[union-attr]

    def fraction_zeroed(self) -> float:
        t = self.total.sum()
        return float(self.zeroed.sum() / t) if t else 0.0

    def _generator(self, device: torch.device) -> torch.Generator:
        if self._gen is None or self._gen.device != device:
            self._gen = torch.Generator(device=device).manual_seed(self.seed)
        return self._gen

    def _make_hook(self, i: int) -> Any:
        def hook(_module: torch.nn.Module, args: Any) -> Any:
            h = args[0]
            if self.mode == "off" or (self.mode != "calibrate" and self.frac == 0.0):
                return None
            if self.mode == "calibrate":
                if h.shape[1] > 1:
                    m = h[:, 1:].detach().abs().amax(dim=(0, 1)).float()
                    prev = self.calib_max[i]
                    self.calib_max[i] = m if prev is None else torch.maximum(prev, m)
                return None
            assert self.amax is not None
            thr = (self.amax[i] * self.frac).to(h.dtype)
            mask = h.abs() < thr
            if self.mode == "random":
                k = mask.sum(dim=-1, keepdim=True)
                r = torch.rand(h.shape, generator=self._generator(h.device), device=h.device)
                kth = r.sort(dim=-1).values.gather(-1, (k - 1).clamp(min=0))
                mask = (r <= kth) & (k > 0)
            if self.track:
                self.zeroed[i] += float(mask.sum())
                self.total[i] += float(mask.numel())
            if self.mode == "scaleweak":
                return (torch.where(mask, h * self.scale, h),)
            return (h.masked_fill(mask, 0),)
        return hook


# ---- calibration / held-out text / PopQA -----------------------------------------------------
def _chunk_stream(token_lists: Any, seq_len: int, n_seqs: int) -> np.ndarray:
    buf: list[int] = []
    out: list[list[int]] = []
    for toks in token_lists:
        buf.extend(toks)
        while len(buf) >= seq_len and len(out) < n_seqs:
            out.append(buf[:seq_len])
            buf = buf[seq_len:]
        if len(out) >= n_seqs:
            break
    if len(out) != n_seqs:
        raise RuntimeError(f"only {len(out)} / {n_seqs} sequences")
    return np.asarray(out, dtype=np.int32)


def fineweb_chunks(tokenizer: Any, seq_len: int, n_calib: int, n_heldout: int,
                   cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint calibration / held-out chunks of HuggingFaceFW/fineweb-edu (sample-10BT), read in
    stream order (deterministic). Same construction as weakabl/data.py."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fwedu_s{seq_len}_c{n_calib}_h{n_heldout}"
    fc, fh = cache_dir / f"{tag}_calib.npy", cache_dir / f"{tag}_heldout.npy"
    if fc.exists() and fh.exists():
        return np.load(fc), np.load(fh)
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",
                      streaming=True)
    it = iter(ds)

    def tok_iter() -> Any:
        for ex in it:
            yield tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + \
                [tokenizer.eos_token_id]

    gen = tok_iter()
    calib = _chunk_stream(gen, seq_len, n_calib)
    heldout = _chunk_stream(gen, seq_len, n_heldout)
    np.save(fc, calib)
    np.save(fh, heldout)
    return calib, heldout


@torch.no_grad()
def calibrate(fm: Any, ablator: Ablator, chunks: np.ndarray, batch_size: int = 4) -> torch.Tensor:
    ablator.set("calibrate")
    for s in tqdm(range(0, len(chunks), batch_size), desc="calibrate"):
        ids = torch.from_numpy(chunks[s : s + batch_size].astype(np.int64)).to(fm.input_device)
        fm.model(input_ids=ids)
    ablator.set("off")
    return ablator.calibration_tensor()


@torch.no_grad()
def lm_loss(fm: Any, chunks: np.ndarray, batch_size: int = 4) -> float:
    """Mean next-token NLL (nats) over the chunks."""
    tot, n = 0.0, 0
    for s in range(0, len(chunks), batch_size):
        ids = torch.from_numpy(chunks[s : s + batch_size].astype(np.int64)).to(fm.input_device)
        logits = fm.model(input_ids=ids).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                              reduction="sum")
        tot += float(nll)
        n += int(ids[:, 1:].numel())
    return tot / n


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def popqa_correct(generation: str, aliases: list[str]) -> bool:
    g = _normalize(generation)
    return any(_normalize(a) and _normalize(a) in g for a in aliases)


def load_popqa_sample(n: int, n_shot: int, seed: int, cache_dir: Path) -> dict[str, Any]:
    """A fixed sample of PopQA questions with an 8-shot prefix built from other questions."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"popqa_n{n}_shot{n_shot}_seed{seed}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split="test")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    ex, qs = order[:n_shot], order[n_shot : n_shot + n]

    def answers(row: Any) -> list[str]:
        v = row["possible_answers"]
        try:
            v = json.loads(v) if isinstance(v, str) else v
        except json.JSONDecodeError:
            v = [v]
        return [str(a) for a in (v if isinstance(v, list) else [v])]

    prefix = "".join(f"Q: {ds[int(i)]['question']}\nA: {answers(ds[int(i)])[0]}\n\n" for i in ex)
    items = [{"id": int(ds[int(i)]["id"]), "question": ds[int(i)]["question"],
              "answers": answers(ds[int(i)]), "s_pop": int(ds[int(i)]["s_pop"])} for i in qs]
    out = {"n_shot": n_shot, "seed": seed, "prefix": prefix, "items": items,
           "source": "akariasai/PopQA test split; metric of Mallen et al. 2023"}
    f.write_text(json.dumps(out), encoding="utf-8")
    return out


@torch.no_grad()
def popqa_accuracy(fm: Any, sample: dict[str, Any], batch_size: int = 32,
                   max_new_tokens: int = 16) -> dict[str, Any]:
    tok = fm.tokenizer
    prompts = [sample["prefix"] + f"Q: {it['question']}\nA:" for it in sample["items"]]
    side = tok.padding_side
    tok.padding_side = "left"
    gens: list[str] = []
    try:
        for s in range(0, len(prompts), batch_size):
            enc = tok(prompts[s : s + batch_size], return_tensors="pt", padding=True
                      ).to(fm.input_device)
            out = fm.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=tok.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            gens += [g.split("\n")[0].strip() for g in tok.batch_decode(new, skip_special_tokens=True)]
    finally:
        tok.padding_side = side
    correct = [popqa_correct(g, it["answers"]) for g, it in zip(gens, sample["items"], strict=True)]
    return {"accuracy": float(np.mean(correct)), "n": len(correct), "correct": correct}


# ---- model-wide readouts per condition -------------------------------------------------------
def run_global(fm: Any, ablator: Ablator, conds: list[Condition], *, out_dir: Path = GLOBAL_DIR,
               n_calib: int = 512, n_heldout: int = 64, seq_len: int = 1024, n_popqa: int = 300,
               lm_batch: int = 4, provider: str = "unknown", instance: str = "unknown"
               ) -> dict[str, Any]:
    """Calibrate (cached), then LM loss + PopQA for every condition (cached per condition)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = ledger.now_iso()
    t0 = time.time()
    calib, heldout = fineweb_chunks(fm.tokenizer, seq_len, n_calib, n_heldout, out_dir / "cache")
    calib_path = out_dir / "calibration_amax.pt"
    if calib_path.exists():
        amax = torch.load(calib_path)
    else:
        amax = calibrate(fm, ablator, calib, lm_batch)
        torch.save(amax, calib_path)
    ablator.load_calibration(amax, fm.input_device)
    sample = load_popqa_sample(n_popqa, 8, 0, out_dir / "cache")
    res_path = out_dir / "global.json"
    res: dict[str, Any] = json.loads(res_path.read_text()) if res_path.exists() else {}
    res.setdefault("conditions", {})
    for c in conds:
        if c.key in res["conditions"]:
            continue
        ablator.set_condition(c, track=True)
        t1 = time.time()
        loss = lm_loss(fm, heldout, lm_batch)
        frac = ablator.fraction_zeroed()
        ablator.set_condition(c, track=False)
        pq = popqa_accuracy(fm, sample)
        res["conditions"][c.key] = {
            **asdict(c), "label": c.label, "description": c.description,
            "lm_loss": loss, "fraction_zeroed": frac, "popqa_accuracy": pq["accuracy"],
            "popqa_n": pq["n"], "elapsed_s": round(time.time() - t1, 1),
        }
        res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"{c.key:14s} loss={loss:.4f} zeroed={frac:.3f} popqa={pq['accuracy']:.3f}")
    ablator.set("off")
    base = res["conditions"]["clean"]["lm_loss"]
    for v in res["conditions"].values():
        v["lm_loss_delta"] = v["lm_loss"] - base
    res["calibration"] = {"n_sequences": n_calib, "seq_len": seq_len,
                          "source": "HuggingFaceFW/fineweb-edu sample-10BT (stream order)",
                          "position_0_excluded": True,
                          "amax_percentiles": {str(p): float(np.percentile(amax.numpy(), p))
                                               for p in (50, 90, 99, 100)}}
    res["heldout"] = {"n_sequences": n_heldout, "seq_len": seq_len}
    res["popqa"] = {"n": n_popqa, "n_shot": 8, "seed": 0}
    res["model_id"] = fm.model_id
    res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round((time.time() - t0) / 3600, 4),
                                         "weak-activation ablation: calibration + LM loss + PopQA"))
    return res


# ---- per-manifold sweep ----------------------------------------------------------------------
def run_manifold(spec: ManifoldSpec, cfg: RunConfig, fm: Any, ablator: Ablator,
                 conds: list[Condition], global_res: dict[str, Any], *,
                 results_dir: Path = RESULTS_DIR, max_points: int = 800, n_kl: int = 400,
                 provider: str = "unknown", instance: str = "unknown") -> dict[str, Any]:
    out_dir = results_dir / spec.name
    ps = PromptSet.load(out_dir / "prompts.json")
    acts = np.load(out_dir / "activations.npy")
    keep = filter_norm_outliers(acts, cfg.analysis.outlier_n_std)
    idx = np.flatnonzero(keep)
    labels = [ps.labels[i] for i in idx]
    prompts = [ps.prompts[i] for i in idx]
    spans = [ps.spans[i] for i in idx] if ps.spans is not None else None
    layer = spec.layer if isinstance(spec.layer, int) else cfg.model.layer
    roles = ROLES[spec.name]
    task = task_for(spec.name, labels)
    cand_ids = first_token_ids(fm.tokenizer, task.candidates) if task and task.candidates else None
    kl_index = subsample(len(idx), n_kl, seed=cfg.seed)
    start = ledger.now_iso()
    t0 = time.time()

    def capture() -> dict[str, Any]:
        return fm.capture_layer_and_logits(prompts, layer + 1, spans=spans, cand_ids=cand_ids,
                                           full_index=kl_index, progress=False)

    ablator.set("off")
    clean = capture()
    a_cached = acts[idx]
    cos = (clean["hidden"] * a_cached).sum(1) / (np.linalg.norm(clean["hidden"], axis=1)
                                                 * np.linalg.norm(a_cached, axis=1) + 1e-9)
    clean_prep = prepare(clean["hidden"], spec.name, labels)
    basis = fit_pca(clean_prep, 3)
    clean_cmp = compare(spec.name, labels, clean_prep, clean_prep, basis)
    clean_task = score_task(task, clean["top1"], clean["cand"], fm.tokenizer) if task else None

    sub = subsample(len(idx), max_points, seed=cfg.seed)
    steps: list[tuple[str, np.ndarray]] = [("clean", clean_cmp["coords_fixed_basis"][sub])]
    rows: dict[str, dict[str, Any]] = {"clean": {
        **asdict(conds[0]), "label": "clean", "description": conds[0].description,
        "status": clean_cmp["status_refit"], "metrics": clean_cmp["metrics_refit"],
        "shape_metric": summary_metric(spec.name, clean_cmp["metrics_refit"]),
        "status_fixed_basis": clean_cmp["status_fixed_basis"],
        "knn_overlap": 1.0, "relative_move": 0.0, "kl_mean": 0.0, "top1_agreement": 1.0,
        "task_accuracy": clean_task["accuracy"] if clean_task else None,
    }}
    for c in conds:
        if c.group == "clean":
            continue
        ablator.set_condition(c)
        cap = capture()
        prep = prepare(cap["hidden"], spec.name, labels)
        cmp = compare(spec.name, labels, clean_prep, prep, basis)
        beh = kl_and_agreement(clean["full"].astype(np.float32), cap["full"].astype(np.float32))
        beh["top1_agreement_all"] = float((clean["top1"] == cap["top1"]).mean())
        tk = score_task(task, cap["top1"], cap["cand"], fm.tokenizer) if task else None
        rows[c.key] = {
            **asdict(c), "label": c.label, "description": c.description,
            "status": cmp["status_refit"], "metrics": cmp["metrics_refit"],
            "shape_metric": summary_metric(spec.name, cmp["metrics_refit"]),
            "status_fixed_basis": cmp["status_fixed_basis"],
            "metrics_fixed_basis": cmp["metrics_fixed_basis"],
            "evr_refit": cmp["evr_refit"], "knn_overlap": cmp["knn_overlap"],
            "knn_overlap_pca3": cmp["knn_overlap_pca3"], "relative_move": cmp["relative_move"],
            "kl_mean": beh["kl_mean"], "kl_median": beh["kl_median"],
            "top1_agreement": beh["top1_agreement_all"],
            "task_accuracy": tk["accuracy"] if tk else None,
        }
        steps.append((c.label, cmp["coords_fixed_basis"][sub]))
        print(f"  {spec.name:20s} {c.key:14s} shape={rows[c.key]['shape_metric']:.3f} "
              f"knn={cmp['knn_overlap']:.2f} agree={beh['top1_agreement_all']:.2f} "
              f"task={rows[c.key]['task_accuracy']}")
    ablator.set("off")
    elapsed = time.time() - t0

    color_key = roles["color_by"]
    color_values = [labels[i][color_key] for i in sub]
    if roles["color_mode"] == "categorical":
        color_values = [str(v) for v in color_values]
    tl = roles["tuning_label"]
    hover = [(prompts[i] if len(prompts[i]) <= 60 else prompts[i][:57] + "…")
             + f"<br>{tl}: {labels[i][tl]}" for i in sub]
    fig = frames_figure(steps, color_values, color_mode=roles["color_mode"], hover_text=hover,
                        active=0, prefix="threshold X (% of neuron max): ")
    report = {
        "name": spec.name, "layer": layer, "n_points": int(len(idx)), "n_plotted": int(len(sub)),
        "n_kl": int(len(kl_index)), "model_id": fm.model_id,
        "consistency_with_cached_activations": {"cosine_min": float(cos.min()),
                                                "cosine_mean": float(cos.mean())},
        "task": task.name if task else None, "task_n": int(task.valid.sum()) if task else 0,
        "clean_task_accuracy": clean_task["accuracy"] if clean_task else None,
        "conditions": rows, "order": [c.key for c in conds],
        "global": {k: {kk: v[kk] for kk in ("lm_loss", "lm_loss_delta", "popqa_accuracy",
                                            "fraction_zeroed")}
                   for k, v in global_res["conditions"].items()},
        "breaking_points": breaking_points(rows, global_res["conditions"]),
        "elapsed_s": round(elapsed, 1),
    }
    (out_dir / "ablation.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out_dir / "ablation_figure.json").write_text(fig.to_json(), encoding="utf-8")
    (out_dir / "ablation_geometry.json").write_text(geometry_figure(report).to_json(),
                                                    encoding="utf-8")
    (out_dir / "ablation_behaviour.json").write_text(behaviour_figure(report).to_json(),
                                                     encoding="utf-8")
    if torch.cuda.is_available():
        ledger.append(ledger.LedgerEntry(provider, instance, torch.cuda.get_device_name(0), start,
                                         ledger.now_iso(), round(elapsed / 3600, 4),
                                         f"weak-activation ablation sweep {spec.name}"))
    return report


def breaking_points(rows: dict[str, dict[str, Any]], glob: dict[str, dict[str, Any]]
                    ) -> dict[str, Any]:
    """First weak threshold X at which each readout has lost half its clean value (shape metric,
    k-NN overlap, task accuracy, PopQA) or LM loss has risen by 0.1 nats."""
    clean = rows["clean"]
    g_clean = glob["clean"]

    def first_x(pred: Any) -> float | None:
        for x in WEAK_X:
            r = rows.get(f"weak_{x:g}")
            if r is not None and pred(r, glob.get(f"weak_{x:g}", {})):
                return x
        return None

    out = {
        "shape_half": first_x(lambda r, g: r["shape_metric"] < 0.5 * clean["shape_metric"]),
        "status_changes": first_x(lambda r, g: r["status"] != clean["status"]),
        "knn_half": first_x(lambda r, g: r["knn_overlap"] < 0.5),
        "agreement_half": first_x(lambda r, g: r["top1_agreement"] < 0.5),
        "lm_loss_plus_0.1": first_x(lambda r, g: g.get("lm_loss", 0) - g_clean["lm_loss"] > 0.1),
        "popqa_half": first_x(lambda r, g: g.get("popqa_accuracy", 1) < 0.5 * g_clean["popqa_accuracy"]),
    }
    if clean.get("task_accuracy") is not None:
        out["task_half"] = first_x(lambda r, g: (r["task_accuracy"] or 0)
                                   < 0.5 * clean["task_accuracy"])
    return out


# ---- 2-D figures -----------------------------------------------------------------------------
def _xcats(rows: dict[str, dict[str, Any]]) -> list[str]:
    return ["clean"] + [f"{x:g}" for x in WEAK_X]


def _series(rows: dict[str, dict[str, Any]], group: str, key: str
            ) -> tuple[list[str], list[Any]]:
    xs: list[str] = []
    ys: list[Any] = []
    if group == "weak":
        xs.append("clean")
        ys.append(rows["clean"].get(key))
    for r in rows.values():
        if r["group"] == group and r.get(key) is not None:
            xs.append(f"{r['x']:g}")
            ys.append(r[key])
    return xs, ys


def geometry_figure(rep: dict[str, Any]) -> go.Figure:
    rows = rep["conditions"]
    fig = go.Figure()
    spec = [("shape_metric", "shape metric (refit PCA)", CATEGORICAL[0]),
            ("knn_overlap", "k-NN overlap with clean", CATEGORICAL[1]),
            ("relative_move", "displacement / spread", CATEGORICAL[2])]
    for key, name, col in spec:
        xs, ys = _series(rows, "weak", key)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=name,
                                 line={"color": col, "width": 2}, marker={"size": 6}))
        xs, ys = _series(rows, "random", key)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{name} · random control",
                                 line={"color": col, "width": 1.5, "dash": "dot"},
                                 marker={"size": 6, "symbol": "x"}))
    for r in rows.values():
        if r["group"] == "scale":
            fig.add_trace(go.Scatter(x=[f"{r['x']:g}"], y=[r["shape_metric"]], mode="markers",
                                     name=f"shape metric · tail ×{r['scale']:g}",
                                     marker={"size": 9, "symbol": "diamond" if r["scale"] < 1
                                             else "diamond-open", "color": CATEGORICAL[0]}))
    lay = line_layout("X: weak threshold, % of each neuron's max (all MLP blocks)", "value",
                      legend_y=-0.35)
    lay["xaxis"]["categoryorder"] = "array"
    lay["xaxis"]["categoryarray"] = _xcats(rows)
    fig.update_layout(**lay)
    return fig


def behaviour_figure(rep: dict[str, Any]) -> go.Figure:
    rows = rep["conditions"]
    glob = rep["global"]
    fig = go.Figure()
    if rep.get("task"):
        xs, ys = _series(rows, "weak", "task_accuracy")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"{rep['task']} accuracy",
                                 line={"color": CATEGORICAL[0], "width": 2}, marker={"size": 6}))
        xs, ys = _series(rows, "random", "task_accuracy")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="accuracy · random control",
                                 line={"color": CATEGORICAL[0], "width": 1.5, "dash": "dot"},
                                 marker={"size": 6, "symbol": "x"}))
    xs, ys = _series(rows, "weak", "top1_agreement")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="top-1 agreement with clean",
                             line={"color": CATEGORICAL[1], "width": 2}, marker={"size": 6}))
    xs, ys = _series(rows, "random", "top1_agreement")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="agreement · random control",
                             line={"color": CATEGORICAL[1], "width": 1.5, "dash": "dot"},
                             marker={"size": 6, "symbol": "x"}))
    # model-wide references
    xs = ["clean"] + [f"{x:g}" for x in WEAK_X if f"weak_{x:g}" in glob]
    keys = ["clean"] + [f"weak_{x:g}" for x in WEAK_X if f"weak_{x:g}" in glob]
    base = glob["clean"]["popqa_accuracy"] or 1e-9
    fig.add_trace(go.Scatter(x=xs, y=[glob[k]["popqa_accuracy"] / base for k in keys],
                             mode="lines+markers", name="PopQA facts (relative to clean)",
                             line={"color": CATEGORICAL[3], "width": 2, "dash": "dash"},
                             marker={"size": 5}))
    fig.add_trace(go.Scatter(x=xs, y=[glob[k]["lm_loss_delta"] for k in keys],
                             mode="lines+markers", name="LM loss increase (nats, right axis)",
                             line={"color": CATEGORICAL[4], "width": 2, "dash": "dash"},
                             marker={"size": 5}, yaxis="y2"))
    lay = line_layout("X: weak threshold, % of each neuron's max (all MLP blocks)",
                      "accuracy / agreement", y_range=(-0.02, 1.05), legend_y=-0.35)
    lay["xaxis"]["categoryorder"] = "array"
    lay["xaxis"]["categoryarray"] = _xcats(rows)
    lay["yaxis2"] = {"title": {"text": "Δ loss (nats)"}, "overlaying": "y", "side": "right",
                     "showgrid": False, "rangemode": "tozero",
                     "tickfont": {"family": lay["font"]["family"], "size": 11}}
    lay["margin"]["r"] = 50
    fig.update_layout(**lay)
    return fig


def global_figure(glob: dict[str, Any]) -> go.Figure:
    """Model-wide curves: LM loss increase and PopQA accuracy vs X, weak vs random."""
    c = glob["conditions"]
    fig = go.Figure()
    for group, dash in (("weak", "solid"), ("random", "dot")):
        ks = ["clean"] + [k for k, v in c.items() if v["group"] == group]
        xs = ["clean"] + [f"{c[k]['x']:g}" for k in ks[1:]]
        fig.add_trace(go.Scatter(x=xs, y=[c[k]["popqa_accuracy"] for k in ks], mode="lines+markers",
                                 name=f"PopQA accuracy · {group}",
                                 line={"color": CATEGORICAL[3], "dash": dash, "width": 2}))
        fig.add_trace(go.Scatter(x=xs, y=[c[k]["lm_loss_delta"] for k in ks], mode="lines+markers",
                                 name=f"LM loss increase · {group}", yaxis="y2",
                                 line={"color": CATEGORICAL[4], "dash": dash, "width": 2}))
    lay = line_layout("X: weak threshold, % of each neuron's max", "PopQA accuracy",
                      y_range=(-0.02, max(0.5, 1.2 * c["clean"]["popqa_accuracy"])), legend_y=-0.35)
    lay["xaxis"]["categoryorder"] = "array"
    lay["xaxis"]["categoryarray"] = ["clean"] + [f"{x:g}" for x in WEAK_X]
    lay["yaxis2"] = {"title": {"text": "Δ loss (nats)"}, "overlaying": "y", "side": "right",
                     "showgrid": False, "rangemode": "tozero"}
    lay["margin"]["r"] = 50
    fig.update_layout(**lay)
    return fig


__all__ = ["Ablator", "Condition", "conditions", "run_global", "run_manifold", "WEAK_X",
           "geometry_figure", "behaviour_figure", "global_figure", "breaking_points"]
