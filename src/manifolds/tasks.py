"""Behavioural readouts for the pilot manifolds: a next-token task per manifold where one is
defined, plus the model-agnostic ones (KL to the clean model, top-1 agreement).

Tasks (all read at the last prompt token, greedy, single next token):
  weekdays_circle        weekday arithmetic: the answer weekday (7-way, restricted to the 7 names)
  months_last            month arithmetic: the answer month (12-way, restricted)
  addition_sum_fourier   addition: the next token must be the sum (Llama 3 tokenises 1-3 digit
                         numbers as one token; other tokenisers are handled by comparing the
                         decoded first token with the sum's first token)
  colors                 hue-family naming: restricted argmax over basic colour names must fall in
                         the hue family of the hex code (approximate, see HUE_FAMILIES)
  geography              country recall: the first token of " {country}" (a *fact*, for contrast)
  political_bias         lean (2-way, restricted to " conservative" / " liberal"; neutral excluded)
  age                    none (KL / agreement only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
# hue (in turns, HLS) -> family; boundaries are conventional, not from any source
HUE_FAMILIES: dict[str, tuple[float, float]] = {
    "red": (0.0, 0.03), "orange": (0.03, 0.11), "yellow": (0.11, 0.19), "green": (0.19, 0.44),
    "cyan": (0.44, 0.53), "blue": (0.53, 0.72), "purple": (0.72, 0.85), "pink": (0.85, 0.95),
    "red2": (0.95, 1.0001),
}
COLOR_WORDS: dict[str, list[str]] = {
    "red": ["red", "crimson", "maroon", "scarlet"], "orange": ["orange", "brown", "tan", "peach"],
    "yellow": ["yellow", "gold", "olive", "mustard"], "green": ["green", "lime", "olive", "mint"],
    "cyan": ["cyan", "teal", "turquoise", "aqua"], "blue": ["blue", "navy", "indigo", "azure"],
    "purple": ["purple", "violet", "lavender", "indigo"], "pink": ["pink", "magenta", "rose"],
}


@dataclass
class Task:
    name: str
    kind: str                         # "restricted" | "exact"
    candidates: list[str] | None      # restricted: candidate continuations (with leading space)
    answers: list[Any]                # per prompt: candidate index (restricted) or string (exact)
    valid: np.ndarray                 # per prompt: whether the prompt has an answer


def _hue_family(h: float) -> str:
    for fam, (lo, hi) in HUE_FAMILIES.items():
        if lo <= h < hi:
            return "red" if fam == "red2" else fam
    return "red"


def task_for(name: str, labels: list[dict[str, Any]]) -> Task | None:
    n = len(labels)
    if name in ("weekdays_circle", "weekdays_last"):
        cands = [" " + d for d in WEEKDAYS]
        ans_w: list[Any] = [WEEKDAYS.index(lab["answer"]) for lab in labels]
        return Task("weekday arithmetic", "restricted", cands, ans_w, np.ones(n, bool))
    if name in ("months_last", "months_circle"):
        cands = [" " + m for m in MONTHS]
        ans_m: list[Any] = [MONTHS.index(lab["answer"]) for lab in labels]
        return Task("month arithmetic", "restricted", cands, ans_m, np.ones(n, bool))
    if name == "addition_sum_fourier":
        return Task("addition (exact sum)", "exact", None, [str(int(lab["sum"])) for lab in labels],
                    np.ones(n, bool))
    if name == "colors":
        words = sorted({w for ws in COLOR_WORDS.values() for w in ws})
        cands = [" " + w for w in words]
        ans_c: list[Any] = []
        for lab in labels:
            fam = _hue_family(float(lab["hue"]))
            ans_c.append([words.index(w) for w in COLOR_WORDS[fam]])
        return Task("hue-family naming", "restricted", cands, ans_c, np.ones(n, bool))
    if name == "geography":
        return Task("country recall (fact)", "exact", None,
                    [" " + str(lab["country"]) for lab in labels], np.ones(n, bool))
    if name == "political_bias":
        cands = [" conservative", " liberal"]
        ans: list[Any] = []
        valid = np.zeros(n, bool)
        for i, lab in enumerate(labels):
            b = int(lab["bias_label"])
            if b in (0, 1):
                ans.append(0)
                valid[i] = True
            elif b in (3, 4):
                ans.append(1)
                valid[i] = True
            else:
                ans.append(-1)
        return Task("lean (conservative vs liberal)", "restricted", cands, ans, valid)
    return None


def first_token_ids(tokenizer: Any, strings: list[str]) -> list[int]:
    """First token id of each continuation string (no BOS)."""
    out = []
    for s in strings:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if not ids:
            raise ValueError(f"empty tokenisation for {s!r}")
        out.append(int(ids[0]))
    return out


def exact_answer_ids(task: Task, tokenizer: Any) -> np.ndarray:
    """For exact tasks: the first token id of each prompt's answer (−1 where undefined)."""
    assert task.kind == "exact"
    out = np.full(len(task.answers), -1, dtype=np.int64)
    cache: dict[str, int] = {}
    for i, a in enumerate(task.answers):
        s = str(a)
        if s not in cache:
            cache[s] = first_token_ids(tokenizer, [s])[0]
        out[i] = cache[s]
    return out


def score_task(task: Task, top1: np.ndarray, cand_logits: np.ndarray | None,
               tokenizer: Any) -> dict[str, Any]:
    """Accuracy from the greedy next token (`top1`, [N]) and, for restricted tasks, the logits of
    the candidate continuations (`cand_logits`, [N, C] in the order of task.candidates)."""
    n = len(task.answers)
    correct = np.zeros(n, bool)
    if task.kind == "restricted":
        assert cand_logits is not None
        pred = cand_logits.argmax(1)
        for i in range(n):
            a = task.answers[i]
            correct[i] = (int(pred[i]) in a) if isinstance(a, list) else (int(pred[i]) == a)
    else:
        want = exact_answer_ids(task, tokenizer)
        correct = np.asarray(top1).astype(np.int64) == want
    v = task.valid
    return {"task": task.name, "kind": task.kind, "n": int(v.sum()),
            "accuracy": float(correct[v].mean()) if v.any() else float("nan"),
            "correct": correct.tolist()}


def kl_and_agreement(clean_logits: np.ndarray, other_logits: np.ndarray) -> dict[str, float]:
    """KL(clean || other) of the next-token distributions (nats, mean over prompts) and top-1
    agreement."""
    def logsoftmax(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        m = x.max(1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))

    lp, lq = logsoftmax(clean_logits), logsoftmax(other_logits)
    kl = (np.exp(lp) * (lp - lq)).sum(1)
    return {"kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
            "top1_agreement": float((clean_logits.argmax(1) == other_logits.argmax(1)).mean())}


__all__ = ["Task", "task_for", "score_task", "kl_and_agreement", "first_token_ids",
           "exact_answer_ids"]
