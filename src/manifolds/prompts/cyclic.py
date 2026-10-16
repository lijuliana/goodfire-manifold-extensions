# ruff: noqa: E501  (reference tables and verbatim templates are long by nature)
"""Cyclic tier: the task templates of Feucht et al. 2026 (arXiv 2605.01148, App. A Table 1).

For weekdays / months / hours the circle is read at the *concept token* (Engels et al. 2024;
Feucht et al. Fig. 3 report it is absent at the last token of layer 18), so these prompt sets carry
character spans of the concept word. The addition task is read at the last token ("=") where the
paper finds base-10 Fourier features of the sum (layer 18).
"""

from __future__ import annotations

import random
from collections.abc import Callable

from manifolds.prompts.types import PromptSet

SRC = "arXiv 2605.01148 App. A Table 1"
NUM_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three",
             "twenty-four"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _num_word(n: int) -> str:
    if n <= 24:
        return NUM_WORDS[n - 1]
    tens, ones = divmod(n, 10)
    base = {2: "twenty", 3: "thirty", 4: "forty"}[tens]
    return base if ones == 0 else f"{base}-{NUM_WORDS[ones - 1]}"


def build_weekdays_circle() -> PromptSet:
    prompts, labels, spans = [], [], []
    for di, day in enumerate(WEEKDAYS):
        for off in range(1, 15):
            p = f"Q: What day is {_num_word(off)} days after {day}?\nA:"
            start = p.index(day)
            prompts.append(p)
            spans.append((start, start + len(day)))
            labels.append(dict(day=day, day_idx=float(di), offset=float(off),
                               answer=WEEKDAYS[(di + off) % 7]))
    return PromptSet("weekdays_circle", prompts, labels,
                     "Q: What day is {offset} days after {concept}?\\nA:", SRC,
                     deviations=["read at the concept token (paper's circle claim); offsets one..fourteen"],
                     spans=spans)


def build_months_circle() -> PromptSet:
    prompts, labels, spans = [], [], []
    for mi, month in enumerate(MONTHS):
        for off in range(1, 25):
            p = f"Q: What month is {_num_word(off)} months after {month}?\nA:"
            start = p.index(month)
            prompts.append(p)
            spans.append((start, start + len(month)))
            labels.append(dict(month=month, month_idx=float(mi), offset=float(off),
                               answer=MONTHS[(mi + off) % 12]))
    return PromptSet("months_circle", prompts, labels,
                     "Q: What month is {offset} months after {concept}?\\nA:", SRC,
                     deviations=["read at the concept token; offsets one..twenty-four"], spans=spans)


def build_hours_circle() -> PromptSet:
    prompts, labels, spans = [], [], []
    for h in range(24):
        for off in range(1, 49):
            hh = f"{h:02d}"
            p = (f"Q: In 24-hour time, it is now {hh}:00. What time will it be in "
                 f"{_num_word(off)} hours?\n A: In 24-hour time, it will be")
            start = p.index(hh)
            prompts.append(p)
            spans.append((start, start + len(hh)))
            labels.append(dict(hour=float(h), offset=float(off), answer=float((h + off) % 24)))
    return PromptSet("hours_circle", prompts, labels,
                     "Q: In 24-hour time, it is now {concept}:00. What time will it be in {offset} hours?\\n A: In 24-hour time, it will be",
                     SRC, deviations=["read at the hour token; offsets one..forty-eight"], spans=spans)


def build_addition_sum_fourier() -> PromptSet:
    rng = random.Random(0)
    pairs = sorted({(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(2000)})[:1200]
    prompts, labels = [], []
    for a, b in pairs:
        prompts.append(f"{a}+{b}=")
        s = a + b
        labels.append(dict(a=float(a), b=float(b), sum=float(s), sum_mod10=float(s % 10),
                           sum_mod5=float(s % 5), sum_mod2=float(s % 2)))
    return PromptSet("addition_sum_fourier", prompts, labels, "{a}+{b}=", SRC,
                     deviations=["1200 random (a, b) pairs from 1..100 instead of all 10k; last token; layer 18"])


def _last_variant(base: PromptSet, name: str, concept_key: str, one_indexed: bool) -> PromptSet:
    """Same prompts read at the last token (block 18): the paper reports no concept circle there
    but base-10 Fourier features of the pre-modulo sum concept + offset (Feucht et al. Figs 3, 5)."""
    labels = []
    for lab in base.labels:
        c = float(lab[concept_key]) + (1.0 if one_indexed else 0.0)
        labels.append(dict(lab, premod_sum=c + float(lab["offset"]), concept_index=c))
    return PromptSet(name, list(base.prompts), labels, base.template, SRC + " + Fig. 3 / Fig. 5",
                     deviations=["read at the last token, block 18 (paper's addition-module site)"],
                     spans=None)


def build_weekdays_last() -> PromptSet:
    return _last_variant(build_weekdays_circle(), "weekdays_last", "day_idx", True)


def build_months_last() -> PromptSet:
    return _last_variant(build_months_circle(), "months_last", "month_idx", True)


def build_hours_last() -> PromptSet:
    return _last_variant(build_hours_circle(), "hours_last", "hour", False)


CYCLIC_BUILDERS: dict[str, Callable[[], PromptSet]] = {
    "weekdays_last": build_weekdays_last,
    "months_last": build_months_last,
    "hours_last": build_hours_last,
    "weekdays_circle": build_weekdays_circle,
    "months_circle": build_months_circle,
    "hours_circle": build_hours_circle,
    "addition_sum_fourier": build_addition_sum_fourier,
}
