"""The nine core manifolds of Bhalla et al. 2026 (arXiv 2604.28119), Table 1.

Builders are line-for-line ports of goodfire-ai/sae-manifold `data.py` (MIT). Where this file
departs from that code, the departure is recorded in `PromptSet.deviations`.
"""

from __future__ import annotations

import colorsys
from collections.abc import Callable

from manifolds.prompts.types import Label, PromptSet

SRC = "arXiv 2604.28119 Table 1; github.com/goodfire-ai/sae-manifold data.py"


def build_colors() -> PromptSet:
    """HEX codes -> hue/lightness/saturation (paraboloid). data.py:51-65."""
    from datasets import load_dataset

    ds = load_dataset("burkelibbey/colors", split="train")
    prompts: list[str] = []
    labels: list[Label] = []
    for i, row in enumerate(ds):
        if i % 10 != 0:
            continue
        h_code = row["color"]
        r, g, b = (int(h_code.lstrip("#")[j : j + 2], 16) / 255 for j in (0, 2, 4))
        h, lgt, s = colorsys.rgb_to_hls(r, g, b)
        if lgt < 0.2 or lgt > 0.8 or s < 0.2 or s > 0.8:
            continue
        prompts.append(f"The hex code {h_code} is for the color")
        labels.append(dict(hex=h_code, hue=h, lightness=lgt, saturation=s))
    return PromptSet("colors", prompts, labels, "The hex code {hex} is for the color", SRC)


def build_age() -> PromptSet:
    """data.py:68-73."""
    prompts, labels = [], []
    for age in range(1, 100):
        prompts.append(f"They are {age} years old. ")
        labels.append(dict(age=float(age)))
    return PromptSet("age", prompts, labels, "They are {age} years old. ", SRC)


def build_temperature() -> PromptSet:
    """data.py:76-81."""
    prompts, labels = [], []
    for f in range(-30, 120):
        prompts.append(f"Today it's {f} degrees Fahrenheit outside")
        labels.append(dict(fahrenheit=float(f), celsius=(f - 32) * 5 / 9))
    return PromptSet(
        "temperature", prompts, labels, "Today it's {f} degrees Fahrenheit outside", SRC
    )


def build_geography() -> PromptSet:
    """World cities -> lat/long/country/continent (hierarchical tree). data.py:84-100."""
    from datasets import load_dataset

    ds = load_dataset("jamescalam/world-cities-geo", split="train")
    prompts, labels = [], []
    for i, row in enumerate(ds):
        if i % 3 != 0:
            continue
        prompts.append(
            f"The geographical coordinates latitude {row['latitude']} "
            f"and longitude {row['longitude']} are in the country of"
        )
        labels.append(
            dict(
                latitude=row["latitude"],
                longitude=row["longitude"],
                country=row["country"],
                region=row["region"],
                continent=row["continent"],
            )
        )
    return PromptSet(
        "geography",
        prompts,
        labels,
        "The geographical coordinates latitude {latitude} and longitude {longitude} "
        "are in the country of",
        SRC,
    )


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = [
    "the crack of dawn",
    "early morning",
    "late morning",
    "noon",
    "afternoon",
    "evening",
    "night",
    "midnight",
]


def build_days() -> PromptSet:
    """7 days x 8 times-of-day = 56 prompts. data.py:103-121.

    Deviation note: the paper's Table 1 lists n=420 for `days`; the released code yields 56.
    """
    prompts, labels = [], []
    for di, day in enumerate(DAYS):
        for ti, time in enumerate(TIMES):
            prompts.append(f"It's {time} on {day}")
            labels.append(dict(day=day, day_idx=di, time=time, time_idx=ti))
    return PromptSet(
        "days",
        prompts,
        labels,
        "It's {time} on {day}",
        SRC,
        deviations=[
            "paper Table 1 reports n=420; released data.py produces 56 prompts (used here)"
        ],
    )


def build_years() -> PromptSet:
    """Years 1800-1998 -> helix (linear century + periodic ones digit). data.py:124-134."""
    prompts, labels = [], []
    for year in range(1800, 1999):
        prompts.append(f"The date is {year}")
        labels.append(
            dict(
                year=float(year),
                decade_digit=float(year % 10),
                decade=float((year % 100) // 10),
            )
        )
    return PromptSet("years", prompts, labels, "The date is {year}", SRC)


def build_formality() -> PromptSet:
    """Pavlick & Tetreault formality scores (line). data.py:137-150."""
    from datasets import load_dataset

    ds = load_dataset("osyvokon/pavlick-formality-scores", split="train")
    prompts, labels = [], []
    for row in ds:
        text = row["sentence"]
        score = row["avg_score"]
        if not text or not text.strip() or len(text.strip()) < 3:
            continue
        prompts.append(text.strip())
        labels.append(dict(formality=float(score)))
        if len(prompts) >= 1000:
            break
    return PromptSet("formality", prompts, labels, "{sentence}", SRC)


def build_sent_length() -> PromptSet:
    """WikiText-103 sentences labelled by whitespace token count 3-80. data.py:153-179.

    Deviation: the dataset is read in streaming mode (same order, same first 5000 sentences)
    to avoid downloading the full 500 MB corpus.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    prompts, labels = [], []
    seen: set[str] = set()
    for row in ds:
        text = row["text"].strip()
        if not text or len(text) < 10 or text.startswith("="):
            continue
        for sent in text.replace("? ", ".\n").replace("! ", ".\n").split(".\n"):
            sent = sent.strip()
            if not sent or len(sent) < 10:
                continue
            if not sent.endswith((".", "?", "!")):
                sent = sent + "."
            n_tokens = len(sent.split())
            if n_tokens < 3 or n_tokens > 80 or sent in seen:
                continue
            seen.add(sent)
            prompts.append(sent)
            labels.append(dict(n_tokens=float(n_tokens), n_chars=float(len(sent))))
            if len(prompts) >= 5000:
                break
        if len(prompts) >= 5000:
            break
    return PromptSet(
        "sent_length",
        prompts,
        labels,
        "{sentence}",
        SRC,
        deviations=["wikitext read in streaming mode (identical order and selection)"],
    )


def build_political_bias() -> PromptSet:
    """cajcodes/political-bias statements, label 0 (conservative) .. 4 (liberal).

    The paper used "GPT-5 augmentations" of this dataset (Table 1); those augmentations were not
    released, so the raw 658 statements are used. Not in the released data.py.
    """
    from datasets import load_dataset

    ds = load_dataset("cajcodes/political-bias", split="train")
    prompts, labels = [], []
    for row in ds:
        text = str(row["text"]).strip()
        if not text:
            continue
        prompts.append(text)
        labels.append(dict(bias_label=int(row["label"])))
    return PromptSet(
        "political_bias",
        prompts,
        labels,
        "{statement}",
        "arXiv 2604.28119 Table 1 (cajcodes_political_bias); not in released data.py",
        deviations=[
            "paper used GPT-5 augmentations of the dataset (not released); raw 658 statements used",
            "prompt template not published; raw statement used as prompt",
        ],
    )


CORE_BUILDERS: dict[str, Callable[[], PromptSet]] = {
    "colors": build_colors,
    "temperature": build_temperature,
    "age": build_age,
    "geography": build_geography,
    "days": build_days,
    "years": build_years,
    "formality": build_formality,
    "sent_length": build_sent_length,
    "political_bias": build_political_bias,
}
