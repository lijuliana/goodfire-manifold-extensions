"""Generate a balanced, graded political-bias statement set with Claude.

Mirrors the augmentation step of Bhalla et al. 2026 (Table 1: "GPT-5 augmentations of
cajcodes_political_bias"), whose outputs were not released. Output: data/political_bias.jsonl
with provenance. Levels follow the source dataset: 0 strongly conservative .. 4 strongly liberal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
LEVELS = {
    0: "strongly conservative",
    1: "moderately conservative",
    2: "centrist / moderate, balancing both sides",
    3: "moderately liberal",
    4: "strongly liberal",
}
TOPICS = [
    "taxes", "healthcare", "immigration", "gun policy", "climate and energy", "public education",
    "policing and crime", "abortion", "welfare and social safety net", "international trade",
    "housing", "labor unions", "minimum wage", "military spending", "foreign aid",
    "drug policy", "criminal justice reform", "voting rules", "religion in public life",
    "business regulation", "big tech and antitrust", "infrastructure", "social security",
    "LGBTQ rights", "free speech online", "higher education and student debt", "agriculture",
    "public transit", "public health mandates", "corporate taxes", "environmental regulation",
    "homelessness", "border security", "national debt", "police funding", "school choice",
    "parental rights in schools", "affirmative action", "nuclear power", "paid family leave",
]
SYSTEM = (
    "You write short political opinion statements for a dataset used to study how language "
    "models represent political lean. Statements must be realistic, first-person-plural or "
    "impersonal, one sentence of 15 to 30 words, declarative, and in the same register at every "
    "level, so that only the political lean varies. Level 2 is a genuine centrist position "
    "(for example incremental reform, means-tested compromise), not a refusal to take a side. "
    "Avoid slurs, named people, and party names. Return only JSON."
)
PROMPT = """Topic: {topic}

Write exactly 4 statements for each of the 5 levels below (20 statements total):
{levels}

Return a JSON array of objects with keys "level" (integer 0-4), "topic" (string) and "text"
(string). No prose before or after the JSON."""


def ask(client: anthropic.Anthropic, topic: str) -> list[dict]:
    levels = "\n".join(f"- level {k}: {v}" for k, v in LEVELS.items())
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=6000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium"},
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT.format(topic=topic, levels=levels)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused on topic {topic!r}: {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON array in response for {topic!r}")
    rows = json.loads(m.group(0))
    out = []
    for r in rows:
        lvl = int(r["level"])
        txt = str(r["text"]).strip()
        if lvl not in LEVELS or not (8 <= len(txt.split()) <= 45):
            continue
        out.append({"level": lvl, "topic": topic, "text": txt})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/political_bias.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--topics", type=int, default=len(TOPICS))
    a = ap.parse_args()
    client = anthropic.Anthropic()
    topics = TOPICS[: a.topics]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda t: ask(client, t), topics))
    rows = [r for rs in results for r in rs]
    seen: set[str] = set()
    uniq = []
    for r in rows:
        k = r["text"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    random.Random(0).shuffle(uniq)
    prov = {
        "generator": MODEL,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16],
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
        "n_topics": len(topics),
        "levels": LEVELS,
        "mirrors": "Bhalla et al. 2026 Table 1 (GPT-5 augmentations of cajcodes_political_bias)",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_provenance": prov}) + "\n")
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    counts = {k: sum(r["level"] == k for r in uniq) for k in LEVELS}
    print(f"wrote {len(uniq)} statements to {out}; per level {counts}")


if __name__ == "__main__":
    main()
