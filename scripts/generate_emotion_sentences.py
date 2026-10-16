"""Generate short story-like sentences for six basic emotions with Claude (data/emotions.jsonl).

Stand-in for the SimpleStories prefills used in Bigelow et al. 2026 ("Meandering on Manifolds");
each sentence should evoke exactly one of: happiness, sadness, anger, fear, disgust, surprise.
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
EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
# Russell circumplex coordinates (valence, arousal) in [-1, 1], approximate, for the shape check
CIRCUMPLEX = {"happiness": (0.8, 0.5), "surprise": (0.2, 0.8), "fear": (-0.6, 0.8),
              "anger": (-0.7, 0.7), "disgust": (-0.7, 0.3), "sadness": (-0.7, -0.5)}
SYSTEM = (
    "You write short narrative sentences for a dataset about how language models represent "
    "emotion. Each sentence is one complete sentence of 12 to 25 words, written in the third "
    "person past tense like a children's story, describing a situation and the character's "
    "feeling, with the named emotion clearly dominant and no other strong emotion. Vary the "
    "characters, settings and events; do not reuse names. Return only JSON."
)
PROMPT = """Emotion: {emotion}
Batch: {batch}

Write exactly 20 distinct sentences in which the character feels {emotion}.
Return a JSON array of objects with keys "emotion" and "text". No prose before or after."""


def ask(client: anthropic.Anthropic, emotion: str, batch: int) -> list[dict]:
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=4000, betas=["server-side-fallback-2026-07-01"],
        fallbacks="default", output_config={"effort": "medium"}, system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT.format(emotion=emotion, batch=batch)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {emotion} {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array")
    out = []
    for r in json.loads(m.group(0)):
        t = str(r["text"]).strip()
        if 8 <= len(t.split()) <= 35:
            out.append({"emotion": emotion, "text": t})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/emotions.jsonl")
    ap.add_argument("--batches", type=int, default=2)   # 2 x 20 = 40 per emotion
    a = ap.parse_args()
    client = anthropic.Anthropic()
    jobs = [(e, b) for e in EMOTIONS for b in range(a.batches)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda j: ask(client, *j), jobs))
    rows = [r for rs in res for r in rs]
    seen: set[str] = set()
    uniq = [r for r in rows if not (r["text"].lower() in seen or seen.add(r["text"].lower()))]
    random.Random(0).shuffle(uniq)
    prov = {"generator": MODEL, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt_sha256": hashlib.sha256((SYSTEM + PROMPT).encode()).hexdigest()[:16],
            "circumplex": CIRCUMPLEX,
            "mirrors": "Bigelow et al. 2026 (Meandering on Manifolds): emotions in short stories"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_provenance": prov}) + "\n")
        for r in uniq:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(uniq)} sentences to {out}; per emotion",
          {e: sum(r["emotion"] == e for r in uniq) for e in EMOTIONS})


if __name__ == "__main__":
    main()
