"""Ask Claude for a one-line interpretation of each principal axis from its lens readouts.

Input: results/<manifold>/depth.json (pc_readouts). Output: results/<manifold>/axis_labels.json
with provenance. The label is an LLM's reading of the token tables, shown on the site as a
"likely meaning" with a confidence flag; the tables remain the evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import anthropic

MODEL = "claude-opus-5-5"
SYSTEM = (
    "You interpret principal axes of a language model's activation manifold. For each axis you "
    "get tokens that four readouts associate with its positive and negative ends: three lenses "
    "that project the axis direction through the model's output head (Jacobian, logit, tuned) "
    "and a lens-free correlation with the model's real next-token log-probabilities. Give a short "
    "label of the form '<negative end> ↔ <positive end>' (3-10 words), a one-sentence rationale "
    "citing the tokens that convinced you, and a confidence: high if at least two readouts agree "
    "on a clear concept, medium if one readout is clear, low if the tokens look like fragments or "
    "noise. If the axis is uninterpretable, say so with low confidence. Return only JSON."
)


def ask(client: anthropic.Anthropic, name: str, variable: str, readouts: list[dict]) -> list[dict]:
    axes = []
    for e in readouts:
        d = {"pc": e["pc"],
             "explained_variance_pct": round(100 * e.get("explained_variance_ratio", 0))}
        for k in ("jacobian_lens", "logit_lens", "tuned_lens"):
            if k in e:
                d[k] = {"positive": e[k]["up"][:10], "negative": e[k]["down"][:10]}
        emp = e.get("empirical", {})
        d["real_logit_correlation"] = {"positive": emp.get("positive", [])[:10],
                                       "negative": emp.get("negative", [])[:10]}
        axes.append(d)
    prompt = (f"Manifold: {name} (prompts vary the variable '{variable}').\n\nAxes:\n"
              f"{json.dumps(axes, ensure_ascii=False, indent=1)}\n\n"
              "Return a JSON array with one object per axis: {\"pc\": int, \"label\": str, "
              "\"rationale\": str, \"confidence\": \"high\"|\"medium\"|\"low\"}.")
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=2000, betas=["server-side-fallback-2026-07-01"],
        fallbacks="default", output_config={"effort": "medium"}, system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refused: {name}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"no JSON for {name}")
    out = json.loads(m.group(0))
    for o in out:
        o["confidence"] = str(o.get("confidence", "low")).lower()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--manifold", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    client = anthropic.Anthropic()
    dirs = sorted(Path(a.results).glob("*/depth.json"))
    for dj in dirs:
        name = dj.parent.name
        if a.manifold and name not in a.manifold:
            continue
        out = dj.parent / "axis_labels.json"
        if out.exists() and not a.force:
            continue
        d = json.loads(dj.read_text(encoding="utf-8"))
        readouts = d.get("pc_readouts", [])
        if not readouts:
            continue
        rep = json.loads((dj.parent / "report.json").read_text(encoding="utf-8"))
        variable = rep.get("color", {}).get("color_by", "")
        labels = ask(client, name, variable, readouts)
        out.write_text(json.dumps({
            "_provenance": {"model": MODEL,
                            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                            "note": "LLM reading of the lens token tables; not a measurement"},
            "axes": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name + ": " + " | ".join(f"PC{x['pc']} {x['label']} ({x['confidence']})"
                                        for x in labels))


if __name__ == "__main__":
    main()
