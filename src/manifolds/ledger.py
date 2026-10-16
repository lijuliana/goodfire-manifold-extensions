"""GPU-hour ledger.

One JSON line per GPU session, appended by the pipeline and by hand via the CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from manifolds import LOGS_DIR

LEDGER_PATH = LOGS_DIR / "gpu_ledger.jsonl"


@dataclass(frozen=True)
class LedgerEntry:
    provider: str          # azure | aws | gcp | local
    instance: str          # VM / instance name
    gpu: str               # e.g. "Tesla T4"
    start: str             # ISO-8601 UTC
    end: str               # ISO-8601 UTC
    hours: float
    purpose: str           # e.g. "extract colors"
    notes: str = ""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def hours_between(start_iso: str, end_iso: str) -> float:
    s = datetime.fromisoformat(start_iso)
    e = datetime.fromisoformat(end_iso)
    return round((e - s).total_seconds() / 3600.0, 4)


def append(entry: LedgerEntry, path: Path = LEDGER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read(path: Path = LEDGER_PATH) -> list[LedgerEntry]:
    if not path.exists():
        return []
    out: list[LedgerEntry] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(LedgerEntry(**json.loads(line)))
    return out


def merge(src: Path, dst: Path = LEDGER_PATH) -> int:
    """Append entries from `src` that are not already in `dst` (keyed by start+purpose)."""
    have = {(e.start, e.purpose) for e in read(dst)}
    n = 0
    for e in read(src):
        if (e.start, e.purpose) not in have:
            append(e, dst)
            have.add((e.start, e.purpose))
            n += 1
    return n


def summarize(path: Path = LEDGER_PATH) -> dict[str, float]:
    """Total GPU hours per provider plus an 'all' total."""
    totals: dict[str, float] = {}
    for e in read(path):
        totals[e.provider] = round(totals.get(e.provider, 0.0) + e.hours, 4)
    totals["all"] = round(sum(v for k, v in totals.items() if k != "all"), 4)
    return totals
