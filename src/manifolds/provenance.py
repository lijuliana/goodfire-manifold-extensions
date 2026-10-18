"""Provenance stamp written into every per-page result, so pages produced by different code,
hardware or settings are visible in an audit (scripts/audit_pages.py)."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from manifolds import ROOT, ledger

PIPELINE_VERSION = "2026-10-18.1"   # bump when a per-page method changes


def code_commit() -> str:
    env = os.environ.get("CODE_COMMIT")   # set by the launch scripts (rsync copies have no .git)
    if env:
        return env
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def stamp(fm: Any | None = None, **params: Any) -> dict[str, Any]:
    gpu = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    out: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "code_commit": code_commit(),
                           "gpu": gpu, "run_at": ledger.now_iso()}
    if fm is not None:
        out |= {"model_id": getattr(fm, "model_id", None),
                "dtype": getattr(getattr(fm, "cfg", None), "dtype", None),
                "batch_size": getattr(getattr(fm, "cfg", None), "batch_size", None)}
    out |= params
    return out


__all__ = ["stamp", "PIPELINE_VERSION", "code_commit"]
