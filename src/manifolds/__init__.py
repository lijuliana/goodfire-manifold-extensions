"""Goodfire concept-manifold replication package."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "site"
LOGS_DIR = ROOT / "logs"
MANIFEST_PATH = ROOT / "manifest.yaml"

__all__ = ["ROOT", "RESULTS_DIR", "SITE_DIR", "LOGS_DIR", "MANIFEST_PATH"]
