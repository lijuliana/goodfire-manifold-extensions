"""Typed loader for manifest.yaml (Step 0 of the replication)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from manifolds import MANIFEST_PATH

Tier = Literal["core", "extended", "cyclic", "vision"]
ReplicationMode = Literal["exact-template", "supervised-approx", "probe-based", "requires-bsf"]
Status = Literal["pending", "matches", "partial", "failed"]

# normalized geometry keys used by the shape checker, matched by substring (first hit wins)
_GEOMETRY_KEYS: tuple[tuple[str, str], ...] = (
    ("paraboloid", "paraboloid"),
    ("helix", "helix"),
    ("circle", "circle"),
    ("hierarchical tree", "tree"),
    ("1d continuous", "ordinal"),
    ("1d ordered curve", "ordinal"),
    ("line", "line"),
)


class ManifoldSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    tier: Tier
    model: str
    layer: int | str
    expected_geometry: str
    replication_mode: ReplicationMode
    status: Status = "pending"
    source: list[str] | str
    prompt_template: str | None = None
    dataset: str | None = None
    labels: list[str] | None = None
    intrinsic_coords: list[str] | None = None
    n_paper: int | str | None = None
    token_position: str | None = None
    archived: bool = False          # hidden from the site and from default runs; results are kept

    @property
    def sources(self) -> list[str]:
        return [self.source] if isinstance(self.source, str) else list(self.source)

    @property
    def geometry_key(self) -> str:
        g = self.expected_geometry.lower()
        for needle, key in _GEOMETRY_KEYS:
            if needle in g:
                return key
        return "unknown"


class Manifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str | float
    defaults: dict[str, Any]
    manifolds: list[ManifoldSpec]
    skipped_candidates: list[dict[str, Any]] = []

    @field_validator("manifolds")
    @classmethod
    def _unique_names(cls, v: list[ManifoldSpec]) -> list[ManifoldSpec]:
        names = [m.name for m in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate manifold names: {sorted(dupes)}")
        return v

    def get(self, name: str) -> ManifoldSpec:
        for m in self.manifolds:
            if m.name == name:
                return m
        raise KeyError(name)

    def by_tier(self, tier: Tier) -> list[ManifoldSpec]:
        return [m for m in self.manifolds if m.tier == tier]

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.manifolds]


def load_manifest(path: Path | str = MANIFEST_PATH) -> Manifest:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Manifest.model_validate(raw)
