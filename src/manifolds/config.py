"""Typed run configuration loaded from configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from manifolds import ROOT


class ModelConfig(BaseModel):
    id: str
    fallback_id: str | None = None
    layer: int = 19
    dtype: str = "float16"
    truncate_to_layer: bool = True
    batch_size: int = 16
    max_length: int = 1024


class AnalysisConfig(BaseModel):
    outlier_n_std: float = 3.0
    pca_components: int = 3


class PrecisionCheckConfig(BaseModel):
    n_prompts: int = 16
    min_cosine: float = 0.999


class SiteConfig(BaseModel):
    preview_max_points: int = 1500


class RunConfig(BaseModel):
    seed: int = 0
    model: ModelConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    precision_check: PrecisionCheckConfig = Field(default_factory=PrecisionCheckConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> RunConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return RunConfig.model_validate(raw)
