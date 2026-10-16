"""Shared prompt-set container."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Label = dict[str, Any]


@dataclass
class PromptSet:
    name: str
    prompts: list[str]
    labels: list[Label]
    template: str
    source: str
    deviations: list[str] = field(default_factory=list)
    spans: list[tuple[int, int]] | None = None   # char span whose last token is read; None = last token

    def __post_init__(self) -> None:
        if len(self.prompts) != len(self.labels):
            raise ValueError("prompts and labels must have the same length")
        if self.spans is not None and len(self.spans) != len(self.prompts):
            raise ValueError("spans must match prompts")

    def __len__(self) -> int:
        return len(self.prompts)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": self.name,
                    "template": self.template,
                    "source": self.source,
                    "deviations": self.deviations,
                    "n": len(self),
                    "prompts": self.prompts,
                    "labels": self.labels,
                    "spans": self.spans,
                },
                f,
                indent=1,
            )

    @classmethod
    def load(cls, path: Path) -> PromptSet:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            name=d["name"],
            prompts=d["prompts"],
            labels=d["labels"],
            template=d["template"],
            source=d["source"],
            deviations=d.get("deviations", []),
            spans=[tuple(x) for x in d["spans"]] if d.get("spans") else None,
        )
