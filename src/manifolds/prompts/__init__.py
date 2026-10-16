"""Prompt-set builders.

Every builder reproduces a published template verbatim or documents its deviation.
"""

from collections.abc import Callable

from manifolds.prompts.core import CORE_BUILDERS
from manifolds.prompts.cyclic import CYCLIC_BUILDERS
from manifolds.prompts.extended import EXTENDED_BUILDERS
from manifolds.prompts.types import PromptSet

BUILDERS: dict[str, Callable[[], PromptSet]] = {**CORE_BUILDERS, **EXTENDED_BUILDERS,
                                                 **CYCLIC_BUILDERS}


def build(name: str) -> PromptSet:
    if name not in BUILDERS:
        raise KeyError(
            f"no prompt builder registered for manifold {name!r}; known: {sorted(BUILDERS)}"
        )
    return BUILDERS[name]()


__all__ = ["BUILDERS", "PromptSet", "build"]
