from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorMatchStage:
    enabled: bool = False

    def describe(self) -> str:
        return "color_match stage is reserved for phase 3"
