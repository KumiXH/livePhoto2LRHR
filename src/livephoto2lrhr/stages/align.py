from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignStage:
    enabled: bool = False

    def describe(self) -> str:
        return "align stage is reserved for phase 2"
