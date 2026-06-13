from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class AlignmentContext:
    sample_id: str
    lr_path: Path
    hr_path: Path
    metadata: dict[str, Any]
    config: dict[str, Any]
    artifact_root: Path
    device: str


@dataclass(frozen=True)
class AlignResult:
    aligned_lr_rgb: np.ndarray | None
    status: str
    confidence: float
    message: str = ""
    transforms: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Aligner(Protocol):
    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        raise NotImplementedError
