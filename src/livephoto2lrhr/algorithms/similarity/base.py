from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    timestamp_sec: float
    score: float


@dataclass(frozen=True)
class FrameSelectionResult:
    frame_rgb: np.ndarray
    selected: FrameCandidate
    top_k: list[FrameCandidate]
    diagnostics: dict[str, object]


class FrameSelector(Protocol):
    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        """Return the selected RGB frame and ranked candidate metadata."""
