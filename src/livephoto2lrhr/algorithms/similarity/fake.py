from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult


class FakeFrameSelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.top_k = int(config.get("top_k", 1))
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        candidates = [
            FrameCandidate(frame_index=index, timestamp_sec=index / 30.0, score=1.0 / (index + 1))
            for index in range(max(self.top_k, 1))
        ]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        return FrameSelectionResult(
            frame_rgb=frame,
            selected=candidates[0],
            top_k=candidates[: self.top_k],
            diagnostics={"algorithm": "fake_selector"},
        )
