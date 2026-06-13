from __future__ import annotations

from typing import Any

import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult


class IdentityAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        return AlignResult(
            aligned_lr_rgb=lr_rgb.copy(),
            status="success",
            confidence=1.0,
            transforms=[{"type": "identity", "coordinate_system": "lr_to_hr"}],
            artifacts=[],
            diagnostics={"algorithm": "identity_alignment"},
        )
