from __future__ import annotations

from typing import Any

import numpy as np

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult


class IdentityColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        return ColorMatchResult(
            matched_lr_rgb=lr_rgb.copy(),
            status="success",
            confidence=1.0,
            transforms=[{"type": "identity_color", "color_space": "rgb"}],
            artifacts=[],
            diagnostics={"algorithm": "identity_color_match"},
        )
