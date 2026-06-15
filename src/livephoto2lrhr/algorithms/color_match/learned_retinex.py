from __future__ import annotations

from typing import Any

import cv2

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.shared import (
    confidence_from_errors,
    from_work_space,
    gaussian_low_high_split,
    mean_abs_delta,
    resize_to_lr,
    to_work_space,
)


class LearnedRetinexColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        learned_config = config.get("learned_retinex", {})
        self.sigma = float(_value(learned_config, "sigma", 15.0))
        self.eps = float(_value(learned_config, "eps", 1.0e-3))
        self.base_mix = float(_value(learned_config, "base_mix", 0.65))

    def match(self, lr_rgb, hr_rgb, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            lr_lab = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2LAB).astype("float32")
            hr_lab = cv2.cvtColor(hr_reference, cv2.COLOR_RGB2LAB).astype("float32")
            lr_l = lr_lab[..., 0]
            hr_l = hr_lab[..., 0]
            lr_base = cv2.GaussianBlur(lr_l, (0, 0), self.sigma)
            hr_base = cv2.GaussianBlur(hr_l, (0, 0), self.sigma)
            reflectance = (lr_l + self.eps) / (lr_base + self.eps)
            target_base = lr_base * (1.0 - self.base_mix) + hr_base * self.base_mix
            matched_l = reflectance * target_base
            matched = lr_lab.copy()
            matched[..., 0] = matched_l
            matched[..., 1:] = lr_lab[..., 1:] * 0.3 + hr_lab[..., 1:] * 0.7
            matched_rgb = from_work_space(matched, "lab")
            pre_error = mean_abs_delta(lr_lab, hr_lab)
            post_error = mean_abs_delta(to_work_space(matched_rgb, "lab"), hr_lab)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "learned_retinex_color_match"},
            )

        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[{"type": "learned_retinex_color_transfer"}],
            diagnostics={
                "algorithm": "learned_retinex_color_match",
                "implementation_mode": "proxy",
                "sigma": self.sigma,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
                "replay_reference_required": True,
            },
        )


def _value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
