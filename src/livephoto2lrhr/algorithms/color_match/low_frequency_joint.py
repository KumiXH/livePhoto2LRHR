from __future__ import annotations

from typing import Any

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.shared import (
    confidence_from_errors,
    from_work_space,
    gaussian_low_high_split,
    mean_abs_delta,
    resize_to_lr,
    to_work_space,
)


class LowFrequencyJointAppearanceMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        joint_config = config.get("low_frequency_joint", {})
        self.color_space = str(_value(joint_config, "color_space", "lab")).lower()
        self.sigma = float(_value(joint_config, "sigma", 5.0))
        self.base_mix = float(_value(joint_config, "base_mix", 0.85))
        self.detail_preservation = float(_value(joint_config, "detail_preservation", 1.0))
        self.chroma_strength = float(_value(joint_config, "chroma_strength", 0.7))

    def match(self, lr_rgb, hr_rgb, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            lr_work = to_work_space(lr_rgb, self.color_space)
            hr_work = to_work_space(hr_reference, self.color_space)
            lr_base, lr_detail = gaussian_low_high_split(lr_work, self.sigma)
            hr_base, _ = gaussian_low_high_split(hr_work, self.sigma)
            matched_base = lr_base * (1.0 - self.base_mix) + hr_base * self.base_mix
            matched = matched_base + lr_detail * self.detail_preservation
            matched[..., 1:] = lr_work[..., 1:] * (1.0 - self.chroma_strength) + hr_work[..., 1:] * self.chroma_strength
            matched_rgb = from_work_space(matched, self.color_space)
            pre_error = mean_abs_delta(lr_work, hr_work)
            post_error = mean_abs_delta(to_work_space(matched_rgb, self.color_space), hr_work)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "low_frequency_joint_appearance_match"},
            )

        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[{"type": "low_frequency_joint_appearance_transfer"}],
            diagnostics={
                "algorithm": "low_frequency_joint_appearance_match",
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
