from __future__ import annotations

from typing import Any

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.shared import (
    blend_with_soft_mask,
    build_difference_mask,
    channel_transfer,
    confidence_from_errors,
    from_work_space,
    gaussian_low_high_split,
    mean_abs_delta,
    resize_to_lr,
    to_work_space,
)


class MaskAwareHarmonizationNetworkMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        harmonization_config = config.get("mask_aware_harmonization", {})
        self.color_space = str(_value(harmonization_config, "color_space", "lab")).lower()
        self.difference_threshold = float(_value(harmonization_config, "difference_threshold", 12.0))
        self.min_mask_fraction = float(_value(harmonization_config, "min_mask_fraction", 0.01))
        self.max_mask_fraction = float(_value(harmonization_config, "max_mask_fraction", 0.85))
        self.morphology_kernel_size = int(_value(harmonization_config, "morphology_kernel_size", 5))
        self.low_frequency_sigma = float(_value(harmonization_config, "low_frequency_sigma", 5.0))

    def match(self, lr_rgb, hr_rgb, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            lr_work = to_work_space(lr_rgb, self.color_space)
            hr_work = to_work_space(hr_reference, self.color_space)
            mask = build_difference_mask(
                lr_work,
                hr_work,
                difference_threshold=self.difference_threshold,
                morphology_kernel_size=self.morphology_kernel_size,
            )
            global_match = channel_transfer(lr_work, hr_work)
            mask_fraction = float(mask.mean())
            if self.min_mask_fraction <= mask_fraction <= self.max_mask_fraction:
                base_lr, detail_lr = gaussian_low_high_split(lr_work, self.low_frequency_sigma)
                base_hr, _ = gaussian_low_high_split(hr_work, self.low_frequency_sigma)
                region_base = channel_transfer(base_lr, base_hr, mask=mask)
                region_match = region_base + detail_lr
                matched = blend_with_soft_mask(
                    global_match,
                    region_match,
                    mask,
                    blur_radius=self.morphology_kernel_size * 2 + 1,
                )
            else:
                matched = global_match
            matched_rgb = from_work_space(matched, self.color_space)
            pre_error = mean_abs_delta(lr_work, hr_work)
            post_error = mean_abs_delta(to_work_space(matched_rgb, self.color_space), hr_work)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "mask_aware_harmonization_network"},
            )

        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[{"type": "mask_aware_harmonization_transfer"}],
            diagnostics={
                "algorithm": "mask_aware_harmonization_network",
                "implementation_mode": "proxy",
                "mask_fraction": mask_fraction,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
                "replay_reference_required": True,
            },
        )


def _value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
