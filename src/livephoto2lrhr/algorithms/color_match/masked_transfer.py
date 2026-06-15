from __future__ import annotations

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.shared import (
    blend_with_soft_mask,
    build_difference_mask,
    channel_transfer,
    confidence_from_errors,
    from_work_space,
    mean_abs_delta,
    resize_to_lr,
    to_work_space,
)


class MaskedColorTransferMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        masked_config = config.get("masked_transfer", {})
        self.color_space = str(self._value(masked_config, "color_space", "lab")).lower()
        self.difference_threshold = float(self._value(masked_config, "difference_threshold", 12.0))
        self.min_mask_fraction = float(self._value(masked_config, "min_mask_fraction", 0.01))
        self.max_mask_fraction = float(self._value(masked_config, "max_mask_fraction", 0.85))
        self.morphology_kernel_size = int(self._value(masked_config, "morphology_kernel_size", 5))
        self.eps = 1.0e-6
        if self.color_space not in {"lab", "rgb"}:
            raise ValueError(f"unsupported masked_transfer color space: {self.color_space}")
        if self.difference_threshold <= 0:
            raise ValueError("masked_transfer.difference_threshold must be greater than 0")
        if not 0.0 <= self.min_mask_fraction <= 1.0:
            raise ValueError("masked_transfer.min_mask_fraction must be in [0, 1]")
        if not 0.0 <= self.max_mask_fraction <= 1.0:
            raise ValueError("masked_transfer.max_mask_fraction must be in [0, 1]")
        if self.min_mask_fraction > self.max_mask_fraction:
            raise ValueError("masked_transfer.min_mask_fraction cannot exceed max_mask_fraction")
        if self.morphology_kernel_size < 1:
            raise ValueError("masked_transfer.morphology_kernel_size must be >= 1")

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            lr_work = to_work_space(lr_rgb, self.color_space)
            hr_work = to_work_space(hr_reference, self.color_space)
            pre_error = mean_abs_delta(lr_work, hr_work)

            mask = build_difference_mask(
                lr_work,
                hr_work,
                difference_threshold=self.difference_threshold,
                morphology_kernel_size=self.morphology_kernel_size,
            )
            mask_fraction = float(mask.mean())
            global_matched = channel_transfer(lr_work, hr_work, eps=self.eps)
            if self.min_mask_fraction <= mask_fraction <= self.max_mask_fraction:
                foreground_matched = channel_transfer(lr_work, hr_work, eps=self.eps, mask=mask)
                background_matched = channel_transfer(lr_work, hr_work, eps=self.eps, mask=~mask)
                combined = background_matched
                combined[mask] = foreground_matched[mask]
                matched_work = blend_with_soft_mask(
                    global_matched,
                    combined,
                    mask,
                    blur_radius=self.morphology_kernel_size * 2 + 1,
                )
            else:
                matched_work = global_matched
            matched_rgb = from_work_space(matched_work, self.color_space)
            post_error = mean_abs_delta(to_work_space(matched_rgb, self.color_space), hr_work)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "masked_color_transfer"},
            )

        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[
                {
                    "type": "masked_mean_std_color_transfer",
                    "color_space": self.color_space,
                }
            ],
            diagnostics={
                "algorithm": "masked_color_transfer",
                "color_space": self.color_space,
                "difference_threshold": self.difference_threshold,
                "mask_fraction": mask_fraction,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
                "replay_reference_required": True,
            },
        )

    def _value(self, config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
