from __future__ import annotations

from typing import Any

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.image_adaptive_3d_lut import ImageAdaptive3DLUTColorMatcher
from livephoto2lrhr.algorithms.color_match.low_frequency_joint import LowFrequencyJointAppearanceMatcher
from livephoto2lrhr.algorithms.color_match.shared import confidence_from_errors, mean_abs_delta, resize_to_lr


class DiffusionHarmonizationMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        diffusion_config = config.get("diffusion_harmonization", {})
        self.color_space = str(_value(diffusion_config, "color_space", "lab")).lower()
        self.num_steps = int(_value(diffusion_config, "num_steps", 4))
        self.guidance_strength = float(_value(diffusion_config, "guidance_strength", 0.6))
        self.low_frequency_sigma = float(_value(diffusion_config, "low_frequency_sigma", 4.0))
        self.lut_identity_mix = float(_value(diffusion_config, "lut_identity_mix", 0.15))
        self.low_frequency_matcher = LowFrequencyJointAppearanceMatcher(
            {
                "low_frequency_joint": {
                    "color_space": self.color_space,
                    "sigma": self.low_frequency_sigma,
                    "base_mix": self.guidance_strength,
                    "detail_preservation": 1.0,
                    "chroma_strength": self.guidance_strength,
                }
            }
        )
        self.lut_matcher = ImageAdaptive3DLUTColorMatcher(
            {
                "adaptive_3d_lut": {
                    "color_space": "rgb",
                    "grid_size": 9,
                    "smoothing_sigma": 1.0,
                    "identity_mix": self.lut_identity_mix,
                }
            }
        )

    def match(self, lr_rgb, hr_rgb, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            current = lr_rgb
            for _ in range(max(self.num_steps, 1)):
                low_result = self.low_frequency_matcher.match(current, hr_reference, context)
                if low_result.matched_lr_rgb is None:
                    raise ValueError(low_result.message or "low-frequency harmonization failed")
                lut_result = self.lut_matcher.match(low_result.matched_lr_rgb, hr_reference, context)
                if lut_result.matched_lr_rgb is None:
                    raise ValueError(lut_result.message or "adaptive LUT harmonization failed")
                current = lut_result.matched_lr_rgb
            pre_error = mean_abs_delta(lr_rgb, hr_reference)
            post_error = mean_abs_delta(current, hr_reference)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "diffusion_harmonization"},
            )

        return ColorMatchResult(
            matched_lr_rgb=current,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[{"type": "diffusion_harmonization_transfer"}],
            diagnostics={
                "algorithm": "diffusion_harmonization",
                "implementation_mode": "proxy",
                "num_steps": self.num_steps,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
                "replay_reference_required": True,
            },
        )


def _value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
