from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult


class HistogramMatchColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        histogram_config = config.get("histogram_match", {})
        self.color_space = str(self._value(histogram_config, "color_space", "lab")).lower()
        self.bins = int(self._value(histogram_config, "bins", 256))
        if self.color_space not in {"lab", "rgb"}:
            raise ValueError(f"unsupported histogram_match color space: {self.color_space}")
        if self.bins < 2 or self.bins > 256:
            raise ValueError("histogram_match.bins must be in [2, 256]")

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = self._resize_to_lr(hr_rgb, lr_rgb)
            lr_work = self._to_work_space(lr_rgb)
            hr_work = self._to_work_space(hr_reference)
            pre_error = self._mean_abs_delta(lr_work, hr_work)
            matched_work = self._match_histograms(lr_work, hr_work)
            post_error = self._mean_abs_delta(matched_work, hr_work)
            matched_rgb = self._from_work_space(matched_work)
            confidence = 1.0 if pre_error <= 0 else max(0.0, min(1.0, 1.0 - post_error / pre_error))
            return ColorMatchResult(
                matched_lr_rgb=matched_rgb,
                status="success",
                confidence=confidence,
                transforms=[
                    {
                        "type": "histogram_color_transfer",
                        "color_space": self.color_space,
                        "bins": self.bins,
                    }
                ],
                diagnostics={
                    "algorithm": "histogram_match_lab",
                    "color_space": self.color_space,
                    "bins": self.bins,
                    "pre_color_error": pre_error,
                    "post_color_error": post_error,
                    "replay_reference_required": True,
                },
            )
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "histogram_match_lab", "color_space": self.color_space},
            )

    def _resize_to_lr(self, hr_rgb: np.ndarray, lr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = lr_rgb.shape[:2]
        if hr_rgb.shape[:2] == (target_height, target_width):
            return hr_rgb
        return cv2.resize(hr_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _to_work_space(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.color_space == "lab":
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        return image_rgb.copy()

    def _from_work_space(self, image: np.ndarray) -> np.ndarray:
        if self.color_space == "lab":
            return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        return image

    def _match_histograms(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        matched_channels = [
            self._match_single_channel(source[..., channel], target[..., channel])
            for channel in range(source.shape[2])
        ]
        return np.stack(matched_channels, axis=2)

    def _match_single_channel(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_values = source.reshape(-1)
        target_values = target.reshape(-1)
        source_hist, _ = np.histogram(source_values, bins=self.bins, range=(0, 256), density=False)
        target_hist, _ = np.histogram(target_values, bins=self.bins, range=(0, 256), density=False)
        source_cdf = np.cumsum(source_hist).astype(np.float64)
        target_cdf = np.cumsum(target_hist).astype(np.float64)
        if source_cdf[-1] <= 0 or target_cdf[-1] <= 0:
            return source.copy()
        source_cdf /= source_cdf[-1]
        target_cdf /= target_cdf[-1]
        source_bin_indices = np.clip((source_values.astype(np.float32) / 256.0 * self.bins).astype(np.int32), 0, self.bins - 1)
        source_quantiles = source_cdf[source_bin_indices]
        target_bin_indices = np.searchsorted(target_cdf, source_quantiles, side="left")
        target_bin_indices = np.clip(target_bin_indices, 0, self.bins - 1)
        bin_width = 256.0 / self.bins
        mapped = np.clip((target_bin_indices.astype(np.float32) + 0.5) * bin_width, 0, 255).astype(np.uint8)
        return mapped.reshape(source.shape)

    def _mean_abs_delta(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))

    def _value(self, config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
