from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult


class MeanStdColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        self.color_space = str(config.get("color_space", "lab")).lower()
        self.eps = float(config.get("eps", 1.0e-6))
        if self.color_space not in {"lab", "rgb"}:
            raise ValueError(f"unsupported mean/std color space: {self.color_space}")
        if self.eps <= 0:
            raise ValueError("eps must be greater than 0")

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = self._resize_to_lr(hr_rgb, lr_rgb)
            lr_work = self._to_work_space(lr_rgb)
            hr_work = self._to_work_space(hr_reference)
            pre_error = self._mean_abs_delta(lr_work, hr_work)

            lr_mean, lr_std = self._channel_stats(lr_work)
            hr_mean, hr_std = self._channel_stats(hr_work)
            matched_work = (lr_work - lr_mean) * (hr_std / np.maximum(lr_std, self.eps)) + hr_mean
            matched_work = np.clip(matched_work, 0, 255).astype(np.uint8)
            post_error = self._mean_abs_delta(matched_work, hr_work)
            matched_rgb = self._from_work_space(matched_work)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "mean_std_lab", "color_space": self.color_space},
            )

        confidence = 1.0
        if pre_error > 0:
            confidence = max(0.0, min(1.0, 1.0 - post_error / pre_error))
        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence,
            transforms=[
                {
                    "type": "mean_std_color_transfer",
                    "color_space": self.color_space,
                    "source_mean": lr_mean.reshape(-1).tolist(),
                    "source_std": lr_std.reshape(-1).tolist(),
                    "target_mean": hr_mean.reshape(-1).tolist(),
                    "target_std": hr_std.reshape(-1).tolist(),
                }
            ],
            diagnostics={
                "algorithm": "mean_std_lab",
                "color_space": self.color_space,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
            },
        )

    def _resize_to_lr(self, hr_rgb: np.ndarray, lr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = lr_rgb.shape[:2]
        if hr_rgb.shape[:2] == (target_height, target_width):
            return hr_rgb
        return cv2.resize(hr_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _to_work_space(self, image_rgb: np.ndarray) -> np.ndarray:
        if self.color_space == "lab":
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        return image_rgb.astype(np.float32)

    def _from_work_space(self, image: np.ndarray) -> np.ndarray:
        if self.color_space == "lab":
            return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        return image

    def _channel_stats(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.mean(image, axis=(0, 1), keepdims=True), np.std(image, axis=(0, 1), keepdims=True)

    def _mean_abs_delta(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))
