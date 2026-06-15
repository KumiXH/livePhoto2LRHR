from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult


class RetinexColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        retinex_config = config.get("retinex", {})
        self.sigma = float(self._value(retinex_config, "sigma", 15.0))
        self.eps = float(self._value(retinex_config, "eps", 1.0e-3))
        if self.sigma <= 0:
            raise ValueError("retinex.sigma must be greater than 0")
        if self.eps <= 0:
            raise ValueError("retinex.eps must be greater than 0")

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = self._resize_to_lr(hr_rgb, lr_rgb)
            lr_lab = cv2.cvtColor(lr_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
            hr_lab = cv2.cvtColor(hr_reference, cv2.COLOR_RGB2LAB).astype(np.float32)
            pre_error = self._mean_abs_delta(lr_lab, hr_lab)

            matched_lab = lr_lab.copy()
            matched_lab[..., 0] = self._match_luminance(lr_lab[..., 0], hr_lab[..., 0])
            matched_lab[..., 1:] = self._match_chroma(lr_lab[..., 1:], hr_lab[..., 1:])
            matched_lab = np.clip(matched_lab, 0, 255).astype(np.uint8)
            post_error = self._mean_abs_delta(matched_lab, hr_lab)
            matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "retinex_color_match"},
            )

        confidence = 1.0 if pre_error <= 0 else max(0.0, min(1.0, 1.0 - post_error / pre_error))
        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence,
            transforms=[
                {
                    "type": "retinex_color_transfer",
                    "sigma": self.sigma,
                    "eps": self.eps,
                }
            ],
            diagnostics={
                "algorithm": "retinex_color_match",
                "sigma": self.sigma,
                "eps": self.eps,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
                "replay_reference_required": True,
            },
        )

    def _match_luminance(self, lr_l: np.ndarray, hr_l: np.ndarray) -> np.ndarray:
        lr_base = cv2.GaussianBlur(lr_l, (0, 0), self.sigma)
        hr_base = cv2.GaussianBlur(hr_l, (0, 0), self.sigma)
        reflectance = (lr_l + self.eps) / (lr_base + self.eps)
        matched = reflectance * hr_base
        matched_mean = float(np.mean(matched))
        matched_std = float(np.std(matched))
        target_mean = float(np.mean(hr_l))
        target_std = float(np.std(hr_l))
        if matched_std > self.eps and target_std > 0:
            matched = (matched - matched_mean) * (target_std / matched_std) + target_mean
        else:
            matched = matched - matched_mean + target_mean
        return matched

    def _match_chroma(self, lr_ab: np.ndarray, hr_ab: np.ndarray) -> np.ndarray:
        lr_mean = np.mean(lr_ab, axis=(0, 1), keepdims=True)
        lr_std = np.std(lr_ab, axis=(0, 1), keepdims=True)
        hr_mean = np.mean(hr_ab, axis=(0, 1), keepdims=True)
        hr_std = np.std(hr_ab, axis=(0, 1), keepdims=True)
        return (lr_ab - lr_mean) * (hr_std / np.maximum(lr_std, self.eps)) + hr_mean

    def _resize_to_lr(self, hr_rgb: np.ndarray, lr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = lr_rgb.shape[:2]
        if hr_rgb.shape[:2] == (target_height, target_width):
            return hr_rgb
        return cv2.resize(hr_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _mean_abs_delta(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))

    def _value(self, config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
