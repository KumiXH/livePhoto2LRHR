from __future__ import annotations

from typing import Any

import numpy as np

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.shared import confidence_from_errors, mean_abs_delta, resize_to_lr


class ImageAdaptive3DLUTColorMatcher:
    def __init__(self, config: dict[str, Any]) -> None:
        lut_config = config.get("adaptive_3d_lut", {})
        self.color_space = str(_value(lut_config, "color_space", "rgb")).lower()
        self.grid_size = int(_value(lut_config, "grid_size", 9))
        self.smoothing_sigma = float(_value(lut_config, "smoothing_sigma", 1.0))
        self.identity_mix = float(_value(lut_config, "identity_mix", 0.1))
        if self.color_space != "rgb":
            raise ValueError("adaptive_3d_lut currently supports only rgb")
        if self.grid_size < 2:
            raise ValueError("adaptive_3d_lut.grid_size must be >= 2")

    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        try:
            hr_reference = resize_to_lr(hr_rgb, lr_rgb)
            lut = self._build_lut(lr_rgb, hr_reference)
            matched_rgb = self._apply_lut(lr_rgb, lut)
            pre_error = mean_abs_delta(lr_rgb, hr_reference)
            post_error = mean_abs_delta(matched_rgb, hr_reference)
        except Exception as exc:
            return ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "image_adaptive_3d_lut_color_match"},
            )

        return ColorMatchResult(
            matched_lr_rgb=matched_rgb,
            status="success",
            confidence=confidence_from_errors(pre_error, post_error),
            transforms=[
                {
                    "type": "adaptive_3d_lut_color_transfer",
                    "color_space": self.color_space,
                    "grid_size": self.grid_size,
                    "lut": lut.tolist(),
                }
            ],
            diagnostics={
                "algorithm": "image_adaptive_3d_lut_color_match",
                "grid_size": self.grid_size,
                "pre_color_error": pre_error,
                "post_color_error": post_error,
            },
        )

    def _build_lut(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> np.ndarray:
        bins = np.linspace(0.0, 1.0, self.grid_size, dtype=np.float32)
        bin_ids = np.clip(np.round((lr_rgb.astype(np.float32) / 255.0) * (self.grid_size - 1)), 0, self.grid_size - 1).astype(np.int32)
        sum_lut = np.zeros((self.grid_size, self.grid_size, self.grid_size, 3), dtype=np.float32)
        count_lut = np.zeros((self.grid_size, self.grid_size, self.grid_size, 1), dtype=np.float32)
        for source_idx, target_value in zip(bin_ids.reshape(-1, 3), hr_rgb.reshape(-1, 3).astype(np.float32) / 255.0, strict=False):
            r, g, b = source_idx.tolist()
            sum_lut[r, g, b] += target_value
            count_lut[r, g, b] += 1.0
        identity = np.stack(np.meshgrid(bins, bins, bins, indexing="ij"), axis=3)
        lut = np.divide(sum_lut, np.maximum(count_lut, 1.0), where=count_lut > 0)
        lut[count_lut[..., 0] == 0] = identity[count_lut[..., 0] == 0]
        lut = lut * (1.0 - self.identity_mix) + identity * self.identity_mix
        return lut.astype(np.float32)

    def _apply_lut(self, image_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
        normalized = image_rgb.astype(np.float32) / 255.0
        coords = normalized * (self.grid_size - 1)
        low = np.floor(coords).astype(np.int32)
        high = np.clip(low + 1, 0, self.grid_size - 1)
        frac = coords - low
        out = np.zeros_like(normalized)
        for dr in (0, 1):
            wr = (1.0 - frac[..., 0]) if dr == 0 else frac[..., 0]
            rr = low[..., 0] if dr == 0 else high[..., 0]
            for dg in (0, 1):
                wg = (1.0 - frac[..., 1]) if dg == 0 else frac[..., 1]
                gg = low[..., 1] if dg == 0 else high[..., 1]
                for db in (0, 1):
                    wb = (1.0 - frac[..., 2]) if db == 0 else frac[..., 2]
                    bb = low[..., 2] if db == 0 else high[..., 2]
                    weight = (wr * wg * wb)[..., None]
                    out += lut[rr, gg, bb] * weight
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
