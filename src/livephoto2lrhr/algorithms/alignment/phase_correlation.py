from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult


class PhaseCorrelationTranslationAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.resize_short_side = int(config.get("resize_short_side", 512))
        if self.resize_short_side < 1:
            raise ValueError("resize_short_side must be at least 1")

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        lr_rgb_work = self._resize_rgb_to_hr(lr_rgb, hr_rgb)
        lr_gray = self._prepare_gray(lr_rgb_work)
        hr_gray = self._prepare_gray(hr_rgb)
        lr_small, hr_small, scale_x, scale_y = self._resize_pair(lr_gray, hr_gray)
        pre_error = self._mse(lr_gray, hr_gray)

        (shift_x, shift_y), response = cv2.phaseCorrelate(
            np.float32(lr_small),
            np.float32(hr_small),
        )
        dx = float(shift_x * scale_x)
        dy = float(shift_y * scale_y)
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        aligned = cv2.warpAffine(
            lr_rgb_work,
            matrix,
            (hr_rgb.shape[1], hr_rgb.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        post_error = self._mse(self._prepare_gray(aligned), hr_gray)

        return AlignResult(
            aligned_lr_rgb=aligned,
            status="success",
            confidence=max(0.0, min(float(response), 1.0)),
            transforms=[
                {
                    "type": "translation",
                    "coordinate_system": "lr_to_hr",
                    "dx": dx,
                    "dy": dy,
                    "matrix": matrix.tolist(),
                }
            ],
            diagnostics={
                "algorithm": "phase_correlation_translation",
                "response": float(response),
                "resize_short_side": self.resize_short_side,
                "pre_alignment_error": pre_error,
                "post_alignment_error": post_error,
            },
        )

    def _prepare_gray(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def _resize_pair(
        self, lr_gray: np.ndarray, hr_gray: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        target_height, target_width = hr_gray.shape[:2]
        if lr_gray.shape[:2] != (target_height, target_width):
            lr_gray = cv2.resize(lr_gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

        short_side = min(target_height, target_width)
        if short_side <= self.resize_short_side:
            return lr_gray, hr_gray, 1.0, 1.0

        scale = self.resize_short_side / short_side
        new_width = max(int(round(target_width * scale)), 1)
        new_height = max(int(round(target_height * scale)), 1)
        lr_small = cv2.resize(lr_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        hr_small = cv2.resize(hr_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return lr_small, hr_small, target_width / new_width, target_height / new_height

    def _resize_rgb_to_hr(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = hr_rgb.shape[:2]
        if lr_rgb.shape[:2] == (target_height, target_width):
            return lr_rgb
        return cv2.resize(lr_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _mse(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
