from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult


MOTION_MODELS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


class ECCAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.motion_model = str(config.get("motion_model", "affine"))
        if self.motion_model not in MOTION_MODELS:
            raise ValueError(f"unsupported ECC motion_model: {self.motion_model}")
        self.number_of_iterations = int(config.get("number_of_iterations", 100))
        self.termination_eps = float(config.get("termination_eps", 1.0e-5))
        self.gaussian_filter_size = int(config.get("gaussian_filter_size", 5))
        if self.number_of_iterations < 1:
            raise ValueError("number_of_iterations must be at least 1")
        if self.termination_eps <= 0:
            raise ValueError("termination_eps must be greater than 0")

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        try:
            lr_rgb_work = self._resize_rgb_to_hr(lr_rgb, hr_rgb)
            lr_gray = self._gray_float(lr_rgb_work)
            hr_gray = self._gray_float(hr_rgb)
            pre_error = self._mse(lr_gray, hr_gray)
            warp_matrix = self._initial_matrix()
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.number_of_iterations,
                self.termination_eps,
            )
            score, warp_matrix = cv2.findTransformECC(
                hr_gray,
                lr_gray,
                warp_matrix,
                MOTION_MODELS[self.motion_model],
                criteria,
                None,
                self.gaussian_filter_size,
            )
            aligned = self._warp(lr_rgb_work, hr_rgb, warp_matrix)
            post_error = self._mse(self._gray_float(aligned), hr_gray)
        except Exception as exc:
            return AlignResult(
                aligned_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
                diagnostics={"algorithm": "ecc_alignment", "motion_model": self.motion_model},
            )

        return AlignResult(
            aligned_lr_rgb=aligned,
            status="success",
            confidence=max(0.0, min(float(score), 1.0)),
            transforms=[
                {
                    "type": f"ecc_{self.motion_model}",
                    "coordinate_system": "lr_to_hr",
                    "matrix": warp_matrix.tolist(),
                }
            ],
            diagnostics={
                "algorithm": "ecc_alignment",
                "motion_model": self.motion_model,
                "ecc_score": float(score),
                "number_of_iterations": self.number_of_iterations,
                "termination_eps": self.termination_eps,
                "gaussian_filter_size": self.gaussian_filter_size,
                "pre_alignment_error": pre_error,
                "post_alignment_error": post_error,
            },
        )

    def _gray_float(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32) / 255.0

    def _initial_matrix(self) -> np.ndarray:
        if self.motion_model == "homography":
            return np.eye(3, 3, dtype=np.float32)
        return np.eye(2, 3, dtype=np.float32)

    def _warp(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        output_size = (hr_rgb.shape[1], hr_rgb.shape[0])
        if self.motion_model == "homography":
            return cv2.warpPerspective(
                lr_rgb,
                warp_matrix,
                output_size,
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
        return cv2.warpAffine(
            lr_rgb,
            warp_matrix,
            output_size,
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _resize_rgb_to_hr(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = hr_rgb.shape[:2]
        if lr_rgb.shape[:2] == (target_height, target_width):
            return lr_rgb
        return cv2.resize(lr_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _mse(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
