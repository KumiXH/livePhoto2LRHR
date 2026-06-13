from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult
from livephoto2lrhr.algorithms.alignment.ecc import ECCAligner
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.phase_correlation import PhaseCorrelationTranslationAligner


class CoarseToFlowAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.coarse_algorithm = str(config.get("coarse_algorithm", "phase_correlation_translation"))
        self.optical_flow = config.get("optical_flow", {})
        self.coarse_aligner = self._create_coarse_aligner(self.coarse_algorithm, config)

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        coarse_result = self.coarse_aligner.align(lr_rgb, hr_rgb, context)
        diagnostics = dict(coarse_result.diagnostics)
        diagnostics["algorithm"] = "coarse_to_flow"
        diagnostics["coarse_algorithm"] = self.coarse_algorithm
        diagnostics["flow_used"] = False
        diagnostics["flow_algorithm"] = self._flow_value("algorithm", "dis")
        message = coarse_result.message
        if self._flow_enabled() and coarse_result.status == "success" and coarse_result.aligned_lr_rgb is not None:
            flow_result = self._refine_with_flow(coarse_result.aligned_lr_rgb, hr_rgb)
            diagnostics.update(flow_result["diagnostics"])
            if flow_result["accepted"]:
                diagnostics["flow_used"] = True
                return AlignResult(
                    aligned_lr_rgb=flow_result["aligned"],
                    status=coarse_result.status,
                    confidence=coarse_result.confidence,
                    message=message,
                    transforms=[
                        *coarse_result.transforms,
                        {
                            "type": "dense_flow",
                            "coordinate_system": "lr_to_hr",
                            "algorithm": diagnostics["flow_algorithm"],
                        },
                    ],
                    artifacts=coarse_result.artifacts,
                    diagnostics=diagnostics,
                )
            message = coarse_result.message or "optical flow refinement did not improve error; returned coarse result"
        elif self._flow_enabled():
            diagnostics["flow_status"] = "skipped_coarse_failed"
        else:
            diagnostics["flow_status"] = "disabled"
        return AlignResult(
            aligned_lr_rgb=coarse_result.aligned_lr_rgb,
            status=coarse_result.status,
            confidence=coarse_result.confidence,
            message=message,
            transforms=coarse_result.transforms,
            artifacts=coarse_result.artifacts,
            diagnostics=diagnostics,
        )

    def _create_coarse_aligner(self, name: str, config: dict[str, Any]):
        if name == "identity_alignment":
            return IdentityAligner(config)
        if name == "phase_correlation_translation":
            return PhaseCorrelationTranslationAligner(config)
        if name == "ecc_alignment":
            return ECCAligner(config)
        raise KeyError(f"unknown coarse alignment algorithm: {name}")

    def _flow_enabled(self) -> bool:
        return bool(self._flow_value("enabled", False))

    def _flow_value(self, key: str, default: Any) -> Any:
        if isinstance(self.optical_flow, dict):
            return self.optical_flow.get(key, default)
        return getattr(self.optical_flow, key, default)

    def _refine_with_flow(self, coarse_lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> dict[str, Any]:
        coarse_lr_rgb = self._resize_rgb_to_hr(coarse_lr_rgb, hr_rgb)
        lr_gray = self._gray_float(coarse_lr_rgb)
        hr_gray = self._gray_float(hr_rgb)
        pre_error = self._mse(lr_gray, hr_gray)
        algorithm = str(self._flow_value("algorithm", "dis")).lower()
        try:
            flow = self._compute_flow(lr_gray, hr_gray, algorithm)
            refined = self._warp_with_flow(coarse_lr_rgb, flow)
            post_error = self._mse(self._gray_float(refined), hr_gray)
        except Exception as exc:
            return {
                "accepted": False,
                "aligned": coarse_lr_rgb,
                "diagnostics": {
                    "flow_status": "failed",
                    "flow_error": str(exc),
                    "pre_flow_error": pre_error,
                },
            }
        accepted = post_error < pre_error
        return {
            "accepted": accepted,
            "aligned": refined if accepted else coarse_lr_rgb,
            "diagnostics": {
                "flow_status": "accepted" if accepted else "rejected_no_improvement",
                "pre_flow_error": pre_error,
                "post_flow_error": post_error,
                "mean_flow_magnitude": float(np.mean(np.linalg.norm(flow, axis=2))),
            },
        }

    def _compute_flow(self, lr_gray: np.ndarray, hr_gray: np.ndarray, algorithm: str) -> np.ndarray:
        if algorithm == "dis":
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            return dis.calc(self._float_gray_to_uint8(lr_gray), self._float_gray_to_uint8(hr_gray), None)
        if algorithm == "farneback":
            return cv2.calcOpticalFlowFarneback(
                lr_gray,
                hr_gray,
                None,
                0.5,
                3,
                21,
                5,
                7,
                1.5,
                0,
            )
        raise ValueError(f"unsupported optical flow algorithm: {algorithm}")

    def _warp_with_flow(self, image_rgb: np.ndarray, flow: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_x = grid_x - flow[..., 0].astype(np.float32)
        map_y = grid_y - flow[..., 1].astype(np.float32)
        return cv2.remap(
            image_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _resize_rgb_to_hr(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = hr_rgb.shape[:2]
        if lr_rgb.shape[:2] == (target_height, target_width):
            return lr_rgb
        return cv2.resize(lr_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _gray_float(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    def _float_gray_to_uint8(self, image: np.ndarray) -> np.ndarray:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def _mse(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))
