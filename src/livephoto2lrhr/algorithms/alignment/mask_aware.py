from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult
from livephoto2lrhr.algorithms.alignment.ecc import ECCAligner
from livephoto2lrhr.algorithms.alignment.feature_match_transform import FeatureMatchTransformAligner
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.phase_correlation import PhaseCorrelationTranslationAligner


class MaskAwareAlignmentAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.coarse_algorithm = str(config.get("coarse_algorithm", "phase_correlation_translation"))
        mask_aware_config = config.get("mask_aware", {})
        self.motion_model = str(self._value(mask_aware_config, "motion_model", "translation")).lower()
        self.difference_threshold = float(self._value(mask_aware_config, "difference_threshold", 18.0))
        self.min_mask_fraction = float(self._value(mask_aware_config, "min_mask_fraction", 0.02))
        self.blend_blur_ksize = int(self._value(mask_aware_config, "blend_blur_ksize", 9))
        self.morphology_kernel_size = int(self._value(mask_aware_config, "morphology_kernel_size", 5))
        self.coarse_aligner = self._create_coarse_aligner(self.coarse_algorithm, config)
        if self.motion_model not in {"translation", "affine"}:
            raise ValueError(f"unsupported mask_aware motion_model: {self.motion_model}")

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        coarse_result = self.coarse_aligner.align(lr_rgb, hr_rgb, context)
        diagnostics = dict(coarse_result.diagnostics)
        diagnostics["algorithm"] = "mask_aware_alignment"
        diagnostics["coarse_algorithm"] = self.coarse_algorithm
        diagnostics["mask_refinement_used"] = False
        if coarse_result.status != "success" or coarse_result.aligned_lr_rgb is None:
            return AlignResult(
                aligned_lr_rgb=coarse_result.aligned_lr_rgb,
                status=coarse_result.status,
                confidence=coarse_result.confidence,
                message=coarse_result.message,
                transforms=coarse_result.transforms,
                artifacts=coarse_result.artifacts,
                diagnostics=diagnostics,
            )

        coarse_rgb = coarse_result.aligned_lr_rgb
        pre_error = self._mse(self._gray_float(coarse_rgb), self._gray_float(hr_rgb))
        try:
            refinement = self._refine_subject_region(coarse_rgb, hr_rgb)
        except Exception as exc:
            diagnostics["mask_refinement_error"] = str(exc)
            return AlignResult(
                aligned_lr_rgb=coarse_rgb,
                status=coarse_result.status,
                confidence=coarse_result.confidence,
                message=coarse_result.message,
                transforms=coarse_result.transforms,
                artifacts=coarse_result.artifacts,
                diagnostics=diagnostics,
            )

        diagnostics.update(refinement["diagnostics"])
        if not refinement["accepted"]:
            return AlignResult(
                aligned_lr_rgb=coarse_rgb,
                status=coarse_result.status,
                confidence=coarse_result.confidence,
                message=coarse_result.message,
                transforms=coarse_result.transforms,
                artifacts=coarse_result.artifacts,
                diagnostics=diagnostics,
            )

        diagnostics["mask_refinement_used"] = True
        transforms = [
            *coarse_result.transforms,
            {
                "type": f"mask_aware_{self.motion_model}",
                "coordinate_system": "lr_to_hr",
                "difference_threshold": self.difference_threshold,
            },
        ]
        return AlignResult(
            aligned_lr_rgb=refinement["aligned"],
            status="success",
            confidence=max(coarse_result.confidence, refinement["confidence"]),
            message=coarse_result.message,
            transforms=transforms,
            artifacts=coarse_result.artifacts,
            diagnostics=diagnostics,
        )

    def _refine_subject_region(self, coarse_rgb: np.ndarray, hr_rgb: np.ndarray) -> dict[str, Any]:
        coarse_gray = self._gray_uint8(coarse_rgb)
        hr_gray = self._gray_uint8(hr_rgb)
        diff_gray = cv2.absdiff(coarse_gray, hr_gray)
        diff_rgb = np.max(cv2.absdiff(coarse_rgb, hr_rgb), axis=2).astype(np.uint8)
        diff = np.maximum(diff_gray, diff_rgb)
        _, mask = cv2.threshold(diff, self.difference_threshold, 255, cv2.THRESH_BINARY)
        kernel_size = max(self.morphology_kernel_size, 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if mask_fraction < self.min_mask_fraction:
            return {
                "accepted": False,
                "aligned": coarse_rgb,
                "confidence": 0.0,
                "diagnostics": {
                    "mask_fraction": mask_fraction,
                    "mask_refinement_status": "skipped_small_mask",
                },
            }

        points = cv2.findNonZero(mask)
        if points is None:
            return {
                "accepted": False,
                "aligned": coarse_rgb,
                "confidence": 0.0,
                "diagnostics": {
                    "mask_fraction": mask_fraction,
                    "mask_refinement_status": "skipped_empty_mask",
                },
            }
        x, y, w, h = cv2.boundingRect(points)
        pad = max(4, kernel_size)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, coarse_rgb.shape[1])
        y1 = min(y + h + pad, coarse_rgb.shape[0])
        source_patch = coarse_gray[y0:y1, x0:x1].astype(np.float32) / 255.0
        target_patch = hr_gray[y0:y1, x0:x1].astype(np.float32) / 255.0
        patch_matrix = self._estimate_patch_transform(source_patch, target_patch)
        if patch_matrix is None:
            return {
                "accepted": False,
                "aligned": coarse_rgb,
                "confidence": 0.0,
                "diagnostics": {
                    "mask_fraction": mask_fraction,
                    "mask_refinement_status": "failed_transform_estimation",
                },
            }

        refined_patch = cv2.warpAffine(
            coarse_rgb[y0:y1, x0:x1],
            patch_matrix,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
        blend_mask = cv2.GaussianBlur(mask[y0:y1, x0:x1].astype(np.float32) / 255.0, (self._odd(self.blend_blur_ksize), self._odd(self.blend_blur_ksize)), 0)
        blend_mask = np.clip(blend_mask[..., None], 0.0, 1.0)
        refined = coarse_rgb.copy().astype(np.float32)
        refined_region = (
            refined_patch.astype(np.float32) * blend_mask
            + coarse_rgb[y0:y1, x0:x1].astype(np.float32) * (1.0 - blend_mask)
        )
        refined[y0:y1, x0:x1] = refined_region
        refined = np.clip(refined, 0, 255).astype(np.uint8)
        pre_error = self._mse(self._gray_float(coarse_rgb), self._gray_float(hr_rgb))
        post_error = self._mse(self._gray_float(refined), self._gray_float(hr_rgb))
        accepted = post_error < pre_error
        return {
            "accepted": accepted,
            "aligned": refined if accepted else coarse_rgb,
            "confidence": max(0.0, min(1.0, 1.0 - post_error / max(pre_error, 1.0e-6))),
            "diagnostics": {
                "mask_fraction": mask_fraction,
                "mask_refinement_status": "accepted" if accepted else "rejected_no_improvement",
                "mask_pre_alignment_error": pre_error,
                "mask_post_alignment_error": post_error,
                "mask_bbox": [int(x0), int(y0), int(x1), int(y1)],
            },
        }

    def _estimate_patch_transform(self, source_patch: np.ndarray, target_patch: np.ndarray) -> np.ndarray | None:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        motion_model = cv2.MOTION_TRANSLATION if self.motion_model == "translation" else cv2.MOTION_AFFINE
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1.0e-5)
        try:
            _, warp_matrix = cv2.findTransformECC(
                target_patch,
                source_patch,
                warp_matrix,
                motion_model,
                criteria,
                None,
                3,
            )
        except cv2.error:
            return None
        return warp_matrix

    def _create_coarse_aligner(self, name: str, config: dict[str, Any]):
        if name == "identity_alignment":
            return IdentityAligner(config)
        if name == "phase_correlation_translation":
            effective_config = dict(config)
            phase_config = config.get("phase_correlation")
            if phase_config is not None:
                resize_short_side = self._value(phase_config, "resize_short_side", config.get("resize_short_side", 512))
                effective_config["resize_short_side"] = resize_short_side
            return PhaseCorrelationTranslationAligner(effective_config)
        if name == "ecc_alignment":
            return ECCAligner(config)
        if name in {"feature_match_transform", "feature_match_homography"}:
            effective_config = dict(config)
            if name == "feature_match_homography":
                feature_match_config = dict(config.get("feature_match", {}))
                feature_match_config["transform_model"] = "homography"
                effective_config["feature_match"] = feature_match_config
            return FeatureMatchTransformAligner(effective_config)
        raise KeyError(f"unknown coarse alignment algorithm: {name}")

    def _gray_uint8(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def _gray_float(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._gray_uint8(image_rgb).astype(np.float32) / 255.0

    def _mse(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))

    def _odd(self, value: int) -> int:
        value = max(int(value), 1)
        return value if value % 2 == 1 else value + 1

    def _value(self, config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
