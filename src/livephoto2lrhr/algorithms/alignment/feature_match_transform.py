from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult


class FeatureMatchTransformAligner:
    def __init__(self, config: dict[str, Any]) -> None:
        feature_match_config = config.get("feature_match", {})
        self.detector = str(self._value(feature_match_config, "detector", "orb")).lower()
        self.transform_model = str(self._value(feature_match_config, "transform_model", "affine")).lower()
        self.max_keypoints = int(self._value(feature_match_config, "max_keypoints", 4000))
        self.ratio_test = float(self._value(feature_match_config, "ratio_test", 0.6))
        self.min_matches = int(self._value(feature_match_config, "min_matches", 10))
        self.ransac_reproj_threshold = float(self._value(feature_match_config, "ransac_reproj_threshold", 3.0))
        if self.detector not in {"orb", "akaze"}:
            raise ValueError(f"unsupported feature_match detector: {self.detector}")
        if self.transform_model not in {"affine", "similarity", "homography"}:
            raise ValueError(f"unsupported feature_match transform_model: {self.transform_model}")
        if self.max_keypoints < 1:
            raise ValueError("feature_match.max_keypoints must be at least 1")
        if not (0.0 < self.ratio_test <= 1.0):
            raise ValueError("feature_match.ratio_test must be in (0, 1]")
        if self.min_matches < 4:
            raise ValueError("feature_match.min_matches must be at least 4")
        if self.ransac_reproj_threshold <= 0.0:
            raise ValueError("feature_match.ransac_reproj_threshold must be greater than 0")

    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        try:
            lr_rgb_work = self._resize_rgb_to_hr(lr_rgb, hr_rgb)
            lr_gray = self._gray_uint8(lr_rgb_work)
            hr_gray = self._gray_uint8(hr_rgb)
            pre_error = self._mse(lr_gray, hr_gray)

            detector = self._create_detector()
            lr_keypoints, lr_descriptors = detector.detectAndCompute(lr_gray, None)
            hr_keypoints, hr_descriptors = detector.detectAndCompute(hr_gray, None)
            if (
                lr_descriptors is None
                or hr_descriptors is None
                or len(lr_keypoints) < self.min_matches
                or len(hr_keypoints) < self.min_matches
            ):
                return self._failed_result(
                    "not enough keypoints for feature matching",
                    diagnostics={
                        "algorithm": "feature_match_transform",
                        "detector": self.detector,
                        "transform_model": self.transform_model,
                        "keypoints_lr": len(lr_keypoints),
                        "keypoints_hr": len(hr_keypoints),
                    },
                )

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn_matches = matcher.knnMatch(lr_descriptors, hr_descriptors, k=2)
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) < 2:
                    continue
                first, second = match_pair
                if first.distance < self.ratio_test * second.distance:
                    good_matches.append(first)
            if len(good_matches) < self.min_matches:
                return self._failed_result(
                    "not enough good matches for transform estimation",
                    diagnostics={
                        "algorithm": "feature_match_transform",
                        "detector": self.detector,
                        "transform_model": self.transform_model,
                        "match_count": len(good_matches),
                        "keypoints_lr": len(lr_keypoints),
                        "keypoints_hr": len(hr_keypoints),
                    },
                )

            src_points = np.float32([lr_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_points = np.float32([hr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            transform_result = self._estimate_transform(src_points, dst_points)
            if transform_result is None:
                return self._failed_result(
                    "transform estimation failed",
                    diagnostics={
                        "algorithm": "feature_match_transform",
                        "detector": self.detector,
                        "transform_model": self.transform_model,
                        "match_count": len(good_matches),
                    },
                )
            matrix, inlier_mask = transform_result
            aligned = self._warp_image(lr_rgb_work, hr_rgb, matrix)
            post_error = self._mse(self._gray_uint8(aligned), hr_gray)
            inlier_count = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else len(good_matches)
            if post_error >= pre_error:
                return self._failed_result(
                    "feature matching transform did not improve alignment error",
                    diagnostics={
                        "algorithm": "feature_match_transform",
                        "detector": self.detector,
                        "transform_model": self.transform_model,
                        "keypoints_lr": len(lr_keypoints),
                        "keypoints_hr": len(hr_keypoints),
                        "match_count": len(good_matches),
                        "inlier_count": inlier_count,
                        "pre_alignment_error": pre_error,
                        "post_alignment_error": post_error,
                        "ransac_reproj_threshold": self.ransac_reproj_threshold,
                    },
                )
            confidence = max(0.0, min(1.0, inlier_count / max(len(good_matches), 1)))
            transform_type = "feature_match_homography" if matrix.shape == (3, 3) else f"feature_match_{self.transform_model}"
            return AlignResult(
                aligned_lr_rgb=aligned,
                status="success",
                confidence=confidence,
                transforms=[
                    {
                        "type": transform_type,
                        "coordinate_system": "lr_to_hr",
                        "matrix": matrix.tolist(),
                        "detector": self.detector,
                        "transform_model": self.transform_model,
                    }
                ],
                diagnostics={
                    "algorithm": "feature_match_transform",
                    "detector": self.detector,
                    "transform_model": self.transform_model,
                    "keypoints_lr": len(lr_keypoints),
                    "keypoints_hr": len(hr_keypoints),
                    "match_count": len(good_matches),
                    "inlier_count": inlier_count,
                    "pre_alignment_error": pre_error,
                    "post_alignment_error": post_error,
                    "ransac_reproj_threshold": self.ransac_reproj_threshold,
                },
            )
        except Exception as exc:
            return self._failed_result(
                str(exc),
                diagnostics={
                    "algorithm": "feature_match_transform",
                    "detector": self.detector,
                    "transform_model": self.transform_model,
                },
            )

    def _estimate_transform(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        if self.transform_model == "homography":
            matrix, inlier_mask = cv2.findHomography(
                src_points,
                dst_points,
                cv2.RANSAC,
                self.ransac_reproj_threshold,
            )
            if matrix is None:
                return None
            return matrix.astype(np.float32), inlier_mask

        full_affine = self.transform_model == "affine"
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        if full_affine:
            matrix, inlier_mask = cv2.estimateAffine2D(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold,
            )
        if matrix is None:
            return None
        return matrix.astype(np.float32), inlier_mask

    def _warp_image(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        output_size = (hr_rgb.shape[1], hr_rgb.shape[0])
        if matrix.shape == (3, 3):
            return cv2.warpPerspective(
                lr_rgb,
                matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        return cv2.warpAffine(
            lr_rgb,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def _create_detector(self):
        if self.detector == "akaze":
            return cv2.AKAZE_create()
        return cv2.ORB_create(nfeatures=self.max_keypoints)

    def _resize_rgb_to_hr(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = hr_rgb.shape[:2]
        if lr_rgb.shape[:2] == (target_height, target_width):
            return lr_rgb
        return cv2.resize(lr_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)

    def _gray_uint8(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def _mse(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))

    def _failed_result(self, message: str, diagnostics: dict[str, Any]) -> AlignResult:
        return AlignResult(
            aligned_lr_rgb=None,
            status="failed",
            confidence=0.0,
            message=message,
            diagnostics=diagnostics,
        )

    def _value(self, config: Any, key: str, default: Any) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)
