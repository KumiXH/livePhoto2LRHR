from __future__ import annotations

from typing import Any

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult
from livephoto2lrhr.algorithms.alignment.coarse_to_flow import CoarseToFlowAligner
from livephoto2lrhr.algorithms.alignment.ecc import ECCAligner
from livephoto2lrhr.algorithms.alignment.feature_match_transform import FeatureMatchTransformAligner
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.phase_correlation import PhaseCorrelationTranslationAligner


class HybridFeatureFlowAligner(CoarseToFlowAligner):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def align(self, lr_rgb, hr_rgb, context: AlignmentContext) -> AlignResult:
        result = super().align(lr_rgb, hr_rgb, context)
        diagnostics = dict(result.diagnostics)
        diagnostics["algorithm"] = "hybrid_feature_flow"
        if result.transforms:
            transforms = list(result.transforms)
        else:
            transforms = []
        return AlignResult(
            aligned_lr_rgb=result.aligned_lr_rgb,
            status=result.status,
            confidence=result.confidence,
            message=result.message,
            transforms=transforms,
            artifacts=result.artifacts,
            diagnostics=diagnostics,
        )

    def _create_coarse_aligner(self, name: str, config: dict[str, Any]):
        if name == "identity_alignment":
            return IdentityAligner(config)
        if name == "phase_correlation_translation":
            return PhaseCorrelationTranslationAligner(config)
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
