from __future__ import annotations

from typing import Any

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
        if self._flow_enabled():
            diagnostics["flow_status"] = "unsupported"
            message = coarse_result.message or "optical flow refinement is not implemented; returned coarse result"
        else:
            diagnostics["flow_status"] = "disabled"
            message = coarse_result.message
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
