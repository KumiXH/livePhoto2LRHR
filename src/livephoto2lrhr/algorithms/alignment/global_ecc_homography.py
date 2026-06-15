from __future__ import annotations

from typing import Any

from livephoto2lrhr.algorithms.alignment.ecc import ECCAligner


class GlobalECCHomographyAligner(ECCAligner):
    def __init__(self, config: dict[str, Any]) -> None:
        effective_config = dict(config)
        effective_config["motion_model"] = "homography"
        super().__init__(effective_config)

    def align(self, lr_rgb, hr_rgb, context):
        result = super().align(lr_rgb, hr_rgb, context)
        diagnostics = dict(result.diagnostics)
        diagnostics["algorithm"] = "global_ecc_homography"
        return type(result)(
            aligned_lr_rgb=result.aligned_lr_rgb,
            status=result.status,
            confidence=result.confidence,
            message=result.message,
            transforms=result.transforms,
            artifacts=result.artifacts,
            diagnostics=diagnostics,
        )
