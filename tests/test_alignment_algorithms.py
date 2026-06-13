from pathlib import Path

import cv2
import numpy as np

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.alignment.base import AlignmentContext


def make_context(tmp_path: Path, config: dict | None = None) -> AlignmentContext:
    return AlignmentContext(
        sample_id="sample",
        lr_path=tmp_path / "LR" / "sample.png",
        hr_path=tmp_path / "HR" / "sample.png",
        metadata={},
        config=config or {},
        artifact_root=tmp_path / "artifacts" / "alignment" / "sample",
        device="cpu",
    )


def make_feature_image(size: int = 64) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.rectangle(image, (18, 20), (42, 44), (220, 220, 220), -1)
    cv2.circle(image, (45, 18), 6, (120, 120, 120), -1)
    cv2.line(image, (12, 50), (52, 55), (180, 180, 180), 2)
    return image


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


def mse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2))


def test_phase_correlation_translation_improves_shifted_lr(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create("phase_correlation_translation", {"resize_short_side": 64})
    hr = make_feature_image()
    lr = shift_image(hr, dx=5, dy=-3)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.confidence > 0.1
    assert result.aligned_lr_rgb is not None
    assert mse(result.aligned_lr_rgb, hr) < mse(lr, hr)
    assert result.transforms[0]["type"] == "translation"
    assert result.transforms[0]["coordinate_system"] == "lr_to_hr"
    assert result.diagnostics["algorithm"] == "phase_correlation_translation"
