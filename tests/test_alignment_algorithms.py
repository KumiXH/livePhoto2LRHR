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


def make_texture_image(size: int = 96) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    base = 127 + 50 * np.sin(x / 3) + 40 * np.cos(y / 4) + 30 * np.sin((x + y) / 5)
    image = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 11, axis=0)], axis=2)
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.circle(image, (30, 30), 12, (230, 40, 80), -1)
    cv2.rectangle(image, (55, 45), (82, 75), (40, 220, 120), -1)
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


def test_ecc_alignment_improves_shifted_lr(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create(
        "ecc_alignment",
        {
            "motion_model": "translation",
            "number_of_iterations": 100,
            "termination_eps": 1.0e-6,
            "gaussian_filter_size": 3,
        },
    )
    hr = make_feature_image()
    lr = shift_image(hr, dx=4, dy=3)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.confidence > 0.0
    assert result.aligned_lr_rgb is not None
    assert mse(result.aligned_lr_rgb, hr) < mse(lr, hr)
    assert result.transforms[0]["type"] == "ecc_translation"
    assert result.transforms[0]["coordinate_system"] == "lr_to_hr"
    assert result.diagnostics["algorithm"] == "ecc_alignment"


def test_ecc_alignment_returns_failed_for_unalignable_images(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create("ecc_alignment", {"motion_model": "translation"})
    lr = np.zeros((32, 32, 3), dtype=np.uint8)
    hr = np.zeros((32, 32, 3), dtype=np.uint8)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "failed"
    assert result.aligned_lr_rgb is None
    assert result.confidence == 0.0
    assert result.message


def test_phase_correlation_handles_mixed_lr_hr_sizes(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create("phase_correlation_translation", {"resize_short_side": 64})
    hr = make_feature_image(96)
    lr = cv2.resize(hr, (48, 48), interpolation=cv2.INTER_AREA)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.aligned_lr_rgb is not None
    assert result.aligned_lr_rgb.shape == hr.shape
    assert "pre_alignment_error" in result.diagnostics
    assert "post_alignment_error" in result.diagnostics


def test_ecc_alignment_reports_quality_metrics_for_mixed_sizes(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create("ecc_alignment", {"motion_model": "translation"})
    hr = make_feature_image(96)
    lr = cv2.resize(hr, (48, 48), interpolation=cv2.INTER_AREA)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.aligned_lr_rgb is not None
    assert result.aligned_lr_rgb.shape == hr.shape
    assert "pre_alignment_error" in result.diagnostics
    assert "post_alignment_error" in result.diagnostics


def test_coarse_to_flow_registry_falls_back_to_coarse_result(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create(
        "coarse_to_flow",
        {
            "coarse_algorithm": "phase_correlation_translation",
            "optical_flow": {"enabled": False, "algorithm": "dis"},
        },
    )
    hr = make_feature_image()
    lr = shift_image(hr, dx=3, dy=2)

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.aligned_lr_rgb is not None
    assert result.diagnostics["algorithm"] == "coarse_to_flow"
    assert result.diagnostics["coarse_algorithm"] == "phase_correlation_translation"
    assert result.diagnostics["flow_used"] is False


def test_coarse_to_flow_applies_dense_flow_when_it_improves_error(tmp_path: Path):
    registry = build_alignment_registry()
    aligner = registry.create(
        "coarse_to_flow",
        {
            "coarse_algorithm": "identity_alignment",
            "optical_flow": {"enabled": True, "algorithm": "dis"},
        },
    )
    hr = make_texture_image(96)
    height, width = hr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    local_flow = np.zeros((height, width, 2), dtype=np.float32)
    right_half = grid_x > width / 2
    local_flow[..., 0] = np.where(right_half, 4 * np.sin((grid_y / height) * np.pi), 0)
    local_flow[..., 1] = np.where(right_half, 2 * np.sin((grid_x / width) * np.pi), 0)
    lr = cv2.remap(
        hr,
        grid_x - local_flow[..., 0],
        grid_y - local_flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    result = aligner.align(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.aligned_lr_rgb is not None
    assert result.aligned_lr_rgb.shape == hr.shape
    assert result.diagnostics["algorithm"] == "coarse_to_flow"
    assert result.diagnostics["flow_used"] is True
    assert result.diagnostics["flow_status"] == "accepted"
    assert result.diagnostics["post_flow_error"] < result.diagnostics["pre_flow_error"]
    assert mse(result.aligned_lr_rgb, hr) < mse(lr, hr)
