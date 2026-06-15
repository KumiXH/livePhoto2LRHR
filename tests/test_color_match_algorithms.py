from pathlib import Path

import numpy as np

from livephoto2lrhr.algorithms.color_match import build_color_match_registry
from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext


def make_context(tmp_path: Path, config: dict | None = None) -> ColorMatchContext:
    return ColorMatchContext(
        sample_id="sample",
        lr_path=tmp_path / "LR" / "sample.png",
        hr_path=tmp_path / "HR" / "sample.png",
        metadata={},
        config=config or {},
        artifact_root=tmp_path / "artifacts" / "color_match" / "sample",
        device="cpu",
    )


def mean_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def test_color_match_registry_creates_identity_matcher(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create("identity_color_match", {})
    lr = np.full((4, 5, 3), (10, 20, 30), dtype=np.uint8)
    hr = np.full((8, 10, 3), (200, 180, 160), dtype=np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.confidence == 1.0
    assert result.matched_lr_rgb is not None
    assert np.array_equal(result.matched_lr_rgb, lr)
    assert result.diagnostics["algorithm"] == "identity_color_match"


def test_mean_std_lab_matcher_moves_lr_color_toward_hr(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create("mean_std_lab", {})
    lr = np.full((16, 16, 3), (20, 60, 120), dtype=np.uint8)
    hr = np.full((32, 32, 3), (180, 140, 70), dtype=np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert result.matched_lr_rgb.shape == lr.shape
    assert mean_abs_delta(result.matched_lr_rgb, hr[:16, :16]) < mean_abs_delta(lr, hr[:16, :16])
    assert result.diagnostics["algorithm"] == "mean_std_lab"
    assert "pre_color_error" in result.diagnostics
    assert "post_color_error" in result.diagnostics


def test_histogram_match_lab_moves_lr_histogram_toward_hr(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "histogram_match_lab",
        {
            "histogram_match": {
                "color_space": "lab",
                "bins": 256,
            }
        },
    )
    y = np.linspace(0, 1, 48, dtype=np.float32)
    x = np.linspace(0, 1, 48, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    lr = np.stack(
        [
            20 + grid_x * 30,
            40 + grid_y * 50,
            80 + (grid_x + grid_y) * 20,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    hr = np.stack(
        [
            150 + grid_x * 60,
            110 + grid_y * 70,
            40 + (grid_x * 0.4 + grid_y * 0.8) * 90,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert result.matched_lr_rgb.shape == lr.shape
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "histogram_match_lab"
    assert result.transforms[0]["type"] == "histogram_color_transfer"
    assert result.transforms[0]["color_space"] == "lab"


def test_retinex_color_match_improves_luminance_toward_hr(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "retinex_color_match",
        {
            "retinex": {
                "sigma": 12.0,
                "eps": 1.0e-3,
            }
        },
    )
    y = np.linspace(0, 1, 48, dtype=np.float32)
    x = np.linspace(0, 1, 48, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    base = 20 + grid_x * 35 + grid_y * 25
    lr = np.stack(
        [
            base + 5,
            base + 15,
            base + 25,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    hr = np.stack(
        [
            80 + grid_x * 70 + grid_y * 50,
            95 + grid_x * 65 + grid_y * 45,
            110 + grid_x * 60 + grid_y * 40,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert result.matched_lr_rgb.shape == lr.shape
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "retinex_color_match"
    assert result.diagnostics["replay_reference_required"] is True
    assert result.transforms[0]["type"] == "retinex_color_transfer"


def test_masked_color_transfer_focuses_on_foreground_region(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "masked_color_transfer",
        {
            "masked_transfer": {
                "color_space": "lab",
                "difference_threshold": 10.0,
                "min_mask_fraction": 0.02,
                "max_mask_fraction": 0.8,
                "morphology_kernel_size": 3,
            }
        },
    )
    lr = np.full((48, 48, 3), (90, 110, 120), dtype=np.uint8)
    hr = np.full((48, 48, 3), (90, 110, 120), dtype=np.uint8)
    lr[12:36, 12:36] = (25, 180, 35)
    hr[12:36, 12:36] = (220, 60, 160)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    foreground_before = mean_abs_delta(lr[12:36, 12:36], hr[12:36, 12:36])
    foreground_after = mean_abs_delta(result.matched_lr_rgb[12:36, 12:36], hr[12:36, 12:36])
    assert foreground_after < foreground_before
    assert result.diagnostics["algorithm"] == "masked_color_transfer"
    assert result.diagnostics["replay_reference_required"] is True
    assert result.transforms[0]["type"] == "masked_mean_std_color_transfer"


def test_image_adaptive_3d_lut_color_match_moves_lr_toward_hr(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "image_adaptive_3d_lut_color_match",
        {
            "adaptive_3d_lut": {
                "color_space": "rgb",
                "grid_size": 9,
                "smoothing_sigma": 1.0,
                "identity_mix": 0.1,
            }
        },
    )
    y = np.linspace(0, 1, 40, dtype=np.float32)
    x = np.linspace(0, 1, 40, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    lr = np.stack(
        [
            20 + grid_x * 80,
            30 + grid_y * 60,
            50 + (grid_x * 0.4 + grid_y * 0.3) * 120,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    hr = np.stack(
        [
            35 + np.power(grid_x, 0.8) * 140,
            25 + np.power(grid_y, 1.1) * 120,
            40 + (grid_x * 0.2 + grid_y * 0.9) * 150,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.transforms[0]["type"] == "adaptive_3d_lut_color_transfer"
    assert result.diagnostics["algorithm"] == "image_adaptive_3d_lut_color_match"


def test_low_frequency_joint_appearance_match_preserves_detail_while_fixing_base(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "low_frequency_joint_appearance_match",
        {
            "low_frequency_joint": {
                "color_space": "lab",
                "sigma": 5.0,
                "base_mix": 0.85,
                "detail_preservation": 1.0,
                "chroma_strength": 0.7,
            }
        },
    )
    y = np.linspace(0, 1, 48, dtype=np.float32)
    x = np.linspace(0, 1, 48, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    base_lr = 35 + grid_x * 35 + grid_y * 20
    base_hr = 80 + grid_x * 70 + grid_y * 45
    detail = ((np.sin(grid_x * 30) + np.cos(grid_y * 28)) * 6.0)
    lr = np.stack([base_lr + detail, base_lr + 8 + detail, base_lr + 16 + detail], axis=2).clip(0, 255).astype(np.uint8)
    hr = np.stack([base_hr + detail, base_hr + 6 + detail, base_hr + 10 + detail], axis=2).clip(0, 255).astype(np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "low_frequency_joint_appearance_match"
    assert result.diagnostics["replay_reference_required"] is True


def test_learned_retinex_color_match_reports_proxy_mode_and_improves(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "learned_retinex_color_match",
        {
            "learned_retinex": {
                "sigma": 13.0,
                "eps": 1.0e-3,
                "base_mix": 0.65,
            }
        },
    )
    lr = np.full((32, 32, 3), (30, 50, 70), dtype=np.uint8)
    hr = np.full((32, 32, 3), (130, 145, 160), dtype=np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "learned_retinex_color_match"
    assert result.diagnostics["implementation_mode"] == "proxy"
    assert result.transforms[0]["type"] == "learned_retinex_color_transfer"


def test_mask_aware_harmonization_network_reports_proxy_mode_and_improves(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "mask_aware_harmonization_network",
        {
            "mask_aware_harmonization": {
                "color_space": "lab",
                "difference_threshold": 10.0,
                "min_mask_fraction": 0.02,
                "max_mask_fraction": 0.8,
                "morphology_kernel_size": 3,
                "low_frequency_sigma": 4.0,
            }
        },
    )
    lr = np.full((48, 48, 3), (65, 85, 95), dtype=np.uint8)
    hr = np.full((48, 48, 3), (80, 95, 105), dtype=np.uint8)
    lr[12:36, 12:36] = (20, 170, 50)
    hr[12:36, 12:36] = (210, 80, 165)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "mask_aware_harmonization_network"
    assert result.diagnostics["implementation_mode"] == "proxy"
    assert result.transforms[0]["type"] == "mask_aware_harmonization_transfer"


def test_diffusion_harmonization_reports_proxy_mode_and_improves(tmp_path: Path):
    registry = build_color_match_registry()
    matcher = registry.create(
        "diffusion_harmonization",
        {
            "diffusion_harmonization": {
                "color_space": "lab",
                "num_steps": 4,
                "guidance_strength": 0.6,
                "low_frequency_sigma": 4.0,
                "lut_identity_mix": 0.15,
            }
        },
    )
    y = np.linspace(0, 1, 36, dtype=np.float32)
    x = np.linspace(0, 1, 36, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    lr = np.stack(
        [
            25 + grid_x * 45,
            35 + grid_y * 40,
            55 + (grid_x + grid_y) * 25,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    hr = np.stack(
        [
            85 + grid_x * 90,
            75 + grid_y * 70,
            65 + (grid_x * 0.2 + grid_y * 0.9) * 110,
        ],
        axis=2,
    ).clip(0, 255).astype(np.uint8)

    result = matcher.match(lr, hr, make_context(tmp_path))

    assert result.status == "success"
    assert result.matched_lr_rgb is not None
    assert mean_abs_delta(result.matched_lr_rgb, hr) < mean_abs_delta(lr, hr)
    assert result.diagnostics["algorithm"] == "diffusion_harmonization"
    assert result.diagnostics["implementation_mode"] == "proxy"
    assert result.transforms[0]["type"] == "diffusion_harmonization_transfer"
