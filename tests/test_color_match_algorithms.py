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
