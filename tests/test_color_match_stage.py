from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from livephoto2lrhr.algorithms.color_match.base import ColorMatchContext, ColorMatchResult
from livephoto2lrhr.algorithms.color_match.identity import IdentityColorMatcher
from livephoto2lrhr.data.io import metadata_path, output_image_path, save_rgb_array, write_yaml
from livephoto2lrhr.data.pairing import SamplePair
from livephoto2lrhr.stages.color_match import ColorMatchStage


class ContextEchoColorMatcher:
    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        return ColorMatchResult(
            matched_lr_rgb=np.full_like(lr_rgb, 123),
            status="success",
            confidence=0.9,
            diagnostics={"context_config": context.config, "input": str(context.lr_path)},
        )


class FailingColorMatcher:
    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        return ColorMatchResult(
            matched_lr_rgb=None,
            status="failed",
            confidence=0.0,
            message="synthetic color failure",
            diagnostics={"reason": "synthetic"},
        )


class LowConfidenceColorMatcher:
    def match(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: ColorMatchContext) -> ColorMatchResult:
        return ColorMatchResult(
            matched_lr_rgb=np.full_like(lr_rgb, 222),
            status="success",
            confidence=0.1,
            diagnostics={"reason": "low_confidence"},
        )


def write_color_inputs(output_dir: Path, relative_stem: Path) -> tuple[Path, Path, Path, Path]:
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    aligned_path = output_image_path(output_dir, "LR_aligned", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    meta_path = metadata_path(output_dir, relative_stem)
    save_rgb_array(np.full((4, 5, 3), 10, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((4, 5, 3), 77, dtype=np.uint8), aligned_path)
    save_rgb_array(np.full((8, 10, 3), 200, dtype=np.uint8), hr_path)
    write_yaml(
        meta_path,
        {
            "sample_id": relative_stem.as_posix(),
            "output": {"lr": str(lr_path), "lr_aligned": str(aligned_path), "hr": str(hr_path)},
            "status": {"aligned": True, "color_matched": False},
        },
    )
    return lr_path, aligned_path, hr_path, meta_path


def write_color_inputs_with_custom_aligned_folder(
    output_dir: Path,
    relative_stem: Path,
    *,
    aligned_folder: str,
) -> tuple[Path, Path, Path, Path]:
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    aligned_path = output_image_path(output_dir, aligned_folder, relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    meta_path = metadata_path(output_dir, relative_stem)
    save_rgb_array(np.full((4, 5, 3), 10, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((4, 5, 3), 177, dtype=np.uint8), aligned_path)
    save_rgb_array(np.full((8, 10, 3), 200, dtype=np.uint8), hr_path)
    write_yaml(
        meta_path,
        {
            "sample_id": relative_stem.as_posix(),
            "output": {"lr": str(lr_path), "lr_aligned": str(aligned_path), "hr": str(hr_path)},
            "status": {"aligned": True, "color_matched": False},
        },
    )
    return lr_path, aligned_path, hr_path, meta_path


def test_color_match_stage_prefers_aligned_lr_and_updates_metadata(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("nested") / "sample"
    _, aligned_path, _, meta_path = write_color_inputs(output_dir, relative_stem)
    pair = SamplePair("nested/sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = ColorMatchStage(
        output_dir=output_dir,
        output_ext=".png",
        input_folder="auto",
        output_folder="LR_color_matched",
        overwrite=False,
        save_metadata=True,
        matcher=ContextEchoColorMatcher(),
        algorithm_name="context_echo",
        algorithm_config={"model": {"name": "future-color-net"}},
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    output_path = output_dir / "LR_color_matched" / "nested" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "color_match_success"
    assert output_path.exists()
    assert Image.open(output_path).getpixel((0, 0)) == (123, 123, 123)
    assert metadata["status"]["color_matched"] is True
    assert metadata["color_match"]["algorithm"] == "context_echo"
    assert metadata["color_match"]["input"]["lr"] == str(aligned_path)
    assert metadata["color_match"]["output"]["lr_color_matched"] == str(output_path)
    assert metadata["color_match"]["diagnostics"]["context_config"] == {"model": {"name": "future-color-net"}}


def test_color_match_stage_falls_back_to_lr_when_aligned_missing(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    save_rgb_array(np.full((4, 5, 3), 11, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((8, 10, 3), 200, dtype=np.uint8), hr_path)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = ColorMatchStage(
        output_dir=output_dir,
        output_ext=".png",
        input_folder="auto",
        output_folder="LR_color_matched",
        overwrite=False,
        save_metadata=False,
        matcher=IdentityColorMatcher({}),
        algorithm_name="identity_color_match",
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    output_path = output_dir / "LR_color_matched" / "sample.png"
    assert result.status == "color_match_success"
    assert output_path.exists()
    assert Image.open(output_path).getpixel((0, 0)) == (11, 11, 11)


def test_color_match_stage_auto_uses_metadata_aligned_output_path(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    _, aligned_path, _, meta_path = write_color_inputs_with_custom_aligned_folder(
        output_dir,
        relative_stem,
        aligned_folder="LR_aligned_flow",
    )
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = ColorMatchStage(
        output_dir=output_dir,
        output_ext=".png",
        input_folder="auto",
        output_folder="LR_color_matched",
        overwrite=False,
        save_metadata=True,
        matcher=ContextEchoColorMatcher(),
        algorithm_name="context_echo",
        algorithm_config={},
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    output_path = output_dir / "LR_color_matched" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "color_match_success"
    assert output_path.exists()
    assert Image.open(output_path).getpixel((0, 0)) == (123, 123, 123)
    assert metadata["color_match"]["input"]["lr"] == str(aligned_path)


def test_color_match_stage_skip_on_failure_does_not_write_output(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    _, _, _, meta_path = write_color_inputs(output_dir, relative_stem)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = ColorMatchStage(
        output_dir=output_dir,
        output_ext=".png",
        input_folder="auto",
        output_folder="LR_color_matched",
        overwrite=False,
        save_metadata=True,
        matcher=FailingColorMatcher(),
        algorithm_name="failing_color",
        on_failure="skip",
        device="cpu",
    )

    result = stage.run(pair)

    output_path = output_dir / "LR_color_matched" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "color_match_failed"
    assert not output_path.exists()
    assert metadata["status"]["color_matched"] is False
    assert metadata["color_match"]["status"] == "failed"


def test_color_match_stage_skip_on_low_confidence_does_not_write_rejected_output(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    _, _, _, meta_path = write_color_inputs(output_dir, relative_stem)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = ColorMatchStage(
        output_dir=output_dir,
        output_ext=".png",
        input_folder="auto",
        output_folder="LR_color_matched",
        overwrite=False,
        save_metadata=True,
        matcher=LowConfidenceColorMatcher(),
        algorithm_name="low_confidence_color",
        confidence_threshold=0.5,
        on_failure="skip",
        device="cpu",
    )

    result = stage.run(pair)

    output_path = output_dir / "LR_color_matched" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "color_match_skipped_low_confidence"
    assert not output_path.exists()
    assert metadata["status"]["color_matched"] is False
    assert metadata["color_match"]["status"] == "skipped_low_confidence"
