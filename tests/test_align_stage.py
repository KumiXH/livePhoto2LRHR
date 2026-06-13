from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from livephoto2lrhr.algorithms.alignment.base import AlignmentContext, AlignResult
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.data.io import metadata_path, output_image_path, save_rgb_array, write_yaml
from livephoto2lrhr.data.pairing import SamplePair
from livephoto2lrhr.stages.align import AlignStage


class FailingAligner:
    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        return AlignResult(
            aligned_lr_rgb=None,
            status="failed",
            confidence=0.0,
            message="alignment failed",
            diagnostics={"reason": "synthetic"},
        )


class LowConfidenceAligner:
    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        return AlignResult(
            aligned_lr_rgb=np.full_like(lr_rgb, 99),
            status="success",
            confidence=0.1,
            transforms=[{"type": "synthetic"}],
            diagnostics={"context_config": context.config},
        )


class ContextEchoAligner:
    def align(self, lr_rgb: np.ndarray, hr_rgb: np.ndarray, context: AlignmentContext) -> AlignResult:
        return AlignResult(
            aligned_lr_rgb=lr_rgb.copy(),
            status="success",
            confidence=1.0,
            diagnostics={"context_config": context.config},
        )


def write_phase1_outputs(output_dir: Path, relative_stem: Path) -> tuple[Path, Path, Path]:
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    meta_path = metadata_path(output_dir, relative_stem)
    save_rgb_array(np.full((4, 5, 3), 10, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((8, 10, 3), 20, dtype=np.uint8), hr_path)
    write_yaml(
        meta_path,
        {
            "sample_id": relative_stem.as_posix(),
            "output": {"lr": str(lr_path), "hr": str(hr_path)},
            "status": {"aligned": False, "color_matched": False},
        },
    )
    return lr_path, hr_path, meta_path


def test_align_stage_writes_aligned_lr_and_updates_metadata(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("nested") / "IMG.0001"
    lr_path, hr_path, meta_path = write_phase1_outputs(output_dir, relative_stem)
    pair = SamplePair(
        sample_id="nested/IMG.0001",
        image_path=tmp_path / "source.jpg",
        video_path=tmp_path / "source.mp4",
        relative_stem=relative_stem,
    )
    original_lr = Image.open(lr_path).copy()
    original_hr = Image.open(hr_path).copy()
    stage = AlignStage(
        output_dir=output_dir,
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=True,
        aligner=IdentityAligner({}),
        algorithm_name="identity_alignment",
        confidence_threshold=0.3,
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    aligned_path = output_dir / "LR_aligned" / "nested" / "IMG.0001.png"
    assert result.status == "align_success"
    assert aligned_path.exists()
    assert Image.open(aligned_path).size == (5, 4)
    assert list(original_lr.getdata()) == list(Image.open(lr_path).getdata())
    assert list(original_hr.getdata()) == list(Image.open(hr_path).getdata())
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert metadata["status"]["aligned"] is True
    assert metadata["align"]["algorithm"] == "identity_alignment"
    assert metadata["align"]["status"] == "success"
    assert metadata["align"]["confidence"] == 1.0
    assert metadata["align"]["output"]["lr_aligned"] == str(aligned_path)
    assert metadata["align"]["transforms"] == [{"type": "identity", "coordinate_system": "lr_to_hr"}]
    assert metadata["align"]["diagnostics"] == {"algorithm": "identity_alignment"}
    assert not (output_dir / "LR_aligned" / "nested" / "IMG.png").exists()


def test_align_stage_reports_missing_phase1_outputs(tmp_path: Path):
    pair = SamplePair(
        sample_id="missing",
        image_path=tmp_path / "source.jpg",
        video_path=tmp_path / "source.mp4",
        relative_stem=Path("missing"),
    )
    stage = AlignStage(
        output_dir=tmp_path / "output",
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=True,
        aligner=IdentityAligner({}),
        algorithm_name="identity_alignment",
        confidence_threshold=0.3,
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    assert result.status == "align_skipped_missing_input"
    assert "missing phase 1 output" in result.message


def test_align_stage_keep_original_on_failure_marks_unaligned(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    _, _, meta_path = write_phase1_outputs(output_dir, relative_stem)
    pair = SamplePair(
        sample_id="sample",
        image_path=tmp_path / "source.jpg",
        video_path=tmp_path / "source.mp4",
        relative_stem=relative_stem,
    )
    stage = AlignStage(
        output_dir=output_dir,
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=True,
        aligner=FailingAligner(),
        algorithm_name="failing_alignment",
        confidence_threshold=0.3,
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    aligned_path = output_dir / "LR_aligned" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "align_failed"
    assert aligned_path.exists()
    assert metadata["status"]["aligned"] is False
    assert metadata["align"]["status"] == "failed"
    assert metadata["align"]["message"] == "alignment failed"


def test_align_stage_skip_on_low_confidence_does_not_write_rejected_alignment(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    lr_path, _, meta_path = write_phase1_outputs(output_dir, relative_stem)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = AlignStage(
        output_dir=output_dir,
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=True,
        aligner=LowConfidenceAligner(),
        algorithm_name="low_confidence",
        confidence_threshold=0.5,
        on_failure="skip",
        device="cpu",
    )

    result = stage.run(pair)

    aligned_path = output_dir / "LR_aligned" / "sample.png"
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "align_skipped_low_confidence"
    assert not aligned_path.exists()
    assert metadata["status"]["aligned"] is False
    assert metadata["align"]["status"] == "skipped_low_confidence"
    assert Image.open(lr_path).getpixel((0, 0)) == (10, 10, 10)


def test_align_stage_save_metadata_false_does_not_require_metadata(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    save_rgb_array(np.full((4, 5, 3), 10, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((8, 10, 3), 20, dtype=np.uint8), hr_path)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = AlignStage(
        output_dir=output_dir,
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=False,
        aligner=IdentityAligner({}),
        algorithm_name="identity_alignment",
        confidence_threshold=0.3,
        on_failure="keep_original",
        device="cpu",
    )

    result = stage.run(pair)

    assert result.status == "align_success"
    assert (output_dir / "LR_aligned" / "sample.png").exists()


def test_align_stage_passes_algorithm_config_to_context(tmp_path: Path):
    output_dir = tmp_path / "output"
    relative_stem = Path("sample")
    _, _, meta_path = write_phase1_outputs(output_dir, relative_stem)
    pair = SamplePair("sample", tmp_path / "source.jpg", tmp_path / "source.mp4", relative_stem)
    stage = AlignStage(
        output_dir=output_dir,
        output_ext=".png",
        output_folder="LR_aligned",
        overwrite=False,
        save_metadata=True,
        aligner=ContextEchoAligner(),
        algorithm_name="context_echo",
        confidence_threshold=0.3,
        on_failure="keep_original",
        device="cpu",
        algorithm_config={"model": {"path": "D:/models/future"}},
    )

    result = stage.run(pair)

    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert result.status == "align_success"
    assert metadata["align"]["diagnostics"]["context_config"] == {"model": {"path": "D:/models/future"}}
