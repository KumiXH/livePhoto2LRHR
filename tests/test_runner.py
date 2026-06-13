from pathlib import Path

import yaml

from livephoto2lrhr.config import (
    AlignConfig,
    AppConfig,
    ColorMatchConfig,
    DataConfig,
    ExportConfig,
    FrameSelectConfig,
    OutputConfig,
    PipelineConfig,
    ReportConfig,
)
from livephoto2lrhr.pipeline.runner import run_pipeline


def test_run_pipeline_writes_outputs_and_summary(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, video_path = tiny_pair
    input_dir = image_path.parent
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=2),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 1
    assert (output_dir / "LR" / "flower.png").exists()
    assert (output_dir / "HR" / "flower.png").exists()
    summary_yaml = yaml.safe_load((output_dir / "run_summary.yaml").read_text(encoding="utf-8"))
    assert summary_yaml["counts"]["success"] == 1
    assert summary_yaml["pair_discovery"]["missing_images"] == []
    assert summary_yaml["pair_discovery"]["missing_videos"] == []


def test_run_pipeline_reports_missing_pairs(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "image_only.jpg").write_bytes(b"x")
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 0
    assert summary["counts"]["skipped_existing"] == 0
    assert summary["counts"]["frame_select_failed"] == 0
    assert summary["counts"]["write_failed"] == 0
    assert summary["pair_discovery"]["missing_videos"] == ["image_only"]


def test_run_pipeline_preserves_ambiguous_pairs_in_summary(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "duplicate.jpg").write_bytes(b"x")
    (input_dir / "duplicate.jpeg").write_bytes(b"x")
    (input_dir / "duplicate.mp4").write_bytes(b"x")
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            image_exts=(".jpg", ".jpeg"),
            video_exts=(".mp4",),
        ),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)

    assert summary["pair_discovery"]["ambiguous"] == ["duplicate"]


def test_run_pipeline_runs_alignment_after_frame_select(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    input_dir = image_path.parent
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        align=AlignConfig(enabled=True, algorithm="identity_alignment"),
    )

    summary = run_pipeline(config)

    aligned_path = output_dir / "LR_aligned" / "flower.png"
    metadata = yaml.safe_load((output_dir / "metadata" / "flower.yaml").read_text(encoding="utf-8"))
    assert summary["counts"]["success"] == 1
    assert summary["counts"]["align_success"] == 1
    assert aligned_path.exists()
    assert metadata["status"]["aligned"] is True
    assert metadata["align"]["algorithm"] == "identity_alignment"


def test_run_pipeline_does_not_align_when_disabled(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    input_dir = image_path.parent
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        align=AlignConfig(enabled=False, algorithm="identity_alignment"),
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 1
    assert summary["counts"]["align_skipped_disabled"] == 1
    assert not (output_dir / "LR_aligned" / "flower.png").exists()


def test_run_pipeline_uses_fallback_alignment(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        align=AlignConfig(
            enabled=True,
            algorithm="ecc_alignment",
            fallback_algorithm="identity_alignment",
            confidence_threshold=0.3,
        ),
    )

    summary = run_pipeline(config)

    metadata = yaml.safe_load((output_dir / "metadata" / "flower.yaml").read_text(encoding="utf-8"))
    assert summary["counts"]["align_success"] == 1
    assert (output_dir / "LR_aligned" / "flower.png").exists()
    assert metadata["align"]["diagnostics"]["fallback_used"] is True


def test_run_pipeline_runs_color_match_after_frame_select(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "color_match")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        color_match=ColorMatchConfig(enabled=True, algorithm="identity_color_match"),
    )

    summary = run_pipeline(config)

    matched_path = output_dir / "LR_color_matched" / "flower.png"
    metadata = yaml.safe_load((output_dir / "metadata" / "flower.yaml").read_text(encoding="utf-8"))
    assert summary["counts"]["color_match_success"] == 1
    assert matched_path.exists()
    assert metadata["status"]["color_matched"] is True
    assert metadata["color_match"]["algorithm"] == "identity_color_match"


def test_run_pipeline_generates_quality_report_when_enabled(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        report=ReportConfig(enabled=True, max_preview_samples=1, thumbnail_size=32),
    )

    summary = run_pipeline(config)

    assert summary["report"]["rows"] == 1
    assert (output_dir / "reports" / "quality_report.csv").exists()
    assert (output_dir / "reports" / "preview_contact_sheet.jpg").exists()


def test_run_pipeline_exports_final_dataset_when_enabled(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=True),
        raw={"test": True},
        align=AlignConfig(enabled=True, algorithm="identity_alignment"),
        report=ReportConfig(enabled=True, aligned_folder="LR_aligned", max_preview_samples=0),
        export=ExportConfig(
            enabled=True,
            input_report="reports/quality_report.csv",
            output_folder="final",
            lr_source="aligned",
            require_align_status="success",
        ),
    )

    summary = run_pipeline(config)

    assert summary["export"]["accepted"] == 1
    assert summary["export"]["rejected"] == 0
    assert (output_dir / "final" / "LR" / "flower.png").exists()
    assert (output_dir / "final" / "HR" / "flower.png").exists()
    assert (output_dir / "final" / "manifest.csv").exists()
