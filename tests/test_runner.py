from pathlib import Path

import cv2
import numpy as np
import PIL.Image
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
from livephoto2lrhr.data.io import output_image_path, save_rgb_array, write_yaml
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
    assert (output_dir / "reports" / "quality_report_zh.csv").exists()
    assert (output_dir / "reports" / "preview_contact_sheet.jpg").exists()
    assert summary["report"]["csv_zh"] == str(output_dir / "reports" / "quality_report_zh.csv")


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
            final_lr_source="raw",
            gate_lr_source="aligned",
            final_lr_resize_mode="copy",
            require_align_status="success",
        ),
    )

    summary = run_pipeline(config)

    assert summary["export"]["accepted"] == 1
    assert summary["export"]["rejected"] == 0
    assert (output_dir / "final" / "LR" / "flower.png").exists()
    assert (output_dir / "final" / "HR" / "flower.png").exists()
    assert (output_dir / "final" / "manifest.csv").exists()


def test_run_pipeline_exports_with_extended_quality_gate_fields(tmp_path: Path, tiny_pair: tuple[Path, Path]):
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
            final_lr_source="raw",
            gate_lr_source="aligned",
            require_align_status="success",
            min_source_to_hr_psnr=0.0,
        ),
    )

    summary = run_pipeline(config)

    assert summary["export"]["accepted"] == 1


def test_run_pipeline_resume_summary_counts_existing_outputs(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align", "color_match")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        align=AlignConfig(enabled=True, algorithm="identity_alignment"),
        color_match=ColorMatchConfig(enabled=True, algorithm="identity_color_match"),
    )

    first_summary = run_pipeline(config)
    second_summary = run_pipeline(config)

    assert first_summary["counts"]["success"] == 1
    assert second_summary["counts"]["skipped_existing"] == 1
    assert second_summary["counts"]["align_skipped_existing"] == 1
    assert second_summary["counts"]["color_match_skipped_existing"] == 1
    assert second_summary["execution"]["resumed_from_existing_outputs"] == 3


def test_run_pipeline_retries_failed_alignment_when_retry_failed_enabled(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    relative_stem = Path("flower")
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    meta_path = output_dir / "metadata" / "flower.yaml"
    aligned_path = output_image_path(output_dir, "LR_aligned", relative_stem, ".png")
    save_rgb_array(np.full((4, 5, 3), 10, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((8, 10, 3), 20, dtype=np.uint8), hr_path)
    save_rgb_array(np.full((4, 5, 3), 99, dtype=np.uint8), aligned_path)
    write_yaml(
        meta_path,
        {
            "sample_id": "flower",
            "output": {"lr": str(lr_path), "hr": str(hr_path)},
            "status": {"aligned": False, "color_matched": False},
            "align": {"status": "failed", "message": "synthetic failure"},
        },
    )
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("align",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True, "runtime": {"retry_failed_samples": True}},
        align=AlignConfig(enabled=True, algorithm="identity_alignment"),
    )

    summary = run_pipeline(config)

    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    assert summary["counts"]["align_success"] == 1
    assert summary["counts"]["align_skipped_existing"] == 0
    assert summary["execution"]["retried_failed_samples"] == 1
    assert metadata["align"]["status"] == "success"


def test_run_pipeline_writes_failed_samples_manifest(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select", "align")),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
        align=AlignConfig(enabled=True, algorithm="ecc_alignment", fallback_algorithm="ecc_alignment"),
    )

    summary = run_pipeline(config)
    failed_samples_path = output_dir / "failed_samples.yaml"
    failed_samples = yaml.safe_load(failed_samples_path.read_text(encoding="utf-8"))

    assert summary["counts"]["align_failed"] == 1
    assert summary["execution"]["failed_samples_manifest"] == str(failed_samples_path)
    assert failed_samples["failed_samples"][0]["sample_id"] == "flower"
    assert failed_samples["failed_samples"][0]["status"] == "align_failed"


def test_run_pipeline_writes_empty_failed_samples_manifest_when_all_success(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)
    failed_samples_path = output_dir / "failed_samples.yaml"
    failed_samples = yaml.safe_load(failed_samples_path.read_text(encoding="utf-8"))

    assert summary["counts"]["success"] == 1
    assert failed_samples["failed_samples"] == []


def test_run_pipeline_records_parallel_runtime_configuration(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={
            "test": True,
            "runtime": {
                "parallel": {
                    "num_workers": 4,
                    "gpu_ids": ["0", "1", "2", "3"],
                }
            },
        },
    )

    summary = run_pipeline(config)

    assert summary["execution"]["parallel"]["enabled"] is True
    assert summary["execution"]["parallel"]["requested_workers"] == 4
    assert summary["execution"]["parallel"]["gpu_ids"] == ["0", "1", "2", "3"]
    assert len(summary["execution"]["parallel"]["worker_assignments"]) == 4


def test_run_pipeline_builds_parallel_worker_assignments(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={
            "test": True,
            "runtime": {
                "parallel": {
                    "num_workers": 2,
                    "gpu_ids": ["0", "1"],
                }
            },
        },
    )

    summary = run_pipeline(config)

    assignments = summary["execution"]["parallel"]["worker_assignments"]
    assert len(assignments) == 2
    assert assignments[0]["worker_index"] == 0
    assert assignments[0]["gpu_id"] == "0"
    assert assignments[1]["worker_index"] == 1
    assert assignments[1]["gpu_id"] == "1"


def test_run_pipeline_marks_parallel_enabled_for_multi_worker_frame_select(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=image_path.parent, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={
            "test": True,
            "runtime": {
                "parallel": {
                    "num_workers": 2,
                    "gpu_ids": ["0", "1"],
                }
            },
        },
    )

    summary = run_pipeline(config)

    assert summary["execution"]["parallel"]["enabled"] is True


def test_run_pipeline_parallel_frame_select_reports_used_workers(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("flower_a", "flower_b"):
        (input_dir / f"{name}.jpg").write_bytes(b"x")
        (input_dir / f"{name}.mp4").write_bytes(b"x")
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={
            "test": True,
            "runtime": {
                "parallel": {
                    "num_workers": 2,
                    "gpu_ids": ["0", "1"],
                }
            },
        },
    )

    summary = run_pipeline(config)

    assert summary["execution"]["parallel"]["enabled"] is True
    assert summary["execution"]["parallel"]["used_workers"] == 2


def test_run_pipeline_parallel_frame_select_uses_multiple_processes(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, color in (("flower_a", (20, 40, 200)), ("flower_b", (200, 40, 20))):
        image_path = input_dir / f"{name}.jpg"
        video_path = input_dir / f"{name}.mp4"
        PIL.Image.new("RGB", (8, 8), color=color).save(image_path)
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (8, 8),
        )
        frame_bgr = np.full((8, 8, 3), 100, dtype=np.uint8)
        writer.write(frame_bgr)
        writer.release()
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={
            "test": True,
            "runtime": {
                "parallel": {
                    "num_workers": 2,
                    "gpu_ids": ["0", "1"],
                }
            },
        },
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 2
    assert len(summary["execution"]["parallel"]["worker_pids"]) == 2
    assert (output_dir / "LR" / "flower_a.png").exists()
    assert (output_dir / "LR" / "flower_b.png").exists()
