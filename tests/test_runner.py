from pathlib import Path

import yaml

from livephoto2lrhr.config import AppConfig, DataConfig, FrameSelectConfig, OutputConfig, PipelineConfig
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

    assert summary["counts"] == {
        "success": 0,
        "skipped_existing": 0,
        "frame_select_failed": 0,
        "write_failed": 0,
    }
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
