from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.data.pairing import SamplePair
import livephoto2lrhr.stages.frame_select as frame_select
from livephoto2lrhr.stages.frame_select import FrameSelectStage


class TrackingSelector:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        self.calls += 1
        candidate = FrameCandidate(frame_index=0, timestamp_sec=0.0, score=1.0)
        return FrameSelectionResult(
            frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            selected=candidate,
            top_k=[candidate],
            diagnostics={},
        )


class DiagnosticsSelector:
    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        candidate = FrameCandidate(frame_index=0, timestamp_sec=0.0, score=1.0)
        return FrameSelectionResult(
            frame_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            selected=candidate,
            top_k=[candidate],
            diagnostics={
                "score": np.float32(0.5),
                "vector": np.array([1, 2, 3], dtype=np.int64),
                "path": Path("diagnostics") / "source.npy",
            },
        )


class RaisingSelector:
    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        raise RuntimeError("selector exploded")


def test_frame_select_stage_writes_mirrored_lr_hr_and_metadata(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="nested/flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("nested") / "flower",
    )
    selector = FakeFrameSelector({"top_k": 3})
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=selector,
        algorithm_name="fake_selector",
    )

    result = stage.run(pair)

    lr_path = output_dir / "LR" / "nested" / "flower.png"
    hr_path = output_dir / "HR" / "nested" / "flower.png"
    metadata_path = output_dir / "metadata" / "nested" / "flower.yaml"
    assert result.status == "success"
    assert lr_path.exists()
    assert hr_path.exists()
    assert metadata_path.exists()
    assert Image.open(lr_path).size == (4, 4)
    assert Image.open(hr_path).size == (8, 8)

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_id"] == "nested/flower"
    assert metadata["frame_select"]["algorithm"] == "fake_selector"
    assert metadata["frame_select"]["selected"]["frame_index"] == 0
    assert [item["frame_index"] for item in metadata["frame_select"]["top_k"]] == [0, 1, 2]
    assert sorted(path.name for path in (output_dir / "LR" / "nested").iterdir()) == ["flower.png"]


def test_frame_select_stage_preserves_dotted_relative_stems(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="nested/IMG.0001",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("nested") / "IMG.0001",
    )
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=FakeFrameSelector({"top_k": 1}),
        algorithm_name="fake_selector",
    )

    result = stage.run(pair)

    assert result.status == "success"
    assert (output_dir / "LR" / "nested" / "IMG.0001.png").exists()
    assert (output_dir / "HR" / "nested" / "IMG.0001.png").exists()
    assert (output_dir / "metadata" / "nested" / "IMG.0001.yaml").exists()
    assert not (output_dir / "LR" / "nested" / "IMG.png").exists()


def test_frame_select_stage_respects_overwrite_false(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("flower"),
    )
    selector = TrackingSelector()
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=selector,
        algorithm_name="fake_selector",
    )

    first = stage.run(pair)
    second = stage.run(pair)

    assert first.status == "success"
    assert second.status == "skipped_existing"
    assert selector.calls == 1


def test_frame_select_stage_writes_yaml_safe_diagnostics(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("flower"),
    )
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=DiagnosticsSelector(),
        algorithm_name="diagnostics_selector",
    )

    result = stage.run(pair)

    metadata_path = output_dir / "metadata" / "flower.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    diagnostics = metadata["frame_select"]["diagnostics"]
    assert result.status == "success"
    assert diagnostics == {
        "score": 0.5,
        "vector": [1, 2, 3],
        "path": str(Path("diagnostics") / "source.npy"),
    }


def test_frame_select_stage_reports_selector_errors_as_frame_select_failed(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    stage = FrameSelectStage(
        output_dir=tmp_path / "output",
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=RaisingSelector(),
        algorithm_name="raising_selector",
    )

    result = stage.run(
        SamplePair(
            sample_id="flower",
            image_path=image_path,
            video_path=video_path,
            relative_stem=Path("flower"),
        )
    )

    assert result.status == "frame_select_failed"
    assert "selector exploded" in result.message


def test_frame_select_stage_reports_write_errors_as_write_failed(
    monkeypatch,
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair

    def raise_write_error(path: Path, data: dict[str, object]) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(frame_select, "write_yaml", raise_write_error)
    stage = FrameSelectStage(
        output_dir=tmp_path / "output",
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=FakeFrameSelector({"top_k": 1}),
        algorithm_name="fake_selector",
    )

    result = stage.run(
        SamplePair(
            sample_id="flower",
            image_path=image_path,
            video_path=video_path,
            relative_stem=Path("flower"),
        )
    )

    assert result.status == "write_failed"
    assert "disk full" in result.message


def test_frame_select_stage_recreates_missing_metadata_when_overwrite_false(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("flower"),
    )
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=FakeFrameSelector({"top_k": 1}),
        algorithm_name="fake_selector",
    )

    first = stage.run(pair)
    metadata_path = output_dir / "metadata" / "flower.yaml"
    metadata_path.unlink()
    second = stage.run(pair)

    assert first.status == "success"
    assert second.status == "success"
    assert metadata_path.exists()
