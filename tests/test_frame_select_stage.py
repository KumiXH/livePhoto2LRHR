from pathlib import Path

import yaml
from PIL import Image

from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.data.pairing import SamplePair
from livephoto2lrhr.stages.frame_select import FrameSelectStage


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
    selector = FakeFrameSelector({"top_k": 1})
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
