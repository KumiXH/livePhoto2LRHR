from pathlib import Path

from livephoto2lrhr.data.pairing import discover_pairs


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_discover_pairs_matches_by_relative_stem(tmp_path: Path):
    touch(tmp_path / "IMG_0001.jpg")
    touch(tmp_path / "IMG_0001.mp4")
    touch(tmp_path / "trip" / "a.jpg")
    touch(tmp_path / "trip" / "a.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg",),
        video_exts=(".mp4",),
        recursive=True,
    )

    assert [sample.sample_id for sample in result.pairs] == ["IMG_0001", "trip/a"]
    assert result.missing_images == []
    assert result.missing_videos == []
    assert result.ambiguous == []


def test_discover_pairs_reports_missing_and_ambiguous_files(tmp_path: Path):
    touch(tmp_path / "ok.jpg")
    touch(tmp_path / "ok.mp4")
    touch(tmp_path / "image_only.jpg")
    touch(tmp_path / "video_only.mp4")
    touch(tmp_path / "dup.jpg")
    touch(tmp_path / "dup.png")
    touch(tmp_path / "dup.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg", ".png"),
        video_exts=(".mp4",),
        recursive=True,
    )

    assert [sample.sample_id for sample in result.pairs] == ["ok"]
    assert result.missing_videos == ["image_only"]
    assert result.missing_images == ["video_only"]
    assert result.ambiguous == ["dup"]


def test_discover_pairs_can_ignore_nested_files_when_not_recursive(tmp_path: Path):
    touch(tmp_path / "root.jpg")
    touch(tmp_path / "root.mp4")
    touch(tmp_path / "nested" / "child.jpg")
    touch(tmp_path / "nested" / "child.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg",),
        video_exts=(".mp4",),
        recursive=False,
    )

    assert [sample.sample_id for sample in result.pairs] == ["root"]
