from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector


def make_video(path: Path, frames_rgb: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"could not create test video: {path}"
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def test_opencv_selector_prefers_frame_closest_to_photo(tmp_path: Path):
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "target.mp4"
    target = np.full((32, 32, 3), (20, 120, 220), dtype=np.uint8)
    Image.fromarray(target).save(image_path)
    frames = [
        np.full((32, 32, 3), (200, 20, 20), dtype=np.uint8),
        target.copy(),
        np.full((32, 32, 3), (10, 10, 10), dtype=np.uint8),
    ]
    make_video(video_path, frames)

    selector = OpenCVSimilaritySelector(
        {
            "sample_fps": 30,
            "top_k": 2,
            "resize_short_side": 64,
            "score_fusion": {"feature_weight": 0.7, "edge_weight": 0.3},
        }
    )
    result = selector.select(image_path, video_path)

    assert result.selected.frame_index == 1
    assert result.frame_rgb.shape == (32, 32, 3)
    assert [candidate.frame_index for candidate in result.top_k] == [1, 2]


def test_opencv_selector_raises_for_unreadable_video(tmp_path: Path):
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "bad.mp4"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
    video_path.write_bytes(b"not a video")

    selector = OpenCVSimilaritySelector({"top_k": 1})

    try:
        selector.select(image_path, video_path)
    except ValueError as exc:
        assert "could not open video" in str(exc)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"sample_fps": 0}, "sample_fps must be greater than 0"),
        ({"sample_fps": float("inf")}, "sample_fps must be finite"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"resize_short_side": 0}, "resize_short_side must be at least 1"),
        ({"score_fusion": {"feature_weight": float("nan")}}, "feature_weight must be finite"),
        ({"score_fusion": {"feature_weight": -0.1}}, "feature_weight must be non-negative"),
        ({"score_fusion": {"edge_weight": float("inf")}}, "edge_weight must be finite"),
        ({"score_fusion": {"edge_weight": -0.1}}, "edge_weight must be non-negative"),
        (
            {"score_fusion": {"feature_weight": 0, "edge_weight": 0}},
            "at least one score fusion weight must be greater than 0",
        ),
    ],
)
def test_opencv_selector_rejects_invalid_config(config: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        OpenCVSimilaritySelector(config)


def test_opencv_selector_releases_capture_when_scoring_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "target.mp4"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
    released = False

    class FakeCapture:
        def __init__(self, path: str) -> None:
            self.read_count = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop_id: int) -> float:
            if prop_id == cv2.CAP_PROP_FPS:
                return 30.0
            return 0.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.read_count += 1
            if self.read_count == 1:
                frame_bgr = np.full((8, 8, 3), (3, 2, 1), dtype=np.uint8)
                return True, frame_bgr
            return False, None

        def release(self) -> None:
            nonlocal released
            released = True

    selector = OpenCVSimilaritySelector({"sample_fps": 30, "top_k": 1})

    def raise_score(target_small: np.ndarray, target_edges: np.ndarray, frame_rgb: np.ndarray) -> float:
        raise RuntimeError("score failed")

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(selector, "_score", raise_score)

    with pytest.raises(RuntimeError, match="score failed"):
        selector.select(image_path, video_path)

    assert released
