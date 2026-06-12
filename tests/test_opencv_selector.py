from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector


def make_video(path: Path, frames_rgb: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
