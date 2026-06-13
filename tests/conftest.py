from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "input" / "flower.jpg"
    video_path = tmp_path / "input" / "flower.mp4"
    image_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (8, 8), color=(20, 40, 200))
    image.save(image_path)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (8, 8),
    )
    for value in (0, 80, 160):
        frame_bgr = np.full((8, 8, 3), value, dtype=np.uint8)
        writer.write(frame_bgr)
    writer.release()

    return image_path, video_path
