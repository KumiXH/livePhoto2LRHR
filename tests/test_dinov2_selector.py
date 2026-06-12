import builtins
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.algorithms.similarity.base import FrameCandidate
from livephoto2lrhr.algorithms.similarity.dinov2 import DINOv2SimilaritySelector
from livephoto2lrhr.utils.device import resolve_device


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def to(self, device):
        return self

    def unsqueeze(self, dim):
        return self

    def cpu(self):
        return self

    def flatten(self):
        return self.value.reshape(-1)


class FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, tensor):
        return tensor


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class no_grad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class hub:
        @staticmethod
        def load(repo, model_name):
            return FakeModel()


class FakeTransform:
    def __call__(self, image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return FakeTensor([rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()])


def fake_cosine_similarity(left, right):
    left_array = left.flatten()
    right_array = right.flatten()
    return float(np.dot(left_array, right_array) / (np.linalg.norm(left_array) * np.linalg.norm(right_array)))


def patch_fake_dinov2(monkeypatch):
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._import_torch", lambda: FakeTorch)
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._build_transform", lambda size: FakeTransform())
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._cosine_similarity", fake_cosine_similarity)


def make_video(path: Path, frames_rgb: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), f"could not create test video: {path}"
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def test_dinov2_selector_reports_missing_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Install the dinov2 extra"):
        DINOv2SimilaritySelector({"device": "cpu"})


def test_dinov2_ranks_frames_by_feature_similarity(monkeypatch, tmp_path: Path):
    patch_fake_dinov2(monkeypatch)

    selector = DINOv2SimilaritySelector({"device": "cpu", "top_k": 2, "resize_short_side": 32})
    ranked = selector._rank_features(
        FakeTensor([0.0, 1.0, 0.0]),
        [
            (FrameCandidate(frame_index=0, timestamp_sec=0.0, score=0.0), FakeTensor([1.0, 0.0, 0.0])),
            (FrameCandidate(frame_index=1, timestamp_sec=1.0, score=0.0), FakeTensor([0.0, 1.0, 0.0])),
            (FrameCandidate(frame_index=2, timestamp_sec=2.0, score=0.0), FakeTensor([0.0, 0.5, 0.5])),
        ],
    )

    assert [candidate.frame_index for candidate in ranked] == [1, 2, 0]
    assert ranked[0].score == pytest.approx(1.0)


def test_dinov2_selector_releases_capture_when_video_open_fails(tmp_path: Path, monkeypatch):
    patch_fake_dinov2(monkeypatch)
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "bad.mp4"
    Image.new("RGB", (8, 8), color=(20, 220, 20)).save(image_path)
    released = False

    class FakeCapture:
        def __init__(self, path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return False

        def release(self) -> None:
            nonlocal released
            released = True

    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2.cv2.VideoCapture", FakeCapture)
    selector = DINOv2SimilaritySelector({"device": "cpu", "top_k": 1})

    with pytest.raises(ValueError, match="could not open video"):
        selector.select(image_path, video_path)

    assert released


def test_dinov2_selector_selects_best_frame_from_video(monkeypatch, tmp_path: Path):
    patch_fake_dinov2(monkeypatch)
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "target.mp4"
    target = np.full((32, 32, 3), (20, 220, 20), dtype=np.uint8)
    Image.fromarray(target).save(image_path)
    frames = [
        np.full((32, 32, 3), (220, 20, 20), dtype=np.uint8),
        target.copy(),
        np.full((32, 32, 3), (20, 120, 160), dtype=np.uint8),
    ]
    make_video(video_path, frames)

    selector = DINOv2SimilaritySelector({"device": "cpu", "sample_fps": 30, "top_k": 2, "resize_short_side": 32})
    result = selector.select(image_path, video_path)

    assert result.selected.frame_index == 1
    assert [candidate.frame_index for candidate in result.top_k] == [1, 2]
    assert len(result.top_k) == 2
    assert result.frame_rgb.shape == (32, 32, 3)
    assert result.diagnostics["algorithm"] == "dinov2_similarity"
    assert result.diagnostics["model"] == "dinov2_vits14"
    assert result.diagnostics["scored_frames"] == 3


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"sample_fps": 0}, "sample_fps must be greater than 0"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"resize_short_side": 0}, "resize_short_side must be at least 1"),
    ],
)
def test_dinov2_selector_rejects_invalid_config_before_model_load(
    monkeypatch, config: dict[str, object], message: str
):
    def fail_import():
        raise AssertionError("model dependencies should not be imported for invalid config")

    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._import_torch", fail_import)

    with pytest.raises(ValueError, match=message):
        DINOv2SimilaritySelector(config)


def test_resolve_device_rejects_unsupported_device():
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("unsupported")


def test_similarity_registry_includes_dinov2_without_instantiating():
    registry = build_similarity_registry()

    assert "dinov2_similarity" in registry._items
