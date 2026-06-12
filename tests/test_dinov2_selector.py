import builtins
from pathlib import Path

import numpy as np
import pytest

from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.algorithms.similarity.base import FrameCandidate
from livephoto2lrhr.algorithms.similarity.dinov2 import DINOv2SimilaritySelector
from livephoto2lrhr.utils.device import resolve_device


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

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, tensor):
            return tensor

    class FakeTransform:
        def __call__(self, image):
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            return FakeTensor([rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()])

    def fake_cosine_similarity(left, right):
        left_array = left.flatten()
        right_array = right.flatten()
        return float(np.dot(left_array, right_array) / (np.linalg.norm(left_array) * np.linalg.norm(right_array)))

    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._import_torch", lambda: FakeTorch)
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._build_transform", lambda size: FakeTransform())
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._cosine_similarity", fake_cosine_similarity)

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
