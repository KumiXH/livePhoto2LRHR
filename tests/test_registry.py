from pathlib import Path

import numpy as np
import pytest

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.pipeline.registry import Registry


def test_registry_creates_registered_algorithm():
    registry = Registry()
    registry.register("fake_selector", FakeFrameSelector)

    selector = registry.create("fake_selector", {"top_k": 2})

    assert isinstance(selector, FakeFrameSelector)
    assert selector.top_k == 2


def test_registry_is_subscriptable_at_runtime():
    assert Registry[FakeFrameSelector]


def test_registry_rejects_unknown_algorithm():
    registry = Registry()

    with pytest.raises(KeyError, match="unknown algorithm"):
        registry.create("missing", {})


def test_fake_selector_returns_one_selected_frame_and_top_k(tmp_path: Path):
    selector = FakeFrameSelector({"top_k": 2})

    result = selector.select(tmp_path / "hr.png", tmp_path / "video.mp4")

    assert result.selected.frame_index == 0
    assert result.frame_rgb.shape == (4, 4, 3)
    assert result.frame_rgb.dtype == np.uint8
    assert result.top_k == [
        FrameCandidate(frame_index=0, timestamp_sec=0.0, score=1.0),
        FrameCandidate(frame_index=1, timestamp_sec=1.0 / 30.0, score=0.5),
    ]


def test_fake_selector_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        FakeFrameSelector({"top_k": 0})
