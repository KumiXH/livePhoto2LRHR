from livephoto2lrhr.algorithms.similarity.base import FrameSelector
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector
from livephoto2lrhr.pipeline.registry import Registry


def build_similarity_registry() -> Registry[FrameSelector]:
    registry: Registry[FrameSelector] = Registry()
    registry.register("fake_selector", FakeFrameSelector)
    registry.register("opencv_similarity", OpenCVSimilaritySelector)
    return registry
