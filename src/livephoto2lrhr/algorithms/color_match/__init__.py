from livephoto2lrhr.algorithms.color_match.identity import IdentityColorMatcher
from livephoto2lrhr.algorithms.color_match.mean_std import MeanStdColorMatcher
from livephoto2lrhr.pipeline.registry import Registry


def build_color_match_registry() -> Registry:
    registry = Registry()
    registry.register("identity_color_match", IdentityColorMatcher)
    registry.register("mean_std_lab", MeanStdColorMatcher)
    return registry
