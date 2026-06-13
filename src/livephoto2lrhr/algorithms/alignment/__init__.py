from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.phase_correlation import PhaseCorrelationTranslationAligner
from livephoto2lrhr.pipeline.registry import Registry


def build_alignment_registry() -> Registry:
    registry = Registry()
    registry.register("identity_alignment", IdentityAligner)
    registry.register("phase_correlation_translation", PhaseCorrelationTranslationAligner)
    return registry
