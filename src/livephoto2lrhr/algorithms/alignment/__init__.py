from livephoto2lrhr.algorithms.alignment.coarse_to_flow import CoarseToFlowAligner
from livephoto2lrhr.algorithms.alignment.ecc import ECCAligner
from livephoto2lrhr.algorithms.alignment.feature_match_homography import FeatureMatchHomographyAligner
from livephoto2lrhr.algorithms.alignment.feature_match_transform import FeatureMatchTransformAligner
from livephoto2lrhr.algorithms.alignment.global_ecc_homography import GlobalECCHomographyAligner
from livephoto2lrhr.algorithms.alignment.hybrid_feature_flow import HybridFeatureFlowAligner
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.mask_aware import MaskAwareAlignmentAligner
from livephoto2lrhr.algorithms.alignment.phase_correlation import PhaseCorrelationTranslationAligner
from livephoto2lrhr.pipeline.registry import Registry


def build_alignment_registry() -> Registry:
    registry = Registry()
    registry.register("coarse_to_flow", CoarseToFlowAligner)
    registry.register("ecc_alignment", ECCAligner)
    registry.register("feature_match_transform", FeatureMatchTransformAligner)
    registry.register("feature_match_homography", FeatureMatchHomographyAligner)
    registry.register("global_ecc_homography", GlobalECCHomographyAligner)
    registry.register("hybrid_feature_flow", HybridFeatureFlowAligner)
    registry.register("identity_alignment", IdentityAligner)
    registry.register("mask_aware_alignment", MaskAwareAlignmentAligner)
    registry.register("phase_correlation_translation", PhaseCorrelationTranslationAligner)
    return registry
