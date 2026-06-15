from livephoto2lrhr.algorithms.color_match.diffusion_harmonization import DiffusionHarmonizationMatcher
from livephoto2lrhr.algorithms.color_match.histogram_match import HistogramMatchColorMatcher
from livephoto2lrhr.algorithms.color_match.image_adaptive_3d_lut import ImageAdaptive3DLUTColorMatcher
from livephoto2lrhr.algorithms.color_match.identity import IdentityColorMatcher
from livephoto2lrhr.algorithms.color_match.learned_retinex import LearnedRetinexColorMatcher
from livephoto2lrhr.algorithms.color_match.low_frequency_joint import LowFrequencyJointAppearanceMatcher
from livephoto2lrhr.algorithms.color_match.mask_aware_harmonization import MaskAwareHarmonizationNetworkMatcher
from livephoto2lrhr.algorithms.color_match.masked_transfer import MaskedColorTransferMatcher
from livephoto2lrhr.algorithms.color_match.mean_std import MeanStdColorMatcher
from livephoto2lrhr.algorithms.color_match.retinex import RetinexColorMatcher
from livephoto2lrhr.pipeline.registry import Registry


def build_color_match_registry() -> Registry:
    registry = Registry()
    registry.register("diffusion_harmonization", DiffusionHarmonizationMatcher)
    registry.register("histogram_match_lab", HistogramMatchColorMatcher)
    registry.register("image_adaptive_3d_lut_color_match", ImageAdaptive3DLUTColorMatcher)
    registry.register("identity_color_match", IdentityColorMatcher)
    registry.register("learned_retinex_color_match", LearnedRetinexColorMatcher)
    registry.register("low_frequency_joint_appearance_match", LowFrequencyJointAppearanceMatcher)
    registry.register("mask_aware_harmonization_network", MaskAwareHarmonizationNetworkMatcher)
    registry.register("masked_color_transfer", MaskedColorTransferMatcher)
    registry.register("mean_std_lab", MeanStdColorMatcher)
    registry.register("retinex_color_match", RetinexColorMatcher)
    return registry
