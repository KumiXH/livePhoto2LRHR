from livephoto2lrhr.algorithms.color_match import build_color_match_registry
from livephoto2lrhr.algorithms.color_match.diffusion_harmonization import DiffusionHarmonizationMatcher
from livephoto2lrhr.algorithms.color_match.histogram_match import HistogramMatchColorMatcher
from livephoto2lrhr.algorithms.color_match.image_adaptive_3d_lut import ImageAdaptive3DLUTColorMatcher
from livephoto2lrhr.algorithms.color_match.learned_retinex import LearnedRetinexColorMatcher
from livephoto2lrhr.algorithms.color_match.low_frequency_joint import LowFrequencyJointAppearanceMatcher
from livephoto2lrhr.algorithms.color_match.mask_aware_harmonization import MaskAwareHarmonizationNetworkMatcher
from livephoto2lrhr.algorithms.color_match.masked_transfer import MaskedColorTransferMatcher
from livephoto2lrhr.algorithms.color_match.retinex import RetinexColorMatcher


def test_color_match_registry_creates_histogram_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("histogram_match_lab", {})

    assert isinstance(matcher, HistogramMatchColorMatcher)


def test_color_match_registry_creates_retinex_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("retinex_color_match", {})

    assert isinstance(matcher, RetinexColorMatcher)


def test_color_match_registry_creates_masked_transfer_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("masked_color_transfer", {})

    assert isinstance(matcher, MaskedColorTransferMatcher)


def test_color_match_registry_creates_adaptive_3d_lut_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("image_adaptive_3d_lut_color_match", {})

    assert isinstance(matcher, ImageAdaptive3DLUTColorMatcher)


def test_color_match_registry_creates_low_frequency_joint_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("low_frequency_joint_appearance_match", {})

    assert isinstance(matcher, LowFrequencyJointAppearanceMatcher)


def test_color_match_registry_creates_learned_retinex_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("learned_retinex_color_match", {})

    assert isinstance(matcher, LearnedRetinexColorMatcher)


def test_color_match_registry_creates_mask_aware_harmonization_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("mask_aware_harmonization_network", {})

    assert isinstance(matcher, MaskAwareHarmonizationNetworkMatcher)


def test_color_match_registry_creates_diffusion_harmonization_matcher():
    registry = build_color_match_registry()

    matcher = registry.create("diffusion_harmonization", {})

    assert isinstance(matcher, DiffusionHarmonizationMatcher)
