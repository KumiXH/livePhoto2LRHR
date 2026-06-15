from pathlib import Path

import numpy as np

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.alignment.base import AlignmentContext
from livephoto2lrhr.algorithms.alignment.feature_match_homography import FeatureMatchHomographyAligner
from livephoto2lrhr.algorithms.alignment.feature_match_transform import FeatureMatchTransformAligner
from livephoto2lrhr.algorithms.alignment.global_ecc_homography import GlobalECCHomographyAligner
from livephoto2lrhr.algorithms.alignment.hybrid_feature_flow import HybridFeatureFlowAligner
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner
from livephoto2lrhr.algorithms.alignment.mask_aware import MaskAwareAlignmentAligner


def test_alignment_registry_creates_identity_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("identity_alignment", {})

    assert isinstance(aligner, IdentityAligner)


def test_identity_aligner_returns_copy_through_result(tmp_path: Path):
    lr = np.full((4, 5, 3), 10, dtype=np.uint8)
    hr = np.full((8, 10, 3), 20, dtype=np.uint8)
    context = AlignmentContext(
        sample_id="sample",
        lr_path=tmp_path / "LR" / "sample.png",
        hr_path=tmp_path / "HR" / "sample.png",
        metadata={},
        config={"device": "cpu"},
        artifact_root=tmp_path / "artifacts" / "alignment" / "sample",
        device="cpu",
    )
    aligner = IdentityAligner({})

    result = aligner.align(lr, hr, context)

    assert result.status == "success"
    assert result.confidence == 1.0
    assert np.array_equal(result.aligned_lr_rgb, lr)
    assert result.aligned_lr_rgb is not lr
    assert result.transforms == [{"type": "identity", "coordinate_system": "lr_to_hr"}]
    assert result.artifacts == []
    assert result.diagnostics == {"algorithm": "identity_alignment"}


def test_alignment_registry_creates_feature_match_homography_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("feature_match_homography", {})

    assert isinstance(aligner, FeatureMatchHomographyAligner)


def test_alignment_registry_creates_feature_match_transform_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("feature_match_transform", {})

    assert isinstance(aligner, FeatureMatchTransformAligner)


def test_alignment_registry_creates_global_ecc_homography_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("global_ecc_homography", {})

    assert isinstance(aligner, GlobalECCHomographyAligner)


def test_alignment_registry_creates_hybrid_feature_flow_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("hybrid_feature_flow", {})

    assert isinstance(aligner, HybridFeatureFlowAligner)


def test_alignment_registry_creates_mask_aware_alignment_aligner():
    registry = build_alignment_registry()

    aligner = registry.create("mask_aware_alignment", {})

    assert isinstance(aligner, MaskAwareAlignmentAligner)
