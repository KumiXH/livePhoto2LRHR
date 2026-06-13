from pathlib import Path

import numpy as np

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.alignment.base import AlignmentContext
from livephoto2lrhr.algorithms.alignment.identity import IdentityAligner


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
