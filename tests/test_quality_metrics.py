import numpy as np

from livephoto2lrhr.reports.metrics import border_mae, branch_metrics_to_hr, psnr, ssim


def test_psnr_is_higher_for_identical_images():
    image = np.full((8, 8, 3), 100, dtype=np.uint8)
    assert psnr(image, image) > 90.0


def test_ssim_is_lower_for_shifted_brightness():
    left = np.full((8, 8, 3), 100, dtype=np.uint8)
    right = np.full((8, 8, 3), 140, dtype=np.uint8)
    assert ssim(left, left) > ssim(left, right)


def test_branch_metrics_resize_hr_to_candidate_shape():
    candidate = np.full((4, 4, 3), 20, dtype=np.uint8)
    hr = np.full((8, 8, 3), 20, dtype=np.uint8)

    metrics = branch_metrics_to_hr(candidate, hr)

    assert metrics.dimension_match is False
    assert metrics.aspect_ratio_match is True
    assert metrics.mae == 0.0


def test_border_mae_detects_edge_artifacts():
    clean = np.full((8, 8, 3), 50, dtype=np.uint8)
    artifact = clean.copy()
    artifact[[0, -1], :, :] = 200
    artifact[:, [0, -1], :] = 200

    assert border_mae(clean, clean) < border_mae(artifact, clean)
