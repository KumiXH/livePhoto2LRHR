from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BranchMetrics:
    mae: float | None
    psnr: float | None
    ssim: float | None
    dimension_match: bool | None
    aspect_ratio_match: bool | None
    border_mae: float | None


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float32)
    right_f = right.astype(np.float32)
    mse = float(np.mean((left_f - right_f) ** 2))
    if mse == 0.0:
        return 99.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def ssim(left: np.ndarray, right: np.ndarray) -> float:
    left_f = left.astype(np.float32)
    right_f = right.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_x = float(np.mean(left_f))
    mu_y = float(np.mean(right_f))
    sigma_x = float(np.var(left_f))
    sigma_y = float(np.var(right_f))
    sigma_xy = float(np.mean((left_f - mu_x) * (right_f - mu_y)))
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def border_mae(candidate: np.ndarray, hr: np.ndarray, border_ratio: float = 0.1) -> float:
    height, width = candidate.shape[:2]
    border = max(1, int(min(height, width) * border_ratio))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    diff = np.abs(candidate.astype(np.float32) - hr.astype(np.float32))
    return float(np.mean(diff[mask]))


def branch_metrics_to_hr(candidate: np.ndarray, hr: np.ndarray) -> BranchMetrics:
    dimension_match = candidate.shape[:2] == hr.shape[:2]
    aspect_ratio_match = candidate.shape[1] * hr.shape[0] == hr.shape[1] * candidate.shape[0]
    if not dimension_match:
        with Image.fromarray(hr) as hr_image:
            resized = hr_image.resize((candidate.shape[1], candidate.shape[0]), Image.Resampling.BICUBIC)
            hr = np.asarray(resized)
    candidate_f = candidate.astype(np.float32)
    hr_f = hr.astype(np.float32)
    mae = float(np.mean(np.abs(candidate_f - hr_f)))
    return BranchMetrics(
        mae=mae,
        psnr=psnr(candidate, hr),
        ssim=ssim(candidate, hr),
        dimension_match=dimension_match,
        aspect_ratio_match=aspect_ratio_match,
        border_mae=border_mae(candidate, hr),
    )
