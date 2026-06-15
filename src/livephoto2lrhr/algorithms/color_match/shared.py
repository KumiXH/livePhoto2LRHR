from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def config_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def resize_to_lr(hr_rgb: np.ndarray, lr_rgb: np.ndarray) -> np.ndarray:
    target_height, target_width = lr_rgb.shape[:2]
    if hr_rgb.shape[:2] == (target_height, target_width):
        return hr_rgb
    return cv2.resize(hr_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def to_work_space(image_rgb: np.ndarray, color_space: str) -> np.ndarray:
    if color_space == "lab":
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return image_rgb.astype(np.float32)


def from_work_space(image: np.ndarray, color_space: str) -> np.ndarray:
    output = np.clip(image, 0, 255).astype(np.uint8)
    if color_space == "lab":
        return cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
    return output


def mean_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def confidence_from_errors(pre_error: float, post_error: float) -> float:
    if pre_error <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - post_error / pre_error))


def channel_transfer(
    source: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1.0e-6,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    matched = source.copy()
    if mask is None:
        source_pixels = source.reshape(-1, source.shape[2])
        target_pixels = target.reshape(-1, target.shape[2])
        transformed = _transfer_pixels(source_pixels, target_pixels, eps=eps)
        return transformed.reshape(source.shape)
    if not np.any(mask):
        return matched
    source_pixels = source[mask]
    target_pixels = target[mask]
    matched[mask] = _transfer_pixels(source_pixels, target_pixels, eps=eps)
    return matched


def _transfer_pixels(source_pixels: np.ndarray, target_pixels: np.ndarray, *, eps: float) -> np.ndarray:
    source_mean = np.mean(source_pixels, axis=0, keepdims=True)
    source_std = np.std(source_pixels, axis=0, keepdims=True)
    target_mean = np.mean(target_pixels, axis=0, keepdims=True)
    target_std = np.std(target_pixels, axis=0, keepdims=True)
    return (source_pixels - source_mean) * (target_std / np.maximum(source_std, eps)) + target_mean


def build_difference_mask(
    lr_work: np.ndarray,
    hr_work: np.ndarray,
    *,
    difference_threshold: float,
    morphology_kernel_size: int,
) -> np.ndarray:
    diff = np.mean(np.abs(hr_work - lr_work), axis=2)
    mask = (diff >= difference_threshold).astype(np.uint8) * 255
    kernel_size = max(1, morphology_kernel_size)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def blend_with_soft_mask(
    base_image: np.ndarray,
    refined_image: np.ndarray,
    mask: np.ndarray,
    *,
    blur_radius: int,
) -> np.ndarray:
    soft = mask.astype(np.float32)
    blur_size = max(3, blur_radius)
    if blur_size % 2 == 0:
        blur_size += 1
    soft = cv2.GaussianBlur(soft, (blur_size, blur_size), 0)
    soft = soft[..., None]
    return refined_image * soft + base_image * (1.0 - soft)


def gaussian_low_high_split(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    base = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    detail = image - base
    return base, detail
