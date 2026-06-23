from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from livephoto2lrhr.data.image_io import open_pil_image


@dataclass(frozen=True)
class ExportDatasetConfig:
    input_report: str = "reports/quality_report.csv"
    output_folder: str = "final"
    final_lr_source: str = "raw"
    gate_lr_source: str = "aligned"
    final_lr_resize_mode: str = "copy"
    min_align_confidence: float = 0.0
    require_align_status: str | None = "success"
    require_flow_status: str | None = None
    max_source_to_hr_mae: float | None = None
    min_source_to_hr_psnr: float | None = None
    min_source_to_hr_ssim: float | None = None
    require_source_to_hr_dimension_match: bool = False
    require_source_to_hr_aspect_ratio_match: bool = False
    max_source_to_hr_border_mae: float | None = None
    max_mean_flow_magnitude: float | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class ExportDatasetResult:
    accepted: int
    rejected: int
    manifest_path: Path


MANIFEST_FIELDS = [
    "sample_id",
    "accepted",
    "reason",
    "lr_source",
    "gate_lr_source",
    "final_lr_resize_mode",
    "gate_source_to_hr_mae",
    "final_source_lr_path",
    "gate_source_lr_path",
    "raw_lr_path",
    "hr_path",
    "final_lr_path",
    "final_hr_path",
    "align_status",
    "align_confidence",
    "flow_status",
]

LR_SOURCE_PATH_FIELDS = {
    "raw": "lr_path",
    "aligned": "aligned_path",
    "color_matched": "color_matched_path",
}

LR_SOURCE_MAE_FIELDS = {
    "raw": "lr_to_hr_mae",
    "aligned": "aligned_to_hr_mae",
    "color_matched": "color_matched_to_hr_mae",
}

LR_SOURCE_METRIC_FIELDS = {
    "raw": {
        "mae": ("raw_to_hr_mae", "lr_to_hr_mae"),
        "psnr": ("raw_to_hr_psnr",),
        "ssim": ("raw_to_hr_ssim",),
        "dimension_match": ("raw_to_hr_dimension_match",),
        "aspect_ratio_match": ("raw_to_hr_aspect_ratio_match",),
        "border_mae": ("raw_to_hr_border_mae",),
    },
    "aligned": {
        "mae": ("aligned_to_hr_mae",),
        "psnr": ("aligned_to_hr_psnr",),
        "ssim": ("aligned_to_hr_ssim",),
        "dimension_match": ("aligned_to_hr_dimension_match",),
        "aspect_ratio_match": ("aligned_to_hr_aspect_ratio_match",),
        "border_mae": ("aligned_to_hr_border_mae",),
    },
    "color_matched": {
        "mae": ("color_matched_to_hr_mae",),
        "psnr": ("color_matched_to_hr_psnr",),
        "ssim": ("color_matched_to_hr_ssim",),
        "dimension_match": ("color_matched_to_hr_dimension_match",),
        "aspect_ratio_match": ("color_matched_to_hr_aspect_ratio_match",),
        "border_mae": ("color_matched_to_hr_border_mae",),
    },
}

LR_SOURCE_EXISTS_FIELDS = {
    "raw": "lr_exists",
    "aligned": "aligned_exists",
    "color_matched": "color_matched_exists",
}


def export_dataset(output_dir: Path, config: ExportDatasetConfig) -> ExportDatasetResult:
    report_path = _resolve_under_output(output_dir, config.input_report)
    rows = _read_report(report_path)
    export_dir = output_dir / config.output_folder
    manifest_rows = [_manifest_row(output_dir, export_dir, row, config) for row in rows]

    for manifest_row in manifest_rows:
        if manifest_row["accepted"] != "true":
            continue
        _copy_pair(
            raw_lr=_path_or_none(manifest_row["raw_lr_path"]),
            source_lr=Path(manifest_row["final_source_lr_path"]),
            source_hr=Path(manifest_row["hr_path"]),
            final_lr=Path(manifest_row["final_lr_path"]),
            final_hr=Path(manifest_row["final_hr_path"]),
            resize_mode=manifest_row["final_lr_resize_mode"],
            overwrite=config.overwrite,
        )

    manifest_path = export_dir / "manifest.csv"
    _write_manifest(manifest_path, manifest_rows)
    accepted = sum(row["accepted"] == "true" for row in manifest_rows)
    rejected = len(manifest_rows) - accepted
    return ExportDatasetResult(accepted=accepted, rejected=rejected, manifest_path=manifest_path)


def _read_report(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _manifest_row(
    output_dir: Path,
    export_dir: Path,
    row: dict[str, str],
    config: ExportDatasetConfig,
) -> dict[str, str]:
    sample_id = row.get("sample_id", "").strip()
    raw_lr = _source_lr_path(row, "raw")
    final_source_lr = _source_lr_path(row, config.final_lr_source)
    gate_source_lr = _source_lr_path(row, config.gate_lr_source)
    gate_source_mae = row.get(LR_SOURCE_MAE_FIELDS[config.gate_lr_source], "")
    hr_path = _path_or_none(row.get("hr_path", ""))
    relative_output = _relative_output_path(final_source_lr, output_dir, config.final_lr_source, sample_id)
    final_lr = export_dir / "LR" / relative_output
    final_hr = export_dir / "HR" / relative_output
    reason = _acceptance_reason(
        row=row,
        final_source_lr=final_source_lr,
        gate_source_lr=gate_source_lr,
        gate_source_mae=gate_source_mae,
        hr_path=hr_path,
        final_lr=final_lr,
        final_hr=final_hr,
        config=config,
    )
    return {
        "sample_id": sample_id,
        "accepted": "true" if reason == "accepted" else "false",
        "reason": reason,
        "lr_source": config.final_lr_source,
        "gate_lr_source": config.gate_lr_source,
        "final_lr_resize_mode": config.final_lr_resize_mode,
        "gate_source_to_hr_mae": gate_source_mae,
        "final_source_lr_path": _path_to_str(final_source_lr),
        "gate_source_lr_path": _path_to_str(gate_source_lr),
        "raw_lr_path": _path_to_str(raw_lr),
        "hr_path": _path_to_str(hr_path),
        "final_lr_path": str(final_lr),
        "final_hr_path": str(final_hr),
        "align_status": row.get("align_status", ""),
        "align_confidence": row.get("align_confidence", ""),
        "flow_status": row.get("flow_status", ""),
    }


def _acceptance_reason(
    *,
    row: dict[str, str],
    final_source_lr: Path | None,
    gate_source_lr: Path | None,
    gate_source_mae: str,
    hr_path: Path | None,
    final_lr: Path,
    final_hr: Path,
    config: ExportDatasetConfig,
) -> str:
    if config.require_align_status and row.get("align_status", "") != config.require_align_status:
        return "align_status_mismatch"
    align_confidence = _float_or_none(row.get("align_confidence", ""))
    if config.min_align_confidence > 0.0 and align_confidence is None:
        return "align_confidence_missing"
    if align_confidence is not None and align_confidence < config.min_align_confidence:
        return "align_confidence_below_min"
    if config.require_flow_status and row.get("flow_status", "") != config.require_flow_status:
        return "flow_status_mismatch"
    if (
        _is_false(row.get(LR_SOURCE_EXISTS_FIELDS[config.gate_lr_source], ""))
        or gate_source_lr is None
        or not gate_source_lr.exists()
    ):
        return "missing_gate_lr_source"
    if (
        _is_false(row.get(LR_SOURCE_EXISTS_FIELDS[config.final_lr_source], ""))
        or final_source_lr is None
        or not final_source_lr.exists()
    ):
        return "missing_final_lr_source"
    if _is_false(row.get("hr_exists", "")) or hr_path is None or not hr_path.exists():
        return "missing_hr"
    if config.max_source_to_hr_mae is not None:
        mae = _float_or_none(_metric_value(row, config.gate_lr_source, "mae") or gate_source_mae)
        if mae is None:
            return "gate_source_to_hr_mae_missing"
        if mae > config.max_source_to_hr_mae:
            return "gate_source_to_hr_mae_above_max"
    if config.min_source_to_hr_psnr is not None:
        psnr = _float_or_none(_metric_value(row, config.gate_lr_source, "psnr"))
        if psnr is None:
            return "gate_source_to_hr_psnr_missing"
        if psnr < config.min_source_to_hr_psnr:
            return "gate_source_to_hr_psnr_below_min"
    if config.min_source_to_hr_ssim is not None:
        ssim = _float_or_none(_metric_value(row, config.gate_lr_source, "ssim"))
        if ssim is None:
            return "gate_source_to_hr_ssim_missing"
        if ssim < config.min_source_to_hr_ssim:
            return "gate_source_to_hr_ssim_below_min"
    if config.require_source_to_hr_dimension_match:
        dimension_match = _bool_or_none(_metric_value(row, config.gate_lr_source, "dimension_match"))
        if dimension_match is None:
            return "gate_source_to_hr_dimension_match_missing"
        if not dimension_match:
            return "gate_source_to_hr_dimension_mismatch"
    if config.require_source_to_hr_aspect_ratio_match:
        aspect_ratio_match = _bool_or_none(_metric_value(row, config.gate_lr_source, "aspect_ratio_match"))
        if aspect_ratio_match is None:
            return "gate_source_to_hr_aspect_ratio_match_missing"
        if not aspect_ratio_match:
            return "gate_source_to_hr_aspect_ratio_mismatch"
    if config.max_source_to_hr_border_mae is not None:
        border_mae = _float_or_none(_metric_value(row, config.gate_lr_source, "border_mae"))
        if border_mae is None:
            return "gate_source_to_hr_border_mae_missing"
        if border_mae > config.max_source_to_hr_border_mae:
            return "gate_source_to_hr_border_mae_above_max"
    if config.max_mean_flow_magnitude is not None:
        mean_flow_magnitude = _float_or_none(row.get("mean_flow_magnitude", ""))
        if mean_flow_magnitude is None:
            return "mean_flow_magnitude_missing"
        if mean_flow_magnitude > config.max_mean_flow_magnitude:
            return "mean_flow_magnitude_above_max"
    if not config.overwrite and (final_lr.exists() or final_hr.exists()):
        return "destination_exists"
    return "accepted"


def _copy_pair(
    *,
    raw_lr: Path | None,
    source_lr: Path,
    source_hr: Path,
    final_lr: Path,
    final_hr: Path,
    resize_mode: str,
    overwrite: bool,
) -> None:
    if not overwrite and (final_lr.exists() or final_hr.exists()):
        return
    final_lr.parent.mkdir(parents=True, exist_ok=True)
    final_hr.parent.mkdir(parents=True, exist_ok=True)
    _write_final_lr(
        source_lr=source_lr,
        raw_lr=raw_lr,
        source_hr=source_hr,
        destination=final_lr,
        resize_mode=resize_mode,
    )
    shutil.copy2(source_hr, final_hr)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _source_lr_path(row: dict[str, str], lr_source: str) -> Path | None:
    return _path_or_none(row.get(LR_SOURCE_PATH_FIELDS[lr_source], ""))


def _write_final_lr(
    *,
    source_lr: Path,
    raw_lr: Path | None,
    source_hr: Path,
    destination: Path,
    resize_mode: str,
) -> None:
    if resize_mode == "copy":
        shutil.copy2(source_lr, destination)
        return
    replayed = _replay_source_to_low_res(
        source_lr=source_lr,
        raw_lr=raw_lr,
        hr_path=source_hr,
        resize_mode=resize_mode,
    )
    if replayed is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(replayed, mode="RGB").save(destination)
        return
    with open_pil_image(source_lr) as source_image:
        source_rgb = ImageOps.exif_transpose(source_image).convert("RGB")
        if resize_mode in {"match_raw", "raw"}:
            if raw_lr is None or not raw_lr.exists():
                raise ValueError("raw LR source is required for final_lr_resize_mode=match_raw/raw")
            with open_pil_image(raw_lr) as raw_image:
                raw_rgb = ImageOps.exif_transpose(raw_image).convert("RGB")
                target_size = raw_rgb.size
        elif resize_mode in {"1.0", "0.75", "0.5"}:
            scale = float(resize_mode)
            target_size = (
                max(1, int(round(source_rgb.width * scale))),
                max(1, int(round(source_rgb.height * scale))),
            )
        else:
            raise ValueError(f"unsupported final LR resize mode: {resize_mode}")
        resized = source_rgb.resize(target_size, Image.Resampling.LANCZOS)
        resized.save(destination)


def _replay_source_to_low_res(
    *,
    source_lr: Path,
    raw_lr: Path | None,
    hr_path: Path,
    resize_mode: str,
) -> np.ndarray | None:
    if resize_mode not in {"match_raw", "raw"}:
        return None
    if raw_lr is None or not raw_lr.exists():
        return None
    metadata = _load_metadata_for_export(raw_lr)
    if metadata is None:
        return None
    if source_lr == raw_lr:
        return None
    with open_pil_image(raw_lr) as raw_image:
        raw_rgb = np.asarray(ImageOps.exif_transpose(raw_image).convert("RGB"))
    with open_pil_image(hr_path) as hr_image:
        hr_rgb = np.asarray(ImageOps.exif_transpose(hr_image).convert("RGB"))
    replayed = _apply_alignment_replay(raw_rgb=raw_rgb, hr_rgb=hr_rgb, metadata=metadata)
    color_matched_path = _metadata_color_matched_path(metadata)
    if source_lr == color_matched_path:
        color_input = raw_rgb if replayed is None else replayed
        return _apply_color_match_replay(color_input, metadata)
    return replayed


def _load_metadata_for_export(raw_lr: Path) -> dict[str, object] | None:
    output_root = raw_lr.parents[1]
    lr_root = output_root / "LR"
    try:
        relative = raw_lr.relative_to(lr_root).with_suffix(".yaml")
    except ValueError:
        return None
    metadata_path = output_root / "metadata" / relative
    if not metadata_path.exists():
        return None
    return yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}


def _metadata_color_matched_path(metadata: dict[str, object]) -> Path | None:
    output = metadata.get("color_match", {}).get("output", {})
    matched = output.get("lr_color_matched")
    if not matched:
        return None
    return Path(str(matched))


def _apply_alignment_replay(
    *,
    raw_rgb: np.ndarray,
    hr_rgb: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray | None:
    align = metadata.get("align")
    if not isinstance(align, dict):
        return None
    transforms = align.get("transforms")
    if not isinstance(transforms, list):
        return None
    output_rgb = raw_rgb.copy()
    scale_x = hr_rgb.shape[1] / raw_rgb.shape[1]
    scale_y = hr_rgb.shape[0] / raw_rgb.shape[0]
    for transform in transforms:
        if not isinstance(transform, dict):
            return None
        transform_type = str(transform.get("type", ""))
        if transform_type == "translation":
            dx_hr = float(transform.get("dx", 0.0))
            dy_hr = float(transform.get("dy", 0.0))
            matrix = np.array(
                [[1.0, 0.0, dx_hr / scale_x], [0.0, 1.0, dy_hr / scale_y]],
                dtype=np.float32,
            )
            output_rgb = cv2.warpAffine(
                output_rgb,
                matrix,
                (raw_rgb.shape[1], raw_rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            continue
        if transform_type.startswith("ecc_"):
            matrix = _scaled_ecc_matrix(transform, scale_x=scale_x, scale_y=scale_y)
            if matrix is None:
                return None
            if matrix.shape == (2, 3):
                output_rgb = cv2.warpAffine(
                    output_rgb,
                    matrix,
                    (raw_rgb.shape[1], raw_rgb.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT,
                )
            else:
                output_rgb = cv2.warpPerspective(
                    output_rgb,
                    matrix,
                    (raw_rgb.shape[1], raw_rgb.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT,
                )
            continue
        if transform_type == "feature_match_homography":
            matrix = _scaled_homography_matrix(transform, scale_x=scale_x, scale_y=scale_y)
            if matrix is None:
                return None
            output_rgb = cv2.warpPerspective(
                output_rgb,
                matrix,
                (raw_rgb.shape[1], raw_rgb.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            continue
        if transform_type in {"feature_match_affine", "feature_match_similarity"}:
            matrix = _scaled_affine_matrix(transform, scale_x=scale_x, scale_y=scale_y)
            if matrix is None:
                return None
            output_rgb = cv2.warpAffine(
                output_rgb,
                matrix,
                (raw_rgb.shape[1], raw_rgb.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            continue
        if transform_type == "identity":
            continue
        if transform_type == "dense_flow":
            return None
        if transform_type.startswith("mask_aware_"):
            return None
        return None
    return output_rgb


def _scaled_ecc_matrix(transform: dict[str, object], *, scale_x: float, scale_y: float) -> np.ndarray | None:
    matrix_value = transform.get("matrix")
    if not isinstance(matrix_value, list):
        return None
    matrix = np.asarray(matrix_value, dtype=np.float32)
    if matrix.shape == (2, 3):
        matrix = matrix.copy()
        matrix[0, 2] /= scale_x
        matrix[1, 2] /= scale_y
        return matrix
    if matrix.shape == (3, 3):
        left = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        right = np.array(
            [[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return right @ matrix @ left
    return None


def _apply_color_match_replay(image_rgb: np.ndarray, metadata: dict[str, object]) -> np.ndarray:
    color_match = metadata.get("color_match")
    if not isinstance(color_match, dict):
        return image_rgb
    transforms = color_match.get("transforms")
    if not isinstance(transforms, list):
        return image_rgb
    output_rgb = image_rgb.copy()
    for transform in transforms:
        if not isinstance(transform, dict):
            return image_rgb
        transform_type = str(transform.get("type", ""))
        if transform_type == "mean_std_color_transfer":
            output_rgb = _replay_mean_std_color_transfer(output_rgb, transform)
            continue
        if transform_type == "histogram_color_transfer":
            replayed = _replay_histogram_color_transfer(output_rgb, metadata)
            if replayed is None:
                return image_rgb
            output_rgb = replayed
            continue
        if transform_type == "retinex_color_transfer":
            replayed = _replay_reference_color_transfer(output_rgb, metadata)
            if replayed is None:
                return image_rgb
            output_rgb = replayed
            continue
        if transform_type == "adaptive_3d_lut_color_transfer":
            output_rgb = _replay_adaptive_3d_lut_color_transfer(output_rgb, transform)
            continue
        if transform_type == "masked_mean_std_color_transfer":
            replayed = _replay_reference_color_transfer(output_rgb, metadata)
            if replayed is None:
                return image_rgb
            output_rgb = replayed
            continue
        if transform_type in {
            "low_frequency_joint_appearance_transfer",
            "learned_retinex_color_transfer",
            "mask_aware_harmonization_transfer",
            "diffusion_harmonization_transfer",
        }:
            replayed = _replay_reference_color_transfer(output_rgb, metadata)
            if replayed is None:
                return image_rgb
            output_rgb = replayed
            continue
        return image_rgb
    return output_rgb


def _replay_mean_std_color_transfer(image_rgb: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    color_space = str(transform.get("color_space", "lab")).lower()
    source_mean = np.asarray(transform.get("source_mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
    source_std = np.asarray(transform.get("source_std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
    target_mean = np.asarray(transform.get("target_mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
    target_std = np.asarray(transform.get("target_std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
    if color_space == "lab":
        work = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    else:
        work = image_rgb.astype(np.float32)
    matched = (work - source_mean) * (target_std / np.maximum(source_std, 1.0e-6)) + target_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    if color_space == "lab":
        return cv2.cvtColor(matched, cv2.COLOR_LAB2RGB)
    return matched


def _replay_adaptive_3d_lut_color_transfer(image_rgb: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    grid_size = int(transform.get("grid_size", 2))
    lut_value = transform.get("lut")
    if not isinstance(lut_value, list):
        return image_rgb
    lut = np.asarray(lut_value, dtype=np.float32)
    if lut.shape != (grid_size, grid_size, grid_size, 3):
        return image_rgb
    normalized = image_rgb.astype(np.float32) / 255.0
    coords = normalized * (grid_size - 1)
    low = np.floor(coords).astype(np.int32)
    high = np.clip(low + 1, 0, grid_size - 1)
    frac = coords - low
    output = np.zeros_like(normalized)
    for dr in (0, 1):
        wr = (1.0 - frac[..., 0]) if dr == 0 else frac[..., 0]
        rr = low[..., 0] if dr == 0 else high[..., 0]
        for dg in (0, 1):
            wg = (1.0 - frac[..., 1]) if dg == 0 else frac[..., 1]
            gg = low[..., 1] if dg == 0 else high[..., 1]
            for db in (0, 1):
                wb = (1.0 - frac[..., 2]) if db == 0 else frac[..., 2]
                bb = low[..., 2] if db == 0 else high[..., 2]
                weight = (wr * wg * wb)[..., None]
                output += lut[rr, gg, bb] * weight
    return np.clip(np.round(output * 255.0), 0, 255).astype(np.uint8)


def _scaled_homography_matrix(transform: dict[str, object], *, scale_x: float, scale_y: float) -> np.ndarray | None:
    matrix_value = transform.get("matrix")
    if not isinstance(matrix_value, list):
        return None
    matrix = np.asarray(matrix_value, dtype=np.float32)
    if matrix.shape != (3, 3):
        return None
    left = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    right = np.array(
        [[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return right @ matrix @ left


def _scaled_affine_matrix(transform: dict[str, object], *, scale_x: float, scale_y: float) -> np.ndarray | None:
    matrix_value = transform.get("matrix")
    if not isinstance(matrix_value, list):
        return None
    matrix = np.asarray(matrix_value, dtype=np.float32)
    if matrix.shape != (2, 3):
        return None
    matrix = matrix.copy()
    matrix[0, 2] /= scale_x
    matrix[1, 2] /= scale_y
    return matrix


def _replay_histogram_color_transfer(image_rgb: np.ndarray, metadata: dict[str, object]) -> np.ndarray | None:
    return _replay_reference_color_transfer(image_rgb, metadata)


def _replay_reference_color_transfer(image_rgb: np.ndarray, metadata: dict[str, object]) -> np.ndarray | None:
    color_match = metadata.get("color_match")
    if not isinstance(color_match, dict):
        return None
    diagnostics = color_match.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    replay_reference_path = diagnostics.get("replay_reference_path")
    if not replay_reference_path:
        return None
    reference_path = Path(str(replay_reference_path))
    if not reference_path.exists():
        return None
    with open_pil_image(reference_path) as reference_image:
        reference_rgb = np.asarray(ImageOps.exif_transpose(reference_image).convert("RGB"))
    return cv2.resize(
        reference_rgb,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


def _relative_output_path(source_lr: Path | None, output_dir: Path, lr_source: str, sample_id: str) -> Path:
    if source_lr is None:
        safe_sample = Path(sample_id)
        return safe_sample.parent / f"{safe_sample.name}.png"
    source_root = output_dir / _source_folder(lr_source)
    try:
        return source_lr.relative_to(source_root)
    except ValueError:
        safe_sample = Path(sample_id)
        suffix = source_lr.suffix or ".png"
        return safe_sample.parent / f"{safe_sample.name}{suffix}"


def _source_folder(lr_source: str) -> str:
    if lr_source == "raw":
        return "LR"
    if lr_source == "aligned":
        return "LR_aligned"
    return "LR_color_matched"


def _resolve_under_output(output_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    return path.resolve()


def _path_or_none(value: str | None) -> Path | None:
    if value is None or not value.strip():
        return None
    return Path(value)


def _path_to_str(value: Path | None) -> str:
    return "" if value is None else str(value)


def _float_or_none(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_value(row: dict[str, str], lr_source: str, metric: str) -> str:
    for field in LR_SOURCE_METRIC_FIELDS[lr_source][metric]:
        value = row.get(field, "")
        if value != "":
            return value
    return ""


def _bool_or_none(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "":
        return None
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _is_false(value: str) -> bool:
    return value.lower() in {"false", "0", "no"}
