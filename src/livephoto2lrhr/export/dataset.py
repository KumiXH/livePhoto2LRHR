from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


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
        mae = _float_or_none(gate_source_mae)
        if mae is None:
            return "gate_source_to_hr_mae_missing"
        if mae > config.max_source_to_hr_mae:
            return "gate_source_to_hr_mae_above_max"
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
    _write_final_lr(source_lr=source_lr, raw_lr=raw_lr, destination=final_lr, resize_mode=resize_mode)
    shutil.copy2(source_hr, final_hr)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _source_lr_path(row: dict[str, str], lr_source: str) -> Path | None:
    return _path_or_none(row.get(LR_SOURCE_PATH_FIELDS[lr_source], ""))


def _write_final_lr(*, source_lr: Path, raw_lr: Path | None, destination: Path, resize_mode: str) -> None:
    if resize_mode == "copy":
        shutil.copy2(source_lr, destination)
        return
    if resize_mode != "match_raw":
        raise ValueError(f"unsupported final LR resize mode: {resize_mode}")
    if raw_lr is None or not raw_lr.exists():
        raise ValueError("raw LR source is required for final_lr_resize_mode=match_raw")
    with Image.open(source_lr) as source_image, Image.open(raw_lr) as raw_image:
        source_rgb = ImageOps.exif_transpose(source_image).convert("RGB")
        raw_rgb = ImageOps.exif_transpose(raw_image).convert("RGB")
        resized = source_rgb.resize(raw_rgb.size, Image.Resampling.LANCZOS)
        resized.save(destination)


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


def _is_false(value: str) -> bool:
    return value.lower() in {"false", "0", "no"}
