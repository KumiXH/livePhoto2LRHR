from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from livephoto2lrhr.data.image_io import open_pil_image
from livephoto2lrhr.data.io import output_image_path, read_rgb_array
from livephoto2lrhr.reports.metrics import branch_metrics_to_hr


@dataclass(frozen=True)
class QualityReportConfig:
    output_folder: str = "reports"
    aligned_folder: str = "LR_aligned"
    color_matched_folder: str = "LR_color_matched"
    max_preview_samples: int = 24
    thumbnail_size: int = 160


@dataclass(frozen=True)
class QualityReportResult:
    rows: int
    csv_path: Path
    csv_zh_path: Path
    preview_path: Path | None


CSV_FIELDS = [
    "sample_id",
    "frame_select_algorithm",
    "frame_index",
    "timestamp_sec",
    "frame_select_score",
    "align_algorithm",
    "align_status",
    "align_confidence",
    "align_pre_error",
    "align_post_error",
    "flow_used",
    "flow_status",
    "pre_flow_error",
    "post_flow_error",
    "mean_flow_magnitude",
    "color_match_algorithm",
    "color_match_status",
    "color_match_confidence",
    "color_match_pre_error",
    "color_match_post_error",
    "lr_exists",
    "aligned_exists",
    "color_matched_exists",
    "hr_exists",
    "raw_to_hr_mae",
    "raw_to_hr_psnr",
    "raw_to_hr_ssim",
    "raw_to_hr_dimension_match",
    "raw_to_hr_aspect_ratio_match",
    "raw_to_hr_border_mae",
    "aligned_to_hr_psnr",
    "aligned_to_hr_ssim",
    "aligned_to_hr_dimension_match",
    "aligned_to_hr_aspect_ratio_match",
    "aligned_to_hr_border_mae",
    "color_matched_to_hr_psnr",
    "color_matched_to_hr_ssim",
    "color_matched_to_hr_dimension_match",
    "color_matched_to_hr_aspect_ratio_match",
    "color_matched_to_hr_border_mae",
    "lr_to_hr_mae",
    "aligned_to_hr_mae",
    "color_matched_to_hr_mae",
    "lr_path",
    "aligned_path",
    "color_matched_path",
    "hr_path",
]

CSV_HEADER_ZH = {
    "sample_id": "样本ID",
    "frame_select_algorithm": "抽帧算法",
    "frame_index": "帧索引",
    "timestamp_sec": "时间戳_秒",
    "frame_select_score": "抽帧分数",
    "align_algorithm": "对齐算法",
    "align_status": "对齐状态",
    "align_confidence": "对齐置信度",
    "align_pre_error": "对齐前误差",
    "align_post_error": "对齐后误差",
    "flow_used": "是否使用光流",
    "flow_status": "光流状态",
    "pre_flow_error": "光流前误差",
    "post_flow_error": "光流后误差",
    "mean_flow_magnitude": "平均光流幅度",
    "color_match_algorithm": "调色算法",
    "color_match_status": "调色状态",
    "color_match_confidence": "调色置信度",
    "color_match_pre_error": "调色前误差",
    "color_match_post_error": "调色后误差",
    "lr_exists": "原始LR存在",
    "aligned_exists": "对齐LR存在",
    "color_matched_exists": "调色LR存在",
    "hr_exists": "HR存在",
    "raw_to_hr_mae": "原始LR到HR_MAE",
    "raw_to_hr_psnr": "原始LR到HR_PSNR",
    "raw_to_hr_ssim": "原始LR到HR_SSIM",
    "raw_to_hr_dimension_match": "原始LR到HR_尺寸一致",
    "raw_to_hr_aspect_ratio_match": "原始LR到HR_比例一致",
    "raw_to_hr_border_mae": "原始LR到HR_边缘伪影分数",
    "aligned_to_hr_psnr": "对齐LR到HR_PSNR",
    "aligned_to_hr_ssim": "对齐LR到HR_SSIM",
    "aligned_to_hr_dimension_match": "对齐LR到HR_尺寸一致",
    "aligned_to_hr_aspect_ratio_match": "对齐LR到HR_比例一致",
    "aligned_to_hr_border_mae": "对齐LR到HR_边缘伪影分数",
    "color_matched_to_hr_psnr": "调色LR到HR_PSNR",
    "color_matched_to_hr_ssim": "调色LR到HR_SSIM",
    "color_matched_to_hr_dimension_match": "调色LR到HR_尺寸一致",
    "color_matched_to_hr_aspect_ratio_match": "调色LR到HR_比例一致",
    "color_matched_to_hr_border_mae": "调色LR到HR_边缘伪影分数",
    "lr_to_hr_mae": "LR到HR_MAE_兼容列",
    "aligned_to_hr_mae": "对齐LR到HR_MAE",
    "color_matched_to_hr_mae": "调色LR到HR_MAE",
    "lr_path": "原始LR路径",
    "aligned_path": "对齐LR路径",
    "color_matched_path": "调色LR路径",
    "hr_path": "HR路径",
}


def generate_quality_report(output_dir: Path, config: QualityReportConfig) -> QualityReportResult:
    report_dir = output_dir / config.output_folder
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _row_from_metadata(output_dir, path, config=config)
        for path in sorted((output_dir / "metadata").rglob("*.yaml"))
    ]
    csv_path = report_dir / "quality_report.csv"
    _write_csv(csv_path, rows)
    zh_csv_path = report_dir / "quality_report_zh.csv"
    _write_csv_zh(zh_csv_path, rows)
    preview_path = None
    if rows and config.max_preview_samples > 0:
        preview_path = report_dir / "preview_contact_sheet.jpg"
        _write_contact_sheet(preview_path, rows[: config.max_preview_samples], config.thumbnail_size)
    return QualityReportResult(rows=len(rows), csv_path=csv_path, csv_zh_path=zh_csv_path, preview_path=preview_path)


def _row_from_metadata(output_dir: Path, meta_path: Path, *, config: QualityReportConfig) -> dict[str, str]:
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    relative_stem = meta_path.relative_to(output_dir / "metadata").with_suffix("")
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    aligned_path = output_image_path(output_dir, config.aligned_folder, relative_stem, ".png")
    matched_path = output_image_path(output_dir, config.color_matched_folder, relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    selected = _nested(metadata, "frame_select", "selected") or {}
    align = metadata.get("align") or {}
    color_match = metadata.get("color_match") or {}
    align_diag = align.get("diagnostics") or {}
    color_diag = color_match.get("diagnostics") or {}
    raw_metrics = _metrics_to_hr(lr_path, hr_path)
    aligned_metrics = _metrics_to_hr(aligned_path, hr_path)
    color_matched_metrics = _metrics_to_hr(matched_path, hr_path)
    row = {
        "sample_id": str(metadata.get("sample_id") or relative_stem.as_posix()),
        "frame_select_algorithm": _as_str(_nested(metadata, "frame_select", "algorithm")),
        "frame_index": _as_str(selected.get("frame_index")),
        "timestamp_sec": _as_str(selected.get("timestamp_sec")),
        "frame_select_score": _as_str(selected.get("score")),
        "align_algorithm": _as_str(align.get("algorithm")),
        "align_status": _as_str(align.get("status")),
        "align_confidence": _as_str(align.get("confidence")),
        "align_pre_error": _as_str(align_diag.get("pre_alignment_error")),
        "align_post_error": _as_str(align_diag.get("post_alignment_error")),
        "flow_used": _as_str(align_diag.get("flow_used")),
        "flow_status": _as_str(align_diag.get("flow_status")),
        "pre_flow_error": _as_str(align_diag.get("pre_flow_error")),
        "post_flow_error": _as_str(align_diag.get("post_flow_error")),
        "mean_flow_magnitude": _as_str(align_diag.get("mean_flow_magnitude")),
        "color_match_algorithm": _as_str(color_match.get("algorithm")),
        "color_match_status": _as_str(color_match.get("status")),
        "color_match_confidence": _as_str(color_match.get("confidence")),
        "color_match_pre_error": _as_str(color_diag.get("pre_color_error")),
        "color_match_post_error": _as_str(color_diag.get("post_color_error")),
        "lr_exists": _bool_str(lr_path.exists()),
        "aligned_exists": _bool_str(aligned_path.exists()),
        "color_matched_exists": _bool_str(matched_path.exists()),
        "hr_exists": _bool_str(hr_path.exists()),
        "raw_to_hr_mae": raw_metrics["mae"],
        "raw_to_hr_psnr": raw_metrics["psnr"],
        "raw_to_hr_ssim": raw_metrics["ssim"],
        "raw_to_hr_dimension_match": raw_metrics["dimension_match"],
        "raw_to_hr_aspect_ratio_match": raw_metrics["aspect_ratio_match"],
        "raw_to_hr_border_mae": raw_metrics["border_mae"],
        "aligned_to_hr_mae": aligned_metrics["mae"],
        "aligned_to_hr_psnr": aligned_metrics["psnr"],
        "aligned_to_hr_ssim": aligned_metrics["ssim"],
        "aligned_to_hr_dimension_match": aligned_metrics["dimension_match"],
        "aligned_to_hr_aspect_ratio_match": aligned_metrics["aspect_ratio_match"],
        "aligned_to_hr_border_mae": aligned_metrics["border_mae"],
        "color_matched_to_hr_mae": color_matched_metrics["mae"],
        "color_matched_to_hr_psnr": color_matched_metrics["psnr"],
        "color_matched_to_hr_ssim": color_matched_metrics["ssim"],
        "color_matched_to_hr_dimension_match": color_matched_metrics["dimension_match"],
        "color_matched_to_hr_aspect_ratio_match": color_matched_metrics["aspect_ratio_match"],
        "color_matched_to_hr_border_mae": color_matched_metrics["border_mae"],
        "lr_to_hr_mae": raw_metrics["mae"],
        "lr_path": str(lr_path),
        "aligned_path": str(aligned_path),
        "color_matched_path": str(matched_path),
        "hr_path": str(hr_path),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_zh(path: Path, rows: list[dict[str, str]]) -> None:
    zh_fields = [CSV_HEADER_ZH.get(field, field) for field in CSV_FIELDS]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(zh_fields)
        for row in rows:
            writer.writerow([row.get(field, "") for field in CSV_FIELDS])


def _write_contact_sheet(path: Path, rows: list[dict[str, str]], thumbnail_size: int) -> None:
    labels = ["LR", "LR_aligned", "LR_color", "HR"]
    columns = len(labels)
    tile_w = max(thumbnail_size, 1)
    label_h = 18
    tile_h = tile_w + label_h
    sheet = Image.new("RGB", (columns * tile_w, max(len(rows), 1) * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    for row_idx, row in enumerate(rows):
        paths = [row["lr_path"], row["aligned_path"], row["color_matched_path"], row["hr_path"]]
        for col_idx, image_path in enumerate(paths):
            x = col_idx * tile_w
            y = row_idx * tile_h
            draw.text((x + 3, y + 2), labels[col_idx], fill=(0, 0, 0))
            thumb = _thumbnail(Path(image_path), tile_w)
            sheet.paste(thumb, (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def _thumbnail(path: Path, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size), (235, 235, 235))
    if not path.exists():
        return canvas
    with open_pil_image(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def _metrics_to_hr(candidate_path: Path, hr_path: Path) -> dict[str, str]:
    if not candidate_path.exists() or not hr_path.exists():
        return {
            "mae": "",
            "psnr": "",
            "ssim": "",
            "dimension_match": "",
            "aspect_ratio_match": "",
            "border_mae": "",
        }
    candidate = read_rgb_array(candidate_path)
    hr = read_rgb_array(hr_path)
    metrics = branch_metrics_to_hr(candidate, hr)
    return {
        "mae": _as_str(metrics.mae),
        "psnr": _as_str(metrics.psnr),
        "ssim": _as_str(metrics.ssim),
        "dimension_match": _bool_str(bool(metrics.dimension_match)),
        "aspect_ratio_match": _bool_str(bool(metrics.aspect_ratio_match)),
        "border_mae": _as_str(metrics.border_mae),
    }


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _bool_str(value: bool) -> str:
    return "true" if value else "false"
