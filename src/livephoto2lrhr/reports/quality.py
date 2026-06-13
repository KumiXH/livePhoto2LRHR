from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from livephoto2lrhr.data.io import output_image_path, read_rgb_array


@dataclass(frozen=True)
class QualityReportConfig:
    output_folder: str = "reports"
    max_preview_samples: int = 24
    thumbnail_size: int = 160


@dataclass(frozen=True)
class QualityReportResult:
    rows: int
    csv_path: Path
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
    "color_match_algorithm",
    "color_match_status",
    "color_match_confidence",
    "color_match_pre_error",
    "color_match_post_error",
    "lr_exists",
    "aligned_exists",
    "color_matched_exists",
    "hr_exists",
    "lr_to_hr_mae",
    "aligned_to_hr_mae",
    "color_matched_to_hr_mae",
    "lr_path",
    "aligned_path",
    "color_matched_path",
    "hr_path",
]


def generate_quality_report(output_dir: Path, config: QualityReportConfig) -> QualityReportResult:
    report_dir = output_dir / config.output_folder
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [_row_from_metadata(output_dir, path) for path in sorted((output_dir / "metadata").rglob("*.yaml"))]
    csv_path = report_dir / "quality_report.csv"
    _write_csv(csv_path, rows)
    preview_path = None
    if rows and config.max_preview_samples > 0:
        preview_path = report_dir / "preview_contact_sheet.jpg"
        _write_contact_sheet(preview_path, rows[: config.max_preview_samples], config.thumbnail_size)
    return QualityReportResult(rows=len(rows), csv_path=csv_path, preview_path=preview_path)


def _row_from_metadata(output_dir: Path, meta_path: Path) -> dict[str, str]:
    metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    relative_stem = meta_path.relative_to(output_dir / "metadata").with_suffix("")
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    aligned_path = output_image_path(output_dir, "LR_aligned", relative_stem, ".png")
    matched_path = output_image_path(output_dir, "LR_color_matched", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    selected = _nested(metadata, "frame_select", "selected") or {}
    align = metadata.get("align") or {}
    color_match = metadata.get("color_match") or {}
    align_diag = align.get("diagnostics") or {}
    color_diag = color_match.get("diagnostics") or {}
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
        "color_match_algorithm": _as_str(color_match.get("algorithm")),
        "color_match_status": _as_str(color_match.get("status")),
        "color_match_confidence": _as_str(color_match.get("confidence")),
        "color_match_pre_error": _as_str(color_diag.get("pre_color_error")),
        "color_match_post_error": _as_str(color_diag.get("post_color_error")),
        "lr_exists": _bool_str(lr_path.exists()),
        "aligned_exists": _bool_str(aligned_path.exists()),
        "color_matched_exists": _bool_str(matched_path.exists()),
        "hr_exists": _bool_str(hr_path.exists()),
        "lr_to_hr_mae": _as_str(_mae_to_hr(lr_path, hr_path)),
        "aligned_to_hr_mae": _as_str(_mae_to_hr(aligned_path, hr_path)),
        "color_matched_to_hr_mae": _as_str(_mae_to_hr(matched_path, hr_path)),
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
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size))
        x = (size - image.width) // 2
        y = (size - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def _mae_to_hr(candidate_path: Path, hr_path: Path) -> float | None:
    if not candidate_path.exists() or not hr_path.exists():
        return None
    candidate = read_rgb_array(candidate_path)
    hr = read_rgb_array(hr_path)
    if candidate.shape[:2] != hr.shape[:2]:
        with Image.fromarray(hr) as hr_image:
            resized = hr_image.resize((candidate.shape[1], candidate.shape[0]), Image.Resampling.BICUBIC)
            hr = np.asarray(resized)
    return float(np.mean(np.abs(candidate.astype(np.float32) - hr.astype(np.float32))))


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
