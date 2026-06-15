from pathlib import Path

import csv
import numpy as np
import yaml
from PIL import Image

from livephoto2lrhr.data.io import metadata_path, output_image_path, save_rgb_array, write_yaml
from livephoto2lrhr.reports.quality import QualityReportConfig, generate_quality_report


def write_report_sample(output_dir: Path, sample: str = "sample") -> Path:
    relative_stem = Path(sample)
    lr_path = output_image_path(output_dir, "LR", relative_stem, ".png")
    aligned_path = output_image_path(output_dir, "LR_aligned", relative_stem, ".png")
    matched_path = output_image_path(output_dir, "LR_color_matched", relative_stem, ".png")
    hr_path = output_image_path(output_dir, "HR", relative_stem, ".png")
    meta_path = metadata_path(output_dir, relative_stem)
    save_rgb_array(np.full((4, 4, 3), 20, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((4, 4, 3), 30, dtype=np.uint8), aligned_path)
    save_rgb_array(np.full((4, 4, 3), 40, dtype=np.uint8), matched_path)
    save_rgb_array(np.full((8, 8, 3), 80, dtype=np.uint8), hr_path)
    write_yaml(
        meta_path,
        {
            "sample_id": sample,
            "output": {"lr": str(lr_path), "hr": str(hr_path)},
            "frame_select": {
                "algorithm": "fake_selector",
                "selected": {"frame_index": 3, "timestamp_sec": 0.1, "score": 0.9},
            },
            "align": {
                "algorithm": "identity_alignment",
                "status": "success",
                "confidence": 1.0,
                "diagnostics": {
                    "pre_alignment_error": 5.0,
                    "post_alignment_error": 1.0,
                    "flow_used": True,
                    "flow_status": "accepted",
                    "pre_flow_error": 0.5,
                    "post_flow_error": 0.1,
                    "mean_flow_magnitude": 2.0,
                },
            },
            "color_match": {
                "algorithm": "mean_std_lab",
                "status": "success",
                "confidence": 0.5,
                "diagnostics": {"pre_color_error": 10.0, "post_color_error": 2.0},
            },
            "status": {"aligned": True, "color_matched": True},
        },
    )
    return meta_path


def test_generate_quality_report_writes_csv_and_contact_sheet(tmp_path: Path):
    output_dir = tmp_path / "output"
    write_report_sample(output_dir)

    result = generate_quality_report(
        output_dir=output_dir,
        config=QualityReportConfig(
            output_folder="reports",
            aligned_folder="LR_aligned",
            color_matched_folder="LR_color_matched",
            max_preview_samples=1,
            thumbnail_size=32,
        ),
    )

    assert result.rows == 1
    assert result.csv_path == output_dir / "reports" / "quality_report.csv"
    assert result.preview_path == output_dir / "reports" / "preview_contact_sheet.jpg"
    assert result.csv_path.exists()
    assert result.preview_path.exists()
    zh_csv_path = output_dir / "reports" / "quality_report_zh.csv"
    assert zh_csv_path.exists()
    with result.csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    with zh_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        zh_reader = csv.reader(file)
        zh_header = next(zh_reader)
    assert rows[0]["sample_id"] == "sample"
    assert "样本ID" in zh_header
    assert "原始LR到HR_PSNR" in zh_header
    assert "对齐LR到HR_SSIM" in zh_header
    assert rows[0]["frame_select_score"] == "0.9"
    assert rows[0]["align_status"] == "success"
    assert rows[0]["align_post_error"] == "1.0"
    assert rows[0]["flow_used"] == "True"
    assert rows[0]["flow_status"] == "accepted"
    assert rows[0]["post_flow_error"] == "0.1"
    assert rows[0]["color_match_status"] == "success"
    assert rows[0]["color_match_post_error"] == "2.0"
    assert float(rows[0]["lr_to_hr_mae"]) > float(rows[0]["color_matched_to_hr_mae"])
    assert rows[0]["raw_to_hr_mae"] != ""
    assert rows[0]["raw_to_hr_psnr"] != ""
    assert rows[0]["raw_to_hr_ssim"] != ""
    assert rows[0]["aligned_to_hr_dimension_match"] == "false"
    assert rows[0]["aligned_to_hr_aspect_ratio_match"] == "true"
    assert rows[0]["color_matched_to_hr_border_mae"] != ""
    preview = Image.open(result.preview_path)
    assert preview.size[0] > 32
    assert preview.size[1] > 32


def test_generate_quality_report_handles_missing_optional_outputs(tmp_path: Path):
    output_dir = tmp_path / "output"
    write_report_sample(output_dir)
    (output_dir / "LR_aligned" / "sample.png").unlink()
    (output_dir / "LR_color_matched" / "sample.png").unlink()

    result = generate_quality_report(
        output_dir=output_dir,
        config=QualityReportConfig(output_folder="reports", max_preview_samples=1, thumbnail_size=32),
    )

    with result.csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["lr_exists"] == "true"
    assert rows[0]["aligned_exists"] == "false"
    assert rows[0]["color_matched_exists"] == "false"
    assert rows[0]["aligned_to_hr_mae"] == ""
    assert rows[0]["aligned_to_hr_psnr"] == ""
    assert rows[0]["aligned_to_hr_ssim"] == ""


def test_generate_quality_report_uses_configured_stage_folders(tmp_path: Path):
    output_dir = tmp_path / "output"
    write_report_sample(output_dir)
    flow_path = output_image_path(output_dir, "LR_aligned_flow", Path("sample"), ".png")
    save_rgb_array(np.full((4, 4, 3), 60, dtype=np.uint8), flow_path)

    result = generate_quality_report(
        output_dir=output_dir,
        config=QualityReportConfig(output_folder="reports", aligned_folder="LR_aligned_flow", max_preview_samples=0),
    )

    with result.csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["aligned_exists"] == "true"
    assert rows[0]["aligned_path"] == str(flow_path)
