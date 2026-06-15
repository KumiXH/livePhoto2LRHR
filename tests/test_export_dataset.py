from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image
import pytest
import yaml

from livephoto2lrhr.data.io import save_rgb_array
from livephoto2lrhr.export.dataset import ExportDatasetConfig, export_dataset


def write_quality_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "align_status",
        "align_confidence",
        "flow_status",
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
        "lr_to_hr_mae",
        "aligned_to_hr_mae",
        "aligned_to_hr_psnr",
        "aligned_to_hr_ssim",
        "aligned_to_hr_dimension_match",
        "aligned_to_hr_aspect_ratio_match",
        "aligned_to_hr_border_mae",
        "color_matched_to_hr_mae",
        "color_matched_to_hr_psnr",
        "color_matched_to_hr_ssim",
        "color_matched_to_hr_dimension_match",
        "color_matched_to_hr_aspect_ratio_match",
        "color_matched_to_hr_border_mae",
        "lr_path",
        "aligned_path",
        "color_matched_path",
        "hr_path",
        "mean_flow_magnitude",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def base_report_row(output_dir: Path, sample: str, **overrides: str) -> dict[str, str]:
    relative = Path(sample)
    lr_path = output_dir / "LR" / f"{sample}.png"
    aligned_path = output_dir / "LR_aligned_flow" / f"{sample}.png"
    matched_path = output_dir / "LR_color_matched" / f"{sample}.png"
    hr_path = output_dir / "HR" / f"{sample}.png"
    row = {
        "sample_id": relative.as_posix(),
        "align_status": "success",
        "align_confidence": "0.9",
        "flow_status": "accepted",
        "lr_exists": "true",
        "aligned_exists": "true",
        "color_matched_exists": "false",
        "hr_exists": "true",
        "raw_to_hr_mae": "50.0",
        "raw_to_hr_psnr": "12.0",
        "raw_to_hr_ssim": "0.20",
        "raw_to_hr_dimension_match": "false",
        "raw_to_hr_aspect_ratio_match": "true",
        "raw_to_hr_border_mae": "40.0",
        "lr_to_hr_mae": "50.0",
        "aligned_to_hr_mae": "12.5",
        "aligned_to_hr_psnr": "22.0",
        "aligned_to_hr_ssim": "0.80",
        "aligned_to_hr_dimension_match": "true",
        "aligned_to_hr_aspect_ratio_match": "true",
        "aligned_to_hr_border_mae": "5.0",
        "color_matched_to_hr_mae": "",
        "color_matched_to_hr_psnr": "",
        "color_matched_to_hr_ssim": "",
        "color_matched_to_hr_dimension_match": "",
        "color_matched_to_hr_aspect_ratio_match": "",
        "color_matched_to_hr_border_mae": "",
        "lr_path": str(lr_path),
        "aligned_path": str(aligned_path),
        "color_matched_path": str(matched_path),
        "hr_path": str(hr_path),
        "mean_flow_magnitude": "2.0",
    }
    row.update(overrides)
    return row


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_metadata(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_export_dataset_copies_only_accepted_rows_and_preserves_structure(tmp_path: Path):
    output_dir = tmp_path / "output"
    accepted_raw_lr = output_dir / "LR" / "nested" / "good.png"
    accepted_lr = output_dir / "LR_aligned_flow" / "nested" / "good.png"
    accepted_hr = output_dir / "HR" / "nested" / "good.png"
    rejected_raw_lr = output_dir / "LR" / "nested" / "bad.png"
    rejected_lr = output_dir / "LR_aligned_flow" / "nested" / "bad.png"
    rejected_hr = output_dir / "HR" / "nested" / "bad.png"
    save_rgb_array(np.full((2, 3, 3), 15, dtype=np.uint8), accepted_raw_lr)
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), accepted_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), accepted_hr)
    save_rgb_array(np.full((2, 3, 3), 18, dtype=np.uint8), rejected_raw_lr)
    save_rgb_array(np.full((3, 4, 3), 35, dtype=np.uint8), rejected_lr)
    save_rgb_array(np.full((6, 8, 3), 210, dtype=np.uint8), rejected_hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(output_dir, "nested/good"),
            base_report_row(output_dir, "nested/bad", align_confidence="0.1"),
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            min_align_confidence=0.5,
            require_align_status="success",
            require_flow_status="accepted",
            max_source_to_hr_mae=20.0,
            overwrite=False,
        ),
    )

    final_lr = output_dir / "final" / "LR" / "nested" / "good.png"
    final_hr = output_dir / "final" / "HR" / "nested" / "good.png"
    assert result.accepted == 1
    assert result.rejected == 1
    assert final_lr.exists()
    assert final_hr.exists()
    first_row = read_manifest(result.manifest_path)[0]
    assert first_row["lr_source"] == "raw"
    assert first_row["final_source_lr_path"] == str(accepted_raw_lr)
    assert first_row["gate_source_lr_path"] == str(accepted_lr)
    assert not (output_dir / "final" / "LR" / "nested" / "bad.png").exists()
    assert first_row["accepted"] == "true"
    assert read_manifest(result.manifest_path)[1]["reason"] == "align_confidence_below_min"


def test_export_dataset_records_rejection_reasons(tmp_path: Path):
    output_dir = tmp_path / "output"
    ok_raw_lr = output_dir / "LR" / "ok.png"
    ok_lr = output_dir / "LR_aligned_flow" / "ok.png"
    ok_hr = output_dir / "HR" / "ok.png"
    save_rgb_array(np.full((2, 3, 3), 15, dtype=np.uint8), ok_raw_lr)
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), ok_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), ok_hr)
    missing_hr_lr = output_dir / "LR_aligned_flow" / "missing_hr.png"
    missing_hr_raw_lr = output_dir / "LR" / "missing_hr.png"
    save_rgb_array(np.full((2, 3, 3), 15, dtype=np.uint8), missing_hr_raw_lr)
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), missing_hr_lr)
    for sample in ("failed", "flow_rejected", "high_mae"):
        save_rgb_array(np.full((2, 3, 3), 15, dtype=np.uint8), output_dir / "LR" / f"{sample}.png")
        save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), output_dir / "LR_aligned_flow" / f"{sample}.png")
        save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), output_dir / "HR" / f"{sample}.png")
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), output_dir / "LR_aligned_flow" / "missing_final_lr.png")
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), output_dir / "HR" / "missing_final_lr.png")
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(output_dir, "ok"),
            base_report_row(output_dir, "failed", align_status="failed"),
            base_report_row(output_dir, "flow_rejected", flow_status="rejected"),
            base_report_row(output_dir, "high_mae", aligned_to_hr_mae="99.9"),
            base_report_row(output_dir, "missing_lr"),
            base_report_row(output_dir, "missing_final_lr"),
            base_report_row(output_dir, "missing_hr", hr_path=str(output_dir / "HR" / "missing_hr.png")),
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            min_align_confidence=0.5,
            require_align_status="success",
            require_flow_status="accepted",
            max_source_to_hr_mae=20.0,
            overwrite=False,
        ),
    )

    reasons = {row["sample_id"]: row["reason"] for row in read_manifest(result.manifest_path)}
    assert result.accepted == 1
    assert result.rejected == 6
    assert reasons["ok"] == "accepted"
    assert reasons["failed"] == "align_status_mismatch"
    assert reasons["flow_rejected"] == "flow_status_mismatch"
    assert reasons["high_mae"] == "gate_source_to_hr_mae_above_max"
    assert reasons["missing_lr"] == "missing_gate_lr_source"
    assert reasons["missing_final_lr"] == "missing_final_lr_source"
    assert reasons["missing_hr"] == "missing_hr"


def test_export_dataset_rejects_existing_destination_without_overwrite(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr_path = output_dir / "LR" / "sample.png"
    lr_path = output_dir / "LR_aligned_flow" / "sample.png"
    hr_path = output_dir / "HR" / "sample.png"
    final_lr = output_dir / "final" / "LR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 15, dtype=np.uint8), raw_lr_path)
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), lr_path)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), hr_path)
    save_rgb_array(np.full((3, 4, 3), 99, dtype=np.uint8), final_lr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(report_path, [base_report_row(output_dir, "sample")])

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            overwrite=False,
        ),
    )

    rows = read_manifest(result.manifest_path)
    assert result.accepted == 0
    assert result.rejected == 1
    assert rows[0]["reason"] == "destination_exists"


def test_export_dataset_rejects_blank_source_paths(tmp_path: Path):
    output_dir = tmp_path / "output"
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    save_rgb_array(np.full((2, 3, 3), 25, dtype=np.uint8), output_dir / "LR" / "blank_hr.png")
    save_rgb_array(np.full((3, 4, 3), 25, dtype=np.uint8), output_dir / "LR_aligned_flow" / "blank_hr.png")
    write_quality_report(
        report_path,
        [
            base_report_row(output_dir, "blank_lr", aligned_path=""),
            base_report_row(output_dir, "blank_hr", hr_path=""),
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
        ),
    )

    rows = read_manifest(result.manifest_path)
    assert result.accepted == 0
    assert result.rejected == 2
    assert rows[0]["reason"] == "missing_gate_lr_source"
    assert rows[1]["reason"] == "missing_hr"


def test_export_dataset_can_gate_on_aligned_but_export_raw_lr(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 11, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((6, 8, 3), 22, dtype=np.uint8), aligned_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(report_path, [base_report_row(output_dir, "sample", aligned_to_hr_mae="10.0", lr_to_hr_mae="80.0")])

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            max_source_to_hr_mae=20.0,
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    rows = read_manifest(result.manifest_path)
    assert result.accepted == 1
    assert rows[0]["lr_source"] == "raw"
    assert rows[0]["gate_lr_source"] == "aligned"
    assert rows[0]["final_source_lr_path"] == str(raw_lr)
    assert rows[0]["gate_source_lr_path"] == str(aligned_lr)
    assert rows[0]["gate_source_to_hr_mae"] == "10.0"
    assert final_lr.exists()


def test_export_dataset_can_export_aligned_content_at_raw_resolution(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 10, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((6, 8, 3), 80, dtype=np.uint8), aligned_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(report_path, [base_report_row(output_dir, "sample", aligned_to_hr_mae="10.0")])

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="aligned",
            gate_lr_source="aligned",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    rows = read_manifest(result.manifest_path)
    exported = np.asarray(Image.open(final_lr))
    assert result.accepted == 1
    assert rows[0]["lr_source"] == "aligned"
    assert rows[0]["final_lr_resize_mode"] == "match_raw"
    assert exported.shape[:2] == (2, 3)
    assert np.all(exported == 80)


def test_export_dataset_match_raw_replays_alignment_on_raw_grid(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    raw_rgb[:, :2] = 20
    raw_rgb[:, 2:] = 220
    hr_rgb = cv2.resize(raw_rgb, (8, 8), interpolation=cv2.INTER_CUBIC)
    hr_aligned = cv2.warpAffine(
        hr_rgb,
        np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (8, 8),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(hr_aligned, aligned_lr)
    save_rgb_array(hr_rgb, hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "align": {
                "transforms": [
                    {
                        "type": "translation",
                        "coordinate_system": "lr_to_hr",
                        "dx": 2.0,
                        "dy": 0.0,
                        "matrix": [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]],
                    }
                ]
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(report_path, [base_report_row(output_dir, "sample", aligned_to_hr_mae="10.0")])

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="aligned",
            gate_lr_source="aligned",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = cv2.warpAffine(
        raw_rgb,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (4, 4),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    naive = cv2.resize(hr_aligned, (4, 4), interpolation=cv2.INTER_AREA)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)
    assert not np.array_equal(exported, naive)


def test_export_dataset_can_export_color_matched_content_at_raw_resolution(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 10, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((6, 8, 3), 60, dtype=np.uint8), aligned_lr)
    save_rgb_array(np.full((6, 8, 3), 120, dtype=np.uint8), matched_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    rows = read_manifest(result.manifest_path)
    exported = np.asarray(Image.open(final_lr))
    assert result.accepted == 1
    assert rows[0]["lr_source"] == "color_matched"
    assert rows[0]["final_lr_resize_mode"] == "match_raw"
    assert exported.shape[:2] == (2, 3)
    assert np.all(exported == 120)


def test_export_dataset_match_raw_replays_color_transfer_on_raw_grid(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((3, 3, 3), 40, dtype=np.uint8)
    raw_rgb[..., 1] = 80
    raw_rgb[..., 2] = 120
    matched_rgb = np.full((6, 6, 3), 0, dtype=np.uint8)
    matched_rgb[..., 0] = 90
    matched_rgb[..., 1] = 120
    matched_rgb[..., 2] = 150
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((6, 6, 3), 180, dtype=np.uint8), hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [
                    {
                        "type": "mean_std_color_transfer",
                        "color_space": "rgb",
                        "source_mean": [40.0, 80.0, 120.0],
                        "source_std": [1.0, 1.0, 1.0],
                        "target_mean": [90.0, 120.0, 150.0],
                        "target_std": [1.0, 1.0, 1.0],
                    }
                ],
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = np.full((3, 3, 3), 0, dtype=np.uint8)
    expected[..., 0] = 90
    expected[..., 1] = 120
    expected[..., 2] = 150
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


def test_export_dataset_match_raw_replays_histogram_transfer_when_metadata_records_reference(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((3, 3, 3), 30, dtype=np.uint8)
    raw_rgb[..., 1] = 60
    raw_rgb[..., 2] = 90
    matched_rgb = np.full((6, 6, 3), 0, dtype=np.uint8)
    matched_rgb[..., 0] = 120
    matched_rgb[..., 1] = 140
    matched_rgb[..., 2] = 160
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((6, 6, 3), 180, dtype=np.uint8), hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [
                    {
                        "type": "histogram_color_transfer",
                        "color_space": "rgb",
                        "bins": 256,
                    }
                ],
                "diagnostics": {
                    "replay_reference_path": str(matched_lr),
                },
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = cv2.resize(matched_rgb, (3, 3), interpolation=cv2.INTER_AREA)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


def test_export_dataset_match_raw_replays_retinex_transfer_from_reference(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((4, 4, 3), (40, 60, 80), dtype=np.uint8)
    matched_rgb = np.full((8, 8, 3), (150, 165, 180), dtype=np.uint8)
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((8, 8, 3), 190, dtype=np.uint8), hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [
                    {
                        "type": "retinex_color_transfer",
                        "sigma": 15.0,
                        "eps": 1.0e-3,
                    }
                ],
                "diagnostics": {
                    "replay_reference_path": str(matched_lr),
                },
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = cv2.resize(matched_rgb, (4, 4), interpolation=cv2.INTER_AREA)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


def test_export_dataset_match_raw_replays_masked_transfer_from_reference(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((4, 4, 3), (60, 70, 80), dtype=np.uint8)
    matched_rgb = np.full((8, 8, 3), (70, 75, 80), dtype=np.uint8)
    matched_rgb[2:6, 2:6] = (180, 90, 150)
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((8, 8, 3), 190, dtype=np.uint8), hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [
                    {
                        "type": "masked_mean_std_color_transfer",
                        "color_space": "lab",
                    }
                ],
                "diagnostics": {
                    "replay_reference_path": str(matched_lr),
                },
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = cv2.resize(matched_rgb, (4, 4), interpolation=cv2.INTER_AREA)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


def test_export_dataset_match_raw_replays_adaptive_3d_lut_transform(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((3, 3, 3), (40, 70, 100), dtype=np.uint8)
    matched_rgb = np.full((6, 6, 3), (120, 150, 180), dtype=np.uint8)
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((6, 6, 3), 190, dtype=np.uint8), hr)
    lut = np.zeros((2, 2, 2, 3), dtype=np.float32)
    lut[..., 0] = 120.0 / 255.0
    lut[..., 1] = 150.0 / 255.0
    lut[..., 2] = 180.0 / 255.0
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [
                    {
                        "type": "adaptive_3d_lut_color_transfer",
                        "color_space": "rgb",
                        "grid_size": 2,
                        "lut": lut.tolist(),
                    }
                ],
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = np.full((3, 3, 3), (120, 150, 180), dtype=np.uint8)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


@pytest.mark.parametrize(
    ("transform_type", "output_folder"),
    [
        ("low_frequency_joint_appearance_transfer", "LR_color_lowfreq"),
        ("learned_retinex_color_transfer", "LR_color_learned_retinex"),
        ("mask_aware_harmonization_transfer", "LR_color_harmonization"),
        ("diffusion_harmonization_transfer", "LR_color_diffusion"),
    ],
)
def test_export_dataset_match_raw_replays_reference_based_new_color_backends(
    tmp_path: Path,
    transform_type: str,
    output_folder: str,
):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / output_folder / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    metadata_path = output_dir / "metadata" / "sample.yaml"
    raw_rgb = np.full((4, 4, 3), (50, 60, 70), dtype=np.uint8)
    matched_rgb = np.full((8, 8, 3), (145, 155, 165), dtype=np.uint8)
    save_rgb_array(raw_rgb, raw_lr)
    save_rgb_array(matched_rgb, matched_lr)
    save_rgb_array(np.full((8, 8, 3), 190, dtype=np.uint8), hr)
    write_metadata(
        metadata_path,
        {
            "sample_id": "sample",
            "color_match": {
                "output": {"lr_color_matched": str(matched_lr)},
                "transforms": [{"type": transform_type}],
                "diagnostics": {
                    "replay_reference_path": str(matched_lr),
                },
            },
        },
    )
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_path=str(matched_lr),
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="match_raw",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    exported = np.asarray(Image.open(final_lr))
    expected = cv2.resize(matched_rgb, (4, 4), interpolation=cv2.INTER_AREA)
    assert result.accepted == 1
    assert np.array_equal(exported, expected)


def test_export_dataset_can_export_color_matched_content_at_half_hr_resolution(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    matched_lr = output_dir / "LR_color_matched" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 10, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((6, 8, 3), 120, dtype=np.uint8), matched_lr)
    save_rgb_array(np.full((6, 8, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                color_matched_exists="true",
                color_matched_to_hr_mae="9.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="color_matched",
            gate_lr_source="color_matched",
            max_source_to_hr_mae=20.0,
            final_lr_resize_mode="0.5",
        ),
    )

    final_lr = output_dir / "final" / "LR" / "sample.png"
    rows = read_manifest(result.manifest_path)
    exported = np.asarray(Image.open(final_lr))
    assert result.accepted == 1
    assert rows[0]["lr_source"] == "color_matched"
    assert rows[0]["final_lr_resize_mode"] == "0.5"
    assert exported.shape[:2] == (3, 4)
    assert np.all(exported == 120)


def test_export_dataset_rejects_low_psnr(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 11, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((2, 3, 3), 22, dtype=np.uint8), aligned_lr)
    save_rgb_array(np.full((4, 6, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                aligned_to_hr_psnr="12.0",
                aligned_to_hr_ssim="0.80",
                aligned_to_hr_dimension_match="true",
                aligned_to_hr_aspect_ratio_match="true",
                aligned_to_hr_border_mae="4.0",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            min_source_to_hr_psnr=18.0,
        ),
    )

    rows = read_manifest(result.manifest_path)
    assert rows[0]["reason"] == "gate_source_to_hr_psnr_below_min"


def test_export_dataset_rejects_dimension_mismatch_when_required(tmp_path: Path):
    output_dir = tmp_path / "output"
    raw_lr = output_dir / "LR" / "sample.png"
    aligned_lr = output_dir / "LR_aligned_flow" / "sample.png"
    hr = output_dir / "HR" / "sample.png"
    save_rgb_array(np.full((2, 3, 3), 11, dtype=np.uint8), raw_lr)
    save_rgb_array(np.full((2, 3, 3), 22, dtype=np.uint8), aligned_lr)
    save_rgb_array(np.full((4, 6, 3), 200, dtype=np.uint8), hr)
    report_path = output_dir / "reports_flow" / "quality_report.csv"
    write_quality_report(
        report_path,
        [
            base_report_row(
                output_dir,
                "sample",
                aligned_to_hr_dimension_match="false",
                aligned_to_hr_aspect_ratio_match="true",
            )
        ],
    )

    result = export_dataset(
        output_dir,
        ExportDatasetConfig(
            input_report="reports_flow/quality_report.csv",
            output_folder="final",
            final_lr_source="raw",
            gate_lr_source="aligned",
            require_source_to_hr_dimension_match=True,
        ),
    )

    rows = read_manifest(result.manifest_path)
    assert rows[0]["reason"] == "gate_source_to_hr_dimension_mismatch"
