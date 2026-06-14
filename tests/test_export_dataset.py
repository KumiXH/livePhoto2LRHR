from pathlib import Path
import csv

import numpy as np
from PIL import Image

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
        "lr_to_hr_mae",
        "aligned_to_hr_mae",
        "color_matched_to_hr_mae",
        "lr_path",
        "aligned_path",
        "color_matched_path",
        "hr_path",
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
        "lr_to_hr_mae": "50.0",
        "aligned_to_hr_mae": "12.5",
        "color_matched_to_hr_mae": "",
        "lr_path": str(lr_path),
        "aligned_path": str(aligned_path),
        "color_matched_path": str(matched_path),
        "hr_path": str(hr_path),
    }
    row.update(overrides)
    return row


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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
