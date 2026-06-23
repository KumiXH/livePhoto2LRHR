from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from livephoto2lrhr.data.io import write_yaml


CSV_FIELDS = [
    "sample_id",
    "source_image",
    "source_video",
    "frame_select_status",
    "frame_select_message",
    "frame_select_started_at",
    "frame_select_finished_at",
    "frame_select_duration_sec",
    "frame_select_error_traceback",
    "align_status",
    "align_message",
    "align_started_at",
    "align_finished_at",
    "align_duration_sec",
    "align_error_traceback",
    "color_match_status",
    "color_match_message",
    "color_match_started_at",
    "color_match_finished_at",
    "color_match_duration_sec",
    "color_match_error_traceback",
    "last_status",
]


def write_sample_status_files(output_dir: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    yaml_path = output_dir / "sample_status.yaml"
    csv_path = output_dir / "sample_status.csv"
    write_yaml(yaml_path, {"samples": records})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: _to_csv_str(record.get(field, "")) for field in CSV_FIELDS})
    return yaml_path, csv_path


def build_sample_status_records(
    *,
    pair_dicts: dict[str, dict[str, str]],
    stage_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for sample_id, pair_info in pair_dicts.items():
        records[sample_id] = {
            "sample_id": sample_id,
            "source_image": pair_info.get("source_image", ""),
            "source_video": pair_info.get("source_video", ""),
            "frame_select_status": "",
            "frame_select_message": "",
            "frame_select_started_at": "",
            "frame_select_finished_at": "",
            "frame_select_duration_sec": "",
            "frame_select_error_traceback": "",
            "align_status": "",
            "align_message": "",
            "align_started_at": "",
            "align_finished_at": "",
            "align_duration_sec": "",
            "align_error_traceback": "",
            "color_match_status": "",
            "color_match_message": "",
            "color_match_started_at": "",
            "color_match_finished_at": "",
            "color_match_duration_sec": "",
            "color_match_error_traceback": "",
            "last_status": "",
        }

    for event in stage_events:
        sample_id = str(event.get("sample_id", ""))
        stage = str(event.get("stage", ""))
        if sample_id not in records or stage not in {"frame_select", "align", "color_match"}:
            continue
        prefix = stage
        record = records[sample_id]
        record[f"{prefix}_status"] = str(event.get("status", ""))
        record[f"{prefix}_message"] = str(event.get("message", ""))
        record[f"{prefix}_started_at"] = str(event.get("started_at", ""))
        record[f"{prefix}_finished_at"] = str(event.get("finished_at", ""))
        record[f"{prefix}_duration_sec"] = event.get("duration_sec", "")
        record[f"{prefix}_error_traceback"] = str(event.get("error_traceback", ""))
        record["last_status"] = str(event.get("status", ""))

    return [records[sample_id] for sample_id in sorted(records)]


def _to_csv_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
