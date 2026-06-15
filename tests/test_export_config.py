from pathlib import Path

import pytest
import yaml

from livephoto2lrhr.config import load_config


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def base_config(input_dir: Path, output_dir: Path) -> dict:
    return {
        "data": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
        "pipeline": {"stages": ["frame_select"]},
        "frame_select": {"algorithm": "fake_selector"},
    }


def test_export_config_defaults_to_disabled_final_dataset(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, base_config(input_dir, output_dir))

    config = load_config(config_path)

    assert config.export.enabled is False
    assert config.export.input_report == "reports/quality_report.csv"
    assert config.export.output_folder == "final"
    assert config.export.final_lr_source == "raw"
    assert config.export.gate_lr_source == "aligned"
    assert config.export.final_lr_resize_mode == "0.5"
    assert config.export.min_align_confidence == 0.0
    assert config.export.require_align_status == "success"
    assert config.export.require_flow_status is None
    assert config.export.max_source_to_hr_mae is None
    assert config.export.min_source_to_hr_psnr is None
    assert config.export.min_source_to_hr_ssim is None
    assert config.export.require_source_to_hr_dimension_match is False
    assert config.export.require_source_to_hr_aspect_ratio_match is False
    assert config.export.max_source_to_hr_border_mae is None
    assert config.export.max_mean_flow_magnitude is None


def test_export_config_loads_quality_gate_values(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {
        "enabled": True,
        "input_report": "reports_flow/quality_report.csv",
        "output_folder": "final_flow",
        "final_lr_source": "raw",
        "gate_lr_source": "aligned",
        "final_lr_resize_mode": "match_raw",
        "min_align_confidence": 0.7,
        "require_align_status": "success",
        "require_flow_status": "accepted",
        "max_source_to_hr_mae": 25.0,
        "min_source_to_hr_psnr": 18.0,
        "min_source_to_hr_ssim": 0.45,
        "require_source_to_hr_dimension_match": True,
        "require_source_to_hr_aspect_ratio_match": True,
        "max_source_to_hr_border_mae": 12.0,
        "max_mean_flow_magnitude": 20.0,
    }
    write_yaml(config_path, data)

    config = load_config(config_path)

    assert config.export.enabled is True
    assert config.export.input_report == "reports_flow/quality_report.csv"
    assert config.export.output_folder == "final_flow"
    assert config.export.final_lr_source == "raw"
    assert config.export.gate_lr_source == "aligned"
    assert config.export.final_lr_resize_mode == "match_raw"
    assert config.export.min_align_confidence == 0.7
    assert config.export.require_align_status == "success"
    assert config.export.require_flow_status == "accepted"
    assert config.export.max_source_to_hr_mae == 25.0
    assert config.export.min_source_to_hr_psnr == 18.0
    assert config.export.min_source_to_hr_ssim == 0.45
    assert config.export.require_source_to_hr_dimension_match is True
    assert config.export.require_source_to_hr_aspect_ratio_match is True
    assert config.export.max_source_to_hr_border_mae == 12.0
    assert config.export.max_mean_flow_magnitude == 20.0


@pytest.mark.parametrize("output_folder", ["HR", "LR", "metadata", "reports", "../escape", "/tmp/out", ""])
def test_export_config_rejects_unsafe_output_folder(tmp_path: Path, output_folder: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"output_folder": output_folder}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="export.output_folder"):
        load_config(config_path)


@pytest.mark.parametrize("lr_source", ["raw", "aligned", "color_matched"])
def test_export_config_accepts_supported_lr_sources(tmp_path: Path, lr_source: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"final_lr_source": lr_source, "gate_lr_source": lr_source}
    write_yaml(config_path, data)

    config = load_config(config_path)

    assert config.export.final_lr_source == lr_source
    assert config.export.gate_lr_source == lr_source


@pytest.mark.parametrize("resize_mode", ["copy", "match_raw", "raw", "1.0", "0.75", "0.5"])
def test_export_config_accepts_supported_resize_modes(tmp_path: Path, resize_mode: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"final_lr_resize_mode": resize_mode}
    write_yaml(config_path, data)

    config = load_config(config_path)

    assert config.export.final_lr_resize_mode == resize_mode


def test_export_config_rejects_unknown_final_lr_source(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"final_lr_source": "unknown"}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="export.final_lr_source"):
        load_config(config_path)


def test_export_config_rejects_unknown_gate_lr_source(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"gate_lr_source": "unknown"}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="export.gate_lr_source"):
        load_config(config_path)


def test_export_config_rejects_unknown_final_lr_resize_mode(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["export"] = {"final_lr_resize_mode": "unknown"}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="export.final_lr_resize_mode"):
        load_config(config_path)
