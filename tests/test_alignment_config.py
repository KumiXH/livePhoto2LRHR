from pathlib import Path

import yaml
import pytest

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


def test_align_config_defaults_to_disabled_identity(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, base_config(input_dir, output_dir))

    config = load_config(config_path)

    assert config.align.enabled is False
    assert config.align.algorithm == "identity_alignment"
    assert config.align.device == "auto"
    assert config.align.output_folder == "LR_aligned"
    assert config.align.confidence_threshold == 0.3
    assert config.align.fallback_algorithm == "identity_alignment"
    assert config.align.on_failure == "keep_original"
    assert config.align.coarse_algorithm == "phase_correlation_translation"
    assert config.align.artifacts.save_debug_overlay is False
    assert config.align.artifacts.save_flow is False
    assert config.align.artifacts.save_masks is False
    assert config.align.phase_correlation.resize_short_side == 512
    assert config.align.ecc.motion_model == "affine"
    assert config.align.ecc.number_of_iterations == 100
    assert config.align.ecc.termination_eps == 1.0e-5
    assert config.align.ecc.gaussian_filter_size == 5
    assert config.align.optical_flow.enabled is False
    assert config.align.optical_flow.algorithm == "dis"


def test_align_config_loads_enabled_yaml_values(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["pipeline"]["stages"] = ["frame_select", "align"]
    data["align"] = {
        "enabled": True,
        "algorithm": "phase_correlation_translation",
        "device": "cpu",
        "output_folder": "custom_aligned",
        "confidence_threshold": 0.75,
        "fallback_algorithm": "identity_alignment",
        "on_failure": "skip",
        "coarse_algorithm": "ecc_alignment",
        "artifacts": {
            "save_debug_overlay": True,
            "save_flow": True,
            "save_masks": True,
        },
        "phase_correlation": {"resize_short_side": 256},
        "ecc": {
            "motion_model": "translation",
            "number_of_iterations": 25,
            "termination_eps": 0.001,
            "gaussian_filter_size": 3,
        },
        "optical_flow": {"enabled": True, "algorithm": "farneback"},
    }
    write_yaml(config_path, data)

    config = load_config(config_path)

    assert config.pipeline.stages == ("frame_select", "align")
    assert config.align.enabled is True
    assert config.align.algorithm == "phase_correlation_translation"
    assert config.align.device == "cpu"
    assert config.align.output_folder == "custom_aligned"
    assert config.align.confidence_threshold == 0.75
    assert config.align.on_failure == "skip"
    assert config.align.coarse_algorithm == "ecc_alignment"
    assert config.align.artifacts.save_debug_overlay is True
    assert config.align.artifacts.save_flow is True
    assert config.align.artifacts.save_masks is True
    assert config.align.phase_correlation.resize_short_side == 256
    assert config.align.ecc.motion_model == "translation"
    assert config.align.ecc.number_of_iterations == 25
    assert config.align.ecc.termination_eps == 0.001
    assert config.align.ecc.gaussian_filter_size == 3
    assert config.align.optical_flow.enabled is True
    assert config.align.optical_flow.algorithm == "farneback"


@pytest.mark.parametrize("output_folder", ["HR", "LR", "metadata", "../escape", "/tmp/out", ""])
def test_align_config_rejects_unsafe_output_folder(tmp_path: Path, output_folder: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["align"] = {"output_folder": output_folder}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="align.output_folder"):
        load_config(config_path)
