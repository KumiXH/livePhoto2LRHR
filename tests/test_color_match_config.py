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


def test_color_match_config_defaults_to_disabled_identity(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, base_config(input_dir, output_dir))

    config = load_config(config_path)

    assert config.color_match.enabled is False
    assert config.color_match.algorithm == "identity_color_match"
    assert config.color_match.device == "auto"
    assert config.color_match.input_folder == "auto"
    assert config.color_match.output_folder == "LR_color_matched"
    assert config.color_match.confidence_threshold == 0.0
    assert config.color_match.on_failure == "keep_original"
    assert config.color_match.mean_std.color_space == "lab"
    assert config.color_match.mean_std.eps == 1.0e-6


def test_color_match_config_loads_enabled_yaml_values(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["pipeline"]["stages"] = ["frame_select", "color_match"]
    data["color_match"] = {
        "enabled": True,
        "algorithm": "mean_std_lab",
        "device": "cpu",
        "input_folder": "LR",
        "output_folder": "LR_color",
        "confidence_threshold": 0.25,
        "on_failure": "skip",
        "mean_std": {"color_space": "rgb", "eps": 0.001},
    }
    write_yaml(config_path, data)

    config = load_config(config_path)

    assert config.pipeline.stages == ("frame_select", "color_match")
    assert config.color_match.enabled is True
    assert config.color_match.algorithm == "mean_std_lab"
    assert config.color_match.device == "cpu"
    assert config.color_match.input_folder == "LR"
    assert config.color_match.output_folder == "LR_color"
    assert config.color_match.confidence_threshold == 0.25
    assert config.color_match.on_failure == "skip"
    assert config.color_match.mean_std.color_space == "rgb"
    assert config.color_match.mean_std.eps == 0.001


@pytest.mark.parametrize("output_folder", ["HR", "LR", "metadata", "artifacts", "../escape", "/tmp/out", ""])
def test_color_match_config_rejects_unsafe_output_folder(tmp_path: Path, output_folder: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["color_match"] = {"output_folder": output_folder}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="color_match.output_folder"):
        load_config(config_path)


@pytest.mark.parametrize("input_folder", ["HR", "metadata", "artifacts", "../escape", "/tmp/out", ""])
def test_color_match_config_rejects_unsafe_input_folder(tmp_path: Path, input_folder: str):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    data = base_config(input_dir, output_dir)
    data["color_match"] = {"input_folder": input_folder}
    write_yaml(config_path, data)

    with pytest.raises(ValueError, match="color_match.input_folder"):
        load_config(config_path)
