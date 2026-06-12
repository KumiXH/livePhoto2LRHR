from pathlib import Path

import pytest
import yaml

from livephoto2lrhr.config import load_config


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_config_resolves_paths_and_defaults(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {
                "algorithm": "fake_selector",
                "device": "cpu",
                "top_k": 3,
            },
        },
    )

    config = load_config(config_path)

    assert config.data.input_dir == input_dir.resolve()
    assert config.data.output_dir == output_dir.resolve()
    assert config.data.recursive is True
    assert config.data.image_exts == (".jpg", ".jpeg", ".png", ".heic")
    assert config.data.video_exts == (".mp4", ".mov")
    assert config.pipeline.stages == ("frame_select",)
    assert config.frame_select.algorithm == "fake_selector"
    assert config.frame_select.top_k == 3
    assert config.frame_select.resize_short_side == 518


def test_load_config_resolves_relative_paths_from_config_directory(tmp_path: Path):
    config_dir = tmp_path / "configs"
    input_dir = config_dir / "input"
    output_dir = config_dir / "output"
    input_dir.mkdir(parents=True)
    config_path = config_dir / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": "input",
                "output_dir": "output",
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {"algorithm": "fake_selector"},
        },
    )

    config = load_config(config_path)

    assert config.data.input_dir == input_dir.resolve()
    assert config.data.output_dir == output_dir.resolve()


def test_load_config_rejects_missing_input_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(tmp_path / "missing"),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {"algorithm": "fake_selector"},
        },
    )

    with pytest.raises(ValueError, match="input_dir does not exist"):
        load_config(config_path)


def test_load_config_rejects_unknown_stage(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select", "unknown"]},
            "frame_select": {"algorithm": "fake_selector"},
        },
    )

    with pytest.raises(ValueError, match="unknown pipeline stage"):
        load_config(config_path)


def test_load_config_rejects_missing_frame_select_algorithm(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {},
        },
    )

    with pytest.raises(ValueError, match="frame_select.algorithm is required"):
        load_config(config_path)


def test_load_config_rejects_blank_frame_select_algorithm(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {"algorithm": "   "},
        },
    )

    with pytest.raises(ValueError, match="frame_select.algorithm is required"):
        load_config(config_path)
