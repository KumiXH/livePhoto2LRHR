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
    assert config.color_match.histogram_match.color_space == "lab"
    assert config.color_match.histogram_match.bins == 256
    assert config.color_match.retinex.sigma == 15.0
    assert config.color_match.retinex.eps == 1.0e-3
    assert config.color_match.masked_transfer.color_space == "lab"
    assert config.color_match.masked_transfer.difference_threshold == 12.0
    assert config.color_match.masked_transfer.min_mask_fraction == 0.01
    assert config.color_match.masked_transfer.max_mask_fraction == 0.85
    assert config.color_match.masked_transfer.morphology_kernel_size == 5
    assert config.color_match.adaptive_3d_lut.grid_size == 9
    assert config.color_match.adaptive_3d_lut.identity_mix == 0.1
    assert config.color_match.low_frequency_joint.sigma == 5.0
    assert config.color_match.learned_retinex.base_mix == 0.65
    assert config.color_match.mask_aware_harmonization.low_frequency_sigma == 5.0
    assert config.color_match.diffusion_harmonization.num_steps == 4


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
        "histogram_match": {"color_space": "rgb", "bins": 128},
        "retinex": {"sigma": 21.0, "eps": 0.005},
        "masked_transfer": {
            "color_space": "rgb",
            "difference_threshold": 8.0,
            "min_mask_fraction": 0.03,
            "max_mask_fraction": 0.7,
            "morphology_kernel_size": 7,
        },
        "adaptive_3d_lut": {
            "color_space": "rgb",
            "grid_size": 17,
            "smoothing_sigma": 1.5,
            "identity_mix": 0.2,
        },
        "low_frequency_joint": {
            "color_space": "rgb",
            "sigma": 7.0,
            "base_mix": 0.9,
            "detail_preservation": 0.95,
            "chroma_strength": 0.8,
        },
        "learned_retinex": {
            "sigma": 19.0,
            "eps": 0.002,
            "base_mix": 0.72,
        },
        "mask_aware_harmonization": {
            "color_space": "rgb",
            "difference_threshold": 9.0,
            "min_mask_fraction": 0.04,
            "max_mask_fraction": 0.65,
            "morphology_kernel_size": 9,
            "low_frequency_sigma": 6.0,
        },
        "diffusion_harmonization": {
            "color_space": "rgb",
            "num_steps": 6,
            "guidance_strength": 0.55,
            "low_frequency_sigma": 6.5,
            "lut_identity_mix": 0.25,
        },
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
    assert config.color_match.histogram_match.color_space == "rgb"
    assert config.color_match.histogram_match.bins == 128
    assert config.color_match.retinex.sigma == 21.0
    assert config.color_match.retinex.eps == 0.005
    assert config.color_match.masked_transfer.color_space == "rgb"
    assert config.color_match.masked_transfer.difference_threshold == 8.0
    assert config.color_match.masked_transfer.min_mask_fraction == 0.03
    assert config.color_match.masked_transfer.max_mask_fraction == 0.7
    assert config.color_match.masked_transfer.morphology_kernel_size == 7
    assert config.color_match.adaptive_3d_lut.color_space == "rgb"
    assert config.color_match.adaptive_3d_lut.grid_size == 17
    assert config.color_match.adaptive_3d_lut.smoothing_sigma == 1.5
    assert config.color_match.adaptive_3d_lut.identity_mix == 0.2
    assert config.color_match.low_frequency_joint.color_space == "rgb"
    assert config.color_match.low_frequency_joint.sigma == 7.0
    assert config.color_match.low_frequency_joint.base_mix == 0.9
    assert config.color_match.low_frequency_joint.detail_preservation == 0.95
    assert config.color_match.low_frequency_joint.chroma_strength == 0.8
    assert config.color_match.learned_retinex.sigma == 19.0
    assert config.color_match.learned_retinex.eps == 0.002
    assert config.color_match.learned_retinex.base_mix == 0.72
    assert config.color_match.mask_aware_harmonization.color_space == "rgb"
    assert config.color_match.mask_aware_harmonization.difference_threshold == 9.0
    assert config.color_match.mask_aware_harmonization.min_mask_fraction == 0.04
    assert config.color_match.mask_aware_harmonization.max_mask_fraction == 0.65
    assert config.color_match.mask_aware_harmonization.morphology_kernel_size == 9
    assert config.color_match.mask_aware_harmonization.low_frequency_sigma == 6.0
    assert config.color_match.diffusion_harmonization.color_space == "rgb"
    assert config.color_match.diffusion_harmonization.num_steps == 6
    assert config.color_match.diffusion_harmonization.guidance_strength == 0.55
    assert config.color_match.diffusion_harmonization.low_frequency_sigma == 6.5
    assert config.color_match.diffusion_harmonization.lut_identity_mix == 0.25


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
