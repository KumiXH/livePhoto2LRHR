from pathlib import Path

import yaml

from livephoto2lrhr.cli import main


def test_cli_runs_pipeline_from_config(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "input_dir": str(image_path.parent),
                    "output_dir": str(output_dir),
                    "image_exts": [".jpg"],
                    "video_exts": [".mp4"],
                },
                "pipeline": {"stages": ["frame_select"]},
                "frame_select": {"algorithm": "fake_selector", "top_k": 1},
                "output": {"save_metadata": True, "overwrite": False},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "LR" / "flower.png").exists()
