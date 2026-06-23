from pathlib import Path

import pytest
from PIL import Image

from livephoto2lrhr.config import load_config
from livephoto2lrhr.data import image_io, io


def test_load_config_defaults_include_heif(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  input_dir: input
  output_dir: output
pipeline:
  stages: [frame_select]
frame_select:
  algorithm: fake_selector
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.image_exts == (".jpg", ".jpeg", ".png", ".heic", ".heif")


def test_read_rgb_array_raises_helpful_error_for_missing_heif_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    heic_path = tmp_path / "sample.heic"
    heic_path.write_bytes(b"not-a-real-heic")
    monkeypatch.setattr(image_io, "_HEIF_PLUGIN_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="pillow-heif"):
        io.read_rgb_array(heic_path)


def test_open_pil_image_registers_heif_support_once_for_heic_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_path = tmp_path / "sample.heic"
    image_path.write_bytes(b"fake-heic")

    calls: list[str] = []

    def fake_register_heif_opener() -> None:
        calls.append("registered")

    class DummyImage:
        size = (2, 2)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    def fake_image_open(path: Path) -> DummyImage:
        assert path == image_path
        return DummyImage()

    monkeypatch.setattr(image_io, "_HEIF_PLUGIN_AVAILABLE", True)
    monkeypatch.setattr(image_io, "_HEIF_REGISTERED", False)
    monkeypatch.setattr(image_io, "_register_heif_opener", fake_register_heif_opener)
    monkeypatch.setattr(image_io.Image, "open", fake_image_open)

    with image_io.open_pil_image(image_path) as image:
        assert image.size == (2, 2)

    assert calls == ["registered"]
