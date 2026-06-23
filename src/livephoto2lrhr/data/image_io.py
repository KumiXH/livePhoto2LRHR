from __future__ import annotations

from pathlib import Path

from PIL import Image

HEIF_PLUGIN_MESSAGE = (
    "HEIC/HEIF support requires pillow-heif. Install it with: "
    "pip install pillow-heif"
)

try:
    from pillow_heif import register_heif_opener as _register_heif_opener

    _HEIF_PLUGIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by tests via monkeypatch
    _register_heif_opener = None
    _HEIF_PLUGIN_AVAILABLE = False

_HEIF_REGISTERED = False


def _ensure_heif_support(path: Path) -> None:
    global _HEIF_REGISTERED

    if path.suffix.lower() not in {".heic", ".heif"}:
        return
    if not _HEIF_PLUGIN_AVAILABLE:
        raise RuntimeError(HEIF_PLUGIN_MESSAGE)
    if not _HEIF_REGISTERED:
        assert _register_heif_opener is not None
        _register_heif_opener()
        _HEIF_REGISTERED = True


def open_pil_image(path: Path) -> Image.Image:
    _ensure_heif_support(path)
    return Image.open(path)
