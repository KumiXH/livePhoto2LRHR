from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate


def output_image_path(output_dir: Path, folder: str, relative_stem: Path, output_ext: str) -> Path:
    return output_dir / folder / relative_stem.with_suffix(output_ext)


def metadata_path(output_dir: Path, relative_stem: Path) -> Path:
    return output_dir / "metadata" / relative_stem.with_suffix(".yaml")


def save_pil_image(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    image.save(destination_path)


def save_rgb_array(frame_rgb: np.ndarray, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb, mode="RGB").save(destination_path)


def candidate_to_dict(candidate: FrameCandidate) -> dict[str, float | int]:
    return {
        "frame_index": candidate.frame_index,
        "timestamp_sec": float(candidate.timestamp_sec),
        "score": float(candidate.score),
    }


def write_yaml(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
