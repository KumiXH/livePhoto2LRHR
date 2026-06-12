from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_STAGES = {"frame_select", "align", "color_match"}


def _normalize_ext(ext: str) -> str:
    normalized = ext.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


def _normalize_exts(exts: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_normalize_ext(ext) for ext in exts)


def _resolve_config_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


@dataclass(frozen=True)
class DataConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool = True
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".heic")
    video_exts: tuple[str, ...] = (".mp4", ".mov")
    output_ext: str = ".png"


@dataclass(frozen=True)
class PipelineConfig:
    stages: tuple[str, ...]


@dataclass(frozen=True)
class FrameSelectConfig:
    algorithm: str
    device: str = "auto"
    sample_fps: float = 15.0
    top_k: int = 5
    batch_size: int = 16
    resize_short_side: int = 512
    score_fusion: dict[str, float] | None = None


@dataclass(frozen=True)
class OutputConfig:
    save_metadata: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    pipeline: PipelineConfig
    frame_select: FrameSelectConfig
    output: OutputConfig
    raw: dict[str, Any]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    data_raw = raw.get("data", {})
    input_dir = _resolve_config_path(config_path.parent, data_raw["input_dir"])
    output_dir = _resolve_config_path(config_path.parent, data_raw["output_dir"])

    if not input_dir.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    stages = tuple(raw.get("pipeline", {}).get("stages", ()))
    for stage in stages:
        if stage not in VALID_STAGES:
            raise ValueError(f"unknown pipeline stage: {stage}")

    frame_raw: dict[str, Any] = raw.get("frame_select", {})
    algorithm = str(frame_raw.get("algorithm", "")).strip()
    if "frame_select" in stages and not algorithm:
        raise ValueError("frame_select.algorithm is required when frame_select stage is enabled")

    data_config = DataConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=bool(data_raw.get("recursive", True)),
        image_exts=_normalize_exts(data_raw.get("image_exts", [".jpg", ".jpeg", ".png", ".heic"])),
        video_exts=_normalize_exts(data_raw.get("video_exts", [".mp4", ".mov"])),
        output_ext=_normalize_exts([data_raw.get("output_ext", ".png")])[0],
    )
    pipeline_config = PipelineConfig(stages=stages)
    frame_select_config = FrameSelectConfig(
        algorithm=algorithm,
        device=str(frame_raw.get("device", "auto")),
        sample_fps=float(frame_raw.get("sample_fps", 15.0)),
        top_k=int(frame_raw.get("top_k", 5)),
        batch_size=int(frame_raw.get("batch_size", 16)),
        resize_short_side=int(frame_raw.get("resize_short_side", 512)),
        score_fusion=frame_raw.get("score_fusion"),
    )
    output_raw = raw.get("output", {})
    output_config = OutputConfig(
        save_metadata=bool(output_raw.get("save_metadata", True)),
        overwrite=bool(output_raw.get("overwrite", False)),
    )

    return AppConfig(
        data=data_config,
        pipeline=pipeline_config,
        frame_select=frame_select_config,
        output=output_config,
        raw=raw,
    )
