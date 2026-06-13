from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_STAGES = {"frame_select", "align", "color_match"}
VALID_EXPORT_LR_SOURCES = {"raw", "aligned", "color_matched"}


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


def _validate_output_folder(value: str, *, config_key: str) -> str:
    folder = value.strip()
    path = Path(folder)
    protected_names = {"lr", "hr", "metadata", "artifacts"}
    if (
        not folder
        or path.is_absolute()
        or len(path.parts) != 1
        or folder in {".", ".."}
        or any(part == ".." for part in path.parts)
        or folder.lower() in protected_names
    ):
        raise ValueError(
            f"{config_key} must be a single safe folder name outside protected outputs "
            "(LR, HR, metadata, artifacts)"
        )
    return folder


def _validate_input_folder(value: str, *, config_key: str) -> str:
    folder = value.strip()
    if folder == "auto":
        return folder
    path = Path(folder)
    protected_names = {"hr", "metadata", "artifacts"}
    if (
        not folder
        or path.is_absolute()
        or len(path.parts) != 1
        or folder in {".", ".."}
        or any(part == ".." for part in path.parts)
        or folder.lower() in protected_names
    ):
        raise ValueError(
            f"{config_key} must be 'auto' or a single safe LR-like folder name outside protected outputs "
            "(HR, metadata, artifacts)"
        )
    return folder


def _validate_export_output_folder(value: str, *, config_key: str) -> str:
    folder = value.strip()
    path = Path(folder)
    protected_names = {"lr", "hr", "metadata", "artifacts", "reports"}
    if (
        not folder
        or path.is_absolute()
        or len(path.parts) != 1
        or folder in {".", ".."}
        or any(part == ".." for part in path.parts)
        or folder.lower() in protected_names
    ):
        raise ValueError(
            f"{config_key} must be a single safe folder name outside protected outputs "
            "(LR, HR, metadata, artifacts, reports)"
        )
    return folder


def _validate_report_path(value: str, *, config_key: str) -> str:
    path = Path(value.strip())
    if (
        not str(path)
        or path.is_absolute()
        or any(part == ".." for part in path.parts)
        or path.name.lower() != "quality_report.csv"
    ):
        raise ValueError(f"{config_key} must be a relative path ending in quality_report.csv")
    return path.as_posix()


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
    resize_short_side: int = 518
    score_fusion: dict[str, float] | None = None


@dataclass(frozen=True)
class AlignArtifactsConfig:
    save_debug_overlay: bool = False
    save_flow: bool = False
    save_masks: bool = False


@dataclass(frozen=True)
class PhaseCorrelationConfig:
    resize_short_side: int = 512


@dataclass(frozen=True)
class ECCConfig:
    motion_model: str = "affine"
    number_of_iterations: int = 100
    termination_eps: float = 1.0e-5
    gaussian_filter_size: int = 5


@dataclass(frozen=True)
class OpticalFlowConfig:
    enabled: bool = False
    algorithm: str = "dis"


@dataclass(frozen=True)
class MeanStdColorMatchConfig:
    color_space: str = "lab"
    eps: float = 1.0e-6


@dataclass(frozen=True)
class AlignConfig:
    enabled: bool = False
    algorithm: str = "identity_alignment"
    device: str = "auto"
    output_folder: str = "LR_aligned"
    confidence_threshold: float = 0.3
    fallback_algorithm: str = "identity_alignment"
    on_failure: str = "keep_original"
    coarse_algorithm: str = "phase_correlation_translation"
    artifacts: AlignArtifactsConfig = AlignArtifactsConfig()
    phase_correlation: PhaseCorrelationConfig = PhaseCorrelationConfig()
    ecc: ECCConfig = ECCConfig()
    optical_flow: OpticalFlowConfig = OpticalFlowConfig()


@dataclass(frozen=True)
class ColorMatchConfig:
    enabled: bool = False
    algorithm: str = "identity_color_match"
    device: str = "auto"
    input_folder: str = "auto"
    output_folder: str = "LR_color_matched"
    confidence_threshold: float = 0.0
    on_failure: str = "keep_original"
    mean_std: MeanStdColorMatchConfig = MeanStdColorMatchConfig()


@dataclass(frozen=True)
class ReportConfig:
    enabled: bool = False
    output_folder: str = "reports"
    aligned_folder: str = "LR_aligned"
    color_matched_folder: str = "LR_color_matched"
    max_preview_samples: int = 24
    thumbnail_size: int = 160


@dataclass(frozen=True)
class ExportConfig:
    enabled: bool = False
    input_report: str = "reports/quality_report.csv"
    output_folder: str = "final"
    lr_source: str = "aligned"
    min_align_confidence: float = 0.0
    require_align_status: str | None = "success"
    require_flow_status: str | None = None
    max_source_to_hr_mae: float | None = None


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
    align: AlignConfig = field(default_factory=AlignConfig)
    color_match: ColorMatchConfig = field(default_factory=ColorMatchConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


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
        resize_short_side=int(frame_raw.get("resize_short_side", 518)),
        score_fusion=frame_raw.get("score_fusion"),
    )
    align_raw: dict[str, Any] = raw.get("align", {})
    align_artifacts_raw: dict[str, Any] = align_raw.get("artifacts", {})
    phase_raw: dict[str, Any] = align_raw.get("phase_correlation", {})
    ecc_raw: dict[str, Any] = align_raw.get("ecc", {})
    optical_flow_raw: dict[str, Any] = align_raw.get("optical_flow", {})
    align_config = AlignConfig(
        enabled=bool(align_raw.get("enabled", False)),
        algorithm=str(align_raw.get("algorithm", "identity_alignment")),
        device=str(align_raw.get("device", "auto")),
        output_folder=_validate_output_folder(
            str(align_raw.get("output_folder", "LR_aligned")),
            config_key="align.output_folder",
        ),
        confidence_threshold=float(align_raw.get("confidence_threshold", 0.3)),
        fallback_algorithm=str(align_raw.get("fallback_algorithm", "identity_alignment")),
        on_failure=str(align_raw.get("on_failure", "keep_original")),
        coarse_algorithm=str(align_raw.get("coarse_algorithm", "phase_correlation_translation")),
        artifacts=AlignArtifactsConfig(
            save_debug_overlay=bool(align_artifacts_raw.get("save_debug_overlay", False)),
            save_flow=bool(align_artifacts_raw.get("save_flow", False)),
            save_masks=bool(align_artifacts_raw.get("save_masks", False)),
        ),
        phase_correlation=PhaseCorrelationConfig(
            resize_short_side=int(phase_raw.get("resize_short_side", 512)),
        ),
        ecc=ECCConfig(
            motion_model=str(ecc_raw.get("motion_model", "affine")),
            number_of_iterations=int(ecc_raw.get("number_of_iterations", 100)),
            termination_eps=float(ecc_raw.get("termination_eps", 1.0e-5)),
            gaussian_filter_size=int(ecc_raw.get("gaussian_filter_size", 5)),
        ),
        optical_flow=OpticalFlowConfig(
            enabled=bool(optical_flow_raw.get("enabled", False)),
            algorithm=str(optical_flow_raw.get("algorithm", "dis")),
        ),
    )
    color_raw: dict[str, Any] = raw.get("color_match", {})
    mean_std_raw: dict[str, Any] = color_raw.get("mean_std", {})
    color_match_config = ColorMatchConfig(
        enabled=bool(color_raw.get("enabled", False)),
        algorithm=str(color_raw.get("algorithm", "identity_color_match")),
        device=str(color_raw.get("device", "auto")),
        input_folder=_validate_input_folder(
            str(color_raw.get("input_folder", "auto")),
            config_key="color_match.input_folder",
        ),
        output_folder=_validate_output_folder(
            str(color_raw.get("output_folder", "LR_color_matched")),
            config_key="color_match.output_folder",
        ),
        confidence_threshold=float(color_raw.get("confidence_threshold", 0.0)),
        on_failure=str(color_raw.get("on_failure", "keep_original")),
        mean_std=MeanStdColorMatchConfig(
            color_space=str(mean_std_raw.get("color_space", "lab")),
            eps=float(mean_std_raw.get("eps", 1.0e-6)),
        ),
    )
    report_raw: dict[str, Any] = raw.get("report", {})
    report_config = ReportConfig(
        enabled=bool(report_raw.get("enabled", False)),
        output_folder=_validate_output_folder(str(report_raw.get("output_folder", "reports")), config_key="report.output_folder"),
        aligned_folder=_validate_input_folder(
            str(report_raw.get("aligned_folder", "LR_aligned")),
            config_key="report.aligned_folder",
        ),
        color_matched_folder=_validate_input_folder(
            str(report_raw.get("color_matched_folder", "LR_color_matched")),
            config_key="report.color_matched_folder",
        ),
        max_preview_samples=int(report_raw.get("max_preview_samples", 24)),
        thumbnail_size=int(report_raw.get("thumbnail_size", 160)),
    )
    export_raw: dict[str, Any] = raw.get("export", {})
    export_lr_source = str(export_raw.get("lr_source", "aligned"))
    if export_lr_source not in VALID_EXPORT_LR_SOURCES:
        raise ValueError(f"export.lr_source must be one of: {sorted(VALID_EXPORT_LR_SOURCES)}")
    max_source_to_hr_mae_raw = export_raw.get("max_source_to_hr_mae")
    max_source_to_hr_mae = (
        None if max_source_to_hr_mae_raw is None else float(max_source_to_hr_mae_raw)
    )
    require_flow_status_raw = export_raw.get("require_flow_status")
    require_flow_status = None if require_flow_status_raw is None else str(require_flow_status_raw)
    require_align_status_raw = export_raw.get("require_align_status", "success")
    require_align_status = None if require_align_status_raw is None else str(require_align_status_raw)
    export_config = ExportConfig(
        enabled=bool(export_raw.get("enabled", False)),
        input_report=_validate_report_path(
            str(export_raw.get("input_report", "reports/quality_report.csv")),
            config_key="export.input_report",
        ),
        output_folder=_validate_export_output_folder(
            str(export_raw.get("output_folder", "final")),
            config_key="export.output_folder",
        ),
        lr_source=export_lr_source,
        min_align_confidence=float(export_raw.get("min_align_confidence", 0.0)),
        require_align_status=require_align_status,
        require_flow_status=require_flow_status,
        max_source_to_hr_mae=max_source_to_hr_mae,
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
        align=align_config,
        color_match=color_match_config,
        report=report_config,
        export=export_config,
        output=output_config,
        raw=raw,
    )
