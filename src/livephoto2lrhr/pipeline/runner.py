from __future__ import annotations

from collections import Counter
from typing import Any

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.color_match import build_color_match_registry
from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.config import AppConfig
from livephoto2lrhr.data.io import write_yaml
from livephoto2lrhr.data.pairing import discover_pairs
from livephoto2lrhr.export.dataset import ExportDatasetConfig, export_dataset
from livephoto2lrhr.reports.quality import QualityReportConfig, generate_quality_report
from livephoto2lrhr.stages.align import AlignStage
from livephoto2lrhr.stages.color_match import ColorMatchStage
from livephoto2lrhr.stages.frame_select import FrameSelectStage


KNOWN_PHASE_1_STATUSES = ("success", "skipped_existing", "frame_select_failed", "write_failed")
KNOWN_ALIGN_STATUSES = (
    "align_success",
    "align_skipped_disabled",
    "align_skipped_existing",
    "align_skipped_missing_input",
    "align_skipped_low_confidence",
    "align_failed",
    "align_write_failed",
)
KNOWN_COLOR_MATCH_STATUSES = (
    "color_match_success",
    "color_match_skipped_disabled",
    "color_match_skipped_existing",
    "color_match_skipped_missing_input",
    "color_match_skipped_low_confidence",
    "color_match_failed",
    "color_match_write_failed",
)


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    pair_result = discover_pairs(
        config.data.input_dir,
        image_exts=config.data.image_exts,
        video_exts=config.data.video_exts,
        recursive=config.data.recursive,
    )
    counts: Counter[str] = Counter(
        {status: 0 for status in (*KNOWN_PHASE_1_STATUSES, *KNOWN_ALIGN_STATUSES, *KNOWN_COLOR_MATCH_STATUSES)}
    )
    samples: list[dict[str, str]] = []

    if "frame_select" in config.pipeline.stages:
        registry = build_similarity_registry()
        selector = registry.create(
            config.frame_select.algorithm,
            {
                "sample_fps": config.frame_select.sample_fps,
                "top_k": config.frame_select.top_k,
                "batch_size": config.frame_select.batch_size,
                "resize_short_side": config.frame_select.resize_short_side,
                "score_fusion": config.frame_select.score_fusion,
                "device": config.frame_select.device,
            },
        )
        stage = FrameSelectStage(
            output_dir=config.data.output_dir,
            output_ext=config.data.output_ext,
            overwrite=config.output.overwrite,
            save_metadata=config.output.save_metadata,
            selector=selector,
            algorithm_name=config.frame_select.algorithm,
        )
        for pair in pair_result.pairs:
            result = stage.run(pair)
            counts[result.status] += 1
            samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})

    if "align" in config.pipeline.stages:
        if not config.align.enabled:
            counts["align_skipped_disabled"] += len(pair_result.pairs)
        else:
            registry = build_alignment_registry()
            aligner_config = _alignment_algorithm_config(config)
            aligner = registry.create(config.align.algorithm, aligner_config)
            fallback_aligner = None
            fallback_config = None
            if config.align.fallback_algorithm and config.align.fallback_algorithm != config.align.algorithm:
                fallback_config = _alignment_algorithm_config(config)
                fallback_aligner = registry.create(config.align.fallback_algorithm, fallback_config)
            stage = AlignStage(
                output_dir=config.data.output_dir,
                output_ext=config.data.output_ext,
                output_folder=config.align.output_folder,
                overwrite=config.output.overwrite,
                save_metadata=config.output.save_metadata,
                aligner=aligner,
                algorithm_name=config.align.algorithm,
                algorithm_config=aligner_config,
                fallback_aligner=fallback_aligner,
                fallback_algorithm_name=config.align.fallback_algorithm,
                fallback_algorithm_config=fallback_config,
                confidence_threshold=config.align.confidence_threshold,
                on_failure=config.align.on_failure,
                device=config.align.device,
            )
            for pair in pair_result.pairs:
                result = stage.run(pair)
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})

    if "color_match" in config.pipeline.stages:
        if not config.color_match.enabled:
            counts["color_match_skipped_disabled"] += len(pair_result.pairs)
        else:
            registry = build_color_match_registry()
            matcher_config = _color_match_algorithm_config(config)
            matcher = registry.create(config.color_match.algorithm, matcher_config)
            stage = ColorMatchStage(
                output_dir=config.data.output_dir,
                output_ext=config.data.output_ext,
                input_folder=config.color_match.input_folder,
                output_folder=config.color_match.output_folder,
                overwrite=config.output.overwrite,
                save_metadata=config.output.save_metadata,
                matcher=matcher,
                algorithm_name=config.color_match.algorithm,
                algorithm_config=matcher_config,
                confidence_threshold=config.color_match.confidence_threshold,
                on_failure=config.color_match.on_failure,
                device=config.color_match.device,
            )
            for pair in pair_result.pairs:
                result = stage.run(pair)
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})

    report_summary: dict[str, Any] | None = None
    if config.report.enabled:
        report_result = generate_quality_report(
            output_dir=config.data.output_dir,
            config=QualityReportConfig(
                output_folder=config.report.output_folder,
                aligned_folder=config.report.aligned_folder,
                color_matched_folder=config.report.color_matched_folder,
                max_preview_samples=config.report.max_preview_samples,
                thumbnail_size=config.report.thumbnail_size,
            ),
        )
        report_summary = {
            "rows": report_result.rows,
            "csv": str(report_result.csv_path),
            "preview": str(report_result.preview_path) if report_result.preview_path is not None else "",
        }

    export_summary: dict[str, Any] | None = None
    if config.export.enabled:
        export_result = export_dataset(
            output_dir=config.data.output_dir,
            config=ExportDatasetConfig(
                input_report=config.export.input_report,
                output_folder=config.export.output_folder,
                final_lr_source=config.export.final_lr_source,
                gate_lr_source=config.export.gate_lr_source,
                final_lr_resize_mode=config.export.final_lr_resize_mode,
                min_align_confidence=config.export.min_align_confidence,
                require_align_status=config.export.require_align_status,
                require_flow_status=config.export.require_flow_status,
                max_source_to_hr_mae=config.export.max_source_to_hr_mae,
                overwrite=config.output.overwrite,
            ),
        )
        export_summary = {
            "accepted": export_result.accepted,
            "rejected": export_result.rejected,
            "manifest": str(export_result.manifest_path),
        }

    summary: dict[str, Any] = {
        "counts": dict(counts),
        "pair_discovery": {
            "paired": len(pair_result.pairs),
            "missing_images": pair_result.missing_images,
            "missing_videos": pair_result.missing_videos,
            "ambiguous": pair_result.ambiguous,
        },
        "samples": samples,
        "config": config.raw,
    }
    if report_summary is not None:
        summary["report"] = report_summary
    if export_summary is not None:
        summary["export"] = export_summary
    write_yaml(config.data.output_dir / "run_summary.yaml", summary)
    return summary


def _alignment_algorithm_config(config: AppConfig) -> dict[str, Any]:
    return {
        "device": config.align.device,
        "phase_correlation": config.align.phase_correlation,
        "coarse_algorithm": config.align.coarse_algorithm,
        "motion_model": config.align.ecc.motion_model,
        "number_of_iterations": config.align.ecc.number_of_iterations,
        "termination_eps": config.align.ecc.termination_eps,
        "gaussian_filter_size": config.align.ecc.gaussian_filter_size,
        "resize_short_side": config.align.phase_correlation.resize_short_side,
        "optical_flow": config.align.optical_flow,
        "artifacts": config.align.artifacts,
    }


def _color_match_algorithm_config(config: AppConfig) -> dict[str, Any]:
    return {
        "device": config.color_match.device,
        "color_space": config.color_match.mean_std.color_space,
        "eps": config.color_match.mean_std.eps,
        "mean_std": config.color_match.mean_std,
    }
