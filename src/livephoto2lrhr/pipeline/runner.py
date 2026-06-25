from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from typing import Any

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.color_match import build_color_match_registry
from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.config import AppConfig
from livephoto2lrhr.data.io import write_yaml
from livephoto2lrhr.data.pairing import SamplePair, discover_pairs
from livephoto2lrhr.export.dataset import ExportDatasetConfig, export_dataset
from livephoto2lrhr.reports.quality import QualityReportConfig, generate_quality_report
from livephoto2lrhr.reports.sample_status import build_sample_status_records, write_sample_status_files
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

PARALLEL_STAGE_NAMES = ("frame_select", "align", "color_match")


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    start_time = perf_counter()
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
    stage_events: list[dict[str, Any]] = []
    pair_lookup = {
        pair.sample_id: {
            "source_image": str(pair.image_path),
            "source_video": str(pair.video_path),
        }
        for pair in pair_result.pairs
    }
    runtime = _runtime_config(config)
    execution = {
        "retry_failed_samples": runtime["retry_failed_samples"],
        "retried_failed_samples": 0,
        "resumed_from_existing_outputs": 0,
        "stage_timings_sec": {},
        "total_runtime_sec": 0.0,
        "failed_samples_manifest": "",
        "sample_status_yaml": "",
        "sample_status_csv": "",
        "parallel": {
            "enabled": False,
            "requested_workers": runtime["parallel"]["num_workers"],
            "gpu_ids": runtime["parallel"]["gpu_ids"],
            "worker_assignments": _build_worker_assignments(runtime["parallel"]),
            "used_workers": 0,
            "worker_pids": [],
            "stages": {
                stage_name: {
                    "enabled": False,
                    "used_workers": 0,
                    "worker_pids": [],
                }
                for stage_name in PARALLEL_STAGE_NAMES
            },
        },
    }
    execution["parallel"]["enabled"] = _parallel_enabled(config, runtime)

    if "frame_select" in config.pipeline.stages:
        stage_start = perf_counter()
        if _parallel_enabled_for_stage("frame_select", config, runtime):
            frame_select_results = _run_parallel_frame_select(config, pair_result.pairs, runtime["parallel"])
            worker_pids = sorted(
                {result["worker_pid"] for result in frame_select_results if result.get("worker_pid")}
            )
            execution["parallel"]["stages"]["frame_select"]["enabled"] = True
            execution["parallel"]["stages"]["frame_select"]["worker_pids"] = worker_pids
            execution["parallel"]["stages"]["frame_select"]["used_workers"] = len(worker_pids)
            for result in frame_select_results:
                counts[result["status"]] += 1
                samples.append(
                    {
                        "sample_id": result["sample_id"],
                        "status": result["status"],
                        "message": result["message"],
                    }
                )
                stage_events.append(
                    {
                        "sample_id": result["sample_id"],
                        "stage": "frame_select",
                        "status": result["status"],
                        "message": result["message"],
                        "started_at": result.get("started_at", ""),
                        "finished_at": result.get("finished_at", ""),
                        "duration_sec": result.get("duration_sec", ""),
                        "error_traceback": result.get("error_traceback", ""),
                    }
                )
                if result["status"] == "skipped_existing":
                    execution["resumed_from_existing_outputs"] += 1
        else:
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
                event_start = perf_counter()
                started_at = _timestamp_now()
                result = stage.run(pair)
                finished_at = _timestamp_now()
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})
                stage_events.append(
                    {
                        "sample_id": result.sample_id,
                        "stage": "frame_select",
                        "status": result.status,
                        "message": result.message,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_sec": _elapsed_seconds(event_start),
                        "error_traceback": result.error_traceback,
                    }
                )
                if result.status == "skipped_existing":
                    execution["resumed_from_existing_outputs"] += 1
        execution["stage_timings_sec"]["frame_select"] = _elapsed_seconds(stage_start)

    if "align" in config.pipeline.stages:
        stage_start = perf_counter()
        if not config.align.enabled:
            counts["align_skipped_disabled"] += len(pair_result.pairs)
        elif _parallel_enabled_for_stage("align", config, runtime):
            align_results = _run_parallel_align(config, pair_result.pairs, runtime["parallel"], runtime["retry_failed_samples"])
            worker_pids = sorted({result["worker_pid"] for result in align_results if result.get("worker_pid")})
            execution["parallel"]["stages"]["align"]["enabled"] = True
            execution["parallel"]["stages"]["align"]["worker_pids"] = worker_pids
            execution["parallel"]["stages"]["align"]["used_workers"] = len(worker_pids)
            for result in align_results:
                counts[result["status"]] += 1
                samples.append({"sample_id": result["sample_id"], "status": result["status"], "message": result["message"]})
                stage_events.append(
                    {
                        "sample_id": result["sample_id"],
                        "stage": "align",
                        "status": result["status"],
                        "message": result["message"],
                        "started_at": result.get("started_at", ""),
                        "finished_at": result.get("finished_at", ""),
                        "duration_sec": result.get("duration_sec", ""),
                        "error_traceback": result.get("error_traceback", ""),
                    }
                )
                if result.get("resumed_from_existing"):
                    execution["resumed_from_existing_outputs"] += 1
                if result.get("retried_failed_before"):
                    execution["retried_failed_samples"] += 1
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
                force_retry_failed = runtime["retry_failed_samples"]
                was_failed_before_retry = _was_failed_stage_output(
                    config.data.output_dir,
                    pair.relative_stem,
                    "align",
                )
                event_start = perf_counter()
                started_at = _timestamp_now()
                result = stage.run(pair, force_retry_failed=force_retry_failed)
                finished_at = _timestamp_now()
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})
                stage_events.append(
                    {
                        "sample_id": result.sample_id,
                        "stage": "align",
                        "status": result.status,
                        "message": result.message,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_sec": _elapsed_seconds(event_start),
                        "error_traceback": result.error_traceback,
                    }
                )
                if result.status == "align_skipped_existing":
                    execution["resumed_from_existing_outputs"] += 1
                elif force_retry_failed and result.status != "align_skipped_existing" and was_failed_before_retry:
                    execution["retried_failed_samples"] += 1
        execution["stage_timings_sec"]["align"] = _elapsed_seconds(stage_start)

    if "color_match" in config.pipeline.stages:
        stage_start = perf_counter()
        if not config.color_match.enabled:
            counts["color_match_skipped_disabled"] += len(pair_result.pairs)
        elif _parallel_enabled_for_stage("color_match", config, runtime):
            color_match_results = _run_parallel_color_match(
                config,
                pair_result.pairs,
                runtime["parallel"],
                runtime["retry_failed_samples"],
            )
            worker_pids = sorted({result["worker_pid"] for result in color_match_results if result.get("worker_pid")})
            execution["parallel"]["stages"]["color_match"]["enabled"] = True
            execution["parallel"]["stages"]["color_match"]["worker_pids"] = worker_pids
            execution["parallel"]["stages"]["color_match"]["used_workers"] = len(worker_pids)
            for result in color_match_results:
                counts[result["status"]] += 1
                samples.append({"sample_id": result["sample_id"], "status": result["status"], "message": result["message"]})
                stage_events.append(
                    {
                        "sample_id": result["sample_id"],
                        "stage": "color_match",
                        "status": result["status"],
                        "message": result["message"],
                        "started_at": result.get("started_at", ""),
                        "finished_at": result.get("finished_at", ""),
                        "duration_sec": result.get("duration_sec", ""),
                        "error_traceback": result.get("error_traceback", ""),
                    }
                )
                if result.get("resumed_from_existing"):
                    execution["resumed_from_existing_outputs"] += 1
                if result.get("retried_failed_before"):
                    execution["retried_failed_samples"] += 1
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
                force_retry_failed = runtime["retry_failed_samples"]
                was_failed_before_retry = _was_failed_stage_output(
                    config.data.output_dir,
                    pair.relative_stem,
                    "color_match",
                )
                event_start = perf_counter()
                started_at = _timestamp_now()
                result = stage.run(pair, force_retry_failed=force_retry_failed)
                finished_at = _timestamp_now()
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})
                stage_events.append(
                    {
                        "sample_id": result.sample_id,
                        "stage": "color_match",
                        "status": result.status,
                        "message": result.message,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "duration_sec": _elapsed_seconds(event_start),
                        "error_traceback": result.error_traceback,
                    }
                )
                if result.status == "color_match_skipped_existing":
                    execution["resumed_from_existing_outputs"] += 1
                elif force_retry_failed and result.status != "color_match_skipped_existing" and was_failed_before_retry:
                    execution["retried_failed_samples"] += 1
        execution["stage_timings_sec"]["color_match"] = _elapsed_seconds(stage_start)

    all_parallel_worker_pids = sorted(
        {
            pid
            for stage_info in execution["parallel"]["stages"].values()
            for pid in stage_info["worker_pids"]
        }
    )
    execution["parallel"]["worker_pids"] = all_parallel_worker_pids
    execution["parallel"]["used_workers"] = len(all_parallel_worker_pids)

    report_summary: dict[str, Any] | None = None
    if config.report.enabled:
        stage_start = perf_counter()
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
            "csv_zh": str(report_result.csv_zh_path),
            "preview": str(report_result.preview_path) if report_result.preview_path is not None else "",
        }
        execution["stage_timings_sec"]["report"] = _elapsed_seconds(stage_start)

    export_summary: dict[str, Any] | None = None
    if config.export.enabled:
        stage_start = perf_counter()
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
                min_source_to_hr_psnr=config.export.min_source_to_hr_psnr,
                min_source_to_hr_ssim=config.export.min_source_to_hr_ssim,
                require_source_to_hr_dimension_match=config.export.require_source_to_hr_dimension_match,
                require_source_to_hr_aspect_ratio_match=config.export.require_source_to_hr_aspect_ratio_match,
                max_source_to_hr_border_mae=config.export.max_source_to_hr_border_mae,
                max_mean_flow_magnitude=config.export.max_mean_flow_magnitude,
                overwrite=config.output.overwrite,
            ),
        )
        export_summary = {
            "accepted": export_result.accepted,
            "rejected": export_result.rejected,
            "manifest": str(export_result.manifest_path),
        }
        execution["stage_timings_sec"]["export"] = _elapsed_seconds(stage_start)

    execution["total_runtime_sec"] = _elapsed_seconds(start_time)
    summary: dict[str, Any] = {
        "counts": dict(counts),
        "execution": execution,
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
    failed_samples_manifest = _write_failed_samples_manifest(config.data.output_dir, samples)
    execution["failed_samples_manifest"] = str(failed_samples_manifest)
    sample_status_records = build_sample_status_records(pair_dicts=pair_lookup, stage_events=stage_events)
    sample_status_yaml, sample_status_csv = write_sample_status_files(config.data.output_dir, sample_status_records)
    execution["sample_status_yaml"] = str(sample_status_yaml)
    execution["sample_status_csv"] = str(sample_status_csv)
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
        "feature_match": config.align.feature_match,
        "mask_aware": config.align.mask_aware,
        "artifacts": config.align.artifacts,
    }


def _color_match_algorithm_config(config: AppConfig) -> dict[str, Any]:
    return {
        "device": config.color_match.device,
        "color_space": config.color_match.mean_std.color_space,
        "eps": config.color_match.mean_std.eps,
        "mean_std": config.color_match.mean_std,
        "histogram_match": config.color_match.histogram_match,
        "retinex": config.color_match.retinex,
        "masked_transfer": config.color_match.masked_transfer,
        "adaptive_3d_lut": config.color_match.adaptive_3d_lut,
        "low_frequency_joint": config.color_match.low_frequency_joint,
        "learned_retinex": config.color_match.learned_retinex,
        "mask_aware_harmonization": config.color_match.mask_aware_harmonization,
        "diffusion_harmonization": config.color_match.diffusion_harmonization,
    }


def _runtime_config(config: AppConfig) -> dict[str, bool]:
    runtime = {}
    if isinstance(config.raw, dict):
        raw_runtime = config.raw.get("runtime", {})
        if isinstance(raw_runtime, dict):
            runtime.update(raw_runtime)
    if not isinstance(runtime, dict):
        runtime = {}
    parallel = runtime.get("parallel", {})
    if not isinstance(parallel, dict):
        parallel = {}
    return {
        "retry_failed_samples": bool(runtime.get("retry_failed_samples", False)),
        "parallel": {
            "num_workers": int(parallel.get("num_workers", 1)),
            "gpu_ids": [str(gpu_id) for gpu_id in parallel.get("gpu_ids", [])],
        },
    }


def _elapsed_seconds(start_time: float) -> float:
    return round(perf_counter() - start_time, 6)


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _was_failed_stage_output(output_dir, relative_stem, stage_name: str) -> bool:
    metadata_file = output_dir / "metadata" / relative_stem.parent / f"{relative_stem.name}.yaml"
    if not metadata_file.exists():
        return False
    import yaml

    metadata = yaml.safe_load(metadata_file.read_text(encoding="utf-8")) or {}
    stage = metadata.get(stage_name)
    if not isinstance(stage, dict):
        return False
    return str(stage.get("status", "")).lower() == "failed"


def _write_failed_samples_manifest(output_dir: Path, samples: list[dict[str, str]]) -> Path:
    failed_samples_path = output_dir / "failed_samples.yaml"
    failed_samples = [
        sample
        for sample in samples
        if sample.get("status", "").endswith("_failed") or sample.get("status", "") == "frame_select_failed"
    ]
    write_yaml(
        failed_samples_path,
        {
            "failed_samples": failed_samples,
        },
    )
    return failed_samples_path


def _build_worker_assignments(parallel_runtime: dict[str, Any]) -> list[dict[str, Any]]:
    requested_workers = max(int(parallel_runtime.get("num_workers", 1)), 1)
    gpu_ids = [str(gpu_id) for gpu_id in parallel_runtime.get("gpu_ids", [])]
    assignments: list[dict[str, Any]] = []
    for worker_index in range(requested_workers):
        gpu_id = gpu_ids[worker_index % len(gpu_ids)] if gpu_ids else ""
        assignments.append(
            {
                "worker_index": worker_index,
                "gpu_id": gpu_id,
            }
        )
    return assignments


def _parallel_enabled(config: AppConfig, runtime: dict[str, Any]) -> bool:
    return any(_parallel_enabled_for_stage(stage_name, config, runtime) for stage_name in PARALLEL_STAGE_NAMES)


def _parallel_enabled_for_stage(stage_name: str, config: AppConfig, runtime: dict[str, Any]) -> bool:
    if int(runtime["parallel"]["num_workers"]) <= 1:
        return False
    if stage_name not in config.pipeline.stages:
        return False
    if stage_name == "align":
        return config.align.enabled
    if stage_name == "color_match":
        return config.color_match.enabled
    return stage_name == "frame_select"


def _run_parallel_frame_select(
    config: AppConfig,
    pairs: list[SamplePair],
    parallel_runtime: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks = _chunk_pairs(pairs, max(int(parallel_runtime["num_workers"]), 1))
    assignments = _build_worker_assignments(parallel_runtime)
    futures = []
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        for worker_index, chunk in enumerate(chunks):
            assignment = assignments[worker_index]
            futures.append(
                executor.submit(
                    _frame_select_worker_entry,
                    _frame_select_worker_config(config),
                    [sample_pair_to_dict(pair) for pair in chunk],
                    assignment["worker_index"],
                    assignment["gpu_id"],
                )
            )
        for future in futures:
            results.extend(future.result())
    return results


def _run_parallel_align(
    config: AppConfig,
    pairs: list[SamplePair],
    parallel_runtime: dict[str, Any],
    retry_failed_samples: bool,
) -> list[dict[str, Any]]:
    return _run_parallel_stage(
        config=config,
        pairs=pairs,
        parallel_runtime=parallel_runtime,
        worker_entry=_align_worker_entry,
        worker_config=_align_worker_config(config, retry_failed_samples),
    )


def _run_parallel_color_match(
    config: AppConfig,
    pairs: list[SamplePair],
    parallel_runtime: dict[str, Any],
    retry_failed_samples: bool,
) -> list[dict[str, Any]]:
    return _run_parallel_stage(
        config=config,
        pairs=pairs,
        parallel_runtime=parallel_runtime,
        worker_entry=_color_match_worker_entry,
        worker_config=_color_match_worker_config(config, retry_failed_samples),
    )


def _run_parallel_stage(
    *,
    config: AppConfig,
    pairs: list[SamplePair],
    parallel_runtime: dict[str, Any],
    worker_entry,
    worker_config: dict[str, Any],
) -> list[dict[str, Any]]:
    chunks = _chunk_pairs(pairs, max(int(parallel_runtime["num_workers"]), 1))
    assignments = _build_worker_assignments(parallel_runtime)
    futures = []
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        for worker_index, chunk in enumerate(chunks):
            assignment = assignments[worker_index]
            futures.append(
                executor.submit(
                    worker_entry,
                    worker_config,
                    [sample_pair_to_dict(pair) for pair in chunk],
                    assignment["worker_index"],
                    assignment["gpu_id"],
                )
            )
        for future in futures:
            results.extend(future.result())
    return results


def _chunk_pairs(pairs: list[SamplePair], num_workers: int) -> list[list[SamplePair]]:
    if not pairs:
        return []
    worker_count = min(max(num_workers, 1), len(pairs))
    chunks: list[list[SamplePair]] = [[] for _ in range(worker_count)]
    for index, pair in enumerate(pairs):
        chunks[index % worker_count].append(pair)
    return [chunk for chunk in chunks if chunk]


def sample_pair_to_dict(pair: SamplePair) -> dict[str, str]:
    return {
        "sample_id": pair.sample_id,
        "image_path": str(pair.image_path),
        "video_path": str(pair.video_path),
        "relative_stem": pair.relative_stem.as_posix(),
    }


def _frame_select_worker_entry(
    worker_config: dict[str, Any],
    pair_dicts: list[dict[str, str]],
    worker_index: int,
    gpu_id: str,
) -> list[dict[str, Any]]:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from pathlib import Path

    from livephoto2lrhr.algorithms.similarity import build_similarity_registry
    from livephoto2lrhr.data.pairing import SamplePair
    from livephoto2lrhr.stages.frame_select import FrameSelectStage

    output_dir = Path(str(worker_config["output_dir"]))
    output_ext = str(worker_config["output_ext"])
    overwrite = bool(worker_config["overwrite"])
    save_metadata = bool(worker_config["save_metadata"])
    frame_select_raw = dict(worker_config["frame_select"])
    registry = build_similarity_registry()
    selector = registry.create(
        str(frame_select_raw["algorithm"]),
        {
            "sample_fps": float(frame_select_raw.get("sample_fps", 15.0)),
            "top_k": int(frame_select_raw.get("top_k", 5)),
            "batch_size": int(frame_select_raw.get("batch_size", 16)),
            "resize_short_side": int(frame_select_raw.get("resize_short_side", 518)),
            "score_fusion": frame_select_raw.get("score_fusion"),
            "device": str(frame_select_raw.get("device", "auto")),
        },
    )
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=output_ext,
        overwrite=overwrite,
        save_metadata=save_metadata,
        selector=selector,
        algorithm_name=str(frame_select_raw["algorithm"]),
    )
    results: list[dict[str, Any]] = []
    for pair_dict in pair_dicts:
        pair = SamplePair(
            sample_id=pair_dict["sample_id"],
            image_path=Path(pair_dict["image_path"]),
            video_path=Path(pair_dict["video_path"]),
            relative_stem=Path(pair_dict["relative_stem"]),
        )
        result = stage.run(pair)
        results.append(
            {
                "sample_id": result.sample_id,
                "status": result.status,
                "message": result.message,
                "started_at": "",
                "finished_at": "",
                "duration_sec": "",
                "error_traceback": result.error_traceback,
                "worker_index": worker_index,
                "worker_pid": os.getpid(),
                "gpu_id": gpu_id,
            }
        )
    return results


def _align_worker_entry(
    worker_config: dict[str, Any],
    pair_dicts: list[dict[str, str]],
    worker_index: int,
    gpu_id: str,
) -> list[dict[str, Any]]:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from pathlib import Path

    from livephoto2lrhr.algorithms.alignment import build_alignment_registry
    from livephoto2lrhr.data.pairing import SamplePair
    from livephoto2lrhr.stages.align import AlignStage

    registry = build_alignment_registry()
    align_raw = dict(worker_config["align"])
    aligner_config = align_raw["algorithm_config"]
    aligner = registry.create(str(align_raw["algorithm"]), aligner_config)
    fallback_aligner = None
    fallback_config = None
    if align_raw.get("fallback_algorithm") and align_raw["fallback_algorithm"] != align_raw["algorithm"]:
        fallback_config = align_raw["fallback_algorithm_config"]
        fallback_aligner = registry.create(str(align_raw["fallback_algorithm"]), fallback_config)
    stage = AlignStage(
        output_dir=Path(str(worker_config["output_dir"])),
        output_ext=str(worker_config["output_ext"]),
        output_folder=str(align_raw["output_folder"]),
        overwrite=bool(worker_config["overwrite"]),
        save_metadata=bool(worker_config["save_metadata"]),
        aligner=aligner,
        algorithm_name=str(align_raw["algorithm"]),
        algorithm_config=aligner_config,
        fallback_aligner=fallback_aligner,
        fallback_algorithm_name=align_raw.get("fallback_algorithm"),
        fallback_algorithm_config=fallback_config,
        confidence_threshold=float(align_raw["confidence_threshold"]),
        on_failure=str(align_raw["on_failure"]),
        device=str(align_raw["device"]),
    )
    retry_failed_samples = bool(worker_config["retry_failed_samples"])
    output_dir = Path(str(worker_config["output_dir"]))
    results: list[dict[str, Any]] = []
    for pair_dict in pair_dicts:
        pair = sample_pair_from_dict(pair_dict)
        was_failed_before_retry = _was_failed_stage_output(output_dir, pair.relative_stem, "align")
        event_start = perf_counter()
        started_at = _timestamp_now()
        result = stage.run(pair, force_retry_failed=retry_failed_samples)
        finished_at = _timestamp_now()
        results.append(
            {
                "sample_id": result.sample_id,
                "status": result.status,
                "message": result.message,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": _elapsed_seconds(event_start),
                "error_traceback": result.error_traceback,
                "worker_index": worker_index,
                "worker_pid": os.getpid(),
                "gpu_id": gpu_id,
                "resumed_from_existing": result.status == "align_skipped_existing",
                "retried_failed_before": (
                    retry_failed_samples and result.status != "align_skipped_existing" and was_failed_before_retry
                ),
            }
        )
    return results


def _color_match_worker_entry(
    worker_config: dict[str, Any],
    pair_dicts: list[dict[str, str]],
    worker_index: int,
    gpu_id: str,
) -> list[dict[str, Any]]:
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    from pathlib import Path

    from livephoto2lrhr.algorithms.color_match import build_color_match_registry
    from livephoto2lrhr.data.pairing import SamplePair
    from livephoto2lrhr.stages.color_match import ColorMatchStage

    registry = build_color_match_registry()
    color_raw = dict(worker_config["color_match"])
    matcher = registry.create(str(color_raw["algorithm"]), color_raw["algorithm_config"])
    stage = ColorMatchStage(
        output_dir=Path(str(worker_config["output_dir"])),
        output_ext=str(worker_config["output_ext"]),
        input_folder=str(color_raw["input_folder"]),
        output_folder=str(color_raw["output_folder"]),
        overwrite=bool(worker_config["overwrite"]),
        save_metadata=bool(worker_config["save_metadata"]),
        matcher=matcher,
        algorithm_name=str(color_raw["algorithm"]),
        algorithm_config=color_raw["algorithm_config"],
        confidence_threshold=float(color_raw["confidence_threshold"]),
        on_failure=str(color_raw["on_failure"]),
        device=str(color_raw["device"]),
    )
    retry_failed_samples = bool(worker_config["retry_failed_samples"])
    output_dir = Path(str(worker_config["output_dir"]))
    results: list[dict[str, Any]] = []
    for pair_dict in pair_dicts:
        pair = sample_pair_from_dict(pair_dict)
        was_failed_before_retry = _was_failed_stage_output(output_dir, pair.relative_stem, "color_match")
        event_start = perf_counter()
        started_at = _timestamp_now()
        result = stage.run(pair, force_retry_failed=retry_failed_samples)
        finished_at = _timestamp_now()
        results.append(
            {
                "sample_id": result.sample_id,
                "status": result.status,
                "message": result.message,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": _elapsed_seconds(event_start),
                "error_traceback": result.error_traceback,
                "worker_index": worker_index,
                "worker_pid": os.getpid(),
                "gpu_id": gpu_id,
                "resumed_from_existing": result.status == "color_match_skipped_existing",
                "retried_failed_before": (
                    retry_failed_samples and result.status != "color_match_skipped_existing" and was_failed_before_retry
                ),
            }
        )
    return results


def _frame_select_worker_config(config: AppConfig) -> dict[str, Any]:
    return {
        "output_dir": str(config.data.output_dir),
        "output_ext": config.data.output_ext,
        "overwrite": config.output.overwrite,
        "save_metadata": config.output.save_metadata,
        "frame_select": {
            "algorithm": config.frame_select.algorithm,
            "sample_fps": config.frame_select.sample_fps,
            "top_k": config.frame_select.top_k,
            "batch_size": config.frame_select.batch_size,
            "resize_short_side": config.frame_select.resize_short_side,
            "score_fusion": config.frame_select.score_fusion,
            "device": config.frame_select.device,
        },
    }


def _align_worker_config(config: AppConfig, retry_failed_samples: bool) -> dict[str, Any]:
    return {
        "output_dir": str(config.data.output_dir),
        "output_ext": config.data.output_ext,
        "overwrite": config.output.overwrite,
        "save_metadata": config.output.save_metadata,
        "retry_failed_samples": retry_failed_samples,
        "align": {
            "algorithm": config.align.algorithm,
            "algorithm_config": _alignment_algorithm_config(config),
            "fallback_algorithm": config.align.fallback_algorithm,
            "fallback_algorithm_config": _alignment_algorithm_config(config),
            "output_folder": config.align.output_folder,
            "confidence_threshold": config.align.confidence_threshold,
            "on_failure": config.align.on_failure,
            "device": config.align.device,
        },
    }


def _color_match_worker_config(config: AppConfig, retry_failed_samples: bool) -> dict[str, Any]:
    return {
        "output_dir": str(config.data.output_dir),
        "output_ext": config.data.output_ext,
        "overwrite": config.output.overwrite,
        "save_metadata": config.output.save_metadata,
        "retry_failed_samples": retry_failed_samples,
        "color_match": {
            "algorithm": config.color_match.algorithm,
            "algorithm_config": _color_match_algorithm_config(config),
            "input_folder": config.color_match.input_folder,
            "output_folder": config.color_match.output_folder,
            "confidence_threshold": config.color_match.confidence_threshold,
            "on_failure": config.color_match.on_failure,
            "device": config.color_match.device,
        },
    }


def sample_pair_from_dict(pair_dict: dict[str, str]) -> SamplePair:
    return SamplePair(
        sample_id=pair_dict["sample_id"],
        image_path=Path(pair_dict["image_path"]),
        video_path=Path(pair_dict["video_path"]),
        relative_stem=Path(pair_dict["relative_stem"]),
    )
