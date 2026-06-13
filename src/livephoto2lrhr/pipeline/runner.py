from __future__ import annotations

from collections import Counter
from typing import Any

from livephoto2lrhr.algorithms.alignment import build_alignment_registry
from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.config import AppConfig
from livephoto2lrhr.data.io import write_yaml
from livephoto2lrhr.data.pairing import discover_pairs
from livephoto2lrhr.stages.align import AlignStage
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


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    pair_result = discover_pairs(
        config.data.input_dir,
        image_exts=config.data.image_exts,
        video_exts=config.data.video_exts,
        recursive=config.data.recursive,
    )
    counts: Counter[str] = Counter({status: 0 for status in (*KNOWN_PHASE_1_STATUSES, *KNOWN_ALIGN_STATUSES)})
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
            aligner = registry.create(
                config.align.algorithm,
                {
                    "device": config.align.device,
                    "phase_correlation": config.align.phase_correlation,
                    "motion_model": config.align.ecc.motion_model,
                    "number_of_iterations": config.align.ecc.number_of_iterations,
                    "termination_eps": config.align.ecc.termination_eps,
                    "gaussian_filter_size": config.align.ecc.gaussian_filter_size,
                    "resize_short_side": config.align.phase_correlation.resize_short_side,
                    "optical_flow": config.align.optical_flow,
                    "artifacts": config.align.artifacts,
                },
            )
            stage = AlignStage(
                output_dir=config.data.output_dir,
                output_ext=config.data.output_ext,
                output_folder=config.align.output_folder,
                overwrite=config.output.overwrite,
                save_metadata=config.output.save_metadata,
                aligner=aligner,
                algorithm_name=config.align.algorithm,
                confidence_threshold=config.align.confidence_threshold,
                on_failure=config.align.on_failure,
                device=config.align.device,
            )
            for pair in pair_result.pairs:
                result = stage.run(pair)
                counts[result.status] += 1
                samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})

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
    write_yaml(config.data.output_dir / "run_summary.yaml", summary)
    return summary
