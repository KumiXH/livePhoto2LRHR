from __future__ import annotations

from collections import Counter
from typing import Any

from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.config import AppConfig
from livephoto2lrhr.data.io import write_yaml
from livephoto2lrhr.data.pairing import discover_pairs
from livephoto2lrhr.stages.frame_select import FrameSelectStage


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    pair_result = discover_pairs(
        config.data.input_dir,
        image_exts=config.data.image_exts,
        video_exts=config.data.video_exts,
        recursive=config.data.recursive,
    )
    counts: Counter[str] = Counter({"success": 0})
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
