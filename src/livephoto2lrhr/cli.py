from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace

from livephoto2lrhr.config import load_config
from livephoto2lrhr.pipeline.runner import run_pipeline
from livephoto2lrhr.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LR/HR pairs from Live Photo images and videos.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["frame_select", "align", "color_match"],
        help="Override pipeline stages from config for this run.",
    )
    parser.add_argument("--num-workers", type=int, help="Override runtime.parallel.num_workers for this run.")
    parser.add_argument("--gpu-ids", nargs="+", help="Override runtime.parallel.gpu_ids for this run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.stages:
        updated_raw = dict(config.raw)
        updated_pipeline = dict(updated_raw.get("pipeline", {}))
        updated_pipeline["stages"] = list(args.stages)
        updated_raw["pipeline"] = updated_pipeline
        config = replace(
            config,
            pipeline=replace(config.pipeline, stages=tuple(args.stages)),
            raw=updated_raw,
        )
    if args.num_workers is not None or args.gpu_ids:
        updated_raw = dict(config.raw)
        runtime = dict(updated_raw.get("runtime", {}))
        parallel = dict(runtime.get("parallel", {}))
        if args.num_workers is not None:
            parallel["num_workers"] = args.num_workers
        if args.gpu_ids:
            parallel["gpu_ids"] = list(args.gpu_ids)
        runtime["parallel"] = parallel
        updated_raw["runtime"] = runtime
        config = replace(config, raw=updated_raw)
    summary = run_pipeline(config)
    logging.info("Completed run: %s", summary["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
