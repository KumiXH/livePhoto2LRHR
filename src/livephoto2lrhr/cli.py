from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from livephoto2lrhr.config import load_config
from livephoto2lrhr.pipeline.runner import run_pipeline
from livephoto2lrhr.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LR/HR pairs from Live Photo images and videos.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    summary = run_pipeline(config)
    logging.info("Completed run: %s", summary["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
