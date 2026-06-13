from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from livephoto2lrhr.algorithms.color_match.base import ColorMatcher, ColorMatchContext, ColorMatchResult
from livephoto2lrhr.data.io import (
    metadata_path,
    output_image_path,
    read_rgb_array,
    save_rgb_array,
    write_yaml,
)
from livephoto2lrhr.data.pairing import SamplePair


@dataclass(frozen=True)
class ColorMatchStageResult:
    sample_id: str
    status: str
    message: str = ""


class ColorMatchStage:
    def __init__(
        self,
        *,
        output_dir: Path,
        output_ext: str,
        input_folder: str,
        output_folder: str,
        overwrite: bool,
        save_metadata: bool,
        matcher: ColorMatcher,
        algorithm_name: str,
        algorithm_config: dict[str, Any] | None = None,
        confidence_threshold: float = 0.0,
        on_failure: str,
        device: str,
    ) -> None:
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.overwrite = overwrite
        self.save_metadata = save_metadata
        self.matcher = matcher
        self.algorithm_name = algorithm_name
        self.algorithm_config = algorithm_config or {}
        self.confidence_threshold = confidence_threshold
        self.on_failure = on_failure
        self.device = device

    def run(self, pair: SamplePair) -> ColorMatchStageResult:
        lr_path = self._input_lr_path(pair)
        hr_path = output_image_path(self.output_dir, "HR", pair.relative_stem, self.output_ext)
        meta_path = metadata_path(self.output_dir, pair.relative_stem)
        matched_path = output_image_path(self.output_dir, self.output_folder, pair.relative_stem, self.output_ext)

        if lr_path is None or not hr_path.exists() or (self.save_metadata and not meta_path.exists()):
            return ColorMatchStageResult(
                sample_id=pair.sample_id,
                status="color_match_skipped_missing_input",
                message="missing phase input",
            )
        if not self.overwrite and matched_path.exists():
            return ColorMatchStageResult(sample_id=pair.sample_id, status="color_match_skipped_existing")

        metadata: dict[str, Any] = {}
        lr_rgb = None
        try:
            lr_rgb = read_rgb_array(lr_path)
            hr_rgb = read_rgb_array(hr_path)
            if self.save_metadata and meta_path.exists():
                metadata = self._read_metadata(meta_path)
            context = ColorMatchContext(
                sample_id=pair.sample_id,
                lr_path=lr_path,
                hr_path=hr_path,
                metadata=metadata,
                config=self.algorithm_config,
                artifact_root=self.output_dir / "artifacts" / "color_match" / pair.relative_stem,
                device=self.device,
            )
            match_result = self.matcher.match(lr_rgb, hr_rgb, context)
        except Exception as exc:
            match_result = ColorMatchResult(
                matched_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
            )

        accepted = match_result.status == "success" and match_result.confidence >= self.confidence_threshold
        low_confidence = match_result.status == "success" and match_result.confidence < self.confidence_threshold
        if accepted:
            final_status = "color_match_success"
            metadata_status = "success"
        elif low_confidence:
            final_status = "color_match_skipped_low_confidence"
            metadata_status = "skipped_low_confidence"
        else:
            final_status = "color_match_failed"
            metadata_status = match_result.status
        output_rgb = match_result.matched_lr_rgb
        if not accepted and self.on_failure == "skip":
            output_rgb = None
        elif not accepted and self.on_failure == "keep_original":
            output_rgb = lr_rgb

        try:
            if output_rgb is not None:
                save_rgb_array(output_rgb, matched_path)
            if self.save_metadata:
                self._write_color_match_metadata(
                    meta_path=meta_path,
                    metadata=metadata,
                    input_lr_path=lr_path,
                    matched_path=matched_path if output_rgb is not None else None,
                    match_result=match_result,
                    matched=accepted,
                    metadata_status=metadata_status,
                )
        except Exception as exc:
            return ColorMatchStageResult(sample_id=pair.sample_id, status="color_match_write_failed", message=str(exc))

        return ColorMatchStageResult(sample_id=pair.sample_id, status=final_status, message=match_result.message)

    def _input_lr_path(self, pair: SamplePair) -> Path | None:
        if self.input_folder == "auto":
            aligned_path = output_image_path(self.output_dir, "LR_aligned", pair.relative_stem, self.output_ext)
            if aligned_path.exists():
                return aligned_path
            lr_path = output_image_path(self.output_dir, "LR", pair.relative_stem, self.output_ext)
            return lr_path if lr_path.exists() else None
        lr_path = output_image_path(self.output_dir, self.input_folder, pair.relative_stem, self.output_ext)
        return lr_path if lr_path.exists() else None

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _write_color_match_metadata(
        self,
        *,
        meta_path: Path,
        metadata: dict[str, Any],
        input_lr_path: Path,
        matched_path: Path | None,
        match_result: ColorMatchResult,
        matched: bool,
        metadata_status: str,
    ) -> None:
        status = metadata.setdefault("status", {})
        status["color_matched"] = matched
        output = {"lr_color_matched": str(matched_path)} if matched_path is not None else {}
        metadata["color_match"] = {
            "algorithm": self.algorithm_name,
            "status": metadata_status,
            "confidence": float(match_result.confidence),
            "message": match_result.message,
            "input": {"lr": str(input_lr_path)},
            "output": output,
            "transforms": match_result.transforms,
            "artifacts": match_result.artifacts,
            "diagnostics": match_result.diagnostics,
        }
        write_yaml(meta_path, metadata)
