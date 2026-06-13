from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from livephoto2lrhr.algorithms.alignment.base import Aligner, AlignmentContext, AlignResult
from livephoto2lrhr.data.io import (
    metadata_path,
    output_image_path,
    read_rgb_array,
    save_rgb_array,
    write_yaml,
)
from livephoto2lrhr.data.pairing import SamplePair


@dataclass(frozen=True)
class AlignStageResult:
    sample_id: str
    status: str
    message: str = ""


class AlignStage:
    def __init__(
        self,
        *,
        output_dir: Path,
        output_ext: str,
        output_folder: str,
        overwrite: bool,
        save_metadata: bool,
        aligner: Aligner,
        algorithm_name: str,
        confidence_threshold: float,
        on_failure: str,
        device: str,
    ) -> None:
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.output_folder = output_folder
        self.overwrite = overwrite
        self.save_metadata = save_metadata
        self.aligner = aligner
        self.algorithm_name = algorithm_name
        self.confidence_threshold = confidence_threshold
        self.on_failure = on_failure
        self.device = device

    def run(self, pair: SamplePair) -> AlignStageResult:
        lr_path = output_image_path(self.output_dir, "LR", pair.relative_stem, self.output_ext)
        hr_path = output_image_path(self.output_dir, "HR", pair.relative_stem, self.output_ext)
        meta_path = metadata_path(self.output_dir, pair.relative_stem)
        aligned_path = output_image_path(self.output_dir, self.output_folder, pair.relative_stem, self.output_ext)

        if not lr_path.exists() or not hr_path.exists() or (self.save_metadata and not meta_path.exists()):
            return AlignStageResult(
                sample_id=pair.sample_id,
                status="align_skipped_missing_input",
                message="missing phase 1 output",
            )
        if not self.overwrite and aligned_path.exists():
            return AlignStageResult(sample_id=pair.sample_id, status="align_skipped_existing")

        try:
            metadata = self._read_metadata(meta_path)
            lr_rgb = read_rgb_array(lr_path)
            hr_rgb = read_rgb_array(hr_path)
            context = AlignmentContext(
                sample_id=pair.sample_id,
                lr_path=lr_path,
                hr_path=hr_path,
                metadata=metadata,
                config={},
                artifact_root=self.output_dir / "artifacts" / "alignment" / pair.relative_stem,
                device=self.device,
            )
            align_result = self.aligner.align(lr_rgb, hr_rgb, context)
        except Exception as exc:
            align_result = AlignResult(
                aligned_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
            )

        accepted = align_result.status == "success" and align_result.confidence >= self.confidence_threshold
        final_status = "align_success" if accepted else "align_failed"
        output_rgb = align_result.aligned_lr_rgb
        if not accepted and self.on_failure == "keep_original":
            output_rgb = lr_rgb

        try:
            if output_rgb is not None:
                save_rgb_array(output_rgb, aligned_path)
            if self.save_metadata:
                self._write_alignment_metadata(
                    meta_path=meta_path,
                    metadata=metadata,
                    aligned_path=aligned_path if output_rgb is not None else None,
                    align_result=align_result,
                    aligned=accepted,
                )
        except Exception as exc:
            return AlignStageResult(sample_id=pair.sample_id, status="align_write_failed", message=str(exc))

        return AlignStageResult(sample_id=pair.sample_id, status=final_status, message=align_result.message)

    def _read_metadata(self, path: Path) -> dict[str, Any]:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _write_alignment_metadata(
        self,
        *,
        meta_path: Path,
        metadata: dict[str, Any],
        aligned_path: Path | None,
        align_result: AlignResult,
        aligned: bool,
    ) -> None:
        status = metadata.setdefault("status", {})
        status["aligned"] = aligned
        output = {"lr_aligned": str(aligned_path)} if aligned_path is not None else {}
        metadata["align"] = {
            "algorithm": self.algorithm_name,
            "status": align_result.status,
            "confidence": float(align_result.confidence),
            "message": align_result.message,
            "output": output,
            "transforms": align_result.transforms,
            "artifacts": align_result.artifacts,
            "diagnostics": align_result.diagnostics,
        }
        write_yaml(meta_path, metadata)
