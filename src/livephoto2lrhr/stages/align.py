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
        algorithm_config: dict[str, Any] | None = None,
        fallback_aligner: Aligner | None = None,
        fallback_algorithm_name: str | None = None,
        fallback_algorithm_config: dict[str, Any] | None = None,
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
        self.algorithm_config = algorithm_config or {}
        self.fallback_aligner = fallback_aligner
        self.fallback_algorithm_name = fallback_algorithm_name
        self.fallback_algorithm_config = fallback_algorithm_config or {}
        self.confidence_threshold = confidence_threshold
        self.on_failure = on_failure
        self.device = device

    def run(self, pair: SamplePair, force_retry_failed: bool = False) -> AlignStageResult:
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
        metadata: dict[str, Any] = {}
        if self.save_metadata and meta_path.exists():
            metadata = self._read_metadata(meta_path)
        retry_failed = force_retry_failed and self._should_retry_failed(metadata)
        if not self.overwrite and aligned_path.exists() and not retry_failed:
            return AlignStageResult(sample_id=pair.sample_id, status="align_skipped_existing")

        lr_rgb = None
        try:
            lr_rgb = read_rgb_array(lr_path)
            hr_rgb = read_rgb_array(hr_path)
            align_result, result_algorithm = self._run_alignment(
                aligner=self.aligner,
                algorithm_name=self.algorithm_name,
                algorithm_config=self.algorithm_config,
                lr_rgb=lr_rgb,
                hr_rgb=hr_rgb,
                pair=pair,
                lr_path=lr_path,
                hr_path=hr_path,
                metadata=metadata,
            )
        except Exception as exc:
            align_result = AlignResult(
                aligned_lr_rgb=None,
                status="failed",
                confidence=0.0,
                message=str(exc),
            )
            result_algorithm = self.algorithm_name

        accepted = align_result.status == "success" and align_result.confidence >= self.confidence_threshold
        fallback_used = False
        if not accepted and self.fallback_aligner is not None:
            try:
                fallback_result, fallback_algorithm = self._run_alignment(
                    aligner=self.fallback_aligner,
                    algorithm_name=self.fallback_algorithm_name or "fallback",
                    algorithm_config=self.fallback_algorithm_config,
                    lr_rgb=lr_rgb,
                    hr_rgb=hr_rgb,
                    pair=pair,
                    lr_path=lr_path,
                    hr_path=hr_path,
                    metadata=metadata,
                )
                fallback_accepted = (
                    fallback_result.status == "success" and fallback_result.confidence >= self.confidence_threshold
                )
                if fallback_accepted:
                    align_result = self._with_fallback_diagnostics(fallback_result, fallback_from=result_algorithm)
                    result_algorithm = fallback_algorithm
                    accepted = True
                    fallback_used = True
            except Exception as exc:
                align_result = self._with_fallback_failure(align_result, str(exc))

        low_confidence = align_result.status == "success" and align_result.confidence < self.confidence_threshold
        if accepted:
            final_status = "align_success"
            metadata_status = "success"
        elif low_confidence:
            final_status = "align_skipped_low_confidence"
            metadata_status = "skipped_low_confidence"
        else:
            final_status = "align_failed"
            metadata_status = align_result.status

        output_rgb = align_result.aligned_lr_rgb
        if not accepted and self.on_failure == "skip":
            output_rgb = None
        elif not accepted and self.on_failure == "keep_original":
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
                    algorithm_name=result_algorithm,
                    metadata_status=metadata_status,
                    fallback_used=fallback_used,
                )
        except Exception as exc:
            return AlignStageResult(sample_id=pair.sample_id, status="align_write_failed", message=str(exc))

        return AlignStageResult(sample_id=pair.sample_id, status=final_status, message=align_result.message)

    def _run_alignment(
        self,
        *,
        aligner: Aligner,
        algorithm_name: str,
        algorithm_config: dict[str, Any],
        lr_rgb,
        hr_rgb,
        pair: SamplePair,
        lr_path: Path,
        hr_path: Path,
        metadata: dict[str, Any],
    ) -> tuple[AlignResult, str]:
        if lr_rgb is None or hr_rgb is None:
            raise RuntimeError("phase 1 images were not loaded")
        context = AlignmentContext(
            sample_id=pair.sample_id,
            lr_path=lr_path,
            hr_path=hr_path,
            metadata=metadata,
            config=algorithm_config,
            artifact_root=self.output_dir / "artifacts" / "alignment" / pair.relative_stem,
            device=self.device,
        )
        return aligner.align(lr_rgb, hr_rgb, context), algorithm_name

    def _with_fallback_diagnostics(self, result: AlignResult, *, fallback_from: str) -> AlignResult:
        diagnostics = dict(result.diagnostics)
        diagnostics["fallback_used"] = True
        diagnostics["fallback_from"] = fallback_from
        return AlignResult(
            aligned_lr_rgb=result.aligned_lr_rgb,
            status=result.status,
            confidence=result.confidence,
            message=result.message,
            transforms=result.transforms,
            artifacts=result.artifacts,
            diagnostics=diagnostics,
        )

    def _with_fallback_failure(self, result: AlignResult, message: str) -> AlignResult:
        diagnostics = dict(result.diagnostics)
        diagnostics["fallback_used"] = False
        diagnostics["fallback_error"] = message
        return AlignResult(
            aligned_lr_rgb=result.aligned_lr_rgb,
            status=result.status,
            confidence=result.confidence,
            message=result.message,
            transforms=result.transforms,
            artifacts=result.artifacts,
            diagnostics=diagnostics,
        )

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
        algorithm_name: str,
        metadata_status: str,
        fallback_used: bool,
    ) -> None:
        status = metadata.setdefault("status", {})
        status["aligned"] = aligned
        output = {"lr_aligned": str(aligned_path)} if aligned_path is not None else {}
        diagnostics = dict(align_result.diagnostics)
        if fallback_used:
            diagnostics["fallback_used"] = True
        metadata["align"] = {
            "algorithm": algorithm_name,
            "status": metadata_status,
            "confidence": float(align_result.confidence),
            "message": align_result.message,
            "output": output,
            "transforms": align_result.transforms,
            "artifacts": align_result.artifacts,
            "diagnostics": diagnostics,
        }
        write_yaml(meta_path, metadata)

    def _should_retry_failed(self, metadata: dict[str, Any]) -> bool:
        align = metadata.get("align")
        if not isinstance(align, dict):
            return False
        return str(align.get("status", "")).lower() == "failed"
