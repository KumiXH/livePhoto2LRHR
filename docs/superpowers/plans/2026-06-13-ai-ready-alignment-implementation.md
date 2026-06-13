# AI-Ready Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build phase 2 as an AI-ready alignment framework with OpenCV baseline aligners and pipeline integration.

**Architecture:** Alignment mirrors phase 1: config dataclasses, algorithm registry, focused stage orchestration, per-sample metadata, and mirrored output folders. The pipeline calls registered aligners through a stable `AlignResult` contract so future SAM/RAFT/LoFTR/LLM-fusion implementations can replace OpenCV aligners without changing stage or runner code.

**Tech Stack:** Python 3.10+, PyYAML, Pillow, OpenCV, NumPy, pytest.

---

## File Structure

- Modify `src/livephoto2lrhr/config.py`: add `AlignConfig`, nested artifact/ECC/phase-correlation/optical-flow config, and load defaults.
- Create `src/livephoto2lrhr/algorithms/alignment/__init__.py`: registry builder.
- Create `src/livephoto2lrhr/algorithms/alignment/base.py`: `AlignmentContext`, `TransformRecord`, `ArtifactRecord`, `AlignResult`, `Aligner` protocol.
- Create `src/livephoto2lrhr/algorithms/alignment/identity.py`: copy-through baseline.
- Create `src/livephoto2lrhr/algorithms/alignment/phase_correlation.py`: translation aligner.
- Create `src/livephoto2lrhr/algorithms/alignment/ecc.py`: ECC aligner.
- Modify `src/livephoto2lrhr/data/io.py`: add generic output path helper and image array read helper.
- Replace `src/livephoto2lrhr/stages/align.py`: real align stage.
- Modify `src/livephoto2lrhr/pipeline/runner.py`: run align after frame_select when configured.
- Modify `configs/frame_select.yaml`: add disabled align config block.
- Modify `README.md`: document phase 2 usage.
- Add tests:
  - `tests/test_alignment_config.py`
  - `tests/test_alignment_registry.py`
  - `tests/test_align_stage.py`
  - `tests/test_alignment_algorithms.py`
  - update `tests/test_runner.py`
  - update `tests/test_future_stages.py`

## Task 1: Alignment Config

- [ ] Write failing tests for default disabled align config, enabled align stage config, and validation of unknown pipeline stage behavior.
- [ ] Implement align config dataclasses and YAML loading.
- [ ] Run `python -m pytest tests/test_alignment_config.py tests/test_config.py -v`.
- [ ] Commit `feat: add alignment config`.

## Task 2: Alignment Types and Registry

- [ ] Write failing tests for alignment registry and identity aligner result shape.
- [ ] Implement `algorithms/alignment/base.py`, registry builder, and `identity_alignment`.
- [ ] Run `python -m pytest tests/test_alignment_registry.py -v`.
- [ ] Commit `feat: add alignment registry`.

## Task 3: Align Stage Outputs and Metadata

- [ ] Write failing tests that `AlignStage` reads `LR`/`HR`, writes mirrored `LR_aligned`, preserves dotted stems, updates metadata, and never overwrites original `LR`/`HR`.
- [ ] Implement image IO helpers and real `stages/align.py`.
- [ ] Run `python -m pytest tests/test_align_stage.py tests/test_frame_select_stage.py -v`.
- [ ] Commit `feat: write aligned lr outputs`.

## Task 4: Phase Correlation Aligner

- [ ] Write failing synthetic shifted-image test.
- [ ] Implement `phase_correlation_translation` with confidence, translation transform, and warp output.
- [ ] Run `python -m pytest tests/test_alignment_algorithms.py -v`.
- [ ] Commit `feat: add phase correlation alignment`.

## Task 5: ECC Aligner

- [ ] Write failing synthetic affine/translation test and graceful failure test.
- [ ] Implement `ecc_alignment` with configurable motion model and diagnostics.
- [ ] Run `python -m pytest tests/test_alignment_algorithms.py -v`.
- [ ] Commit `feat: add ecc alignment`.

## Task 6: Runner Integration

- [ ] Write failing runner tests for `frame_select -> align`, disabled align no-op, and align failure summary counts.
- [ ] Modify runner to run align stage after frame selection and write align counts into `run_summary.yaml`.
- [ ] Run `python -m pytest tests/test_runner.py tests/test_align_stage.py -v`.
- [ ] Commit `feat: run alignment stage`.

## Task 7: Docs and Real Smoke

- [ ] Update README and default YAML with disabled align block.
- [ ] Run `python -m pytest -v`.
- [ ] Run a real-data smoke with `identity_alignment` or `phase_correlation_translation` against `D:/SR数据集/花_pairs` without modifying original `LR`/`HR`.
- [ ] Verify `LR_aligned` mirrors successful outputs and metadata contains `align`.
- [ ] Commit `docs: document alignment stage`.

## Review Gates

- [ ] Run final `python -m pytest -v`.
- [ ] Request spec compliance review against `2026-06-13-ai-ready-alignment-design.md`.
- [ ] Request code-quality review.
- [ ] Fix any Critical/Important findings and rerun full tests.
