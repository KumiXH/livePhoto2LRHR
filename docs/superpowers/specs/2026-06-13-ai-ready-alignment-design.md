# AI-Ready Alignment Design

## Goal

Build phase 2 as an extensible alignment framework for LR/HR pairs. The first implementation should support practical OpenCV-based alignment, while the architecture must allow future replacement with SAM-style mask alignment, RAFT/LoFTR/LightGlue-style deep alignment, or an LLM/fusion controller without rewriting the pipeline.

## Current Baseline

Phase 1 is merged into `master` and writes:

```text
output/
  LR/
  HR/
  metadata/
  run_summary.yaml
```

`LR` contains exactly one selected MP4 frame per sample. `HR` contains the corresponding still image. Metadata already records `frame_select.selected`, `frame_select.top_k`, and placeholder status flags:

```yaml
status:
  aligned: false
  color_matched: false
```

Phase 2 starts from those outputs. It must not overwrite original `LR` or `HR`.

## Core Principle

Alignment is a strategy, not one algorithm.

The pipeline should know only this contract:

```text
LR image + HR image + sample metadata + config -> AlignResult
```

The pipeline must not care whether the strategy uses ORB, ECC, optical flow, SAM masks, RAFT, LoFTR, LightGlue, or a future fusion network.

## Alignment Interface

Every alignment algorithm implements the same conceptual interface:

```python
class Aligner:
    def align(self, lr_rgb, hr_rgb, context) -> AlignResult:
        ...
```

`context` includes:

- `sample_id`
- source LR and HR paths
- current metadata dictionary
- algorithm config
- output artifact root
- device preference: `cpu`, `cuda`, or `auto`

`AlignResult` includes:

- `aligned_lr_rgb`: optional RGB image written to `LR_aligned`
- `confidence`: float from `0.0` to `1.0`
- `status`: `success`, `skipped_low_confidence`, or `failed`
- `message`: short failure or skip reason
- `transforms`: list of transform records
- `artifacts`: list of saved artifact records
- `diagnostics`: YAML-safe diagnostics

The interface intentionally allows matrix transforms, dense flow, masks, debug overlays, or future model outputs to coexist.

## Transform Records

Transform records are metadata dictionaries with a required `type` field.

Examples:

```yaml
transforms:
  - type: identity
  - type: affine
    matrix:
      - [1.0, 0.0, 2.5]
      - [0.0, 1.0, -1.0]
    coordinate_system: lr_to_hr
  - type: homography
    matrix:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
    inlier_ratio: 0.72
  - type: dense_flow
    path: artifacts/alignment/IMG_0001/flow.npz
    coordinate_system: lr_to_hr
```

This keeps compatibility with classic and AI aligners. A future LLM/fusion controller can output multiple transform records and explain the selected route in diagnostics.

## Artifact Records

Artifact records describe optional saved files:

```yaml
artifacts:
  - type: mask
    role: foreground
    path: artifacts/alignment/IMG_0001/hr_mask.png
  - type: flow
    path: artifacts/alignment/IMG_0001/flow.npz
  - type: debug_overlay
    path: artifacts/alignment/IMG_0001/overlay.png
```

Artifacts are configurable. The default should save metadata and aligned LR, but not heavy flow tensors unless enabled.

## Output Layout

Phase 2 adds:

```text
output/
  LR/
  LR_aligned/
  HR/
  metadata/
  artifacts/
    alignment/
  run_summary.yaml
```

`LR` and `HR` remain untouched. `LR_aligned` mirrors `LR` and `HR` only for successful or accepted alignment outputs.

Metadata for each sample gains:

```yaml
align:
  algorithm: coarse_to_flow
  status: success
  confidence: 0.87
  message: ""
  output:
    lr_aligned: D:/.../LR_aligned/IMG_0001.png
  transforms: [...]
  artifacts: [...]
  diagnostics:
    fallback_used: false
    coarse_algorithm: ecc
    flow_algorithm: dis
```

`status.aligned` becomes `true` only when an aligned LR output was written and accepted.

## First Implementation Scope

The first implementation should build the framework and provide these strategies:

1. `identity_alignment`

   Writes LR to `LR_aligned` without geometric changes. This validates the stage, output layout, metadata, skip/overwrite behavior, and algorithm registry.

2. `phase_correlation_translation`

   Estimates global x/y translation on grayscale resized images and warps LR into HR coordinates. This is simple, fast, and robust enough as a baseline.

3. `ecc_alignment`

   Uses OpenCV ECC image registration with configurable motion model: `translation`, `euclidean`, `affine`, or `homography`. Default should be `affine`.

4. `coarse_to_flow`

   Runs a configurable coarse aligner first, then optional local optical flow refinement using OpenCV DIS or Farneback. The initial version can implement DIS/Farneback only if dependency support is present in installed OpenCV; otherwise it records a clear unsupported status and falls back to the coarse result.

Deep learning aligners are not implemented in the first pass. They are reserved through the same registry and config shape.

## Future Plugin Slots

These algorithm names are reserved but should not be implemented yet:

- `sam_masked_alignment`
- `raft_flow_alignment`
- `loftr_feature_alignment`
- `lightglue_feature_alignment`
- `llm_fusion_alignment`

Future implementations may add model configs:

```yaml
align:
  algorithm: sam_masked_alignment
  device: cuda
  model:
    source: local
    path: D:/models/sam3
  artifacts:
    save_masks: true
```

The key compatibility requirement is that they still return `AlignResult`.

## Fallback and Confidence

Alignment must be allowed to fail per sample without crashing the batch.

Config:

```yaml
align:
  enabled: true
  algorithm: coarse_to_flow
  fallback_algorithm: phase_correlation_translation
  confidence_threshold: 0.3
  on_failure: keep_original
```

Behavior:

- If primary aligner succeeds with confidence >= threshold: write `LR_aligned`, mark aligned true.
- If primary aligner fails and fallback succeeds: write fallback output, mark `fallback_used: true`.
- If all aligners fail and `on_failure: keep_original`: copy original LR to `LR_aligned`, mark `align.status: failed`, `status.aligned: false`.
- If all aligners fail and `on_failure: skip`: do not write `LR_aligned`, mark failed.

Default should be conservative:

```yaml
align:
  enabled: false
```

The user explicitly enables it after inspecting phase 1 outputs.

## Quality Signals

Every aligner should report at least:

- `confidence`
- `pre_alignment_error` when measurable
- `post_alignment_error` when measurable
- algorithm-specific metrics such as ECC score, inlier ratio, translation magnitude, flow magnitude, or mask coverage

This is important because alignment can make a pair worse. The stage should expose enough information for a later quality report and preview tool.

## Coordinate Convention

The default coordinate convention is `lr_to_hr`: transform LR into HR coordinate space while preserving HR as the target.

This keeps the training pair intuitive:

```text
LR_aligned -> HR
```

The framework may later support `hr_to_lr`, but first implementation should not expose it as default.

## YAML Configuration

Config additions:

```yaml
pipeline:
  stages:
    - frame_select
    - align

align:
  enabled: false
  algorithm: identity_alignment
  device: auto
  output_folder: LR_aligned
  confidence_threshold: 0.3
  fallback_algorithm: identity_alignment
  on_failure: keep_original
  artifacts:
    save_debug_overlay: false
    save_flow: false
    save_masks: false
  phase_correlation:
    resize_short_side: 512
  ecc:
    motion_model: affine
    number_of_iterations: 100
    termination_eps: 1.0e-5
    gaussian_filter_size: 5
  optical_flow:
    enabled: false
    algorithm: dis
```

## Error Handling

The align stage records per-sample failures in metadata and in `run_summary.yaml`.

Known statuses:

- `align_success`
- `align_skipped_disabled`
- `align_skipped_missing_input`
- `align_skipped_low_confidence`
- `align_failed`
- `align_write_failed`

Frame selection statuses remain unchanged.

## Testing Strategy

Tests should cover:

- Config loading defaults and validation.
- Registry creation for aligners.
- Identity alignment output and metadata.
- Dotted relative stems remain preserved in `LR_aligned` and metadata.
- Translation aligner recovers a synthetic shifted image.
- ECC aligner handles synthetic small affine shifts when available.
- Failure fallback writes or skips according to config.
- Pipeline can run `frame_select -> align` on synthetic media.
- Original `LR` and `HR` are not modified.

Real-data smoke should run with `identity_alignment` first, then with a conservative OpenCV aligner on a few samples.

## Non-Goals

The first implementation does not:

- train a neural network
- implement SAM, RAFT, LoFTR, LightGlue, or LLM fusion
- modify HR images
- overwrite phase 1 LR outputs
- require GPU for OpenCV aligners
- decide final dataset filtering policy

## Open Design Decisions

The first pass chooses `LR_aligned -> HR` as the default coordinate convention. If later experiments show HR cropping gives better super-resolution training pairs, that should be added as another strategy rather than changing the current contract.

The first pass records quality metrics but does not automatically discard samples beyond `confidence_threshold`.
