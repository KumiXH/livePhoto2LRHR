# livePhoto2LRHR

Build paired `LR` and `HR` folders from Live Photo-style image/video pairs for super-resolution training.

The pipeline is YAML-driven and organized into replaceable stages:

```text
image + mp4
  -> frame selection
  -> alignment
  -> optional color matching
  -> quality report
  -> final dataset export
```

For the full production workflow, see [docs/operation_manual.md](docs/operation_manual.md).

## Install

DINOv2 frame selector:

```bash
python -m pip install -e .[dev,dinov2]
```

CPU/OpenCV baseline:

```bash
python -m pip install -e .[dev]
```

Run tests:

```bash
python -m pytest -q
```

## Quick Start

Copy and edit the full pipeline template:

```text
configs/full_pipeline_template.yaml
```

Set absolute input/output paths:

```yaml
data:
  input_dir: /path/to/input
  output_dir: /path/to/output
```

Then run:

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

Use forward-slash paths in YAML on both Windows and Linux, for example `D:/datasets/input` or `/data/datasets/input`.

## Input Contract

Image/video pairs are matched by the same relative stem:

```text
input/trip/IMG_0001.jpg
input/trip/IMG_0001.mp4
```

## Output Contract

The pipeline preserves mirrored LR/HR folder structure:

```text
output/
  LR/
  HR/
  metadata/
  LR_aligned_flow/
  reports_flow/
  final_flow/
    LR/
    HR/
    manifest.csv
  run_summary.yaml
```

Original `LR` and `HR` outputs are not overwritten by alignment, color matching, reports, or final export.

## Current Baseline

Available frame selectors:

```text
dinov2_similarity
opencv_similarity
fake_selector
```

Available aligners:

```text
identity_alignment
phase_correlation_translation
ecc_alignment
coarse_to_flow
```

Available color matchers:

```text
identity_color_match
mean_std_lab
```

Final export currently supports gates for `align_status`, `flow_status`, `align_confidence`, source-to-HR MAE, file existence, and overwrite safety.

## Documentation

Read [docs/operation_manual.md](docs/operation_manual.md) for:

- Windows and Linux setup.
- Recommended stage-by-stage workflow.
- Report and final export usage.
- Quality threshold tuning.
- Advanced quality gate roadmap.
- Advanced alignment and color matching roadmap.
