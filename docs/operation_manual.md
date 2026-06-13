# livePhoto2LRHR Operation Manual

This manual describes the recommended production workflow for building LR/HR super-resolution training pairs from Live Photo-style image/video pairs.

The project is intentionally organized as a configurable Python pipeline:

```text
image + mp4
  -> phase 1 frame selection
  -> phase 2 alignment
  -> optional phase 3 color matching
  -> quality report
  -> final dataset export
```

Original `LR` and `HR` outputs are never overwritten by later stages. Each later stage writes its own folder, so failed experiments can be discarded safely.

## 1. Environment

Use Python 3.10 or newer.

CPU/OpenCV baseline:

```bash
python -m pip install -e .[dev]
```

DINOv2 frame selector:

```bash
python -m pip install -e .[dev,dinov2]
```

Run tests:

```bash
python -m pytest -q
```

On Linux, create the environment in the repo root and run the same commands. If you use CUDA, install a PyTorch build matching your CUDA driver before installing this project.

## 2. Input Layout

Put still images and videos in one input directory. A pair is matched by the same relative stem:

```text
input/
  trip_a/
    IMG_0001.jpg
    IMG_0001.mp4
    IMG_0002.jpg
    IMG_0002.mp4
```

The pipeline will write mirrored output folders:

```text
output/
  LR/
  HR/
  metadata/
  run_summary.yaml
```

Nested input folders are preserved. For example `input/trip_a/IMG_0001.jpg` becomes `output/HR/trip_a/IMG_0001.png`.

## 3. Configuration

Start from:

```text
configs/full_pipeline_template.yaml
```

Copy it to a local file that is not shared across machines, then edit:

```yaml
data:
  input_dir: /path/to/input
  output_dir: /path/to/output
```

Windows paths can use forward slashes:

```yaml
input_dir: D:/datasets/livephoto/input
output_dir: D:/datasets/livephoto/output
```

Linux paths are normal absolute paths:

```yaml
input_dir: /data/livephoto/input
output_dir: /data/livephoto/output
```

Avoid committing machine-specific paths unless the config is explicitly a smoke-test file.

## 4. Recommended First Run

For a new dataset, run the full baseline pipeline:

```bash
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

If the package is not installed as editable yet, use:

```bash
python -m pip install -e .[dev,dinov2]
livephoto2lrhr --config configs/full_pipeline_template.yaml
```

This runs:

```yaml
pipeline:
  stages:
    - frame_select
    - align
```

It also generates a report and exports accepted samples when `report.enabled: true` and `export.enabled: true`.

## 5. Phase 1: Frame Selection

Phase 1 reads each MP4 and chooses one LR frame that best matches the still image.

Recommended default:

```yaml
frame_select:
  algorithm: dinov2_similarity
  device: auto
  sample_fps: 15
  top_k: 5
  batch_size: 16
  resize_short_side: 518
```

Outputs:

```text
output/
  LR/
  HR/
  metadata/
```

`LR` contains only the selected best frame. Top-k candidates are metadata only; they are not written into the LR folder.

CPU-only smoke option:

```yaml
frame_select:
  algorithm: opencv_similarity
```

Use this only for fast validation. For real dataset generation, prefer `dinov2_similarity`.

## 6. Phase 2: Alignment

Phase 2 aligns selected LR to HR geometry.

Recommended baseline:

```yaml
align:
  enabled: true
  algorithm: coarse_to_flow
  output_folder: LR_aligned_flow
  confidence_threshold: 0.3
  fallback_algorithm: identity_alignment
  on_failure: keep_original
  coarse_algorithm: phase_correlation_translation
  optical_flow:
    enabled: true
    algorithm: dis
```

Outputs:

```text
output/
  LR_aligned_flow/
  artifacts/
  metadata/
```

The original `LR/` and `HR/` are not modified.

Available aligners:

```text
identity_alignment
phase_correlation_translation
ecc_alignment
coarse_to_flow
```

`coarse_to_flow` first runs a coarse aligner, then accepts dense flow only when it improves measured error. If the flow result is worse, the coarse result is kept.

## 7. Phase 3: Color Matching

Phase 3 is optional. It is useful for experiments but is not recommended as the default final LR source yet.

```yaml
color_match:
  enabled: true
  algorithm: mean_std_lab
  input_folder: auto
  output_folder: LR_color_matched
```

`input_folder: auto` prefers `LR_aligned` and falls back to `LR`. If your alignment output is `LR_aligned_flow`, set:

```yaml
color_match:
  input_folder: LR_aligned_flow
```

Current color matchers:

```text
identity_color_match
mean_std_lab
```

The baseline `mean_std_lab` can improve brightness/color in some samples and worsen others. Keep it as an experiment until quality gates are stronger.

## 8. Quality Report

Enable:

```yaml
report:
  enabled: true
  output_folder: reports_flow
  aligned_folder: LR_aligned_flow
  color_matched_folder: LR_color_matched
```

Outputs:

```text
output/reports_flow/
  quality_report.csv
  preview_contact_sheet.jpg
```

Important CSV columns:

```text
sample_id
frame_index
timestamp_sec
frame_select_score
align_status
align_confidence
flow_status
lr_to_hr_mae
aligned_to_hr_mae
color_matched_to_hr_mae
lr_path
aligned_path
color_matched_path
hr_path
```

The report is intentionally plain CSV so you can build your own viewer, dashboard, or manual review tool around it.

## 9. Final Dataset Export

Export reads a quality report and copies accepted samples into a final training dataset:

```yaml
export:
  enabled: true
  input_report: reports_flow/quality_report.csv
  output_folder: final_flow
  lr_source: aligned
  min_align_confidence: 0.3
  require_align_status: success
  require_flow_status: accepted
  max_source_to_hr_mae: 30.0
```

Outputs:

```text
output/final_flow/
  LR/
  HR/
  manifest.csv
```

`manifest.csv` records every accepted and rejected sample with a reason:

```text
accepted
align_status_mismatch
align_confidence_below_min
flow_status_mismatch
missing_lr_source
missing_hr
source_to_hr_mae_above_max
destination_exists
```

Recommended first export policy:

```yaml
lr_source: aligned
require_align_status: success
require_flow_status: accepted
max_source_to_hr_mae: 30.0
```

Tune `max_source_to_hr_mae` after inspecting your dataset distribution. A lower threshold is cleaner but rejects more samples.

## 10. Running Stages Separately

You can run stages in separate passes.

Phase 1 only:

```yaml
pipeline:
  stages:
    - frame_select
```

Alignment only, assuming phase 1 already exists:

```yaml
pipeline:
  stages:
    - align
```

Export only, assuming a report already exists:

```yaml
pipeline:
  stages: []

report:
  enabled: false

export:
  enabled: true
```

This is useful when tuning export thresholds. You do not need to rerun DINOv2 or optical flow just to change final dataset filtering.

## 11. Linux Notes

Use forward-slash absolute paths in YAML.

Install system libraries needed by OpenCV/Pillow if your Linux image is minimal:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

For headless servers, `opencv-python` is usually enough for this project because it does not open GUI windows. If your environment has GUI library conflicts, replace it with `opencv-python-headless` in your environment.

For GPU DINOv2, verify PyTorch first:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

Then run with:

```yaml
frame_select:
  device: cuda
```

## 12. Common Problems

No pairs found:

```text
Check that image and video share the same relative stem.
Example: input/a/IMG_0001.jpg and input/a/IMG_0001.mp4
```

DINOv2 fails on first run:

```text
The selector currently uses torch.hub. Make sure the machine has network access or a populated torch cache.
```

Alignment output is missing:

```text
Check output/LR and output/HR exist first. Alignment consumes phase 1 outputs.
```

Export rejects too many samples:

```text
Open manifest.csv and group by reason. Then tune max_source_to_hr_mae, min_align_confidence, or require_flow_status.
```

Chinese or non-ASCII paths look corrupted in terminal:

```text
Prefer UTF-8 terminals and forward-slash YAML paths. The pipeline uses Python pathlib and supports Unicode paths, but terminal display can still be misconfigured.
```

## 13. Current Baseline Status

The current project is a usable baseline:

```text
Phase 1: DINOv2/OpenCV frame selection
Phase 2: identity, phase correlation, ECC, coarse-to-flow alignment
Phase 3: identity and mean/std LAB color matching
Report: CSV and contact sheet
Export: quality-gated final LR/HR dataset
```

The baseline is designed to be replaceable. Algorithms live behind registries, and pipeline behavior is YAML-driven.

## 14. Advanced Quality Gates

The current export gate supports:

```text
align_status
flow_status
align_confidence
source-to-HR MAE
file existence
destination overwrite safety
```

Recommended next quality metrics:

```text
SSIM
PSNR
edge similarity
crop/border artifact score
dimension and aspect-ratio checks
flow magnitude outlier checks
local patch error percentiles
face/foreground region weighted error
```

Suggested implementation shape:

```text
reports/quality.py
  add metric columns

export/dataset.py
  add optional thresholds

configs/*.yaml
  expose thresholds
```

Keep all gates optional. Different photo sets will need different thresholds.

## 15. Advanced Alignment Roadmap

The current alignment baselines are useful but not final.

Recommended next aligners:

```text
LoFTR or LightGlue feature matching
RAFT or GMFlow optical flow
SAM-style foreground/object masks
homography + local flow hybrid
fusion controller that chooses per sample
```

The expected contract should remain:

```text
LR image + HR image + metadata + config -> AlignResult
```

Each advanced aligner should return:

```text
aligned_lr_rgb
confidence
status
transforms
artifacts
diagnostics
```

Do not special-case advanced models in the runner. Add them to the alignment registry and configure them by YAML.

## 16. Advanced Color Roadmap

The current `mean_std_lab` matcher is a baseline only.

Recommended next color methods:

```text
histogram matching
masked foreground/background color matching
Retinex-style illumination correction
3D LUT fitting
small neural color transfer model
exposure/white-balance regression
```

Color matching should stay optional until quality gates prove it improves final training data.

## 17. Recommended Production Checklist

Before treating a generated dataset as training-ready:

```text
Run full tests.
Run frame selection on a small subset.
Inspect metadata top-k candidates.
Run alignment and report.
Check report metric distributions.
Export with conservative gates.
Inspect manifest rejection reasons.
Manually sample accepted and rejected pairs.
Freeze the config used for the dataset.
Record git commit and run_summary.yaml.
```

The most important reproducibility files are:

```text
config YAML
run_summary.yaml
metadata/
reports*/quality_report.csv
final*/manifest.csv
git commit SHA
```
