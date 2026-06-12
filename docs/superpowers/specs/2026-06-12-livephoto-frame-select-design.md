# Live Photo Frame Selection Dataset Pipeline Design

## Goal

Build a Python engineering project that converts Live Photo-style inputs into paired
super-resolution training data. The first phase focuses on selecting the best low
resolution frame from each video. Later phases will add geometric alignment and
color matching.

The user provides one input directory containing many photos and videos. Files with
the same stem are treated as a pair. For example, `IMG_0001.jpg` and `IMG_0001.mp4`
belong to the same sample.

## Scope

Phase 1 must implement:

- YAML-driven input and output paths.
- Recursive pairing of images and videos by matching relative path and filename stem.
- AI-based frame selection from MP4 or MOV video.
- Mirrored `LR` and `HR` output folder structures.
- Metadata that records top-k candidate frame indices and scores.
- No top-k candidate images saved into the `LR` folder. `LR` contains only the final
  selected frame for each sample.
- Cross-platform support for Windows and Linux.
- GPU support when the selected algorithm can use CUDA.

Phase 1 does not implement:

- Final geometric alignment.
- Optical flow refinement.
- Final brightness or color matching.
- Model training.

These are reserved for later stages but must be reflected in the project structure.

## Input Contract

The dataset input directory is configured in YAML.

```yaml
data:
  input_dir: D:/datasets/livephoto_raw
  output_dir: D:/datasets/livephoto_pairs
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".heic"]
  video_exts: [".mp4", ".mov"]
  output_ext: ".png"
```

Pairing rules:

- Match image and video by normalized relative path without extension.
- `input/IMG_0001.jpg` pairs with `input/IMG_0001.mp4`.
- `input/trip/a.jpg` pairs with `input/trip/a.mp4`.
- If multiple images or videos share the same stem, mark the sample as ambiguous and
  skip it unless a future config option defines a priority rule.
- Missing pairs are reported in a summary file and skipped.

## Output Contract

The output directory is configured in YAML.

```text
output/
  LR/
    IMG_0001.png
    trip/a.png
  HR/
    IMG_0001.png
    trip/a.png
  metadata/
    IMG_0001.yaml
    trip/a.yaml
```

Rules:

- `HR` stores the source photo converted to the configured output image format.
- `LR` stores exactly one selected video frame per paired sample.
- `LR` and `HR` must have identical relative file structures.
- `metadata` mirrors the same relative structure but uses `.yaml`.
- Top-k candidates are recorded in metadata only. They are not saved as extra LR images.
- Optional preview contact sheets may be added later under `previews/`, but they must
  never change the `LR` or `HR` folder contract.

Example metadata:

```yaml
sample_id: trip/a
source:
  image: D:/datasets/livephoto_raw/trip/a.jpg
  video: D:/datasets/livephoto_raw/trip/a.mp4
output:
  hr: D:/datasets/livephoto_pairs/HR/trip/a.png
  lr: D:/datasets/livephoto_pairs/LR/trip/a.png
frame_select:
  algorithm: dinov2_similarity
  selected:
    frame_index: 42
    timestamp_sec: 1.4
    score: 0.9123
  top_k:
    - frame_index: 42
      timestamp_sec: 1.4
      score: 0.9123
    - frame_index: 41
      timestamp_sec: 1.3667
      score: 0.9011
status:
  aligned: false
  color_matched: false
```

## Pipeline Architecture

The project should use a deep-learning-style Python layout:

```text
configs/
  frame_select.yaml
src/
  livephoto2lrhr/
    cli.py
    config.py
    pipeline/
      runner.py
      registry.py
    data/
      pairing.py
      io.py
    stages/
      frame_select.py
      align.py
      color_match.py
    algorithms/
      similarity/
        base.py
        dinov2.py
        clip.py
      alignment/
        base.py
        homography.py
        raft.py
      color/
        base.py
        histogram.py
        reinhard.py
    utils/
      device.py
      logging.py
tests/
```

Each stage owns orchestration. Each algorithm owns one replaceable strategy.

The pipeline stages are configured as a list:

```yaml
pipeline:
  stages:
    - frame_select
```

Future stages can be enabled without changing the phase 1 output contract:

```yaml
pipeline:
  stages:
    - frame_select
    - align
    - color_match
```

## Algorithm Interfaces

Frame selection algorithms must implement a common interface:

```python
class FrameSelector:
    def select(self, image_path, video_path, output_context):
        """Return selected frame and ranked candidate metadata."""
```

The return value must include:

- Selected frame image data or a pointer to the decoded frame.
- Selected frame index.
- Selected timestamp.
- Selected score.
- Top-k candidate frame indices, timestamps, and scores.
- Algorithm-specific diagnostics.

Alignment algorithms should later implement:

```python
class Aligner:
    def align(self, lr_image, hr_image, output_context):
        """Return aligned LR image and alignment metadata."""
```

Color matching algorithms should later implement:

```python
class ColorMatcher:
    def match(self, lr_image, hr_image, output_context):
        """Return color-adjusted LR image and color metadata."""
```

All algorithms are registered by string name so YAML can select them:

```yaml
frame_select:
  algorithm: dinov2_similarity
```

## Phase 1 Frame Selection

Recommended default:

```yaml
frame_select:
  algorithm: dinov2_similarity
  device: cuda
  sample_fps: 15
  top_k: 5
  batch_size: 16
  resize_short_side: 518
  score_fusion:
    feature_weight: 0.8
    edge_weight: 0.2
```

The initial algorithm should:

- Decode video frames with a stable library such as PyAV or OpenCV.
- Sample frames by configured FPS.
- Normalize HR photo and candidate LR frames to a comparable size.
- Extract visual features with DINOv2 when available.
- Score candidates by feature similarity.
- Optionally use edge similarity or SSIM as a secondary score for top candidates.
- Save only the best frame into `LR`.
- Save the photo into `HR`.
- Record all top-k frame indices and scores in metadata.

DINOv2 is preferred as the default first algorithm because it is more suitable for
instance-level visual matching than a purely semantic model. CLIP can be added as a
second configurable algorithm.

## Error Handling

The pipeline should not fail the entire batch when one sample is bad. Each sample can
end in one of these states:

- `success`
- `missing_pair`
- `ambiguous_pair`
- `decode_failed`
- `frame_select_failed`
- `write_failed`

At the end of a run, write a summary file under the output directory:

```text
output/run_summary.yaml
```

The summary should include counts, skipped samples, failed samples, and config used.

## Testing Strategy

Phase 1 tests should cover:

- Pair discovery with nested folders.
- Missing and ambiguous pair handling.
- Output path mirroring for `LR`, `HR`, and `metadata`.
- Registry lookup for algorithms.
- Metadata schema for selected and top-k frame records.
- A lightweight fake frame selector that avoids requiring GPU in unit tests.

GPU and DINOv2 behavior can be covered by an optional integration test profile.

## Later Phases

Phase 2 alignment:

- Start with OpenCV feature matching, ECC, or homography as initial alignment.
- Add optical flow refinement later through RAFT or a similar model.
- Save aligned results only when the `align` stage is enabled.

Phase 3 color matching:

- Start with histogram matching or Reinhard color transfer.
- Add learnable color transform later if needed.
- Keep the color algorithm configurable through YAML.

## Open Decisions

- Whether preview contact sheets should be implemented in phase 1 or left for phase 2.
- Whether HEIC support should be required immediately or treated as optional depending
  on installed codecs.
- Whether the first default video decoder should be OpenCV for simpler setup or PyAV
  for better timestamp handling.
