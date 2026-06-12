# livePhoto2LRHR

Build paired `LR` and `HR` folders from Live Photo-style image/video pairs.

## Phase 1

Phase 1 pairs files by relative stem, selects one best LR frame from each video, and writes:

```text
output/
  LR/
  HR/
  metadata/
  run_summary.yaml
```

`LR` contains only the selected frame. Top-k candidate frame indices, timestamps, and scores are recorded in metadata only.

## Install

For the default DINOv2 selector:

```bash
python -m pip install -e .[dev,dinov2]
```

For the CPU-only OpenCV baseline:

```bash
python -m pip install -e .[dev]
```

## Run

Edit `configs/frame_select.yaml`, then run:

```bash
livephoto2lrhr --config configs/frame_select.yaml
```

The included default config expects test data at `D:/SR数据集/花` and writes to `D:/SR数据集/花_pairs`.

## Configuration

The main knobs are:

- `data.input_dir`: directory containing image/video pairs with the same relative stem.
- `data.output_dir`: output root where `LR`, `HR`, `metadata`, and `run_summary.yaml` are written.
- `frame_select.algorithm`: selector name, for example `dinov2_similarity` or `opencv_similarity`.
- `frame_select.device`: `auto`, `cpu`, or `cuda` for GPU-capable selectors.
- `frame_select.resize_short_side`: DINOv2 resize and center-crop size. For `dinov2_similarity`, use a multiple of 14 such as `518`.
- `frame_select.top_k`: number of candidate frame records to keep in metadata.
- `output.overwrite`: whether to replace existing `LR`, `HR`, and metadata outputs.

## Output Contract

For an input pair like:

```text
input/trip/IMG_0001.jpg
input/trip/IMG_0001.mp4
```

the pipeline writes:

```text
output/LR/trip/IMG_0001.png
output/HR/trip/IMG_0001.png
output/metadata/trip/IMG_0001.yaml
```

The `LR` and `HR` directory structures are intentionally identical so they can be compared directly by dataset tooling.
