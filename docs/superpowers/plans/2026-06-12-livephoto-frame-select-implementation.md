# Live Photo Frame Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build phase 1 of the Live Photo dataset pipeline: pair photos and videos by name, select one LR frame from each video, and export mirrored `LR`, `HR`, and metadata directories.

**Architecture:** The project uses a Python package under `src/livephoto2lrhr`, driven by YAML config and a CLI. Pipeline stages are small orchestration units; algorithms are registered by string name so frame selection, alignment, and color matching can evolve independently. Tests start with a fake selector and synthetic media, then add an optional real-data integration command using `D:\SR数据集\花`.

**Tech Stack:** Python 3.10+, PyYAML, Pillow, OpenCV, NumPy, pytest, PyTorch, and DINOv2 through `torch.hub`.

---

## File Structure

- Create `pyproject.toml`: package metadata, console script, dependencies, pytest config.
- Create `.gitignore`: Python, test, model cache, and generated dataset output ignores.
- Create `.gitattributes`: normalize text files to LF to avoid Windows CRLF churn.
- Create `configs/frame_select.yaml`: default config using the provided test input path and configurable output path.
- Create `src/livephoto2lrhr/__init__.py`: package marker and version.
- Create `src/livephoto2lrhr/cli.py`: command-line entry point.
- Create `src/livephoto2lrhr/config.py`: load and validate YAML config into dataclasses.
- Create `src/livephoto2lrhr/data/pairing.py`: discover image/video pairs and report missing or ambiguous files.
- Create `src/livephoto2lrhr/data/io.py`: read/write images, create mirrored output paths, write metadata and summary.
- Create `src/livephoto2lrhr/pipeline/registry.py`: algorithm registry.
- Create `src/livephoto2lrhr/pipeline/runner.py`: run configured stages across discovered samples.
- Create `src/livephoto2lrhr/stages/frame_select.py`: orchestrate phase 1 per sample.
- Create `src/livephoto2lrhr/stages/align.py`: disabled placeholder stage that validates future config only.
- Create `src/livephoto2lrhr/stages/color_match.py`: disabled placeholder stage that validates future config only.
- Create `src/livephoto2lrhr/algorithms/similarity/base.py`: shared frame selector protocol and result dataclasses.
- Create `src/livephoto2lrhr/algorithms/similarity/fake.py`: deterministic test selector.
- Create `src/livephoto2lrhr/algorithms/similarity/opencv.py`: dependency-light baseline selector using resized pixel and edge similarity.
- Create `src/livephoto2lrhr/algorithms/similarity/dinov2.py`: default GPU-capable DINOv2 selector.
- Create `src/livephoto2lrhr/utils/device.py`: resolve `cpu`, `cuda`, and `auto`.
- Create `src/livephoto2lrhr/utils/logging.py`: configure project logging.
- Create `tests/conftest.py`: temp media fixtures.
- Create `tests/test_config.py`: config loading behavior.
- Create `tests/test_pairing.py`: pair discovery behavior.
- Create `tests/test_registry.py`: registry behavior.
- Create `tests/test_frame_select_stage.py`: phase 1 output contract.
- Create `tests/test_cli.py`: CLI smoke behavior.

## Task 1: Project Skeleton and Config Loading

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `configs/frame_select.yaml`
- Create: `src/livephoto2lrhr/__init__.py`
- Create: `src/livephoto2lrhr/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest
import yaml

from livephoto2lrhr.config import load_config


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_config_resolves_paths_and_defaults(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {
                "algorithm": "fake_selector",
                "device": "cpu",
                "top_k": 3,
            },
        },
    )

    config = load_config(config_path)

    assert config.data.input_dir == input_dir.resolve()
    assert config.data.output_dir == output_dir.resolve()
    assert config.data.recursive is True
    assert config.data.image_exts == (".jpg", ".jpeg", ".png", ".heic")
    assert config.data.video_exts == (".mp4", ".mov")
    assert config.pipeline.stages == ("frame_select",)
    assert config.frame_select.algorithm == "fake_selector"
    assert config.frame_select.top_k == 3


def test_load_config_rejects_missing_input_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(tmp_path / "missing"),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select"]},
            "frame_select": {"algorithm": "fake_selector"},
        },
    )

    with pytest.raises(ValueError, match="input_dir does not exist"):
        load_config(config_path)


def test_load_config_rejects_unknown_stage(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    write_yaml(
        config_path,
        {
            "data": {
                "input_dir": str(input_dir),
                "output_dir": str(tmp_path / "output"),
            },
            "pipeline": {"stages": ["frame_select", "unknown"]},
            "frame_select": {"algorithm": "fake_selector"},
        },
    )

    with pytest.raises(ValueError, match="unknown pipeline stage"):
        load_config(config_path)
```

- [ ] **Step 2: Run config tests and verify RED**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr'`.

- [ ] **Step 3: Create package metadata and config implementation**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "livephoto2lrhr"
version = "0.1.0"
description = "Build LR/HR training pairs from Live Photo images and videos."
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "opencv-python>=4.8",
  "pillow>=10.0",
  "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
]
dinov2 = [
  "torch>=2.1",
  "torchvision>=0.16",
]

[project.scripts]
livephoto2lrhr = "livephoto2lrhr.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
build/
dist/
*.egg-info/
outputs/
model_cache/
```

Create `.gitattributes`:

```gitattributes
* text=auto eol=lf
*.png binary
*.jpg binary
*.jpeg binary
*.mp4 binary
*.mov binary
```

Create `configs/frame_select.yaml`:

```yaml
task:
  name: livephoto_frame_select
  seed: 42

data:
  input_dir: D:/SR数据集/花
  output_dir: D:/SR数据集/花_pairs
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".heic"]
  video_exts: [".mp4", ".mov"]
  output_ext: ".png"

pipeline:
  stages:
    - frame_select

frame_select:
  algorithm: dinov2_similarity
  device: auto
  sample_fps: 15
  top_k: 5
  batch_size: 16
  resize_short_side: 512
  score_fusion:
    feature_weight: 0.7
    edge_weight: 0.3

output:
  save_metadata: true
  overwrite: false
```

Create `src/livephoto2lrhr/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `src/livephoto2lrhr/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_STAGES = {"frame_select", "align", "color_match"}


@dataclass(frozen=True)
class DataConfig:
    input_dir: Path
    output_dir: Path
    recursive: bool = True
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".heic")
    video_exts: tuple[str, ...] = (".mp4", ".mov")
    output_ext: str = ".png"


@dataclass(frozen=True)
class PipelineConfig:
    stages: tuple[str, ...]


@dataclass(frozen=True)
class FrameSelectConfig:
    algorithm: str
    device: str = "auto"
    sample_fps: float = 15.0
    top_k: int = 5
    batch_size: int = 16
    resize_short_side: int = 512
    score_fusion: dict[str, float] | None = None


@dataclass(frozen=True)
class OutputConfig:
    save_metadata: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    pipeline: PipelineConfig
    frame_select: FrameSelectConfig
    output: OutputConfig
    raw: dict[str, Any]


def _normalize_exts(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data_raw = raw.get("data", {})
    input_dir = Path(data_raw["input_dir"]).expanduser().resolve()
    output_dir = Path(data_raw["output_dir"]).expanduser().resolve()
    if not input_dir.exists():
        raise ValueError(f"input_dir does not exist: {input_dir}")

    stages = tuple(raw.get("pipeline", {}).get("stages", ()))
    for stage in stages:
        if stage not in VALID_STAGES:
            raise ValueError(f"unknown pipeline stage: {stage}")

    frame_raw = raw.get("frame_select", {})
    if "frame_select" in stages and not frame_raw.get("algorithm"):
        raise ValueError("frame_select.algorithm is required when frame_select stage is enabled")

    data = DataConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=bool(data_raw.get("recursive", True)),
        image_exts=_normalize_exts(data_raw.get("image_exts", [".jpg", ".jpeg", ".png", ".heic"])),
        video_exts=_normalize_exts(data_raw.get("video_exts", [".mp4", ".mov"])),
        output_ext=_normalize_exts([data_raw.get("output_ext", ".png")])[0],
    )
    pipeline = PipelineConfig(stages=stages)
    frame_select = FrameSelectConfig(
        algorithm=str(frame_raw.get("algorithm", "")),
        device=str(frame_raw.get("device", "auto")),
        sample_fps=float(frame_raw.get("sample_fps", 15.0)),
        top_k=int(frame_raw.get("top_k", 5)),
        batch_size=int(frame_raw.get("batch_size", 16)),
        resize_short_side=int(frame_raw.get("resize_short_side", 512)),
        score_fusion=frame_raw.get("score_fusion"),
    )
    output = OutputConfig(
        save_metadata=bool(raw.get("output", {}).get("save_metadata", True)),
        overwrite=bool(raw.get("output", {}).get("overwrite", False)),
    )
    return AppConfig(data=data, pipeline=pipeline, frame_select=frame_select, output=output, raw=raw)
```

- [ ] **Step 4: Run config tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add .gitattributes .gitignore pyproject.toml configs/frame_select.yaml src/livephoto2lrhr/__init__.py src/livephoto2lrhr/config.py tests/test_config.py
git commit -m "feat: add config loading"
```

## Task 2: Pair Discovery

**Files:**
- Create: `src/livephoto2lrhr/data/__init__.py`
- Create: `src/livephoto2lrhr/data/pairing.py`
- Test: `tests/test_pairing.py`

- [ ] **Step 1: Write failing pairing tests**

Create `tests/test_pairing.py`:

```python
from pathlib import Path

from livephoto2lrhr.data.pairing import discover_pairs


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_discover_pairs_matches_by_relative_stem(tmp_path: Path):
    touch(tmp_path / "IMG_0001.jpg")
    touch(tmp_path / "IMG_0001.mp4")
    touch(tmp_path / "trip" / "a.jpg")
    touch(tmp_path / "trip" / "a.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg",),
        video_exts=(".mp4",),
        recursive=True,
    )

    assert [sample.sample_id for sample in result.pairs] == ["IMG_0001", "trip/a"]
    assert result.missing_images == []
    assert result.missing_videos == []
    assert result.ambiguous == []


def test_discover_pairs_reports_missing_and_ambiguous_files(tmp_path: Path):
    touch(tmp_path / "ok.jpg")
    touch(tmp_path / "ok.mp4")
    touch(tmp_path / "image_only.jpg")
    touch(tmp_path / "video_only.mp4")
    touch(tmp_path / "dup.jpg")
    touch(tmp_path / "dup.png")
    touch(tmp_path / "dup.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg", ".png"),
        video_exts=(".mp4",),
        recursive=True,
    )

    assert [sample.sample_id for sample in result.pairs] == ["ok"]
    assert result.missing_videos == ["image_only"]
    assert result.missing_images == ["video_only"]
    assert result.ambiguous == ["dup"]


def test_discover_pairs_can_ignore_nested_files_when_not_recursive(tmp_path: Path):
    touch(tmp_path / "root.jpg")
    touch(tmp_path / "root.mp4")
    touch(tmp_path / "nested" / "child.jpg")
    touch(tmp_path / "nested" / "child.mp4")

    result = discover_pairs(
        tmp_path,
        image_exts=(".jpg",),
        video_exts=(".mp4",),
        recursive=False,
    )

    assert [sample.sample_id for sample in result.pairs] == ["root"]
```

- [ ] **Step 2: Run pairing tests and verify RED**

Run:

```bash
python -m pytest tests/test_pairing.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.data'`.

- [ ] **Step 3: Implement pair discovery**

Create `src/livephoto2lrhr/data/__init__.py`:

```python
```

Create `src/livephoto2lrhr/data/pairing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplePair:
    sample_id: str
    image_path: Path
    video_path: Path
    relative_stem: Path


@dataclass(frozen=True)
class PairDiscoveryResult:
    pairs: list[SamplePair]
    missing_images: list[str]
    missing_videos: list[str]
    ambiguous: list[str]


def _iter_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _sample_id(relative_stem: Path) -> str:
    return relative_stem.as_posix()


def discover_pairs(
    input_dir: Path,
    *,
    image_exts: tuple[str, ...],
    video_exts: tuple[str, ...],
    recursive: bool,
) -> PairDiscoveryResult:
    image_ext_set = {ext.lower() for ext in image_exts}
    video_ext_set = {ext.lower() for ext in video_exts}
    images: dict[Path, list[Path]] = {}
    videos: dict[Path, list[Path]] = {}

    for file_path in _iter_files(input_dir, recursive):
        ext = file_path.suffix.lower()
        relative_stem = file_path.relative_to(input_dir).with_suffix("")
        if ext in image_ext_set:
            images.setdefault(relative_stem, []).append(file_path)
        elif ext in video_ext_set:
            videos.setdefault(relative_stem, []).append(file_path)

    all_stems = sorted(set(images) | set(videos), key=lambda path: path.as_posix())
    pairs: list[SamplePair] = []
    missing_images: list[str] = []
    missing_videos: list[str] = []
    ambiguous: list[str] = []

    for relative_stem in all_stems:
        sample_id = _sample_id(relative_stem)
        image_matches = images.get(relative_stem, [])
        video_matches = videos.get(relative_stem, [])
        if not image_matches:
            missing_images.append(sample_id)
            continue
        if not video_matches:
            missing_videos.append(sample_id)
            continue
        if len(image_matches) > 1 or len(video_matches) > 1:
            ambiguous.append(sample_id)
            continue
        pairs.append(
            SamplePair(
                sample_id=sample_id,
                image_path=image_matches[0],
                video_path=video_matches[0],
                relative_stem=relative_stem,
            )
        )

    return PairDiscoveryResult(
        pairs=pairs,
        missing_images=missing_images,
        missing_videos=missing_videos,
        ambiguous=ambiguous,
    )
```

- [ ] **Step 4: Run pairing tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_pairing.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/data/__init__.py src/livephoto2lrhr/data/pairing.py tests/test_pairing.py
git commit -m "feat: discover live photo pairs"
```

## Task 3: Algorithm Registry and Frame Selector Types

**Files:**
- Create: `src/livephoto2lrhr/pipeline/__init__.py`
- Create: `src/livephoto2lrhr/pipeline/registry.py`
- Create: `src/livephoto2lrhr/algorithms/__init__.py`
- Create: `src/livephoto2lrhr/algorithms/similarity/__init__.py`
- Create: `src/livephoto2lrhr/algorithms/similarity/base.py`
- Create: `src/livephoto2lrhr/algorithms/similarity/fake.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_registry.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.pipeline.registry import Registry


def test_registry_creates_registered_algorithm():
    registry = Registry()
    registry.register("fake_selector", FakeFrameSelector)

    selector = registry.create("fake_selector", {"top_k": 2})

    assert isinstance(selector, FakeFrameSelector)


def test_registry_rejects_unknown_algorithm():
    registry = Registry()

    with pytest.raises(KeyError, match="unknown algorithm"):
        registry.create("missing", {})


def test_fake_selector_returns_one_selected_frame_and_top_k(tmp_path: Path):
    selector = FakeFrameSelector({"top_k": 2})

    result = selector.select(tmp_path / "hr.png", tmp_path / "video.mp4")

    assert result.selected.frame_index == 0
    assert result.frame_rgb.shape == (4, 4, 3)
    assert result.frame_rgb.dtype == np.uint8
    assert result.top_k == [
        FrameCandidate(frame_index=0, timestamp_sec=0.0, score=1.0),
        FrameCandidate(frame_index=1, timestamp_sec=1.0 / 30.0, score=0.5),
    ]
```

- [ ] **Step 2: Run registry tests and verify RED**

Run:

```bash
python -m pytest tests/test_registry.py -v
```

Expected: FAIL during import with `ModuleNotFoundError` for the new modules.

- [ ] **Step 3: Implement registry and fake selector**

Create `src/livephoto2lrhr/pipeline/__init__.py`:

```python
```

Create `src/livephoto2lrhr/pipeline/registry.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Callable[[dict[str, Any]], T]] = {}

    def register(self, name: str, factory: Callable[[dict[str, Any]], T]) -> None:
        if name in self._items:
            raise KeyError(f"algorithm already registered: {name}")
        self._items[name] = factory

    def create(self, name: str, config: dict[str, Any]) -> T:
        try:
            factory = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown algorithm: {name}; available: {available}") from exc
        return factory(config)
```

Create `src/livephoto2lrhr/algorithms/__init__.py`:

```python
```

Create `src/livephoto2lrhr/algorithms/similarity/__init__.py`:

```python
```

Create `src/livephoto2lrhr/algorithms/similarity/base.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class FrameCandidate:
    frame_index: int
    timestamp_sec: float
    score: float


@dataclass(frozen=True)
class FrameSelectionResult:
    frame_rgb: np.ndarray
    selected: FrameCandidate
    top_k: list[FrameCandidate]
    diagnostics: dict[str, object]


class FrameSelector(Protocol):
    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        """Return the selected RGB frame and ranked candidate metadata."""
```

Create `src/livephoto2lrhr/algorithms/similarity/fake.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult


class FakeFrameSelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.top_k = int(config.get("top_k", 1))

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        candidates = [
            FrameCandidate(frame_index=index, timestamp_sec=index / 30.0, score=1.0 / (index + 1))
            for index in range(max(self.top_k, 1))
        ]
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 1] = 255
        return FrameSelectionResult(
            frame_rgb=frame,
            selected=candidates[0],
            top_k=candidates[: self.top_k],
            diagnostics={"algorithm": "fake_selector"},
        )
```

- [ ] **Step 4: Run registry tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_registry.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/pipeline src/livephoto2lrhr/algorithms tests/test_registry.py
git commit -m "feat: add frame selector registry"
```

## Task 4: Frame Selection Stage Output Contract

**Files:**
- Create: `src/livephoto2lrhr/data/io.py`
- Create: `src/livephoto2lrhr/stages/__init__.py`
- Create: `src/livephoto2lrhr/stages/frame_select.py`
- Test: `tests/conftest.py`
- Test: `tests/test_frame_select_stage.py`

- [ ] **Step 1: Write failing output contract tests**

Create `tests/conftest.py`:

```python
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "input" / "flower.jpg"
    video_path = tmp_path / "input" / "flower.mp4"
    image_path.parent.mkdir(parents=True)

    image = Image.new("RGB", (8, 8), color=(20, 40, 200))
    image.save(image_path)

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        (8, 8),
    )
    for value in (0, 80, 160):
        frame_bgr = np.full((8, 8, 3), value, dtype=np.uint8)
        writer.write(frame_bgr)
    writer.release()

    return image_path, video_path
```

Create `tests/test_frame_select_stage.py`:

```python
from pathlib import Path

import yaml
from PIL import Image

from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.data.pairing import SamplePair
from livephoto2lrhr.stages.frame_select import FrameSelectStage


def test_frame_select_stage_writes_mirrored_lr_hr_and_metadata(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="nested/flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("nested") / "flower",
    )
    selector = FakeFrameSelector({"top_k": 3})
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=selector,
        algorithm_name="fake_selector",
    )

    result = stage.run(pair)

    lr_path = output_dir / "LR" / "nested" / "flower.png"
    hr_path = output_dir / "HR" / "nested" / "flower.png"
    metadata_path = output_dir / "metadata" / "nested" / "flower.yaml"
    assert result.status == "success"
    assert lr_path.exists()
    assert hr_path.exists()
    assert metadata_path.exists()
    assert Image.open(lr_path).size == (4, 4)
    assert Image.open(hr_path).size == (8, 8)

    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_id"] == "nested/flower"
    assert metadata["frame_select"]["algorithm"] == "fake_selector"
    assert metadata["frame_select"]["selected"]["frame_index"] == 0
    assert [item["frame_index"] for item in metadata["frame_select"]["top_k"]] == [0, 1, 2]
    assert sorted(path.name for path in (output_dir / "LR" / "nested").iterdir()) == ["flower.png"]


def test_frame_select_stage_respects_overwrite_false(
    tmp_path: Path,
    tiny_pair: tuple[Path, Path],
):
    image_path, video_path = tiny_pair
    output_dir = tmp_path / "output"
    pair = SamplePair(
        sample_id="flower",
        image_path=image_path,
        video_path=video_path,
        relative_stem=Path("flower"),
    )
    selector = FakeFrameSelector({"top_k": 1})
    stage = FrameSelectStage(
        output_dir=output_dir,
        output_ext=".png",
        overwrite=False,
        save_metadata=True,
        selector=selector,
        algorithm_name="fake_selector",
    )

    first = stage.run(pair)
    second = stage.run(pair)

    assert first.status == "success"
    assert second.status == "skipped_existing"
```

- [ ] **Step 2: Run frame select stage tests and verify RED**

Run:

```bash
python -m pytest tests/test_frame_select_stage.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.stages'`.

- [ ] **Step 3: Implement image IO and frame selection stage**

Create `src/livephoto2lrhr/data/io.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate


def output_image_path(output_dir: Path, folder: str, relative_stem: Path, output_ext: str) -> Path:
    return output_dir / folder / relative_stem.with_suffix(output_ext)


def metadata_path(output_dir: Path, relative_stem: Path) -> Path:
    return output_dir / "metadata" / relative_stem.with_suffix(".yaml")


def save_pil_image(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    image.save(destination_path)


def save_rgb_array(frame_rgb: np.ndarray, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_rgb, mode="RGB").save(destination_path)


def candidate_to_dict(candidate: FrameCandidate) -> dict[str, float | int]:
    return {
        "frame_index": candidate.frame_index,
        "timestamp_sec": float(candidate.timestamp_sec),
        "score": float(candidate.score),
    }


def write_yaml(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
```

Create `src/livephoto2lrhr/stages/__init__.py`:

```python
```

Create `src/livephoto2lrhr/stages/frame_select.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from livephoto2lrhr.algorithms.similarity.base import FrameSelector
from livephoto2lrhr.data.io import (
    candidate_to_dict,
    metadata_path,
    output_image_path,
    save_pil_image,
    save_rgb_array,
    write_yaml,
)
from livephoto2lrhr.data.pairing import SamplePair


@dataclass(frozen=True)
class StageResult:
    sample_id: str
    status: str
    message: str = ""


class FrameSelectStage:
    def __init__(
        self,
        *,
        output_dir: Path,
        output_ext: str,
        overwrite: bool,
        save_metadata: bool,
        selector: FrameSelector,
        algorithm_name: str,
    ) -> None:
        self.output_dir = output_dir
        self.output_ext = output_ext
        self.overwrite = overwrite
        self.save_metadata = save_metadata
        self.selector = selector
        self.algorithm_name = algorithm_name

    def run(self, pair: SamplePair) -> StageResult:
        lr_path = output_image_path(self.output_dir, "LR", pair.relative_stem, self.output_ext)
        hr_path = output_image_path(self.output_dir, "HR", pair.relative_stem, self.output_ext)
        meta_path = metadata_path(self.output_dir, pair.relative_stem)

        if not self.overwrite and lr_path.exists() and hr_path.exists():
            return StageResult(sample_id=pair.sample_id, status="skipped_existing")

        try:
            selection = self.selector.select(pair.image_path, pair.video_path)
            save_rgb_array(selection.frame_rgb, lr_path)
            save_pil_image(pair.image_path, hr_path)
            if self.save_metadata:
                write_yaml(
                    meta_path,
                    {
                        "sample_id": pair.sample_id,
                        "source": {
                            "image": str(pair.image_path),
                            "video": str(pair.video_path),
                        },
                        "output": {
                            "hr": str(hr_path),
                            "lr": str(lr_path),
                        },
                        "frame_select": {
                            "algorithm": self.algorithm_name,
                            "selected": candidate_to_dict(selection.selected),
                            "top_k": [candidate_to_dict(candidate) for candidate in selection.top_k],
                            "diagnostics": selection.diagnostics,
                        },
                        "status": {
                            "aligned": False,
                            "color_matched": False,
                        },
                    },
                )
        except Exception as exc:
            return StageResult(sample_id=pair.sample_id, status="frame_select_failed", message=str(exc))

        return StageResult(sample_id=pair.sample_id, status="success")
```

- [ ] **Step 4: Run frame select stage tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_frame_select_stage.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/data/io.py src/livephoto2lrhr/stages tests/conftest.py tests/test_frame_select_stage.py
git commit -m "feat: write mirrored frame selection outputs"
```

## Task 5: OpenCV Similarity Selector

**Files:**
- Create: `src/livephoto2lrhr/algorithms/similarity/opencv.py`
- Test: `tests/test_opencv_selector.py`

- [ ] **Step 1: Write failing OpenCV selector tests**

Create `tests/test_opencv_selector.py`:

```python
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector


def make_video(path: Path, frames_rgb: list[np.ndarray], fps: float = 30.0) -> None:
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def test_opencv_selector_prefers_frame_closest_to_photo(tmp_path: Path):
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "target.mp4"
    target = np.full((32, 32, 3), (20, 120, 220), dtype=np.uint8)
    Image.fromarray(target).save(image_path)
    frames = [
        np.full((32, 32, 3), (200, 20, 20), dtype=np.uint8),
        target.copy(),
        np.full((32, 32, 3), (10, 10, 10), dtype=np.uint8),
    ]
    make_video(video_path, frames)

    selector = OpenCVSimilaritySelector(
        {
            "sample_fps": 30,
            "top_k": 2,
            "resize_short_side": 64,
            "score_fusion": {"feature_weight": 0.7, "edge_weight": 0.3},
        }
    )
    result = selector.select(image_path, video_path)

    assert result.selected.frame_index == 1
    assert result.frame_rgb.shape == (32, 32, 3)
    assert [candidate.frame_index for candidate in result.top_k] == [1, 2]


def test_opencv_selector_raises_for_unreadable_video(tmp_path: Path):
    image_path = tmp_path / "target.jpg"
    video_path = tmp_path / "bad.mp4"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(image_path)
    video_path.write_bytes(b"not a video")

    selector = OpenCVSimilaritySelector({"top_k": 1})

    try:
        selector.select(image_path, video_path)
    except ValueError as exc:
        assert "could not open video" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run OpenCV selector tests and verify RED**

Run:

```bash
python -m pytest tests/test_opencv_selector.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.algorithms.similarity.opencv'`.

- [ ] **Step 3: Implement OpenCV similarity selector**

Create `src/livephoto2lrhr/algorithms/similarity/opencv.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult


class OpenCVSimilaritySelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.sample_fps = float(config.get("sample_fps", 15.0))
        self.top_k = int(config.get("top_k", 5))
        self.resize_short_side = int(config.get("resize_short_side", 512))
        fusion = config.get("score_fusion") or {}
        self.feature_weight = float(fusion.get("feature_weight", 0.7))
        self.edge_weight = float(fusion.get("edge_weight", 0.3))

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        target = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
        target_small = self._resize_for_score(target)
        target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY)
        target_edges = cv2.Canny(target_gray, 80, 160)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(int(round(fps / self.sample_fps)), 1)
        candidates: list[tuple[FrameCandidate, np.ndarray]] = []
        frame_index = 0

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                score = self._score(target_small, target_edges, frame_rgb)
                candidates.append(
                    (
                        FrameCandidate(
                            frame_index=frame_index,
                            timestamp_sec=frame_index / fps,
                            score=score,
                        ),
                        frame_rgb,
                    )
                )
            frame_index += 1

        capture.release()

        if not candidates:
            raise ValueError(f"video contained no readable frames: {video_path}")

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        selected, frame_rgb = candidates[0]
        top_k = [candidate for candidate, _ in candidates[: self.top_k]]
        return FrameSelectionResult(
            frame_rgb=frame_rgb,
            selected=selected,
            top_k=top_k,
            diagnostics={
                "algorithm": "opencv_similarity",
                "sample_fps": self.sample_fps,
                "frame_step": frame_step,
                "scored_frames": len(candidates),
            },
        )

    def _resize_for_score(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        short_side = min(height, width)
        if short_side == self.resize_short_side:
            return image_rgb
        scale = self.resize_short_side / short_side
        new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
        return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)

    def _score(self, target_small: np.ndarray, target_edges: np.ndarray, frame_rgb: np.ndarray) -> float:
        frame_small = cv2.resize(frame_rgb, (target_small.shape[1], target_small.shape[0]), interpolation=cv2.INTER_AREA)
        pixel_mse = np.mean((target_small.astype(np.float32) - frame_small.astype(np.float32)) ** 2)
        pixel_score = 1.0 / (1.0 + pixel_mse)

        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
        frame_edges = cv2.Canny(frame_gray, 80, 160)
        edge_mse = np.mean((target_edges.astype(np.float32) - frame_edges.astype(np.float32)) ** 2)
        edge_score = 1.0 / (1.0 + edge_mse)

        return self.feature_weight * pixel_score + self.edge_weight * edge_score
```

- [ ] **Step 4: Run OpenCV selector tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_opencv_selector.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/algorithms/similarity/opencv.py tests/test_opencv_selector.py
git commit -m "feat: add opencv frame selector"
```

## Task 6: Pipeline Runner and Summary

**Files:**
- Create: `src/livephoto2lrhr/pipeline/runner.py`
- Modify: `src/livephoto2lrhr/algorithms/similarity/__init__.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing runner tests**

Create `tests/test_runner.py`:

```python
from pathlib import Path

import yaml

from livephoto2lrhr.config import AppConfig, DataConfig, FrameSelectConfig, OutputConfig, PipelineConfig
from livephoto2lrhr.pipeline.runner import run_pipeline


def test_run_pipeline_writes_outputs_and_summary(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, video_path = tiny_pair
    input_dir = image_path.parent
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=2),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 1
    assert (output_dir / "LR" / "flower.png").exists()
    assert (output_dir / "HR" / "flower.png").exists()
    summary_yaml = yaml.safe_load((output_dir / "run_summary.yaml").read_text(encoding="utf-8"))
    assert summary_yaml["counts"]["success"] == 1
    assert summary_yaml["pair_discovery"]["missing_images"] == []
    assert summary_yaml["pair_discovery"]["missing_videos"] == []


def test_run_pipeline_reports_missing_pairs(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "image_only.jpg").write_bytes(b"x")
    output_dir = tmp_path / "output"
    config = AppConfig(
        data=DataConfig(input_dir=input_dir, output_dir=output_dir, image_exts=(".jpg",), video_exts=(".mp4",)),
        pipeline=PipelineConfig(stages=("frame_select",)),
        frame_select=FrameSelectConfig(algorithm="fake_selector", top_k=1),
        output=OutputConfig(save_metadata=True, overwrite=False),
        raw={"test": True},
    )

    summary = run_pipeline(config)

    assert summary["counts"]["success"] == 0
    assert summary["pair_discovery"]["missing_videos"] == ["image_only"]
```

- [ ] **Step 2: Run runner tests and verify RED**

Run:

```bash
python -m pytest tests/test_runner.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.pipeline.runner'`.

- [ ] **Step 3: Implement runner and default algorithm registry**

Modify `src/livephoto2lrhr/algorithms/similarity/__init__.py`:

```python
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector
from livephoto2lrhr.pipeline.registry import Registry


def build_similarity_registry() -> Registry:
    registry = Registry()
    registry.register("fake_selector", FakeFrameSelector)
    registry.register("opencv_similarity", OpenCVSimilaritySelector)
    return registry
```

Create `src/livephoto2lrhr/pipeline/runner.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any

from livephoto2lrhr.algorithms.similarity import build_similarity_registry
from livephoto2lrhr.config import AppConfig
from livephoto2lrhr.data.io import write_yaml
from livephoto2lrhr.data.pairing import discover_pairs
from livephoto2lrhr.stages.frame_select import FrameSelectStage


def run_pipeline(config: AppConfig) -> dict[str, Any]:
    pair_result = discover_pairs(
        config.data.input_dir,
        image_exts=config.data.image_exts,
        video_exts=config.data.video_exts,
        recursive=config.data.recursive,
    )
    counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    if "frame_select" in config.pipeline.stages:
        registry = build_similarity_registry()
        selector = registry.create(
            config.frame_select.algorithm,
            {
                "sample_fps": config.frame_select.sample_fps,
                "top_k": config.frame_select.top_k,
                "batch_size": config.frame_select.batch_size,
                "resize_short_side": config.frame_select.resize_short_side,
                "score_fusion": config.frame_select.score_fusion,
                "device": config.frame_select.device,
            },
        )
        stage = FrameSelectStage(
            output_dir=config.data.output_dir,
            output_ext=config.data.output_ext,
            overwrite=config.output.overwrite,
            save_metadata=config.output.save_metadata,
            selector=selector,
            algorithm_name=config.frame_select.algorithm,
        )
        for pair in pair_result.pairs:
            result = stage.run(pair)
            counts[result.status] += 1
            samples.append({"sample_id": result.sample_id, "status": result.status, "message": result.message})

    summary: dict[str, Any] = {
        "counts": dict(counts),
        "pair_discovery": {
            "paired": len(pair_result.pairs),
            "missing_images": pair_result.missing_images,
            "missing_videos": pair_result.missing_videos,
            "ambiguous": pair_result.ambiguous,
        },
        "samples": samples,
        "config": config.raw,
    }
    write_yaml(config.data.output_dir / "run_summary.yaml", summary)
    return summary
```

- [ ] **Step 4: Run runner tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_runner.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/algorithms/similarity/__init__.py src/livephoto2lrhr/pipeline/runner.py tests/test_runner.py
git commit -m "feat: run frame selection pipeline"
```

## Task 7: CLI

**Files:**
- Create: `src/livephoto2lrhr/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
from pathlib import Path

import yaml

from livephoto2lrhr.cli import main


def test_cli_runs_pipeline_from_config(tmp_path: Path, tiny_pair: tuple[Path, Path]):
    image_path, _ = tiny_pair
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "input_dir": str(image_path.parent),
                    "output_dir": str(output_dir),
                    "image_exts": [".jpg"],
                    "video_exts": [".mp4"],
                },
                "pipeline": {"stages": ["frame_select"]},
                "frame_select": {"algorithm": "fake_selector", "top_k": 1},
                "output": {"save_metadata": True, "overwrite": False},
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])

    assert exit_code == 0
    assert (output_dir / "LR" / "flower.png").exists()
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.cli'`.

- [ ] **Step 3: Implement CLI**

Create `src/livephoto2lrhr/cli.py`:

```python
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
```

Create `src/livephoto2lrhr/utils/__init__.py`:

```python
```

Create `src/livephoto2lrhr/utils/logging.py`:

```python
from __future__ import annotations

import logging


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
```

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/cli.py src/livephoto2lrhr/utils tests/test_cli.py
git commit -m "feat: add frame selection cli"
```

## Task 8: DINOv2 AI Similarity Selector

**Files:**
- Create: `src/livephoto2lrhr/utils/device.py`
- Create: `src/livephoto2lrhr/algorithms/similarity/dinov2.py`
- Modify: `src/livephoto2lrhr/algorithms/similarity/__init__.py`
- Test: `tests/test_dinov2_selector.py`

- [ ] **Step 1: Write failing DINOv2 selector tests**

Create `tests/test_dinov2_selector.py`:

```python
import builtins
from pathlib import Path

import numpy as np
import pytest

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate
from livephoto2lrhr.algorithms.similarity.dinov2 import DINOv2SimilaritySelector


def test_dinov2_selector_reports_missing_torch(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Install the dinov2 extra"):
        DINOv2SimilaritySelector({"device": "cpu"})


def test_dinov2_ranks_frames_by_feature_similarity(monkeypatch, tmp_path: Path):
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=np.float32)

        def to(self, device):
            return self

        def unsqueeze(self, dim):
            return self

        def cpu(self):
            return self

        def flatten(self):
            return self.value.reshape(-1)

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class hub:
            @staticmethod
            def load(repo, model_name):
                return FakeModel()

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, tensor):
            return tensor

    class FakeTransform:
        def __call__(self, image):
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            return FakeTensor([rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()])

    def fake_cosine_similarity(left, right):
        left_array = left.flatten()
        right_array = right.flatten()
        return float(np.dot(left_array, right_array) / (np.linalg.norm(left_array) * np.linalg.norm(right_array)))

    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._import_torch", lambda: FakeTorch)
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._build_transform", lambda size: FakeTransform())
    monkeypatch.setattr("livephoto2lrhr.algorithms.similarity.dinov2._cosine_similarity", fake_cosine_similarity)

    selector = DINOv2SimilaritySelector({"device": "cpu", "top_k": 2, "resize_short_side": 32})
    ranked = selector._rank_features(
        FakeTensor([0.0, 1.0, 0.0]),
        [
            (FrameCandidate(frame_index=0, timestamp_sec=0.0, score=0.0), FakeTensor([1.0, 0.0, 0.0])),
            (FrameCandidate(frame_index=1, timestamp_sec=1.0, score=0.0), FakeTensor([0.0, 1.0, 0.0])),
            (FrameCandidate(frame_index=2, timestamp_sec=2.0, score=0.0), FakeTensor([0.0, 0.5, 0.5])),
        ],
    )

    assert [candidate.frame_index for candidate in ranked] == [1, 2, 0]
    assert ranked[0].score == pytest.approx(1.0)
```

- [ ] **Step 2: Run DINOv2 tests and verify RED**

Run:

```bash
python -m pytest tests/test_dinov2_selector.py -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'livephoto2lrhr.algorithms.similarity.dinov2'`.

- [ ] **Step 3: Implement DINOv2 feature selector**

Create `src/livephoto2lrhr/utils/device.py`:

```python
from __future__ import annotations


def resolve_device(device: str) -> str:
    normalized = device.lower()
    if normalized == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA requested but torch is not installed.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device: {device}")
    return normalized
```

Create `src/livephoto2lrhr/algorithms/similarity/dinov2.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageOps

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult
from livephoto2lrhr.utils.device import resolve_device


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the dinov2 extra to use dinov2_similarity: pip install -e .[dinov2]") from exc
    return torch


def _build_transform(resize_short_side: int):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install the dinov2 extra to use dinov2_similarity: pip install -e .[dinov2]") from exc

    return transforms.Compose(
        [
            transforms.Resize(resize_short_side, antialias=True),
            transforms.CenterCrop(resize_short_side),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _cosine_similarity(left, right) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


class DINOv2SimilaritySelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.torch = _import_torch()
        self.device = resolve_device(str(config.get("device", "auto")))
        self.sample_fps = float(config.get("sample_fps", 15.0))
        self.top_k = int(config.get("top_k", 5))
        self.resize_short_side = int(config.get("resize_short_side", 518))
        self.transform = _build_transform(self.resize_short_side)
        self.model = self.torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(self.device)

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        image_feature = self._extract_feature(image)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(int(round(fps / self.sample_fps)), 1)
        frame_index = 0
        frames: list[tuple[FrameCandidate, object, object]] = []

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % frame_step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                feature = self._extract_feature(Image.fromarray(frame_rgb))
                frames.append(
                    (
                        FrameCandidate(frame_index=frame_index, timestamp_sec=frame_index / fps, score=0.0),
                        frame_rgb,
                        feature,
                    )
                )
            frame_index += 1

        capture.release()
        if not frames:
            raise ValueError(f"video contained no readable frames: {video_path}")

        ranked = self._rank_features(image_feature, [(candidate, feature) for candidate, _, feature in frames])
        selected = ranked[0]
        frame_lookup = {candidate.frame_index: frame_rgb for candidate, frame_rgb, _ in frames}
        return FrameSelectionResult(
            frame_rgb=frame_lookup[selected.frame_index],
            selected=selected,
            top_k=ranked[: self.top_k],
            diagnostics={
                "algorithm": "dinov2_similarity",
                "device": self.device,
                "model": "dinov2_vits14",
                "sample_fps": self.sample_fps,
                "frame_step": frame_step,
                "scored_frames": len(frames),
            },
        )

    def _extract_feature(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            return self.model(tensor).cpu()

    def _rank_features(self, image_feature, frame_features: list[tuple[FrameCandidate, object]]) -> list[FrameCandidate]:
        ranked = [
            FrameCandidate(
                frame_index=candidate.frame_index,
                timestamp_sec=candidate.timestamp_sec,
                score=_cosine_similarity(image_feature, feature),
            )
            for candidate, feature in frame_features
        ]
        return sorted(ranked, key=lambda candidate: candidate.score, reverse=True)
```

Modify `src/livephoto2lrhr/algorithms/similarity/__init__.py`:

```python
from livephoto2lrhr.algorithms.similarity.dinov2 import DINOv2SimilaritySelector
from livephoto2lrhr.algorithms.similarity.fake import FakeFrameSelector
from livephoto2lrhr.algorithms.similarity.opencv import OpenCVSimilaritySelector
from livephoto2lrhr.pipeline.registry import Registry


def build_similarity_registry() -> Registry:
    registry = Registry()
    registry.register("fake_selector", FakeFrameSelector)
    registry.register("opencv_similarity", OpenCVSimilaritySelector)
    registry.register("dinov2_similarity", DINOv2SimilaritySelector)
    return registry
```

- [ ] **Step 4: Run DINOv2 tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_dinov2_selector.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/utils/device.py src/livephoto2lrhr/algorithms/similarity/dinov2.py src/livephoto2lrhr/algorithms/similarity/__init__.py tests/test_dinov2_selector.py
git commit -m "feat: add dinov2 frame selector"
```

## Task 9: Future Stage Placeholders

**Files:**
- Create: `src/livephoto2lrhr/stages/align.py`
- Create: `src/livephoto2lrhr/stages/color_match.py`
- Test: `tests/test_future_stages.py`

- [ ] **Step 1: Write failing placeholder tests**

Create `tests/test_future_stages.py`:

```python
from livephoto2lrhr.stages.align import AlignStage
from livephoto2lrhr.stages.color_match import ColorMatchStage


def test_align_stage_is_explicitly_not_implemented():
    stage = AlignStage(enabled=False)

    assert stage.enabled is False
    assert stage.describe() == "align stage is reserved for phase 2"


def test_color_match_stage_is_explicitly_not_implemented():
    stage = ColorMatchStage(enabled=False)

    assert stage.enabled is False
    assert stage.describe() == "color_match stage is reserved for phase 3"
```

- [ ] **Step 2: Run future stage tests and verify RED**

Run:

```bash
python -m pytest tests/test_future_stages.py -v
```

Expected: FAIL during import for missing stage modules.

- [ ] **Step 3: Implement placeholders**

Create `src/livephoto2lrhr/stages/align.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlignStage:
    enabled: bool = False

    def describe(self) -> str:
        return "align stage is reserved for phase 2"
```

Create `src/livephoto2lrhr/stages/color_match.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorMatchStage:
    enabled: bool = False

    def describe(self) -> str:
        return "color_match stage is reserved for phase 3"
```

- [ ] **Step 4: Run future stage tests and verify GREEN**

Run:

```bash
python -m pytest tests/test_future_stages.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/livephoto2lrhr/stages/align.py src/livephoto2lrhr/stages/color_match.py tests/test_future_stages.py
git commit -m "feat: reserve future pipeline stages"
```

## Task 10: Full Verification With Synthetic and Real Samples

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write README usage**

Create `README.md`:

```markdown
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

`LR` contains only the selected frame. Top-k candidate frame indices and scores are recorded in metadata only.

## Install

```bash
python -m pip install -e .[dev,dinov2]
```

For a CPU-only OpenCV baseline:

```bash
python -m pip install -e .[dev]
```

## Run

Edit `configs/frame_select.yaml`, then run:

```bash
livephoto2lrhr --config configs/frame_select.yaml
```

The included default config expects test data at `D:/SR数据集/花` and writes to `D:/SR数据集/花_pairs`.
```

- [ ] **Step 2: Run full unit test suite**

Run:

```bash
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Run real-data smoke test using provided dataset**

Run:

```bash
python -m livephoto2lrhr.cli --config configs/frame_select.yaml
```

Expected:

- Command exits 0.
- `D:/SR数据集/花_pairs/LR` exists.
- `D:/SR数据集/花_pairs/HR` exists.
- `D:/SR数据集/花_pairs/metadata` exists.
- `D:/SR数据集/花_pairs/run_summary.yaml` reports about 21 paired samples and about 14 missing videos based on the current test directory.

- [ ] **Step 4: Verify LR and HR structures match exactly**

Run:

```powershell
$out = 'D:\SR数据集\花_pairs'
$lr = Get-ChildItem -LiteralPath "$out\LR" -Recurse -File | ForEach-Object { $_.FullName.Substring((Resolve-Path "$out\LR").Path.Length + 1) } | Sort-Object
$hr = Get-ChildItem -LiteralPath "$out\HR" -Recurse -File | ForEach-Object { $_.FullName.Substring((Resolve-Path "$out\HR").Path.Length + 1) } | Sort-Object
Compare-Object $lr $hr
```

Expected: no output from `Compare-Object`.

- [ ] **Step 5: Verify no top-k images were written into LR**

Run:

```powershell
Get-ChildItem -LiteralPath 'D:\SR数据集\花_pairs\LR' -Recurse -File | Where-Object { $_.BaseName -match 'top|candidate|frame' }
```

Expected: no output.

- [ ] **Step 6: Commit README and verification-ready config**

```bash
git add README.md configs/frame_select.yaml
git commit -m "docs: add frame selection usage"
```

## Self-Review

Spec coverage:

- YAML input and output paths are covered by Task 1 and Task 7.
- Recursive same-stem image/video pairing is covered by Task 2.
- AI-style configurable frame selection is covered by Task 3, Task 6, and Task 8. Task 5 provides a dependency-light OpenCV baseline.
- Mirrored `LR` and `HR` output structure is covered by Task 4 and Task 10.
- Top-k metadata without top-k LR images is covered by Task 4 and Task 10.
- Per-sample failure reporting and `run_summary.yaml` are covered by Task 6.
- Windows and Linux path handling is covered by `pathlib` usage throughout and real Windows path config in Task 1.
- GPU support is implemented through Task 8's device resolver and DINOv2 selector.
- Future alignment and color matching plugin slots are covered by Task 9.

Placeholder scan:

- The plan contains no unfinished placeholder markers.
- The phrase "reserved for phase" is intentional for disabled future stage placeholders and is covered by tests.
- DINOv2 feature extraction is implemented in Task 8 and is the default algorithm in `configs/frame_select.yaml`.

Type consistency:

- `FrameCandidate`, `FrameSelectionResult`, and `FrameSelector.select()` names are consistent across Tasks 3-8.
- `SamplePair.relative_stem` is consistently a `Path`.
- `output_ext` includes the leading dot everywhere.
- `run_pipeline(config)` returns the same summary shape asserted in tests and written to `run_summary.yaml`.
