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
        if len(image_matches) > 1 or len(video_matches) > 1:
            ambiguous.append(sample_id)
            continue
        if not image_matches:
            missing_images.append(sample_id)
            continue
        if not video_matches:
            missing_videos.append(sample_id)
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
