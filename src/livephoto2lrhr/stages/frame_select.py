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
