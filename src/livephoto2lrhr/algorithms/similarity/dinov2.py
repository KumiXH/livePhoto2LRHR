from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult
from livephoto2lrhr.data.image_io import open_pil_image
from livephoto2lrhr.utils.device import resolve_device


DINOV2_EXTRA_MESSAGE = "Install the dinov2 extra to use dinov2_similarity: pip install -e .[dinov2]"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(DINOV2_EXTRA_MESSAGE) from exc
    return torch


def _build_transform(resize_short_side: int) -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(DINOV2_EXTRA_MESSAGE) from exc

    return transforms.Compose(
        [
            transforms.Resize(resize_short_side, antialias=True),
            transforms.CenterCrop(resize_short_side),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _cosine_similarity(left: Any, right: Any) -> float:
    torch = _import_torch()
    return torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()


class DINOv2SimilaritySelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.sample_fps = float(config.get("sample_fps", 15.0))
        self.top_k = int(config.get("top_k", 5))
        self.resize_short_side = int(config.get("resize_short_side", 518))
        self._validate_config()

        requested_device = str(config.get("device", "auto"))
        self.device = resolve_device(requested_device)
        self.torch = _import_torch()
        self.transform = _build_transform(self.resize_short_side)
        self.model_name = "dinov2_vits14"
        self.model = self.torch.hub.load("facebookresearch/dinov2", self.model_name).eval().to(self.device)

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        with open_pil_image(image_path) as image:
            target_image = ImageOps.exif_transpose(image).convert("RGB")
            target_feature = self._extract_feature(target_image)

        capture = cv2.VideoCapture(str(video_path))
        candidates: list[FrameCandidate] = []
        best_candidate: FrameCandidate | None = None
        best_frame_rgb: np.ndarray | None = None
        frame_index = 0

        try:
            if not capture.isOpened():
                raise ValueError(f"could not open video: {video_path}")

            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            frame_step = max(int(round(fps / self.sample_fps)), 1)
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    feature = self._extract_feature(Image.fromarray(frame_rgb))
                    score = _cosine_similarity(target_feature, feature)
                    candidate = FrameCandidate(
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps,
                        score=score,
                    )
                    candidates.append(candidate)
                    if best_candidate is None or score > best_candidate.score:
                        best_candidate = candidate
                        best_frame_rgb = frame_rgb.copy()
                frame_index += 1
        finally:
            capture.release()

        if not candidates:
            raise ValueError(f"video contained no readable frames: {video_path}")

        ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        selected = ranked[0]
        if best_frame_rgb is None or best_candidate is None or best_candidate.frame_index != selected.frame_index:
            best_frame_rgb = self._read_frame(video_path, selected.frame_index)
        return FrameSelectionResult(
            frame_rgb=best_frame_rgb,
            selected=selected,
            top_k=ranked[: self.top_k],
            diagnostics={
                "algorithm": "dinov2_similarity",
                "model": self.model_name,
                "device": self.device,
                "sample_fps": self.sample_fps,
                "frame_step": frame_step,
                "scored_frames": len(candidates),
            },
        )

    def _validate_config(self) -> None:
        if not np.isfinite(self.sample_fps):
            raise ValueError("sample_fps must be finite")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be greater than 0")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.resize_short_side < 1:
            raise ValueError("resize_short_side must be at least 1")
        if self.resize_short_side % 14 != 0:
            raise ValueError("resize_short_side must be divisible by 14 for dinov2_vits14")

    def _extract_feature(self, image: Image.Image) -> Any:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feature = self.model(tensor)
        return feature.cpu()

    def _rank_features(
        self, target_feature: Any, candidate_features: list[tuple[FrameCandidate, Any]]
    ) -> list[FrameCandidate]:
        ranked = [
            FrameCandidate(
                frame_index=candidate.frame_index,
                timestamp_sec=candidate.timestamp_sec,
                score=_cosine_similarity(target_feature, feature),
            )
            for candidate, feature in candidate_features
        ]
        ranked.sort(key=lambda candidate: candidate.score, reverse=True)
        return ranked

    def _read_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"could not open video: {video_path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok:
                raise ValueError(f"could not read selected frame {frame_index} from video: {video_path}")
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
