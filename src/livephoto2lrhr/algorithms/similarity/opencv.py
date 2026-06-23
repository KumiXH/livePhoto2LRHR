from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from livephoto2lrhr.algorithms.similarity.base import FrameCandidate, FrameSelectionResult
from livephoto2lrhr.data.image_io import open_pil_image
from PIL import ImageOps


class OpenCVSimilaritySelector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.sample_fps = float(config.get("sample_fps", 15.0))
        self.top_k = int(config.get("top_k", 5))
        self.resize_short_side = int(config.get("resize_short_side", 512))
        fusion = config.get("score_fusion") or {}
        self.feature_weight = float(fusion.get("feature_weight", 0.7))
        self.edge_weight = float(fusion.get("edge_weight", 0.3))
        self._validate_config()

    def select(self, image_path: Path, video_path: Path) -> FrameSelectionResult:
        with open_pil_image(image_path) as image:
            target = np.array(ImageOps.exif_transpose(image).convert("RGB"))
        target_small = self._resize_for_score(target)
        target_gray = cv2.cvtColor(target_small, cv2.COLOR_RGB2GRAY)
        target_edges = cv2.Canny(target_gray, 80, 160)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(int(round(fps / self.sample_fps)), 1)
        candidates: list[FrameCandidate] = []
        best_candidate: FrameCandidate | None = None
        best_frame_rgb: np.ndarray | None = None
        frame_index = 0

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % frame_step == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    score = self._score(target_small, target_edges, frame_rgb)
                    candidate = FrameCandidate(
                        frame_index=frame_index,
                        timestamp_sec=frame_index / fps,
                        score=score,
                    )
                    candidates.append(candidate)
                    if best_candidate is None or candidate.score > best_candidate.score:
                        best_candidate = candidate
                        best_frame_rgb = frame_rgb.copy()
                frame_index += 1
        finally:
            capture.release()

        if not candidates:
            raise ValueError(f"video contained no readable frames: {video_path}")

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        selected = candidates[0]
        top_k = candidates[: self.top_k]
        if best_frame_rgb is None:
            raise ValueError(f"video contained no readable frames: {video_path}")
        return FrameSelectionResult(
            frame_rgb=best_frame_rgb,
            selected=selected,
            top_k=top_k,
            diagnostics={
                "algorithm": "opencv_similarity",
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
        self._validate_weight("feature_weight", self.feature_weight)
        self._validate_weight("edge_weight", self.edge_weight)
        if self.feature_weight == 0 and self.edge_weight == 0:
            raise ValueError("at least one score fusion weight must be greater than 0")

    def _validate_weight(self, name: str, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    def _resize_for_score(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        short_side = min(height, width)
        if short_side == self.resize_short_side:
            return image_rgb
        scale = self.resize_short_side / short_side
        new_size = (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1))
        return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)

    def _score(self, target_small: np.ndarray, target_edges: np.ndarray, frame_rgb: np.ndarray) -> float:
        frame_small = cv2.resize(
            frame_rgb,
            (target_small.shape[1], target_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        pixel_mse = np.mean((target_small.astype(np.float32) - frame_small.astype(np.float32)) ** 2)
        pixel_score = 1.0 / (1.0 + pixel_mse)

        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_RGB2GRAY)
        frame_edges = cv2.Canny(frame_gray, 80, 160)
        edge_mse = np.mean((target_edges.astype(np.float32) - frame_edges.astype(np.float32)) ** 2)
        edge_score = 1.0 / (1.0 + edge_mse)

        return self.feature_weight * pixel_score + self.edge_weight * edge_score
