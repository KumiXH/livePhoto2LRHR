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
