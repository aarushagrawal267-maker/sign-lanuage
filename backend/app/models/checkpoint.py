"""Checkpoint save/load and a thread-safe predictor used by live and offline inference."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch

from app.features.spec import FeatureConfig
from app.models.networks import build_model


def frame_mask(x: np.ndarray | torch.Tensor, cfg: FeatureConfig):
    """Frames with at least one detected hand. Falls back to all frames if none."""
    flags_start = cfg.dim - cfg.flag_dim
    if isinstance(x, np.ndarray):
        m = (x[..., flags_start] > 0.5) | (x[..., flags_start + 1] > 0.5)
        if m.ndim == 1:
            return m if m.any() else np.ones_like(m)
        m[~m.any(axis=1)] = True
        return m
    m = (x[..., flags_start] > 0.5) | (x[..., flags_start + 1] > 0.5)
    none = ~m.any(dim=1)
    m[none] = True
    return m


def save_checkpoint(path: Path, model: torch.nn.Module, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "meta": meta}, tmp)
    tmp.replace(path)


class Predictor:
    """Loaded model + its feature config. ``predict`` is safe to call from threads."""

    def __init__(self, path: Path, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.meta: dict = ck["meta"]
        self.path = Path(path)
        self.feature_config = FeatureConfig.from_dict(self.meta["feature_config"])
        self.class_names: list[str] = self.meta["class_names"]
        mk = self.meta["model_kwargs"]
        self.model = build_model(self.meta["architecture"], self.feature_config.dim, len(self.class_names), **mk)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self._lock = threading.Lock()
        self.recommended_threshold: float = float(self.meta.get("recommended_threshold", 0.6))

    @torch.inference_mode()
    def predict(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        """feats: (T, D) or (B, T, D). Returns (probs, inference_ms)."""
        single = feats.ndim == 2
        x = torch.from_numpy(feats[None] if single else feats).float().to(self.device)
        m = frame_mask(x, self.feature_config)
        with self._lock:
            t0 = time.perf_counter()
            logits = self.model(x, m)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000
        return (probs[0] if single else probs), ms
