"""Metric-learning sign encoder (docs/ISL_DESIGN.md section 6.3 b).

Same trunk as the closed-set ``RecurrentClassifier`` (input projection -> BiGRU -> masked attention pooling),
followed by a linear embedding head and L2 normalisation. Cosine similarity between embeddings is the
retrieval score used by the nearest-neighbour index over every dictionary entry and every user sign.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.features.spec import FeatureConfig
from app.models.checkpoint import frame_mask
from app.models.networks import InputProjection, MaskedAttentionPool


class SignEncoder(nn.Module):
    """InputProjection -> BiGRU(hidden, layers) -> MaskedAttentionPool -> Linear(2*hidden -> emb_dim) -> L2."""

    def __init__(self, in_dim: int, hidden: int = 192, layers: int = 2, emb_dim: int = 256, dropout: float = 0.3,
                 input_dropout: float = 0.1):
        super().__init__()
        self.inp = InputProjection(in_dim, hidden, input_dropout)
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.pool = MaskedAttentionPool(2 * hidden)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, emb_dim))
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return F.normalize(self.head(self.pool(h, mask)), dim=-1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, replace: torch.Tensor | None = None,
               token: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Embedding and per-frame states ``(B, T, 2*hidden)``. ``replace`` (B, T) bool swaps those frames'
        projected inputs for ``token`` (masked-frame pre-training)."""
        e = self.inp(x)
        if replace is not None and token is not None:
            e = torch.where(replace[..., None], token.to(e.dtype).expand_as(e), e)
        h, _ = self.rnn(e)
        return F.normalize(self.head(self.pool(h, mask)), dim=-1), h


def load_encoder_weights(enc: SignEncoder, path: Path) -> dict:
    """Warm-start ``enc`` from an encoder checkpoint (pre-trained or earlier fine-tune). Parameters whose shape
    differs (another embedding size) are skipped and listed. Returns the checkpoint meta + the load report."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    own = enc.state_dict()
    ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(sd) - set(ok))
    own.update(ok)
    enc.load_state_dict(own)
    return {"meta": ck.get("meta", {}), "loaded": len(ok), "skipped": skipped, "path": str(path)}


def encoder_kwargs(meta: dict) -> dict:
    return {k: meta["model_kwargs"][k] for k in ("hidden", "layers", "emb_dim", "dropout") if k in meta["model_kwargs"]}


class EncoderPredictor:
    """Loaded encoder + feature config; ``embed`` is thread-safe. Checkpoint format = ``save_checkpoint``'s
    ``{"state_dict", "meta"}`` with ``meta["kind"] == "isl-encoder"`` and ``meta["encoder_version"]``."""

    def __init__(self, path: Path, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.meta: dict = ck["meta"]
        if self.meta.get("kind") != "isl-encoder":
            raise ValueError(f"{path} is not an ISL encoder checkpoint (kind={self.meta.get('kind')!r})")
        self.path = Path(path)
        self.feature_config = FeatureConfig.from_dict(self.meta["feature_config"])
        self.version: str = self.meta["encoder_version"]
        self.model = SignEncoder(self.feature_config.dim, **encoder_kwargs(self.meta))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(device)
        self.device = device
        self.dim = self.model.emb_dim
        self._lock = threading.Lock()

    @torch.inference_mode()
    def embed(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        """feats (T, D) or (B, T, D) -> (B, emb_dim) float32 unit vectors, ms."""
        x = torch.from_numpy(feats[None] if feats.ndim == 2 else feats).float().to(self.device)
        with self._lock:
            t0 = time.perf_counter()
            z = self.model(x, frame_mask(x, self.feature_config)).cpu().numpy()
            ms = (time.perf_counter() - t0) * 1000
        return z.astype(np.float32), ms


class FeatureStatsEmbedder:
    """Model-free fallback embedding: per-block mean and standard deviation of the feature sequence over frames
    with a hand, L2-normalised. Used by "My signs" when the pack has no trained encoder yet (template matching
    of a user's own takes works surprisingly well for the same signer and camera)."""

    version = "featstats-1"

    def __init__(self, feature_config: FeatureConfig):
        self.feature_config = feature_config
        self.dim = 2 * feature_config.dim

    def embed(self, feats: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        x = feats[None] if feats.ndim == 2 else feats
        out = []
        for seq in x:
            m = frame_mask(seq, self.feature_config)
            s = seq[m] if m.any() else seq
            v = np.concatenate([s.mean(0), s.std(0)])
            out.append(v / max(1e-8, float(np.linalg.norm(v))))
        return np.asarray(out, np.float32), (time.perf_counter() - t0) * 1000


class RawEmbedder:
    """Landmark clips -> embeddings with one feature config (pack encoder, or the feature-statistics fallback)."""

    def __init__(self, embedder, feature_config: FeatureConfig | None = None):
        self.embedder = embedder
        self.feature_config = feature_config or embedder.feature_config
        self.version: str = embedder.version
        self.dim: int = embedder.dim

    def embed_raws(self, raws: list) -> np.ndarray:
        """Raises ``FeatureError`` when a clip has no body."""
        from app.features.build import build_features

        if not raws:
            return np.zeros((0, self.dim), np.float32)
        feats = np.stack([build_features(r, self.feature_config) for r in raws])
        return self.embedder.embed(feats)[0]
