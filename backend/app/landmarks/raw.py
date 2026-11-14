"""Raw landmark sequences: what MediaPipe produced, before any normalization.

Missing detections are stored as NaN so later stages can tell "not detected"
apart from a real coordinate. Raw sequences are what gets saved to disk, so the
feature definition can change without re-running MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.features.spec import EXTRACTOR_VERSION, N_FACE, N_HAND, N_POSE


@dataclass
class RawSequence:
    pose: np.ndarray  # (T, 33, 4)  x, y, z, visibility
    left: np.ndarray  # (T, 21, 3)
    right: np.ndarray  # (T, 21, 3)
    face: np.ndarray  # (T, N_FACE, 3)
    t_ms: np.ndarray  # (T,) timestamps in milliseconds
    width: int
    height: int
    extractor_version: str = EXTRACTOR_VERSION

    def __len__(self) -> int:
        return int(self.pose.shape[0])

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height) if self.height else 1.0

    @classmethod
    def empty(cls, width: int = 640, height: int = 480) -> RawSequence:
        return cls(
            pose=np.zeros((0, N_POSE, 4), np.float32),
            left=np.zeros((0, N_HAND, 3), np.float32),
            right=np.zeros((0, N_HAND, 3), np.float32),
            face=np.zeros((0, N_FACE, 3), np.float32),
            t_ms=np.zeros((0,), np.float64),
            width=width,
            height=height,
        )

    def slice(self, start: int, end: int) -> RawSequence:
        return RawSequence(
            self.pose[start:end], self.left[start:end], self.right[start:end], self.face[start:end],
            self.t_ms[start:end], self.width, self.height, self.extractor_version,
        )

    def presence(self) -> dict[str, np.ndarray]:
        return {
            "pose": ~np.isnan(self.pose[:, 0, 0]),
            "left": ~np.isnan(self.left[:, 0, 0]),
            "right": ~np.isnan(self.right[:, 0, 0]),
            "face": ~np.isnan(self.face[:, 0, 0]),
        }

    def detection_rates(self) -> dict[str, float]:
        n = max(1, len(self))
        p = self.presence()
        return {
            "pose": float(p["pose"].sum() / n),
            "left": float(p["left"].sum() / n),
            "right": float(p["right"].sum() / n),
            "any_hand": float((p["left"] | p["right"]).sum() / n),
            "face": float(p["face"].sum() / n),
        }

    # --- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp, pose=self.pose.astype(np.float32), left=self.left.astype(np.float32),
            right=self.right.astype(np.float32), face=self.face.astype(np.float32),
            t_ms=self.t_ms.astype(np.float64), width=self.width, height=self.height,
            extractor_version=self.extractor_version,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> RawSequence:
        z = np.load(path, allow_pickle=False)
        return cls(
            pose=z["pose"], left=z["left"], right=z["right"], face=z["face"], t_ms=z["t_ms"],
            width=int(z["width"]), height=int(z["height"]), extractor_version=str(z["extractor_version"]),
        )

    # --- JSON frames (browser) -------------------------------------------
    @classmethod
    def from_frames(cls, frames: list[dict], width: int, height: int) -> RawSequence:
        """Build from browser frames: {t, pose, left, right, face} with null when absent."""
        T = len(frames)
        seq = cls(
            pose=np.full((T, N_POSE, 4), np.nan, np.float32),
            left=np.full((T, N_HAND, 3), np.nan, np.float32),
            right=np.full((T, N_HAND, 3), np.nan, np.float32),
            face=np.full((T, N_FACE, 3), np.nan, np.float32),
            t_ms=np.zeros((T,), np.float64),
            width=width,
            height=height,
        )
        for i, f in enumerate(frames):
            fill_frame(seq, i, f)
        return seq

    def frame_to_json(self, i: int) -> dict:
        def arr(a):
            return None if np.isnan(a[0, 0]) else np.round(a, 6).tolist()

        return {"t": float(self.t_ms[i]), "pose": arr(self.pose[i]), "left": arr(self.left[i]),
                "right": arr(self.right[i]), "face": arr(self.face[i])}


def _put(dst: np.ndarray, value, n: int, k: int) -> None:
    if value is None:
        dst[:] = np.nan
        return
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (n, k):
        raise ValueError(f"expected landmark array of shape ({n}, {k}), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("landmark values must be finite numbers")
    dst[:] = a


def fill_frame(seq: RawSequence, i: int, f: dict) -> None:
    seq.t_ms[i] = float(f.get("t", i * 40.0))
    _put(seq.pose[i], f.get("pose"), N_POSE, 4)
    _put(seq.left[i], f.get("left"), N_HAND, 3)
    _put(seq.right[i], f.get("right"), N_HAND, 3)
    _put(seq.face[i], f.get("face"), N_FACE, 3)
