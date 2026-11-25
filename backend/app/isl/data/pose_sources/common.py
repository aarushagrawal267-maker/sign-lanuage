"""Shared helpers for the pose-native ISL sources (GloSL-ISL, ISL500, OpenHands raw / fingerspelling).

These sources ship landmarks instead of videos, so ingestion is a format conversion into the app's
``RawSequence`` layout (``app/landmarks/raw.py``) with no MediaPipe run. Helpers here are pure and testable:

* :func:`save_compact` writes the usual ``RawSequence`` ``.npz`` keys (same layout as ``RawSequence.save``),
  optionally as ``float16`` and/or temporally strided. ``RawSequence.load`` reads both; the adapters'
  :class:`PoseClip` upcasts to ``float32`` on load so downstream code always sees the usual dtype.
  Storage options are recorded in the manifest (``dtype``, ``stride``, ``fps_native``) and in the file
  (``storage`` key, ignored by ``RawSequence.load``).
* :func:`hand_aspect` estimates a clip's frame aspect ratio (W/H) from the rigidity of MediaPipe hand bones
  (hand x and z share the width scale, y uses the height scale, so the correct aspect makes bone lengths most
  constant over time). Used to sanity-check headers whose width/height may be placeholders.
* :class:`DiskGuard` stops a job before free disk space falls below a floor.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from app.isl.data.sources import Clip
from app.landmarks.raw import RawSequence

DEFAULT_TARGET_FPS = 15.0  # INCLUDE/CISLR are extracted at 15 fps, ISLRTC at 12 fps (scripts/ingest_isl.py)
MIN_FREE_GB = 12.0  # stop before the shared disk drops under the 10 GB floor (other jobs keep writing)

HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


# --------------------------------------------------------------------------------------------- storage

def stride_for(fps: float, target_fps: float) -> int:
    """Frame step that brings ``fps`` closest to ``target_fps`` (same rule as ``scripts/ingest_isl.py``);
    ``target_fps <= 0`` keeps every frame."""
    if not target_fps or target_fps <= 0 or not fps or fps <= 0:
        return 1
    return max(1, int(round(fps / target_fps)))


def stride_seq(seq: RawSequence, stride: int) -> RawSequence:
    if stride <= 1:
        return seq
    return RawSequence(seq.pose[::stride], seq.left[::stride], seq.right[::stride], seq.face[::stride],
                       seq.t_ms[::stride], seq.width, seq.height, seq.extractor_version)


def save_compact(seq: RawSequence, path: Path, dtype: str = "float32", stride: int = 1) -> RawSequence:
    """Save ``seq`` (strided by ``stride``) as a ``RawSequence`` ``.npz`` with landmark arrays in ``dtype``
    (``float32`` or ``float16``; ``t_ms`` stays ``float64``). Atomic (tmp file + rename). Returns the saved
    (strided, not down-cast) sequence."""
    if dtype not in ("float32", "float16"):
        raise ValueError(f"unsupported dtype {dtype!r}")
    seq = stride_seq(seq, stride)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    dt = np.float16 if dtype == "float16" else np.float32
    try:
        np.savez_compressed(
            tmp, pose=seq.pose.astype(dt), left=seq.left.astype(dt), right=seq.right.astype(dt),
            face=seq.face.astype(dt), t_ms=seq.t_ms.astype(np.float64), width=int(seq.width),
            height=int(seq.height), extractor_version=seq.extractor_version,
            storage=f"{dtype};stride={int(stride)}",
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return seq


def load_float32(path: Path) -> RawSequence:
    raw = RawSequence.load(Path(path))
    if raw.pose.dtype != np.float32:
        raw = RawSequence(raw.pose.astype(np.float32), raw.left.astype(np.float32), raw.right.astype(np.float32),
                          raw.face.astype(np.float32), raw.t_ms.astype(np.float64), raw.width, raw.height,
                          raw.extractor_version)
    return raw


@dataclass(frozen=True)
class PoseClip(Clip):
    """A :class:`Clip` whose ``load`` always returns ``float32`` landmarks (pose sources may store float16)."""

    profiles: tuple[str, ...] = field(default=())

    def load(self) -> RawSequence:
        raw = load_float32(self.path)
        if self.start or self.end is not None:
            raw = raw.slice(self.start, self.end if self.end is not None else len(raw))
        return raw

    def with_slice(self, start: int, end: int | None) -> PoseClip:
        return replace(self, start=start, end=end)


# --------------------------------------------------------------------------------------------- quality

def seq_stats(seq: RawSequence) -> dict:
    """Manifest quality fields shared with ``scripts/ingest_isl.py`` (n_frames, hands_frac) plus per-part rates."""
    n = len(seq)
    if not n:
        return {"n_frames": 0, "hands_frac": 0.0}
    r = seq.detection_rates()
    return {"n_frames": n, "hands_frac": round(r["any_hand"], 4), "pose_frac": round(r["pose"], 4),
            "left_frac": round(r["left"], 4), "right_frac": round(r["right"], 4), "face_frac": round(r["face"], 4)}


def hand_aspect(hands: list[np.ndarray], grid: np.ndarray | None = None, min_frames: int = 5) -> tuple[float, float, int] | None:
    """(aspect W/H, mean bone-length coefficient of variation, frames used) or None.

    ``hands`` are (T, 21, 3) arrays in normalised coordinates (NaN = missing frame). Validated on our own
    INCLUDE/ISLRTC extractions (true 16:9 = 1.78): per-clip estimates 1.6-2.6, median about 1.8, so use it
    as a coarse check (square vs landscape vs portrait), not a measurement."""
    grid = np.linspace(0.4, 2.6, 111) if grid is None else grid
    segs = []
    n = 0
    for h in hands:
        h = np.asarray(h, dtype=np.float64)
        ok = ~np.isnan(h[:, 0, 0])
        h = h[ok]
        if len(h) < min_frames:
            continue
        n += len(h)
        segs.append(np.stack([h[:, j] - h[:, i] for i, j in HAND_BONES]))  # (B, T, 3)
    if not segs:
        return None
    a = np.concatenate(segs, axis=1)
    best = None
    for g in grid:
        length = np.sqrt(a[..., 0] ** 2 + (a[..., 1] / g) ** 2 + a[..., 2] ** 2)
        mean = length.mean(axis=1)
        if np.any(mean <= 0):
            continue
        cv = float(np.mean(length.std(axis=1) / mean))
        if best is None or cv < best[1]:
            best = (float(g), cv)
    return None if best is None else (round(best[0], 3), round(best[1], 4), n)


# --------------------------------------------------------------------------------------------- disk

def free_gb(path: Path) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


class DiskFull(RuntimeError):
    pass


@dataclass
class DiskGuard:
    """``check()`` raises :class:`DiskFull` when free space under ``path`` is below ``min_free_gb``.
    ``budget_gb`` (optional) also caps the size of ``budget_dir`` (measured every ``every`` checks)."""

    path: Path
    min_free_gb: float = MIN_FREE_GB
    budget_dir: Path | None = None
    budget_gb: float | None = None
    every: int = 200
    _n: int = 0
    _used: float = 0.0

    def used_gb(self) -> float:
        if self.budget_dir is None or not Path(self.budget_dir).exists():
            return 0.0
        total = 0
        for root, _dirs, files in os.walk(self.budget_dir):
            for f in files:
                try:
                    total += os.stat(os.path.join(root, f)).st_blocks * 512
                except OSError:
                    pass
        return total / 1e9

    def check(self, force: bool = False) -> None:
        self._n += 1
        if not force and self._n % max(1, self.every) != 1:
            return
        free = free_gb(self.path)
        if free < self.min_free_gb:
            raise DiskFull(f"free disk {free:.1f} GB < {self.min_free_gb} GB floor")
        if self.budget_gb is not None:
            self._used = self.used_gb()
            if self._used > self.budget_gb:
                raise DiskFull(f"{self.budget_dir} uses {self._used:.1f} GB > budget {self.budget_gb} GB")


def safe_name(s: str, limit: int = 150) -> str:
    """File-name-safe version of a source id (letters, digits, ``._-``; others -> ``_``)."""
    import re

    out = re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("._")[:limit]
    return out or "x"


def active_window(seq: RawSequence, pad: int = 2) -> dict:
    """Frames where a hand is visible: ``{"active_start", "active_end", "hands_frac_active"}`` (end exclusive,
    padded by ``pad`` frames). Word clips with long rest periods (ISL500: hands visible in ~30 % of the frames)
    are sliced to this window by their adapter; the ``.npz`` keeps every frame."""
    n = len(seq)
    if not n:
        return {"active_start": 0, "active_end": 0, "hands_frac_active": 0.0}
    p = seq.presence()
    hands = p["left"] | p["right"]
    idx = np.flatnonzero(hands)
    if not len(idx):
        return {"active_start": 0, "active_end": n, "hands_frac_active": 0.0}
    s, e = max(0, int(idx[0]) - pad), min(n, int(idx[-1]) + 1 + pad)
    return {"active_start": s, "active_end": e, "hands_frac_active": round(float(hands[s:e].mean()), 4)}
