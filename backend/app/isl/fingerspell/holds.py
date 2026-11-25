"""Hold detection: fingerspelled letters are short still hand shapes separated by transitions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.features.build import to_body_frame
from app.landmarks.raw import RawSequence


@dataclass
class Hold:
    start: int  # frame index (inclusive)
    end: int  # exclusive
    mean_features: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    motion: float = 0.0  # mean hand speed during the hold (shoulder widths / s)
    t_start: float = 0.0
    t_end: float = 0.0


def hand_speed(raw: RawSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame speed (shoulder widths per second) of the hands: wrist speed plus mean fingertip speed relative to
    the wrist, max over the present hands. Returns (speed, any_hand_present, t_ms)."""
    bf = to_body_frame(raw, max_gap=3)
    t = bf.t
    T = len(bf)
    sp = np.zeros(T)
    present = bf.present["left"] | bf.present["right"]
    if T < 2:
        return sp, present, t
    dt = np.maximum(np.diff(t), 1.0) / 1000.0
    for name in ("left", "right"):
        h, p = getattr(bf, name), bf.present[name]
        ok = p[1:] & p[:-1]
        w = np.linalg.norm(np.diff(h[:, 0, :2], axis=0), axis=1)
        rel = h[:, [4, 8, 12, 16, 20], :2] - h[:, :1, :2]
        f = np.linalg.norm(np.diff(rel, axis=0), axis=2).mean(1)
        s = np.where(ok, (w + f) / dt, 0.0)
        sp[1:] = np.maximum(sp[1:], s)
    sp[0] = sp[1]
    # light smoothing (3 frames)
    k = np.ones(3) / 3
    sp = np.convolve(np.pad(sp, 1, mode="edge"), k, mode="valid")
    return sp, present, t


def detect_holds(raw: RawSequence, min_ms: float = 180.0, v_max: float = 0.35, feats: np.ndarray | None = None,
                 max_gap_frames: int = 1) -> list[Hold]:
    """Runs of frames with a hand present and speed below ``v_max`` lasting at least ``min_ms``. Single-frame
    speed spikes inside a run (``max_gap_frames``) do not split it. ``feats`` (T, F) fills ``mean_features``."""
    if len(raw) < 2:
        return []
    sp, present, t = hand_speed(raw)
    still = present & (sp < v_max)
    # close tiny gaps
    T = len(still)
    i = 0
    while i < T:
        if not still[i]:
            j = i
            while j < T and not still[j]:
                j += 1
            if 0 < i and j < T and j - i <= max_gap_frames and present[i:j].all():
                still[i:j] = True
            i = j
        else:
            i += 1
    holds: list[Hold] = []
    i = 0
    while i < T:
        if still[i]:
            j = i
            while j < T and still[j]:
                j += 1
            dur = t[j - 1] - t[i] + (t[1] - t[0] if T > 1 else 0.0)
            if dur >= min_ms:
                mf = feats[i:j].mean(0) if feats is not None else np.zeros(0)
                holds.append(Hold(i, j, mf, float(sp[i:j].mean()), float(t[i]), float(t[j - 1])))
            i = j
        else:
            i += 1
    return holds
