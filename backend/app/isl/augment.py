"""Strong augmentation for ISL (docs/ISL_DESIGN.md section 6.2).

Composes the existing ``augment`` (crop, speed, mirror, similarity transform, jitter, hand dropout) with ops that
simulate other signers and cameras, which matters because most ISL dictionary entries have a single example:

* arm-length scaling about the shoulders (other body proportions)
* per-finger joint-angle noise (hand-shape variation)
* contiguous hand-dropout chunks (detector misses of several frames)
* a 0-2 frame time shift between the hands
* face-subset dropout
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.features.build import AugmentConfig, BodyFrame, augment

# indices within POSE_FEATURE_IDX = [nose, l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist]
_SHOULDER = {"left": 1, "right": 2}
_ARM = {"left": (3, 5), "right": (4, 6)}
_FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


@dataclass(frozen=True)
class StrongAugmentConfig:
    base: AugmentConfig = field(default_factory=lambda: AugmentConfig(
        p_mirror=0.0, rotate_deg=15.0, scale=0.15, shift=0.15, crop_frac=0.15, speed=0.35,
        landmark_dropout=0.08, jitter=0.012))
    arm_scale: float = 0.10
    finger_angle_deg: float = 4.0
    hand_chunk_drop_p: float = 0.2
    hand_chunk_max: int = 4
    hand_shift_max: int = 2
    face_drop_p: float = 0.1


def _rot2(a: float) -> np.ndarray:
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def augment_strong(bf: BodyFrame, rng: np.random.Generator, cfg: StrongAugmentConfig = StrongAugmentConfig()) -> BodyFrame:
    bf = bf.copy()
    T = len(bf)
    # arm-length scaling about each shoulder (elbow, wrist and that hand move together)
    if cfg.arm_scale > 0:
        for side in ("left", "right"):
            k = 1.0 + rng.uniform(-cfg.arm_scale, cfg.arm_scale)
            sh = bf.pose[:, _SHOULDER[side], :2][:, None, :]
            for j in _ARM[side]:
                bf.pose[:, j, :2] = sh[:, 0] + (bf.pose[:, j, :2] - sh[:, 0]) * k
            h = getattr(bf, side)
            pres = bf.present[side]
            moved = sh + (h[..., :2] - sh) * k
            h[..., :2] = np.where(pres[:, None, None], moved, h[..., :2])
    # per-finger joint-angle noise: rotate each finger chain about its base joint in the image plane
    if cfg.finger_angle_deg > 0:
        for side in ("left", "right"):
            h = getattr(bf, side)
            pres = bf.present[side]
            if not pres.any():
                continue
            for chain in _FINGERS:
                ang = np.deg2rad(rng.normal(0, cfg.finger_angle_deg))
                R = _rot2(ang)
                base = h[:, chain[0], :2][:, None, :]
                pts = h[:, chain[1:], :2]
                h[:, chain[1:], :2] = np.where(pres[:, None, None], base + (pts - base) @ R.T, pts)
    # time shift between the hands
    if cfg.hand_shift_max > 0 and T > 4:
        s = int(rng.integers(-cfg.hand_shift_max, cfg.hand_shift_max + 1))
        if s:
            side = "left" if rng.random() < 0.5 else "right"
            h, p = getattr(bf, side), bf.present[side]
            setattr(bf, side, np.roll(h, s, axis=0))
            bf.present[side] = np.roll(p, s)
    # contiguous hand dropout chunks
    if cfg.hand_chunk_drop_p > 0 and T > 4:
        for side in ("left", "right"):
            if rng.random() < cfg.hand_chunk_drop_p:
                n = int(rng.integers(1, cfg.hand_chunk_max + 1))
                a = int(rng.integers(0, max(1, T - n)))
                getattr(bf, side)[a:a + n] = 0.0
                bf.present[side][a:a + n] = False
    if cfg.face_drop_p > 0 and rng.random() < cfg.face_drop_p:
        bf.face[:] = 0.0
        bf.present["face"][:] = False
    return augment(bf, rng, cfg.base)
