"""Raw landmarks -> fixed-length feature sequence.

This module is used unchanged by training, offline video analysis and live
webcam inference, which guarantees identical preprocessing everywhere.

Pipeline
--------
1. Aspect correction: MediaPipe x is a fraction of image width and y of image
   height; x (and hand z) are multiplied by width/height so distances are isotropic.
2. Short gaps (<= ``max_gap`` frames) in each component are linearly
   interpolated. Longer gaps stay missing and are flagged.
3. Body frame: origin = median shoulder midpoint, unit = median shoulder width
   over the sequence. Sequences with no pose detection are rejected.
4. Optional augmentation (training only) in the body frame.
5. Temporal resampling to ``seq_len`` steps using timestamps.
6. Features per step: upper-body pose, wrist-relative scale-normalized hand
   shape, hand position in the body frame, face subset, presence flags and
   (optionally) first differences (velocity).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.features.spec import (
    HAND_MIDDLE_MCP,
    HAND_WRIST,
    POSE_FEATURE_IDX,
    POSE_L_SHOULDER,
    POSE_R_SHOULDER,
    FeatureConfig,
)
from app.landmarks.raw import RawSequence

# Mirror permutations (positions within POSE_FEATURE_IDX / FACE_IDX).
POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5]
FACE_MIRROR = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10, 19, 18, 17, 16]

MIN_SHOULDER_WIDTH = 1e-3


class FeatureError(ValueError):
    """The sequence cannot be turned into features (e.g. no body detected)."""


@dataclass
class BodyFrame:
    """Landmarks expressed in the signer-centred body frame (pre-resampling)."""

    pose: np.ndarray  # (T, 7, 2)
    left: np.ndarray  # (T, 21, 3)
    right: np.ndarray  # (T, 21, 3)
    face: np.ndarray  # (T, K, 2)
    present: dict[str, np.ndarray]  # bool (T,) per component, after gap filling
    t: np.ndarray  # (T,) ms
    center: np.ndarray  # (2,) original aspect-corrected coords
    scale: float

    def __len__(self) -> int:
        return int(self.t.shape[0])

    def copy(self) -> BodyFrame:
        return BodyFrame(self.pose.copy(), self.left.copy(), self.right.copy(), self.face.copy(),
                         {k: v.copy() for k, v in self.present.items()}, self.t.copy(), self.center.copy(), self.scale)


def fill_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate interior runs of missing frames no longer than max_gap.

    ``arr`` is (T, ...) with NaN marking missing frames (checked on element 0).
    Leading/trailing gaps and long gaps are left as NaN.
    """
    out = arr.copy()
    T = arr.shape[0]
    if T == 0 or max_gap <= 0:
        return out
    miss = np.isnan(arr.reshape(T, -1)[:, 0])
    i = 0
    while i < T:
        if not miss[i]:
            i += 1
            continue
        j = i
        while j < T and miss[j]:
            j += 1
        # gap is [i, j)
        if i > 0 and j < T and (j - i) <= max_gap:
            a, b = arr[i - 1], arr[j]
            for k in range(i, j):
                w = (k - (i - 1)) / (j - (i - 1))
                out[k] = (1 - w) * a + w * b
        i = j
    return out


def to_body_frame(raw: RawSequence, max_gap: int = 5) -> BodyFrame:
    if len(raw) == 0:
        raise FeatureError("empty sequence")
    asp = raw.aspect
    pose = raw.pose[..., :3].astype(np.float64).copy()
    left = raw.left.astype(np.float64).copy()
    right = raw.right.astype(np.float64).copy()
    face = raw.face.astype(np.float64).copy()
    for a in (pose, left, right, face):
        a[..., 0] *= asp
    for a in (left, right):
        a[..., 2] *= asp  # hand z shares the x (width) scale

    pose, left, right, face = (fill_gaps(a, max_gap) for a in (pose, left, right, face))
    present = {
        "pose": ~np.isnan(pose[:, 0, 0]),
        "left": ~np.isnan(left[:, 0, 0]),
        "right": ~np.isnan(right[:, 0, 0]),
        "face": ~np.isnan(face[:, 0, 0]),
    }
    if not present["pose"].any():
        raise FeatureError("no body (pose) detected in any frame")

    ls, rs = pose[present["pose"], POSE_L_SHOULDER, :2], pose[present["pose"], POSE_R_SHOULDER, :2]
    center = np.median((ls + rs) / 2.0, axis=0)
    scale = float(np.median(np.linalg.norm(ls - rs, axis=1)))
    if not np.isfinite(scale) or scale < MIN_SHOULDER_WIDTH:
        raise FeatureError("shoulders not resolved (zero shoulder width)")

    def norm_xy(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        out[..., :2] = (a[..., :2] - center) / scale
        if out.shape[-1] == 3:
            out[..., 2] = a[..., 2] / scale
        return out

    bf = BodyFrame(
        pose=norm_xy(pose[:, POSE_FEATURE_IDX, :2]),
        left=norm_xy(left),
        right=norm_xy(right),
        face=norm_xy(face[..., :2]),
        present=present,
        t=raw.t_ms.astype(np.float64).copy(),
        center=center,
        scale=scale,
    )
    for comp in ("left", "right", "face"):
        arr = getattr(bf, comp)
        arr[~present[comp]] = 0.0
    bf.pose[~present["pose"]] = 0.0
    # Keep timestamps strictly increasing.
    if len(bf) > 1 and np.any(np.diff(bf.t) <= 0):
        bf.t = np.arange(len(bf), dtype=np.float64) * 40.0
    return bf


def mirror_body_frame(bf: BodyFrame) -> BodyFrame:
    """The same sign performed with the other dominant hand (left/right swapped, x negated). In place."""
    bf.pose = bf.pose[:, POSE_MIRROR].copy()
    bf.pose[..., 0] *= -1
    bf.face = bf.face[:, FACE_MIRROR].copy()
    bf.face[..., 0] *= -1
    bf.left, bf.right = bf.right.copy(), bf.left.copy()
    bf.left[..., 0] *= -1
    bf.right[..., 0] *= -1
    bf.present["left"], bf.present["right"] = bf.present["right"].copy(), bf.present["left"].copy()
    return bf


def dominant_hand(bf: BodyFrame) -> str | None:
    """'left' / 'right' = the hand whose wrist travels further (summed path length over frames where it is
    present in two consecutive frames); None on a tie (including no hands)."""
    lens = {}
    for name in ("left", "right"):
        w = getattr(bf, name)[:, 0, :2]
        ok = bf.present[name][1:] & bf.present[name][:-1]
        d = np.linalg.norm(np.diff(w, axis=0), axis=1)
        lens[name] = float(d[ok].sum()) if len(d) else 0.0
    if abs(lens["left"] - lens["right"]) < 1e-9:
        return None
    return "left" if lens["left"] > lens["right"] else "right"


def canonicalize(bf: BodyFrame) -> BodyFrame:
    """Mirror when the left hand is dominant (``FeatureConfig.canonical_hand``)."""
    return mirror_body_frame(bf.copy()) if dominant_hand(bf) == "left" else bf


def prepare_body_frame(raw: RawSequence, cfg: FeatureConfig) -> BodyFrame:
    """``to_body_frame`` plus the opt-in canonical handedness: the one entry point used by training caches,
    offline analysis and live inference."""
    bf = to_body_frame(raw, cfg.max_gap)
    return canonicalize(bf) if getattr(cfg, "canonical_hand", False) else bf


# --- augmentation -------------------------------------------------------------

@dataclass(frozen=True)
class AugmentConfig:
    p_mirror: float = 0.5
    rotate_deg: float = 10.0
    scale: float = 0.1
    shift: float = 0.1
    crop_frac: float = 0.15  # drop up to this fraction from each end
    speed: float = 0.2  # +/- relative change of time axis density
    landmark_dropout: float = 0.05  # probability of dropping a hand in a frame
    jitter: float = 0.01


def augment(bf: BodyFrame, rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig()) -> BodyFrame:
    bf = bf.copy()
    T = len(bf)
    # temporal crop
    if T > 8 and cfg.crop_frac > 0:
        a = int(rng.uniform(0, cfg.crop_frac) * T)
        b = T - int(rng.uniform(0, cfg.crop_frac) * T)
        if b - a >= 4:
            sl = slice(a, b)
            bf = BodyFrame(bf.pose[sl], bf.left[sl], bf.right[sl], bf.face[sl],
                           {k: v[sl] for k, v in bf.present.items()}, bf.t[sl], bf.center, bf.scale)
    # non-uniform speed: warp time axis
    if cfg.speed > 0 and len(bf) > 2:
        k = rng.uniform(-cfg.speed, cfg.speed)
        u = np.linspace(0, 1, len(bf))
        warped = u + k * u * (1 - u)
        bf.t = bf.t[0] + warped * (bf.t[-1] - bf.t[0])
        if np.any(np.diff(bf.t) <= 0):
            bf.t = np.sort(bf.t) + np.arange(len(bf)) * 1e-3
    # mirror = same sign performed with the other dominant hand
    if rng.random() < cfg.p_mirror:
        mirror_body_frame(bf)
    # global similarity transform in the image plane
    th = np.deg2rad(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    sc = 1.0 + rng.uniform(-cfg.scale, cfg.scale)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * sc
    sh = rng.uniform(-cfg.shift, cfg.shift, size=2)
    for name in ("pose", "left", "right", "face"):
        arr = getattr(bf, name)
        pres = bf.present[name]
        xy = arr[..., :2] @ R.T + sh
        xy += rng.normal(0, cfg.jitter, size=xy.shape)
        arr[..., :2] = np.where(pres[:, None, None], xy, 0.0)
        if arr.shape[-1] == 3:
            arr[..., 2] *= sc
    # random hand dropout (simulates missed detections)
    if cfg.landmark_dropout > 0:
        for name in ("left", "right"):
            drop = (rng.random(len(bf)) < cfg.landmark_dropout) & bf.present[name]
            getattr(bf, name)[drop] = 0.0
            bf.present[name] = bf.present[name] & ~drop
    return bf


# --- resampling & featurization ----------------------------------------------

def _resample(arr: np.ndarray, present: np.ndarray, t: np.ndarray, tq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resample (T, ...) to query times; linear when both neighbours are present,
    nearest-neighbour otherwise (never blends a detection with a missing frame)."""
    T = arr.shape[0]
    if T == 1:
        return np.repeat(arr, len(tq), axis=0), np.repeat(present, len(tq))
    idx = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, T - 2)
    t0, t1 = t[idx], t[idx + 1]
    w = np.clip((tq - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    p0, p1 = present[idx], present[idx + 1]
    both = p0 & p1
    nearest = np.where(w < 0.5, idx, idx + 1)
    shape = (-1,) + (1,) * (arr.ndim - 1)
    lin = arr[idx] * (1 - w).reshape(shape) + arr[idx + 1] * w.reshape(shape)
    out = np.where(both.reshape(shape), lin, arr[nearest])
    pres = np.where(both, True, present[nearest])
    out[~pres] = 0.0
    return out, pres


def _hand_shape(h: np.ndarray, pres: np.ndarray, floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(L,21,3) body-frame hand -> wrist-relative, hand-size-normalized shape (L,60) and wrist pos (L,2).
    ``floor`` > 0 bounds the size from below (``FeatureConfig.hand_size_floor``); 0 keeps the original behaviour."""
    wrist = h[:, HAND_WRIST : HAND_WRIST + 1, :]
    rel = h - wrist
    size = np.linalg.norm(rel[:, HAND_MIDDLE_MCP, :2], axis=1)
    size = np.maximum(size, floor) if floor > 0 else np.where(size > 1e-6, size, 1.0)
    shape = (rel[:, 1:, :] / size[:, None, None]).reshape(h.shape[0], -1)
    shape[~pres] = 0.0
    pos = h[:, HAND_WRIST, :2].copy()
    pos[~pres] = 0.0
    return shape, pos


def featurize(bf: BodyFrame, cfg: FeatureConfig) -> np.ndarray:
    L = cfg.seq_len
    t = bf.t
    tq = np.linspace(t[0], t[-1], L) if len(bf) > 1 else np.full(L, t[0])
    pose, pp = _resample(bf.pose, bf.present["pose"], t, tq)
    left, lp = _resample(bf.left, bf.present["left"], t, tq)
    right, rp = _resample(bf.right, bf.present["right"], t, tq)
    lshape, lpos = _hand_shape(left, lp, cfg.hand_size_floor)
    rshape, rpos = _hand_shape(right, rp, cfg.hand_size_floor)
    static = [pose.reshape(L, -1), lshape, lpos, rshape, rpos]
    flags = [lp[:, None], rp[:, None]]
    comp_present = [pp, lp, lp, rp, rp]
    if cfg.use_face:
        face, fp = _resample(bf.face, bf.present["face"], t, tq)
        static.append(face.reshape(L, -1))
        flags.append(fp[:, None])
        comp_present.append(fp)
    X = [np.concatenate(static, axis=1)]
    if cfg.use_velocity:
        vel = []
        for block, pres in zip(static, comp_present):
            d = np.diff(block, axis=0, prepend=block[:1])
            ok = pres & np.concatenate([[pres[0]], pres[:-1]])
            d[~ok] = 0.0
            vel.append(d * (L / 16.0))  # keep magnitudes comparable across seq_len
        X.append(np.concatenate(vel, axis=1))
    X.append(np.concatenate(flags, axis=1).astype(np.float64))
    out = np.concatenate(X, axis=1).astype(np.float32)
    assert out.shape == (L, cfg.dim), (out.shape, cfg.dim)
    return out


def build_features(raw: RawSequence, cfg: FeatureConfig, rng: np.random.Generator | None = None,
                   aug: AugmentConfig | None = None) -> np.ndarray:
    bf = prepare_body_frame(raw, cfg)
    if rng is not None:
        bf = augment(bf, rng, aug or AugmentConfig())
    return featurize(bf, cfg)


def sequence_quality(raw: RawSequence) -> dict:
    rates = raw.detection_rates()
    n = len(raw)
    return {
        "frames": n,
        **{f"{k}_rate": round(v, 4) for k, v in rates.items()},
        "usable": n >= 4 and rates["pose"] > 0 and rates["any_hand"] >= 0.2,
    }
