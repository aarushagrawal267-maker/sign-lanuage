"""Per-frame hand-shape features for fingerspelled letters (docs/ISL_DESIGN.md section 6.4).

The ISL manual alphabet is mostly two-handed (one hand touches the other), so the features describe both
hands and how they relate:

* each hand: wrist-relative, hand-size-normalised 3-D shape (20 points x 3 = 60 values; zeros when missing)
* right-minus-left wrist vector (3 values, body-frame units)
* the 5 fingertips of each hand to the other hand's palm centre (10 distances; 0 when a hand is missing)
* presence flags (2)
* motion summary of a 0.6 s window around the frame (dominant wrist dx, dy and path length: 3 values), which
  separates letters that are drawn (for example J) from static ones

Total ``LETTER_DIM`` = 138. Coordinates come from the same body frame as the word features (``to_body_frame``),
with the opt-in canonical handedness (mirrored when the left hand is dominant).
"""

from __future__ import annotations

import numpy as np

from app.features.build import FeatureError, canonicalize, to_body_frame
from app.features.spec import HAND_MIDDLE_MCP, HAND_WRIST
from app.landmarks.raw import RawSequence

TIPS = [4, 8, 12, 16, 20]
PALM = [0, 5, 9, 13, 17]
LETTER_DIM = 60 * 2 + 3 + 10 + 2 + 3
MOTION_WINDOW_MS = 600.0


def _shape(h: np.ndarray, pres: np.ndarray) -> np.ndarray:
    rel = h - h[:, HAND_WRIST : HAND_WRIST + 1, :]
    size = np.linalg.norm(rel[:, HAND_MIDDLE_MCP, :2], axis=1)
    size = np.where(size > 1e-6, size, 1.0)
    out = (rel[:, 1:, :] / size[:, None, None]).reshape(len(h), -1)
    out[~pres] = 0.0
    return out


def letter_features_seq(raw: RawSequence, canonical: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(T, LETTER_DIM) features for every frame and the (T,) timestamps. Raises ``FeatureError`` without a body."""
    bf = to_body_frame(raw, max_gap=3)
    if canonical:
        bf = canonicalize(bf)
    T = len(bf)
    lp, rp = bf.present["left"], bf.present["right"]
    L, R = bf.left, bf.right
    feats = [_shape(L, lp), _shape(R, rp)]
    both = lp & rp
    d = (R[:, HAND_WRIST] - L[:, HAND_WRIST])
    d[~both] = 0.0
    feats.append(d)
    lpalm, rpalm = L[:, PALM].mean(1), R[:, PALM].mean(1)
    rt = np.linalg.norm(R[:, TIPS] - lpalm[:, None, :], axis=2)
    lt = np.linalg.norm(L[:, TIPS] - rpalm[:, None, :], axis=2)
    rt[~both], lt[~both] = 0.0, 0.0
    feats += [rt, lt, np.stack([lp, rp], 1).astype(np.float64)]
    # motion of the dominant (after canonicalisation: right, else left) wrist in a window around each frame
    t = bf.t
    w = np.where(rp[:, None], R[:, HAND_WRIST, :2], np.where(lp[:, None], L[:, HAND_WRIST, :2], np.nan))
    motion = np.zeros((T, 3))
    half = MOTION_WINDOW_MS / 2
    for i in range(T):
        a, b = np.searchsorted(t, t[i] - half), np.searchsorted(t, t[i] + half, side="right")
        seg = w[a:b]
        seg = seg[~np.isnan(seg[:, 0])]
        if len(seg) >= 2:
            motion[i, :2] = seg[-1] - seg[0]
            motion[i, 2] = np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()
    feats.append(motion)
    out = np.concatenate(feats, axis=1).astype(np.float32)
    assert out.shape[1] == LETTER_DIM, out.shape
    return out, t


def letter_frame_features(raw: RawSequence, i: int) -> np.ndarray:
    return letter_features_seq(raw)[0][i]


__all__ = ["LETTER_DIM", "FeatureError", "letter_features_seq", "letter_frame_features"]
