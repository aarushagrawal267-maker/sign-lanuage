"""Synthetic sequences for trim checks: join two landmark recordings in one body frame."""

from __future__ import annotations

import numpy as np

from app.landmarks.raw import RawSequence


def _frame(raw: RawSequence) -> tuple[np.ndarray, float]:
    asp = raw.aspect
    ls, rs = raw.pose[:, 11, :2].astype(np.float64).copy(), raw.pose[:, 12, :2].astype(np.float64).copy()
    ls[:, 0] *= asp
    rs[:, 0] *= asp
    ok = ~np.isnan(ls[:, 0]) & ~np.isnan(rs[:, 0])
    if not ok.any():
        return np.array([0.5 * asp, 0.5]), 0.2
    c = np.median((ls[ok] + rs[ok]) / 2, axis=0)
    s = float(np.median(np.linalg.norm(ls[ok] - rs[ok], axis=1)))
    return c, s


def align_concat(a: RawSequence, b: RawSequence, gap_frames: int = 0) -> RawSequence:
    """A followed by B, with B moved/scaled so its median shoulder centre and width match A's, and ``gap_frames``
    empty frames between them (synthetic 'sign + explanation' or 'part 1 + part 2' sequences)."""
    ca, sa = _frame(a)
    cb, sb = _frame(b)
    k = sa / sb if sb > 0 else 1.0

    def mv(arr: np.ndarray) -> np.ndarray:
        out = arr.astype(np.float64).copy()
        x = out[..., 0] * b.aspect
        y = out[..., 1]
        out[..., 0] = ((x - cb[0]) * k + ca[0]) / a.aspect
        out[..., 1] = (y - cb[1]) * k + ca[1]
        if out.shape[-1] >= 3:
            out[..., 2] = out[..., 2] * k * b.aspect / a.aspect
        return out.astype(np.float32)

    dt = float(np.median(np.diff(a.t_ms))) if len(a) > 1 else 80.0

    def nanf(shape):
        return np.full(shape, np.nan, np.float32)

    pose_b = mv(b.pose)
    pose_b[..., 3] = b.pose[..., 3]
    t_b = a.t_ms[-1] + dt * (gap_frames + 1) + (b.t_ms - b.t_ms[0])
    return RawSequence(
        np.concatenate([a.pose, nanf((gap_frames,) + a.pose.shape[1:]), pose_b]),
        np.concatenate([a.left, nanf((gap_frames,) + a.left.shape[1:]), mv(b.left)]),
        np.concatenate([a.right, nanf((gap_frames,) + a.right.shape[1:]), mv(b.right)]),
        np.concatenate([a.face, nanf((gap_frames,) + a.face.shape[1:]), mv(b.face)]),
        np.concatenate([a.t_ms, a.t_ms[-1] + dt * np.arange(1, gap_frames + 1), t_b]), a.width, a.height,
        a.extractor_version)


def splice_out(raw: RawSequence, a: int, b: int, bridge: int = 2) -> RawSequence:
    """Remove frames [a, b) and bridge the gap with ``bridge`` linearly interpolated frames between frame a-1 and
    frame b (landmarks missing on either side stay missing). Timestamps stay evenly spaced. Used to turn
    'sign, hands down, explanation' into 'sign straight into explanation' with a known boundary at ``a``."""
    n = len(raw)
    if not (0 < a < b < n):
        raise ValueError("need 0 < a < b < len(raw)")
    w = np.linspace(0, 1, bridge + 2)[1:-1]

    def br(arr: np.ndarray) -> np.ndarray:
        x0, x1 = arr[a - 1].astype(np.float64), arr[b].astype(np.float64)
        mid = np.stack([(1 - k) * x0 + k * x1 for k in w]) if bridge else arr[:0].astype(np.float64)
        return np.concatenate([arr[:a], mid.astype(arr.dtype), arr[b:]])

    dt = float(np.median(np.diff(raw.t_ms))) if n > 1 else 80.0
    T = a + bridge + (n - b)
    t = raw.t_ms[0] + dt * np.arange(T)
    return RawSequence(br(raw.pose), br(raw.left), br(raw.right), br(raw.face), t, raw.width, raw.height,
                       raw.extractor_version)
