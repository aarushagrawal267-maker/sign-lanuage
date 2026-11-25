"""Synthetic fixtures for the ISL data tests (no network, no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.features.spec import N_FACE, N_HAND, N_POSE
from app.landmarks.raw import RawSequence

SW = 0.2  # shoulder width in image units (square 480x480 frame, so x and y share the scale)
CENTER = np.array([0.5, 0.45])


def _hand(wrist_xy: np.ndarray, phase: float = 0.0) -> np.ndarray:
    pts = np.zeros((N_HAND, 3), np.float32)
    pts[0, :2] = wrist_xy
    for f in range(5):
        ang = np.deg2rad(-40 + 20 * f) + 0.3 * np.sin(phase)
        for j in range(4):
            r = 0.012 * (j + 1)
            pts[1 + 4 * f + j, :2] = wrist_xy + np.array([np.sin(ang) * r, -np.cos(ang) * r])
    return pts


def body_to_img(bx: float, by: float, center=CENTER, sw=SW) -> np.ndarray:
    return center + sw * np.array([bx, by])


def make_trim_sequence(segments: list[tuple[str, int]], fps: float = 12.0, seed: int = 0, hands_at_rest: bool = False,
                       center=CENTER, sw=SW) -> RawSequence:
    """Build a sequence from (kind, n_frames) segments. kinds:
    rest   both wrists low (y = 1.6 shoulder widths below the shoulder line), hands not detected
    sign   both hands up in signing space, moving
    hold   both hands up, still
    expl   hands up, moving with a different pattern (an 'explanation')
    low    both hands dropped to the waist (y = 0.9), detected, still
    """
    rng = np.random.default_rng(seed)
    T = sum(n for _, n in segments)
    pose = np.full((T, N_POSE, 4), np.nan, np.float32)
    left = np.full((T, N_HAND, 3), np.nan, np.float32)
    right = np.full((T, N_HAND, 3), np.nan, np.float32)
    face = np.full((T, N_FACE, 3), np.nan, np.float32)
    i = 0
    for kind, n in segments:
        for k in range(n):
            ph = k / max(1, n - 1) * 2 * np.pi
            if kind == "rest":
                wl, wr, det = (0.9, 1.6), (-0.9, 1.6), hands_at_rest
            elif kind == "low":
                wl, wr, det = (0.4, 0.9), (-0.4, 0.9), True
            elif kind == "hold":
                wl, wr, det = (0.3, 0.1), (-0.3, 0.1), True
            elif kind == "sign":
                wl, wr, det = (0.3 + 0.35 * np.sin(ph), 0.1 + 0.3 * np.cos(ph)), (-0.3 - 0.35 * np.sin(ph), 0.1), True
            elif kind == "expl":
                wl, wr, det = (0.2 * np.cos(2 * ph), -0.2 + 0.25 * np.sin(2 * ph)), (-0.4, 0.3 * np.sin(3 * ph)), True
            else:
                raise ValueError(kind)
            p = np.zeros((N_POSE, 4), np.float32)
            p[:, 3] = 0.99
            p[:, :2] = center
            p[0, :2] = body_to_img(0, -1.1, center, sw)
            p[11, :2] = body_to_img(0.5, 0, center, sw)
            p[12, :2] = body_to_img(-0.5, 0, center, sw)
            p[15, :2] = body_to_img(*wl, center, sw)
            p[16, :2] = body_to_img(*wr, center, sw)
            p[13, :2] = (p[11, :2] + p[15, :2]) / 2
            p[14, :2] = (p[12, :2] + p[16, :2]) / 2
            p[23, :2] = body_to_img(0.4, 2.0, center, sw)
            p[24, :2] = body_to_img(-0.4, 2.0, center, sw)
            p[:, :2] += rng.normal(0, 0.0005, (N_POSE, 2))
            pose[i] = p
            if det:
                hp = 0.0 if kind in ("hold", "low") else ph  # still hands keep their shape
                left[i] = _hand(p[15, :2], hp)
                right[i] = _hand(p[16, :2], hp)
            face[i, :, :2] = p[0, :2] + np.linspace(-0.02, 0.02, N_FACE)[:, None]
            face[i, :, 2] = 0
            i += 1
    t = np.arange(T) * 1000.0 / fps
    return RawSequence(pose, left, right, face, t, 480, 480)


def write_manifest(path: Path, rows: list[dict], partial_tail: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        if partial_tail:
            fh.write(partial_tail)
