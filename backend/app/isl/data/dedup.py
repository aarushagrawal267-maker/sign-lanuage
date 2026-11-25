"""Near-duplicate detection across (and within) ISL sources (docs/ISL_DESIGN.md section 5.4).

CISLR was collected from public online ISL videos and may contain the ISLRTC dictionary videos themselves;
INCLUDE stores some word folders twice (for example ``1. loud`` under both Adjectives and Electronics). Pairs
with near-identical landmarks are linked so that they always land on the same side of every split and are
never used as query and reference at the same time.

Candidates: cosine similarity of a compact fingerprint > ``cos_threshold`` (vectorised in blocks);
confirmation: mean per-step distance of the body-frame wrist trajectories after DTW alignment < ``threshold``
(shoulder-width units).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.features.build import FeatureError, to_body_frame
from app.features.spec import HAND_MIDDLE_MCP
from app.landmarks.raw import RawSequence

N_STEPS = 16
_POSE_LW, _POSE_RW = 5, 6


def _wrist_tracks(raw: RawSequence, steps: int) -> np.ndarray:
    """(steps, 4): left and right wrist xy in the body frame, resampled on time; missing -> pose wrist -> 0."""
    bf = to_body_frame(raw, max_gap=5)
    n = len(bf)
    out = np.zeros((n, 4))
    for k, (side, pidx) in enumerate((("left", _POSE_LW), ("right", _POSE_RW))):
        hp = bf.present[side]
        tr = bf.pose[:, pidx, :2].copy()
        tr[hp] = getattr(bf, side)[hp, 0, :2]
        out[:, 2 * k: 2 * k + 2] = tr
    t = bf.t
    tq = np.linspace(t[0], t[-1], steps) if n > 1 else np.full(steps, t[0])
    res = np.stack([np.interp(tq, t, out[:, j]) for j in range(4)], axis=1)
    return res


def _mean_shapes(raw: RawSequence) -> np.ndarray:
    out = []
    for h in (raw.left, raw.right):
        ok = ~np.isnan(h[:, 0, 0])
        if not ok.any():
            out.append(np.zeros(40))
            continue
        rel = h[ok, 1:, :2] - h[ok, :1, :2]
        size = np.linalg.norm(rel[:, HAND_MIDDLE_MCP - 1], axis=1)
        size = np.where(size > 1e-6, size, 1.0)
        out.append((rel / size[:, None, None]).mean(0).reshape(-1))
    return np.concatenate(out)


def fingerprint(raw: RawSequence) -> np.ndarray:
    """16-step resampled wrist tracks + mean hand shapes, centred and L2-normalised (float32)."""
    w = _wrist_tracks(raw, N_STEPS)
    w = w - w.mean(0, keepdims=True)
    v = np.concatenate([w.reshape(-1), 0.25 * _mean_shapes(raw)])
    nrm = np.linalg.norm(v)
    return (v / nrm if nrm > 0 else v).astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-step Euclidean distance along the optimal DTW path (a: (n, d), b: (m, d))."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    L = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
            k = int(np.argmin(choices))
            prev = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[k]
            D[i, j] = cost[i - 1, j - 1] + choices[k]
            L[i, j] = L[prev] + 1
    return float(D[n, m] / max(1.0, L[n, m]))


def find_duplicates(clips: list, threshold: float = 0.02, cos_threshold: float = 0.98,
                    loader: Callable | None = None, block: int = 2048,
                    progress: Callable[[int, int], None] | None = None) -> list[tuple[str, str, float]]:
    """Return confirmed pairs (clip_a, clip_b, dtw_distance) sorted by distance. ``clips`` are
    :class:`~app.isl.data.sources.Clip` objects (or anything with ``clip_id`` and ``load()``)."""
    load = loader or (lambda c: c.load())
    ids, fps, tracks = [], [], []
    for i, c in enumerate(clips):
        try:
            raw = load(c)
            fps.append(fingerprint(raw))
            tracks.append(_wrist_tracks(raw, 32))
            ids.append(c.clip_id)
        except (FeatureError, ValueError, OSError):
            continue
        if progress and i % 200 == 0:
            progress(i, len(clips))
    if len(ids) < 2:
        return []
    F = np.stack(fps)
    pairs = []
    for s in range(0, len(F), block):
        S = F[s: s + block] @ F.T
        ii, jj = np.nonzero(S > cos_threshold)
        for i, j in zip(ii + s, jj):
            if j <= i:
                continue
            d = dtw_distance(tracks[i], tracks[j])
            if d < threshold:
                pairs.append((ids[i], ids[j], round(d, 5)))
    return sorted(pairs, key=lambda x: x[2])


def duplicate_clusters(pairs: list[tuple[str, str, float]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for x in parent:
        groups.setdefault(find(x), []).append(x)
    return [sorted(v) for v in groups.values() if len(v) > 1]
