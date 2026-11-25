"""Landmark-content duplicate finder for pre-training data (docs/ISL_DESIGN.md 15.4, item 5).

The OpenHands **Sign Library** channel ships two files (``sign_library_1_min_splits.h5``: 374 one-minute chunks of
76 named videos; ``sign_library_sub_based.h5``: 9,582 hash-named subtitle chunks). Names cannot be matched, so the
same video may be ingested twice, and nothing by name says whether a chunk repeats an evaluation clip either. This
module matches by **pose content** instead:

* a frame is described by 7 upper-body pose points (nose, shoulders, elbows, wrists), x/y in normalised image
  coordinates (``frame_vectors``; frames without both shoulders are skipped);
* a query clip **duplicates** a reference clip when, for at least ``min_score`` of its consecutive frame pairs, both
  frames have one of their 4 nearest reference frames closer than ``eps`` *in that same reference clip* and the
  reference frame index moves forward by 0-2 frames (``FrameIndex.match``). The time-order condition is what separates a re-cut copy of
  the same recording from a different video of the same presenter in the same studio (rest poses match, motion
  sequences do not).

:func:`calibrate` measures the rule before it is trusted: **positive controls** are synthetic re-cuts of reference
chunks (a random window, re-sampled half a frame off the stored frames by interpolation, plus landmark jitter, as a
second extraction / a different stride offset would give), **negative controls** are windows of reference chunks
queried against an index *without* their own video. The score threshold is the one with the best recall among those
with a false-positive rate <= 0.01; the dedupe is called reliable only when that recall is >= 0.95 (``RELIABLE``).
Otherwise the whole channel is dropped from pre-training (``apply_dedup``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

POSE_POINTS = (0, 11, 12, 13, 14, 15, 16)
# Measured on the real Sign Library files (25 Nov 2026): the 16 sub-based chunks that repeat a one-minute chunk
# match it at a median frame distance of 0.014-0.023 with time-order scores 0.5-1.0; every other chunk scores
# <= 0.33 even at eps 0.05 (same presenter / studio, different content). Consecutive stored frames (12.5 fps)
# are 0.04 apart (median).
EPS = 0.04  # normalised units
MIN_SCORE_FLOOR = 0.2  # the calibrated threshold is never set lower than this
RELIABLE = {"recall": 0.95, "false_positive_rate": 0.01}
SIGN_LIBRARY = "Sign_Library"
DEDUP_FILE = "pretrain_dedup.json"


def frame_vectors(raw) -> np.ndarray:
    """(n, 14) float32: x/y of ``POSE_POINTS`` for frames where both shoulders are visible."""
    pose = np.asarray(raw.pose, dtype=np.float32)
    if pose.ndim != 3 or len(pose) == 0:
        return np.zeros((0, 2 * len(POSE_POINTS)), np.float32)
    vis = pose[:, [11, 12], 3] if pose.shape[2] > 3 else np.ones((len(pose), 2), np.float32)
    ok = (vis.min(1) > 0.5) & np.isfinite(pose[:, POSE_POINTS, :2]).all((1, 2)) \
        & (np.abs(pose[:, POSE_POINTS, :2]).sum((1, 2)) > 0)  # missing pose = zeros
    return pose[ok][:, POSE_POINTS, :2].reshape(int(ok.sum()), -1)


def _vectors_of(item) -> np.ndarray:
    if isinstance(item, np.ndarray):
        return item.astype(np.float32)
    return frame_vectors(item.load())


class FrameIndex:
    """Nearest-neighbour index over the frames of many reference clips (``owner`` = reference position)."""

    def __init__(self, refs: list[tuple[str, np.ndarray]]):
        from sklearn.neighbors import NearestNeighbors

        X, own, fi = [], [], []
        self.ids = []
        for k, (rid, v) in enumerate(refs):
            self.ids.append(rid)
            if len(v):
                X.append(v)
                own.append(np.full(len(v), k))
                fi.append(np.arange(len(v)))
        self.n_frames = int(sum(len(x) for x in X))
        self.X = np.concatenate(X) if X else np.zeros((0, 2 * len(POSE_POINTS)), np.float32)
        self.owner = np.concatenate(own) if own else np.zeros(0, int)
        self.fidx = np.concatenate(fi) if fi else np.zeros(0, int)
        self.nn = NearestNeighbors(n_neighbors=1).fit(self.X) if len(self.X) else None

    def match(self, q: np.ndarray, eps: float = EPS, k: int = 4) -> tuple[str | None, float]:
        """(best reference id, score) for one query clip's frame vectors. The best reference is the clip owning
        most close neighbours (``k`` nearest frames per query frame); a query frame counts when one of its
        neighbours within ``eps`` lies in that clip; the score is the fraction of consecutive query-frame pairs that
        both count and move forward 0-2 frames in the reference."""
        if self.nn is None or len(q) < 4:
            return None, 0.0
        k = max(1, min(k, len(self.X)))
        d, j = self.nn.kneighbors(q, n_neighbors=k)
        o, f = self.owner[j], self.fidx[j]
        close = d < eps
        if not close.any():
            return None, 0.0
        best = int(np.bincount(o[close]).argmax())
        inb_all = close & (o == best)
        inb = inb_all.any(1)
        # the closest neighbour inside the best clip gives the reference frame index of each query frame
        dd = np.where(inb_all, d, np.inf)
        fi = f[np.arange(len(q)), dd.argmin(1)]
        step = fi[1:] - fi[:-1]
        ok = inb[1:] & inb[:-1] & (step >= 0) & (step <= 2)
        return self.ids[best], float(ok.mean())


def match_score(index: FrameIndex, q: np.ndarray, eps: float = EPS) -> float:
    return index.match(q, eps)[1]


def _resample_half(v: np.ndarray, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """A synthetic re-cut: frames half-way between the stored ones (another stride offset), plus jitter."""
    mid = (v[:-1] + v[1:]) / 2.0
    return (mid + rng.normal(0.0, jitter, mid.shape)).astype(np.float32)


def calibrate(refs: list[tuple[str, str, np.ndarray]], n: int = 200, seed: int = 0, eps: float = EPS,
              jitter: float = 0.003, win: tuple[int, int] = (25, 60)) -> dict:
    """Recall on synthetic re-cuts and false positives on other videos; picks the score threshold with the best
    recall among those with a false-positive rate <= 1 %. ``refs`` = ``(clip id, video id, vectors)``."""
    rng = np.random.default_rng(seed)
    usable = [r for r in refs if len(r[2]) >= win[0] + 2]
    if len(usable) < 4 or len({r[1] for r in usable}) < 2:
        return {"n": 0, "reliable": False, "reason": "too few reference clips / videos to calibrate"}
    idx = FrameIndex([(r[0], r[2]) for r in refs])
    pos, neg = [], []
    picks = rng.choice(len(usable), min(n, len(usable)), replace=len(usable) < n)
    for k in picks:
        cid, vid, v = usable[int(k)]
        L = int(rng.integers(win[0], min(win[1], len(v) - 1) + 1))
        a = int(rng.integers(0, len(v) - L))
        sub = v[a:a + L + 1]
        rid, sc = idx.match(_resample_half(sub, rng, jitter), eps)
        pos.append(sc if rid == cid else 0.0)
    by_video: dict[str, list] = {}
    for r in refs:
        by_video.setdefault(r[1], []).append(r)
    videos = sorted(v for v in by_video if any(len(x[2]) >= win[0] for x in by_video[v]))
    chosen = rng.choice(videos, min(len(videos), 30), replace=False)
    per_video = max(1, -(-n // max(1, len(chosen))))
    for vid in chosen:
        other = FrameIndex([(r[0], r[2]) for r in refs if r[1] != vid])
        mine = [x for x in by_video[vid] if len(x[2]) >= win[0]]
        for t in range(per_video):
            _, _, v = mine[t % len(mine)]
            L = int(rng.integers(win[0], min(win[1], len(v)) + 1))
            a = int(rng.integers(0, len(v) - L + 1))
            neg.append(other.match(v[a:a + L], eps)[1])
    P, N = np.asarray(pos), np.asarray(neg)
    curve = [{"min_score": round(float(t), 2), "recall": round(float((P >= t).mean()), 4),
              "false_positive_rate": round(float((N >= t).mean()), 4)} for t in np.arange(0.1, 0.95, 0.05)]
    ok = [c for c in curve if c["false_positive_rate"] <= RELIABLE["false_positive_rate"]]
    pick = max(ok, key=lambda c: (c["recall"], -c["min_score"])) if ok else None
    chosen_t = pick["min_score"] if pick else None
    return {"n_positive": len(pos), "n_negative": len(neg), "eps": eps, "jitter": jitter,
            "min_score": chosen_t, "recall": pick["recall"] if pick else None,
            "false_positive_rate": pick["false_positive_rate"] if pick else None,
            "reliable": bool(pick and pick["recall"] >= RELIABLE["recall"]), "rule": RELIABLE, "curve": curve,
            "max_negative_score": round(float(N.max()), 3) if len(N) else None}


@dataclass
class DedupResult:
    reliable: bool
    drop_ids: list[str]
    drop_channels: list[str]
    report: dict
    checked_ids: list[str] = field(default_factory=list)  # every Sign Library chunk this check looked at

    def to_json(self) -> dict:
        return {"reliable": self.reliable, "drop_ids": sorted(self.drop_ids), "drop_channels": self.drop_channels,
                "checked_ids": sorted(self.checked_ids), **self.report}


def dedupe_sign_library(sl_clips: list, eval_clips: Iterable, *, eps: float = EPS, n_calib: int = 300, seed: int = 0,
                        progress=None) -> DedupResult:
    """Find Sign Library chunks that repeat another Sign Library chunk (the two OpenHands files) or contain an
    evaluation clip. Returns the ids to drop from pre-training, or - when the matcher is not shown reliable on
    this data - the whole channel."""
    log = progress or (lambda m: None)
    t0 = datetime.now(UTC)
    one_min = [c for c in sl_clips if "1_min" in str(c.quality.get("member") or "")]
    others = [c for c in sl_clips if c not in one_min]
    vec = {}
    for c in sl_clips:
        try:
            vec[c.clip_id] = _vectors_of(c)
        except (OSError, ValueError, KeyError):
            vec[c.clip_id] = np.zeros((0, 2 * len(POSE_POINTS)), np.float32)
    log(f"Sign Library: {len(one_min)} one-minute chunks, {len(others)} other chunks, "
        f"{sum(len(v) for v in vec.values())} frames")
    cal = calibrate([(c.clip_id, str(c.group), vec[c.clip_id]) for c in one_min], n=n_calib, seed=seed, eps=eps)
    log(f"calibration: { {k: v for k, v in cal.items() if k != 'curve'} }")
    min_score = max(MIN_SCORE_FLOOR, cal.get("min_score") or 1.0)
    rep = {"created": t0.strftime("%Y-%m-%dT%H:%M:%SZ"), "channel": SIGN_LIBRARY, "eps": eps, "min_score": min_score,
           "calibration": cal, "chunks": {"one_minute": len(one_min), "other": len(others)}}
    if not cal.get("reliable"):
        rep["decision"] = ("matcher not shown reliable on this data: the whole Sign Library channel is excluded from "
                           "pre-training")
        return DedupResult(False, [], [SIGN_LIBRARY], rep, [c.clip_id for c in sl_clips])
    drop: set[str] = set()
    within = []
    idx1 = FrameIndex([(c.clip_id, vec[c.clip_id]) for c in one_min])
    for c in others:
        rid, sc = idx1.match(vec[c.clip_id], eps)
        if sc >= min_score:
            drop.add(c.clip_id)
            within.append({"id": c.clip_id, "matches": rid, "score": round(sc, 3)})
    log(f"within-channel copies: {len(within)} of {len(others)} chunks")
    idx_all = FrameIndex([(c.clip_id, vec[c.clip_id]) for c in sl_clips])
    ev_hits, n_eval = [], 0
    for e in eval_clips:
        try:
            q = _vectors_of(e)
        except (OSError, ValueError, KeyError):
            continue
        n_eval += 1
        rid, sc = idx_all.match(q, eps)
        if sc >= min_score and rid is not None:
            drop.add(rid)
            ev_hits.append({"eval_clip": e.clip_id, "chunk": rid, "score": round(sc, 3)})
    log(f"evaluation clips found inside Sign Library chunks: {len(ev_hits)} of {n_eval}")
    rep.update({"within_copies": len(within), "within_examples": within[:20], "eval_clips_checked": n_eval,
                "eval_matches": ev_hits[:200], "n_eval_matches": len(ev_hits),
                "decision": "reliable: drop the matched chunks only",
                "seconds": round((datetime.now(UTC) - t0).total_seconds(), 1)})
    return DedupResult(True, sorted(drop), [], rep, [c.clip_id for c in sl_clips])


def load_dedup(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def apply_dedup(clips: list, dedup: dict | None) -> tuple[list, dict]:
    """Pre-training clips minus the dropped Sign Library chunks. Without a dedupe result the whole Sign Library
    channel is dropped (conservative default: overlap unknown). A chunk the dedupe never looked at (ingested after
    the pretrain-dedup step, left out of a smoke sample, or a result file without ``checked_ids``) is dropped too:
    nothing says it is not a copy of an evaluation clip."""
    def is_sl(c) -> bool:
        return c.source == "openhands_raw" and c.category == SIGN_LIBRARY

    sl = [c for c in clips if is_sl(c)]
    if not sl:
        return clips, {"sign_library_chunks": 0}
    if dedup is None:
        keep = [c for c in clips if not is_sl(c)]
        return keep, {"sign_library_chunks": len(sl), "dropped": len(sl),
                      "why": "no pretrain_dedup.json: Sign Library excluded (overlap not checked)"}
    channels = set(dedup.get("drop_channels") or [])
    ids = set(dedup.get("drop_ids") or [])
    checked = set(dedup.get("checked_ids") or [])
    unchecked = sum(1 for c in sl if c.clip_id not in checked)
    keep = [c for c in clips if not (c.source == "openhands_raw" and (c.category in channels or c.clip_id in ids))
            and not (is_sl(c) and c.clip_id not in checked)]
    return keep, {"sign_library_chunks": len(sl), "dropped": len(clips) - len(keep), "reliable": dedup.get("reliable"),
                  "unchecked_dropped": unchecked, "why": dedup.get("decision") or dedup.get("reason")}
