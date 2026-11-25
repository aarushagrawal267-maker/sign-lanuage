"""ISLRTC first-sign refinement, stage 1 heuristic (docs/ISL_DESIGN.md section 5.3).

Each ISLRTC dictionary video shows the sign first and then, often, an explanation. The ingestion job cut at
the title card when it could, but many clips (``stop=max_s``) still contain part of the explanation. This
module finds the frame range of the **first sign** from the landmarks alone and writes it to a sidecar
(``data/isl/derived/islrtc_trim.jsonl``); the ``.npz`` files are never modified.

Signals (body frame: origin = shoulder midpoint, unit = shoulder width, y grows downwards):

* wrist track per hand: the hand model's wrist when the hand is detected, else the pose model's wrist
  (hands at rest are often outside the hand detector's view but still placed by the pose model);
* ``up``: wrist above ``active_above`` (signing space); ``rest``: both wrists below ``rest_below`` or absent;
  ``dip``: both wrists below ``dip_below`` (hands lowered to the waist);
* ``speed``: max wrist speed of the two hands (units/s, smoothed over ~150 ms, real timestamps);
* ``shape_change``: mean fingertip speed relative to the wrist (units/s): hand-shape changes.

Algorithm: start = first active run (hand up or moving) lasting ``min_active_ms``; end = earliest of
*rest* (both hands down for ``rest_ms``), *dip* (both wrists drop to the waist for ``dip_ms``: in
the ISLRTC videos the signer usually lowers both hands briefly between the sign and the explanation and
then keeps signing, without a title card), *pause* (both still for ``pause_ms`` after ``min_sign_ms``),
*source_end* (the clip ends: title card or end of file) and *cap* (no boundary inside the sign window).

trim-2 changes (measured in docs/DATASETS_ISL.md section 7; tooling in ``trim_bench.py``):

* *cap* no longer takes the lowest hand position in the last 40% of the 4 s window; it scores every frame in
  [0.6, 1.8] x the expected sign length (2.2 s, +0.6 s for multiword names; medians of bounded clips) by a
  log-length prior plus hand height (``cap_scores``). On same-signer synthetic "sign straight into the
  explanation" sequences this lifted cap ends within 300 ms from 35% to 66%.
* every record gets a ``second`` cut (also in ``alternatives``) at least 400 ms from the main cut: the next
  rest/dip (compounds that drop the hands between their parts), the last inner rest/dip of a bounded clip,
  or the best length-prior frame. Training crops and the second index row should use it.
* bounded clips up to 2 x the sign window keep their whole sign (long compounds run 5-8 s).
* ``card`` clips that end with the hands still up are flagged ``truncated`` (the title-card detector of the
  ingestion job also fires on plain backgrounds and then cuts mid-sign; re-extract them with ``--full``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from app.features.build import FeatureError, to_body_frame
from app.landmarks.raw import RawSequence

# positions inside POSE_FEATURE_IDX = [nose, L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist]
_POSE_LW, _POSE_RW = 5, 6
_TIPS = [4, 8, 12, 16, 20]

METHOD_CONFIDENCE = {"rest": 0.9, "dip": 0.75, "pause": 0.7, "source_end": 0.8, "clip_end": 0.5, "cap": 0.3, "empty": 0.0,
                     "full": 0.8, "learned": 0.85, "truncated": 0.1}


@dataclass(frozen=True)
class TrimConfig:
    rest_below: float = 1.1  # wrist y (body frame, shoulder-width units, y down) below this = hand at rest
    active_above: float = 0.75  # a wrist above this line counts as "up" (signing space) for the start
    dip_below: float = 0.7  # both wrists below this line (hands dropped to the waist) = "dip"
    dip_ms: float = 150.0  # a dip this long after min_sign_ms ends the sign
    v_move: float = 0.6  # wrist speed (units/s) above which a hand is "moving"
    v_hold: float = 0.25  # speed below which both hands are "holding"
    min_active_ms: float = 250.0  # first active run must last at least this long
    min_sign_ms: float = 400.0
    pause_ms: float = 450.0  # a hold this long after min_sign_ms ends the sign
    rest_ms: float = 200.0  # both hands down/absent this long ends the sign
    max_sign_ms: float = 4000.0
    per_word_min_ms: float = 600.0  # multiword names (compound signs): min_sign_ms grows per extra word ...
    per_word_max_ms: float = 1500.0  # ... and so does max_sign_ms
    max_min_sign_ms: float = 2400.0
    hard_max_sign_ms: float = 8000.0
    tail_ms: float = 150.0
    smooth_ms: float = 150.0
    pre_roll: int = 2
    # trim-2 (docs/DATASETS_ISL.md 7)
    bounded_max_factor: float = 2.0  # a bounded clip longer than this x the sign window is treated as unbounded
    # no boundary at all (continuous signing): end chosen by a sign-length prior + motion minima (cap_scores)
    cap_prior: bool = True
    cap_mu_ms: float = 2200.0
    cap_mu_extra_ms: float = 600.0
    cap_w_prior: float = 0.25
    cap_w_motion: float = 0.0
    cap_w_low: float = 2.0
    second_sep_ms: float = 400.0  # the second cut must be at least this far from the main cut
    # unbounded: a first rest/dip later than this x the expected length takes its second cut from the prior
    late_factor: float | None = 1.3
    version: str = "trim-2"


@dataclass
class TrimResult:
    start: int
    end: int  # frame slice [start, end)
    t_start_ms: float
    t_end_ms: float
    method: str  # rest | dip | pause | source_end | clip_end | cap | truncated | empty | full | learned
    confidence: float
    alternatives: list[int] = field(default_factory=list)  # up to 2 other plausible end frames
    notes: list[str] = field(default_factory=list)
    # trim-2: the best other cut (exclusive end frame), for a second index row / training crop; None if no
    # plausible second cut exists. Always one of ``alternatives``.
    second: int | None = None

    def to_record(self, clip_id: str, n_frames: int, version: str) -> dict:
        d = asdict(self)
        d.update({"id": clip_id, "n_frames": n_frames, "trim_version": version,
                  "t_start_ms": round(self.t_start_ms, 1), "t_end_ms": round(self.t_end_ms, 1)})
        return d


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where mask is True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _run_ms(t: np.ndarray, a: int, b: int, dt: float) -> float:
    return float(t[b - 1] - t[a]) + dt


def activity_signal(raw: RawSequence, cfg: TrimConfig = TrimConfig()) -> dict[str, np.ndarray]:
    """Per-frame signals: wrist_y_left/right, up_left/up_right, rest, speed, shape_change, present,
    hand_left/hand_right (hand model detected), t (ms)."""
    bf = to_body_frame(raw, max_gap=3)
    n = len(bf)
    t = bf.t.astype(np.float64)
    dts = np.diff(t)
    dt = float(np.median(dts)) if n > 1 else 83.3
    dt = dt if dt > 0 else 83.3
    k = max(1, int(round(cfg.smooth_ms / dt)))
    if k % 2 == 0:
        k += 1
    out: dict[str, np.ndarray] = {"t": t}
    speeds, shapes = [], []
    pose_ok = bf.present["pose"]
    for side, pidx in (("left", _POSE_LW), ("right", _POSE_RW)):
        hand = getattr(bf, side)
        hp = bf.present[side]
        track = np.full((n, 2), np.nan)
        track[pose_ok] = bf.pose[pose_ok, pidx, :2]
        track[hp] = hand[hp, 0, :2]
        y = track[:, 1]
        out[f"wrist_y_{side}"] = y
        out[f"hand_{side}"] = hp
        out[f"up_{side}"] = np.isfinite(y) & (y < cfg.active_above)
        # wrist speed (units/s)
        v = np.zeros(n)
        if n > 1:
            d = np.linalg.norm(np.diff(track, axis=0), axis=1) / (np.maximum(dts, 1e-3) / 1000.0)
            d = np.where(np.isfinite(d), d, 0.0)
            v[1:] = d
        speeds.append(_smooth(v, k))
        # fingertip motion relative to the wrist (hand-shape change)
        s = np.zeros(n)
        if n > 1:
            rel = hand[:, _TIPS, :2] - hand[:, :1, :2]
            dd = np.linalg.norm(np.diff(rel, axis=0), axis=2).mean(axis=1) / (np.maximum(dts, 1e-3) / 1000.0)
            ok = hp[1:] & hp[:-1]
            s[1:] = np.where(ok, dd, 0.0)
        shapes.append(_smooth(s, k))
    out["speed"] = np.maximum(speeds[0], speeds[1])
    out["shape_change"] = np.maximum(shapes[0], shapes[1])
    out["present"] = bf.present["left"] | bf.present["right"]
    yl, yr = out["wrist_y_left"], out["wrist_y_right"]
    out["rest"] = (~np.isfinite(yl) | (yl >= cfg.rest_below)) & (~np.isfinite(yr) | (yr >= cfg.rest_below))
    out["dip"] = (~np.isfinite(yl) | (yl >= cfg.dip_below)) & (~np.isfinite(yr) | (yr >= cfg.dip_below))
    out["dt"] = np.array([dt])
    return out


def candidate_boundaries(sig: dict[str, np.ndarray], t_ms: np.ndarray, after: int = 0,
                         until: int | None = None) -> list[int]:
    """Local minima of (speed + shape_change) in (after, until): plausible sign ends (as exclusive end frames)."""
    m = sig["speed"] + sig["shape_change"]
    n = len(m)
    until = n if until is None else min(n, until)
    out = []
    for i in range(max(1, after + 1), until - 1):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if m[i] <= m[lo:hi].min() + 1e-9 and m[i] < m[lo:hi].max() - 1e-6:
            if not out or i - out[-1] > 2:
                out.append(i)
    return out


def bounded_stop(stop: str | None) -> bool:
    """The ingestion job stopped at a real boundary (title card, end of file, or a pose gap = card)."""
    s = stop or ""
    return any(k in s for k in ("card", "eof", "posegap"))


def _highest_wrist(sig: dict[str, np.ndarray]) -> np.ndarray:
    yl, yr = sig["wrist_y_left"], sig["wrist_y_right"]
    hy = np.fmin(np.where(np.isfinite(yl), yl, np.inf), np.where(np.isfinite(yr), yr, np.inf))
    hy = np.where(np.isfinite(hy), hy, np.nan)
    if len(hy) >= 3:  # 3-frame median against single-frame landmark jitter
        p = np.pad(hy, 1, mode="edge")
        st = np.stack([p[:-2], p[1:-1], p[2:]])
        cnt = np.isfinite(st).sum(axis=0)
        srt = np.sort(np.where(np.isfinite(st), st, np.inf), axis=0)
        mid = np.take_along_axis(srt, np.clip((cnt - 1) // 2, 0, 2)[None], axis=0)[0]
        hy = np.where(cnt > 0, mid, np.nan)
    return hy


def find_first_sign(raw: RawSequence, stop: str | None = None, cfg: TrimConfig = TrimConfig(),
                    n_words: int = 1) -> TrimResult:
    """First-sign frame range. ``stop`` is the ingestion stop reason; ``n_words`` the number of words of a
    multiword name (compound signs are longer, so the minimum and maximum sign lengths grow with it)."""
    n = len(raw)
    if n == 0:
        return TrimResult(0, 0, 0.0, 0.0, "empty", 0.0, [], ["no frames"])
    try:
        sig = activity_signal(raw, cfg)
    except FeatureError as e:
        return TrimResult(0, n, float(raw.t_ms[0]), float(raw.t_ms[-1]), "empty", 0.0, [], [str(e)])
    t = sig["t"]
    dt = float(sig["dt"][0])
    up = sig["up_left"] | sig["up_right"]
    active = up | (sig["speed"] > cfg.v_move)
    start_run = None
    for a, b in _runs(active):
        if _run_ms(t, a, b, dt) >= cfg.min_active_ms:
            start_run = (a, b)
            break
    if start_run is None:
        return TrimResult(0, n, float(t[0]), float(t[-1]), "empty", 0.0, [], ["no active run"])
    s = max(0, start_run[0] - cfg.pre_roll)
    t0 = t[start_run[0]]
    tail = max(1, int(round(cfg.tail_ms / dt)))
    extra = max(0, int(n_words) - 1)
    min_sign = min(cfg.min_sign_ms + cfg.per_word_min_ms * extra, cfg.max_min_sign_ms)
    max_sign = min(cfg.max_sign_ms + cfg.per_word_max_ms * extra, cfg.hard_max_sign_ms)
    bounded = bounded_stop(stop)
    win_end = int(np.searchsorted(t, t0 + max_sign, side="right"))  # exclusive
    s_min = int(np.searchsorted(t, t0 + min_sign, side="left"))
    notes: list[str] = []

    def cut_at(e: int) -> int:
        """Exclusive end for a boundary whose first 'hands down' frame is ``e``."""
        return int(min(n, e + tail))

    # boundary candidates: rest (both hands down) and dip (both hands lowered to the waist)
    bounds: list[tuple[int, str]] = []
    for a, b in _runs(sig["rest"]):
        if a > start_run[0] and (_run_ms(t, a, b, dt) >= cfg.rest_ms or b == n):
            bounds.append((a, "rest"))
    for a, b in _runs(sig["dip"]):
        if b <= s_min or a <= start_run[0] or _run_ms(t, a, b, dt) < cfg.dip_ms:
            continue
        # a dip that continues into a full rest (hands all the way down) is a rest
        full = sig["rest"][a:b].any() and _run_ms(t, a, b, dt) >= cfg.dip_ms + cfg.rest_ms
        bounds.append((a, "rest" if full else "dip"))
    bounds = sorted(set(bounds), key=lambda x: (x[0], x[1] != "rest"))
    in_win = [x for x in bounds if x[0] < win_end]
    # pause: both still (wrist and hand shape) after min_sign_ms. Used only when no rest/dip exists: on the
    # reviewed videos pauses mostly fired on static holds inside the sign ("Air Deccan"), ending it too early.
    hold = (sig["speed"] < cfg.v_hold) & (sig["shape_change"] < cfg.v_hold)
    pause_end = None
    for a, b in _runs(hold):
        if b <= s_min or a >= win_end:
            continue
        a2 = max(a, s_min)
        if _run_ms(t, a2, b, dt) >= cfg.pause_ms:
            pause_end = a2
            break

    end, method = None, None
    late = False
    if bounded:
        # The clip ended at a real boundary (title card / end of file / pose gap), so it normally holds the
        # sign only: keep everything up to the final return of the hands (last frame with a wrist in signing
        # space). Earlier rests/dips become alternatives: compound signs ("Anal Cancer", "Account Closing")
        # drop the hands between their parts, so the first rest would cut them in half.
        idx = np.nonzero(~sig["dip"][start_run[0]:])[0]
        last = start_run[0] + int(idx[-1]) if len(idx) else n - 1
        final = cut_at(last + 1)
        if t[final - 1] - t0 <= cfg.bounded_max_factor * max_sign:
            # rest/dip boundaries inside the clip first (latest first), then motion minima
            alts = [cut_at(e) for e, _ in reversed(bounds) if s + 2 < cut_at(e) < final - 2]
            cb = candidate_boundaries(sig, t, after=s_min, until=final)
            alts += [c + 1 for c in reversed(cb) if s + 2 < c + 1 < final - 2]
            alts = sorted(list(dict.fromkeys(alts))[:2])
            if "card" in (stop or "") and not sig["dip"][max(0, n - 3):].any():
                # The title-card detector of the ingestion job also fires on plain backgrounds, so a "card" clip
                # whose hands are still up in its last frames was usually cut in the middle of the sign (5 of the
                # 8 card clips checked by eye). Keep the frames but flag them: re-extract with --full.
                rd = [cut_at(e) for e, _ in reversed(bounds) if s + 2 < cut_at(e) < n - 2]
                second = _second_cut(sig, t0, s, n, n_words, cfg, rd)
                return TrimResult(s, n, float(t[s]), float(t[n - 1]), "truncated", METHOD_CONFIDENCE["truncated"],
                                  _with_second(alts, second), [f"hands still up at a title-card stop (stop={stop}): "
                                                               "probably cut mid-sign"], second)
            # second cut: the last rest/dip inside the clip (compound first part / sign before an explanation)
            rd = [cut_at(e) for e, _ in reversed(bounds) if s + 2 < cut_at(e) < final - 2]
            second = _second_cut(sig, t0, s, final, n_words, cfg, rd)
            return TrimResult(s, final, float(t[s]), float(t[final - 1]), "source_end",
                              METHOD_CONFIDENCE["source_end"], _with_second(alts, second),
                              [f"bounded clip (stop={stop})"], second)
        # short video that still contains an explanation: compounds take the last boundary in the window
        notes.append(f"bounded clip longer than the sign window (stop={stop})")
        if in_win:
            e, method = in_win[-1] if n_words > 1 else in_win[0]
            end = cut_at(e)
    elif in_win:
        e, method = in_win[0]
        end = cut_at(e)
        if cfg.late_factor and t[e] - t0 > cfg.late_factor * _expected_ms(n_words, cfg):
            # the first hands-down comes much later than a typical sign: either a long sign or the signer went
            # straight on into the explanation and dropped the hands only inside it. Keep the boundary as the
            # main cut (right for long signs) and take the second cut from the length prior (cap_scores).
            notes.append(f"first {method} at {t[e] / 1000:.2f}s is late; second cut from the length prior")
            late = True
    if end is None and pause_end is not None:
        end, method = min(n, pause_end + tail), "pause"
    if end is None:
        last_hand = np.nonzero(sig["present"])[0]
        clip_end = int(last_hand[-1]) + 1 if len(last_hand) else n
        if win_end >= n:  # the clip ends inside the window
            end = clip_end
            method = "clip_end"
            notes.append(f"clip ended inside the window (stop={stop})")
        else:
            end = _cap_end(sig, t0, n_words, cfg) if cfg.cap_prior else _cap_end_v1(sig, t0, max_sign, win_end, cfg)
            method = "cap"
    end = int(max(s + 1, min(n, end)))
    cb = candidate_boundaries(sig, t, after=s_min, until=int(np.searchsorted(t, t0 + 1.5 * max_sign)) + 1)
    alts = [c + 1 for c in cb if c + 1 > end + 2][:2]
    if len(alts) < 2:
        alts += [c + 1 for c in reversed(cb) if c + 1 < end - 2 and c + 1 > s + 2][: 2 - len(alts)]
    later = [cut_at(e) for e, _ in bounds if e < int(np.searchsorted(t, t0 + 1.5 * max_sign)) and cut_at(e) > end]
    preferred = later if method in ("rest", "dip", "pause") and not late else []
    second = _second_cut(sig, t0, s, end, n_words, cfg, preferred)
    conf = METHOD_CONFIDENCE[method]
    return TrimResult(s, end, float(t[s]), float(t[end - 1]), method, conf, _with_second(alts, second), notes,
                      second)


def _with_second(alts: list[int], second: int | None) -> list[int]:
    if second is None:
        return sorted(alts)[:2]
    return sorted([second] + [a for a in alts if a != second][:1])


def _second_cut(sig: dict[str, np.ndarray], t0: float, s: int, end: int, n_words: int, cfg: TrimConfig,
                preferred: list[int]) -> int | None:
    """Best other cut at least ``second_sep_ms`` away from ``end``: the first of ``preferred`` (e.g. the next
    rest after a rest cut: compounds that drop the hands between their parts), else the best cap-score frame
    (sign-length prior + low hands)."""
    t = sig["t"]
    n = len(t)

    def far(e: int) -> bool:
        return s + 2 < e <= n and abs(float(t[min(e, n) - 1] - t[end - 1])) >= cfg.second_sep_ms

    for e in preferred:
        if far(e):
            return int(e)
    lo, sc = cap_scores(sig, t0, n_words, cfg)
    for k in np.argsort(-sc, kind="stable"):
        e = lo + int(k) + 1
        if far(e):
            return int(e)
    return None


def _cap_end_v1(sig: dict[str, np.ndarray], t0: float, max_sign: float, win_end: int, cfg: TrimConfig) -> int:
    """trim-1 fallback: the lowest hand position, slightly favouring slow frames, in the last 40% of the window."""
    t = sig["t"]
    lo = int(np.searchsorted(t, t0 + 0.6 * max_sign))
    yl, yr = sig["wrist_y_left"], sig["wrist_y_right"]
    low = np.fmin(np.where(np.isfinite(yl), yl, np.inf), np.where(np.isfinite(yr), yr, np.inf))
    low = np.where(np.isfinite(low), low, cfg.rest_below)
    score = (low - 0.1 * (sig["speed"] + sig["shape_change"]))[lo:win_end]
    return lo + int(np.argmax(score)) + 1 if len(score) else win_end


def _expected_ms(n_words: int, cfg: TrimConfig) -> float:
    """Typical first-sign length: median of bounded ISLRTC clips (2.2 s one word, ~2.8 s compounds)."""
    return cfg.cap_mu_ms + (cfg.cap_mu_extra_ms if n_words > 1 else 0.0)


def cap_scores(sig: dict[str, np.ndarray], t0: float, n_words: int, cfg: TrimConfig = TrimConfig()):
    """Frame scores for the end of a sign that runs straight into the explanation (no rest/dip/pause): a
    duration prior learned from bounded ISLRTC clips (median sign length 2.2 s for one word, 2.8 s for
    compounds; docs/DATASETS_ISL.md 7) plus a bonus for slow frames (motion minima) and low hands.
    Returns (first frame index, scores) for the search window [0.6, 1.8] x the expected length."""
    t = sig["t"]
    mu = _expected_ms(n_words, cfg)
    lo = int(np.searchsorted(t, t0 + 0.6 * mu))
    hi = int(np.searchsorted(t, t0 + 1.8 * mu, side="right"))
    if hi <= lo:
        return lo, np.zeros(0)
    d = np.maximum(t[lo:hi] - t0, 1.0)
    m = (sig["speed"] + sig["shape_change"])[lo:hi]
    hy = _highest_wrist(sig)[lo:hi]
    hy = np.where(np.isfinite(hy), hy, cfg.rest_below)
    score = (-cfg.cap_w_prior * np.abs(np.log(d / mu)) - cfg.cap_w_motion * m / (np.median(m) + 1e-6)
             + cfg.cap_w_low * hy)
    return lo, score


def _cap_end(sig: dict[str, np.ndarray], t0: float, n_words: int, cfg: TrimConfig) -> int:
    lo, score = cap_scores(sig, t0, n_words, cfg)
    if len(score) == 0:
        return min(len(sig["t"]), lo + 1)
    return lo + int(np.argmax(score)) + 1


def words_for(kind: str | None, text: str | None) -> int:
    """Expected number of sign components: multiword names are compounds; synonym lists are one sign."""
    if kind in ("multiword",) and text:
        return min(4, len(text.split()))
    return 1


def trim_record_for(clip_id: str, raw: RawSequence, stop: str | None, kind: str | None,
                    cfg: TrimConfig = TrimConfig(), n_words: int = 1) -> dict:
    """Sidecar record for one ISLRTC clip (phrases keep the full clip from the first activity)."""
    res = find_first_sign(raw, stop, cfg, n_words=n_words)
    if kind == "phrase" and res.method != "empty":
        last = len(raw)
        res = TrimResult(res.start, last, res.t_start_ms, float(raw.t_ms[last - 1]), "full", METHOD_CONFIDENCE["full"],
                         [], ["phrase: full clip"])
    return res.to_record(clip_id, len(raw), cfg.version)


# --------------------------------------------------------------------------------------------- review plots

def plot_trim(raw: RawSequence, res: TrimResult | dict, out_png, title: str = "", cfg: TrimConfig = TrimConfig(),
              truth_end: int | None = None) -> None:
    """Review sheet panel: wrist height (top), motion (bottom), kept range shaded, alternatives dashed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(res, dict):
        res = TrimResult(res["start"], res["end"], res.get("t_start_ms", 0.0), res.get("t_end_ms", 0.0),
                         res["method"], res.get("confidence", 0.0), res.get("alternatives", []))
    sig = activity_signal(raw, cfg)
    t = (sig["t"] - sig["t"][0]) / 1000.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for side, col in (("left", "#2a6fdb"), ("right", "#d9480f")):
        y = sig[f"wrist_y_{side}"]
        hp = sig[f"hand_{side}"]
        ax1.plot(t, np.where(hp, y, np.nan), "-", color=col, lw=1.6, label=f"{side} wrist (hand)")
        ax1.plot(t, np.where(~hp, y, np.nan), ":", color=col, lw=1.0, label=f"{side} wrist (pose only)")
    ax1.axhline(cfg.rest_below, color="#777", lw=0.8, ls="--")
    ax1.axhline(cfg.dip_below, color="#bbb", lw=0.6, ls=":")
    ax1.text(t[0], cfg.rest_below, " rest line", va="bottom", fontsize=7, color="#555")
    ax1.invert_yaxis()
    ax1.set_ylabel("wrist height\n(shoulder widths, up = higher)", fontsize=7)
    ax2.plot(t, sig["speed"], color="#333", lw=1.2, label="wrist speed")
    ax2.plot(t, sig["shape_change"], color="#2b8a3e", lw=1.0, label="hand-shape change")
    ax2.axhline(cfg.v_hold, color="#999", lw=0.7, ls="--")
    ax2.set_ylabel("units / s", fontsize=7)
    ax2.set_xlabel("seconds", fontsize=8)
    for ax in (ax1, ax2):
        ax.axvspan(t[res.start], t[max(res.start, res.end - 1)], color="#f2c94c", alpha=0.25, lw=0)
        for a in res.alternatives:
            if 0 < a <= len(t):
                ax.axvline(t[a - 1], color="#7048e8", lw=0.9, ls="--")
        if truth_end is not None and 0 < truth_end <= len(t):
            ax.axvline(t[truth_end - 1], color="#c92a2a", lw=1.2, ls="-.")
        ax.tick_params(labelsize=7)
    ax1.legend(fontsize=6, loc="lower right", ncol=2)
    ax2.legend(fontsize=6, loc="upper right")
    ax1.set_title(f"{title}  [{res.method}, conf {res.confidence:.1f}, kept {t[res.start]:.1f}-{t[max(res.start, res.end - 1)]:.1f}s]",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
