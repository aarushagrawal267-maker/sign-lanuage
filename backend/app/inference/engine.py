"""Streaming recognition engine with sign-boundary detection and temporal smoothing.

Used identically for live webcam frames (via WebSocket) and offline video analysis.

Two modes
---------
``segment`` (default)
    A sign starts when hands are detected for a few consecutive frames and ends
    when they leave the frame / drop out of view for ``end_absent_ms`` or the
    segment reaches ``max_segment_ms``. The completed segment is classified as a
    whole - matching how training clips are whole signs - using three temporal
    crops (test-time augmentation). A word is emitted only if the averaged
    confidence passes the threshold, the margin over the runner-up is large
    enough and at least two of the three crops agree. Provisional predictions
    are shown while the sign is in progress but never emitted as text.

``sliding``
    Continuous classification of the last ``window_ms`` every ``interval_ms``.
    Probabilities are smoothed with an exponential moving average; a word is
    emitted after ``agree_n`` consecutive windows agree above threshold. The same
    word is not repeated until the hands leave or the prediction changes, and a
    cooldown applies after every emission.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from app.features.build import FeatureError, build_features
from app.landmarks.raw import RawSequence, fill_frame
from app.models.checkpoint import Predictor


@dataclass
class EngineConfig:
    mode: str = "segment"  # segment | sliding
    threshold: float | None = None  # None -> model's validation-derived threshold
    min_margin: float = 0.10
    min_frames: int = 6
    min_hand_ratio: float = 0.5
    start_frames: int = 3
    pre_roll_ms: float = 240.0
    end_absent_ms: float = 400.0
    max_segment_ms: float = 4000.0
    interval_ms: float = 200.0
    window_ms: float = 2000.0
    ema_alpha: float = 0.5
    agree_n: int = 3
    cooldown_ms: float = 700.0
    top_k: int = 5

    @classmethod
    def from_dict(cls, d: dict | None) -> EngineConfig:
        d = d or {}
        out = cls()
        for k, v in d.items():
            if k in cls.__dataclass_fields__ and v is not None:
                ftype = type(getattr(out, k)) if getattr(out, k) is not None else float
                setattr(out, k, ftype(v) if ftype in (int, float, str) else v)
        if out.mode not in ("segment", "sliding"):
            raise ValueError("mode must be 'segment' or 'sliding'")
        return out


@dataclass
class _Frame:
    t: float
    data: dict
    hands: bool
    pose: bool


class RecognitionEngine:
    def __init__(self, predictor: Predictor | None, width: int = 640, height: int = 480,
                 config: EngineConfig | None = None):
        self.predictor = predictor
        self.cfg = config or EngineConfig()
        self.width, self.height = width, height
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.buf: deque[_Frame] = deque()
        self.segment: list[_Frame] = []
        self.active = False
        self.hand_streak = 0
        self.last_hand_t: float | None = None
        self.last_infer_t = -1e18
        self.last_commit_t = -1e18
        self.last_commit_class: int | None = None
        self.ema: np.ndarray | None = None
        self.streak_class: int | None = None
        self.streak = 0
        self.status = "waiting_for_hands"
        self.last_prediction: dict | None = None

    @property
    def threshold(self) -> float:
        if self.cfg.threshold is not None:
            return float(self.cfg.threshold)
        return self.predictor.recommended_threshold if self.predictor else 0.6

    def set_size(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    # --------------------------------------------------------------- helpers
    def _raw(self, frames: list[_Frame]) -> RawSequence:
        return RawSequence.from_frames([f.data for f in frames], self.width, self.height)

    def _classify(self, frames: list[_Frame], tta: bool) -> dict | None:
        if self.predictor is None:
            return None
        fcfg = self.predictor.feature_config
        t0 = time.perf_counter()
        crops = [frames]
        n = len(frames)
        if tta and n >= 12:
            k = max(1, n // 10)
            crops += [frames[k:], frames[:-k]]
        try:
            feats = np.stack([build_features(self._raw(c), fcfg) for c in crops])
        except FeatureError as e:
            return {"error": str(e)}
        feat_ms = (time.perf_counter() - t0) * 1000
        probs, inf_ms = self.predictor.predict(feats)
        avg = probs.mean(axis=0)
        top1 = int(avg.argmax())
        agree = int(sum(int(p.argmax()) == top1 for p in probs))
        return {"probs": avg, "agree": agree, "crops": len(crops), "feature_ms": feat_ms, "inference_ms": inf_ms}

    def _topk(self, probs: np.ndarray) -> list[dict]:
        names = self.predictor.class_names
        idx = np.argsort(-probs)[: self.cfg.top_k]
        return [{"class_index": int(i), "gloss": names[i], "prob": float(probs[i])} for i in idx]

    def _decision(self, probs: np.ndarray, agree_ok: bool) -> tuple[bool, str]:
        srt = np.sort(probs)[::-1]
        if srt[0] < self.threshold:
            return False, "low_confidence"
        if len(srt) > 1 and srt[0] - srt[1] < self.cfg.min_margin:
            return False, "ambiguous"
        if not agree_ok:
            return False, "unstable"
        return True, "ok"

    # ------------------------------------------------------------------ push
    def push(self, frame: dict) -> dict:
        """Add one frame ({t, pose, left, right, face}); returns a state message."""
        tmp = RawSequence.from_frames([frame], self.width, self.height)  # validates shapes
        hands = bool(not np.isnan(tmp.left[0, 0, 0]) or not np.isnan(tmp.right[0, 0, 0]))
        pose = bool(not np.isnan(tmp.pose[0, 0, 0]))
        t = float(frame.get("t", 0.0))
        f = _Frame(t, frame, hands, pose)
        self.buf.append(f)
        horizon = max(self.cfg.max_segment_ms, self.cfg.window_ms) + self.cfg.pre_roll_ms + 500
        while self.buf and t - self.buf[0].t > horizon:
            self.buf.popleft()
        if hands:
            self.last_hand_t = t

        msg: dict = {"type": "state", "t": t, "hands": hands, "pose": pose}
        if self.predictor is None:
            self.status = "unavailable"
        elif self.cfg.mode == "segment":
            msg.update(self._push_segment(f))
        else:
            msg.update(self._push_sliding(f))
        msg["status"] = self.status
        msg["threshold"] = self.threshold
        return msg

    def _push_segment(self, f: _Frame) -> dict:
        out: dict = {}
        c = self.cfg
        if not self.active:
            self.hand_streak = self.hand_streak + 1 if f.hands else 0
            if self.hand_streak >= c.start_frames:
                self.active = True
                start_t = f.t - c.pre_roll_ms
                # include the streak frames plus a short pre-roll
                self.segment = [x for x in self.buf if x.t >= start_t]
                self.status = "collecting"
            else:
                self.status = "waiting_for_hands" if f.pose else "no_body"
            out["segment_frames"] = len(self.segment) if self.active else 0
            return out

        self.segment.append(f)
        dur = f.t - self.segment[0].t
        absent_for = f.t - (self.last_hand_t if self.last_hand_t is not None else f.t)
        ended = absent_for >= c.end_absent_ms or dur >= c.max_segment_ms
        out["segment_frames"] = len(self.segment)
        out["segment_ms"] = dur

        if ended:
            # trim trailing hand-less frames, keeping a short tail
            frames = list(self.segment)
            while len(frames) > 1 and not frames[-1].hands and (frames[-1].t - (self.last_hand_t or 0)) > 120:
                frames.pop()
            self.active, self.segment, self.hand_streak = False, [], 0
            out.update(self._finalize(frames, f.t))
            return out

        self.status = "collecting"
        if len(self.segment) >= c.min_frames and f.t - self.last_infer_t >= c.interval_ms:
            self.last_infer_t = f.t
            r = self._classify(self.segment, tta=False)
            if r and "probs" in r:
                self.status = "analyzing"
                out["provisional"] = {"top": self._topk(r["probs"]), "feature_ms": r["feature_ms"],
                                      "inference_ms": r["inference_ms"]}
        return out

    def _finalize(self, frames: list[_Frame], t: float) -> dict:
        c = self.cfg
        hand_ratio = sum(fr.hands for fr in frames) / max(1, len(frames))
        if len(frames) < c.min_frames or hand_ratio < c.min_hand_ratio * 0.6:
            self.status = "waiting_for_hands"
            return {"segment_result": {"accepted": False, "reason": "too_short", "frames": len(frames)}}
        r = self._classify(frames, tta=True)
        if r is None or "error" in r:
            self.status = "no_body"
            return {"segment_result": {"accepted": False, "reason": (r or {}).get("error", "no model")}}
        probs = r["probs"]
        need = 2 if r["crops"] >= 3 else 1
        ok, reason = self._decision(probs, r["agree"] >= need)
        if ok and t - self.last_commit_t < c.cooldown_ms:
            ok, reason = False, "cooldown"
        top = self._topk(probs)
        res = {"accepted": ok, "reason": reason, "frames": len(frames), "duration_ms": frames[-1].t - frames[0].t,
               "agreement": f"{r['agree']}/{r['crops']}", "top": top, "feature_ms": r["feature_ms"],
               "inference_ms": r["inference_ms"]}
        out: dict = {"segment_result": res}
        if ok:
            self.last_commit_t, self.last_commit_class = t, top[0]["class_index"]
            self.status = "recognized"
            out["commit"] = {"gloss": top[0]["gloss"], "class_index": top[0]["class_index"], "score": top[0]["prob"], "t": t}
        else:
            self.status = "uncertain"
        self.last_prediction = res
        return out

    def _push_sliding(self, f: _Frame) -> dict:
        c = self.cfg
        out: dict = {}
        if self.last_hand_t is None or f.t - self.last_hand_t >= c.end_absent_ms:
            # sign boundary: hands gone -> reset smoothing and allow repeats
            self.ema, self.streak, self.streak_class, self.last_commit_class = None, 0, None, None
            self.status = "waiting_for_hands" if f.pose else "no_body"
            return out
        if f.t - self.last_infer_t < c.interval_ms:
            return out
        window = [x for x in self.buf if x.t >= f.t - c.window_ms]
        hand_ratio = sum(x.hands for x in window) / max(1, len(window))
        out["window_frames"] = len(window)
        if len(window) < c.min_frames or hand_ratio < c.min_hand_ratio:
            self.status = "collecting"
            return out
        self.last_infer_t = f.t
        r = self._classify(window, tta=False)
        if r is None or "error" in r:
            self.status = "no_body"
            return out
        p = r["probs"]
        self.ema = p if self.ema is None else c.ema_alpha * p + (1 - c.ema_alpha) * self.ema
        top = self._topk(self.ema)
        cls = top[0]["class_index"]
        self.streak = self.streak + 1 if cls == self.streak_class else 1
        self.streak_class = cls
        ok, reason = self._decision(self.ema, self.streak >= c.agree_n)
        if ok and cls == self.last_commit_class:
            ok, reason = False, "repeat_suppressed"
        if ok and f.t - self.last_commit_t < c.cooldown_ms:
            ok, reason = False, "cooldown"
        out["provisional"] = {"top": top, "feature_ms": r["feature_ms"], "inference_ms": r["inference_ms"],
                              "streak": self.streak, "reason": reason}
        if ok:
            self.last_commit_t, self.last_commit_class = f.t, cls
            self.status = "recognized"
            out["commit"] = {"gloss": top[0]["gloss"], "class_index": cls, "score": top[0]["prob"], "t": f.t}
        else:
            self.status = "analyzing" if reason in ("unstable",) else "uncertain"
        return out

    def flush(self) -> dict:
        """End of stream (offline video): finalize any open segment."""
        if self.cfg.mode == "segment" and self.active and self.segment:
            frames = list(self.segment)
            self.active, self.segment = False, []
            return self._finalize(frames, frames[-1].t)
        return {}

    def describe(self) -> dict:
        return {"config": asdict(self.cfg), "threshold": self.threshold}
