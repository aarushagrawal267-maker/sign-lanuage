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

``spell`` (language packs with fingerspelling, e.g. ISL)
    Letters are holds; the word ends when the hands are down for ``spell_end_ms``. Live letter posteriors and
    lexicon suggestions are reported in ``state.spell``; the finished word arrives as a ``segment_result`` with
    candidates whose ``source`` is ``"spelling"``.

With a ``Recognizer`` (language packs, docs/ISL_DESIGN.md 4.3) a finished segment gets a three-way decision
(``accept`` / ``suggest`` top-5 / ``abstain``), candidates with English and Hindi text and a ``segment_id``
that the client can answer with ``choose`` / ``reject``. Without one, the engine behaves exactly as before.

``sliding``
    Continuous classification of the last ``window_ms`` every ``interval_ms``.
    Probabilities are smoothed with an exponential moving average; a word is
    emitted after ``agree_n`` consecutive windows agree above threshold. The same
    word is not repeated until the hands leave or the prediction changes, and a
    cooldown applies after every emission.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from app.features.build import FeatureError, build_features
from app.landmarks.raw import RawSequence
from app.models.checkpoint import Predictor

MODES = ("segment", "sliding", "spell")
POLICIES = ("strict", "balanced", "lenient")
# Fields only used with a language-pack recogniser; left out of ``describe()`` for the classic (ASL) engine so
# its hello/config_ack messages stay exactly as before.
RECOGNIZER_FIELDS = ("policy", "use_my_signs", "spell_end_ms", "max_spell_ms")


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
    policy: str = "balanced"  # strict | balanced | lenient (maps to the pack's fitted thresholds)
    use_my_signs: bool = True
    spell_end_ms: float = 600.0
    max_spell_ms: float = 15000.0

    @classmethod
    def from_dict(cls, d: dict | None) -> EngineConfig:
        d = d or {}
        out = cls()
        for k, v in d.items():
            if k in cls.__dataclass_fields__ and v is not None:
                ftype = type(getattr(out, k)) if getattr(out, k) is not None else float
                if ftype is bool:
                    if not isinstance(v, bool):
                        raise ValueError(f"{k} must be true or false")
                    setattr(out, k, v)
                    continue
                setattr(out, k, ftype(v) if ftype in (int, float, str) else v)
        if out.mode not in MODES:
            raise ValueError("mode must be 'segment', 'sliding' or 'spell'")
        if out.policy not in POLICIES:
            raise ValueError("policy must be 'strict', 'balanced' or 'lenient'")
        return out


@dataclass
class _Frame:
    t: float
    data: dict
    hands: bool
    pose: bool


class RecognitionEngine:
    def __init__(self, predictor: Predictor | None, width: int = 640, height: int = 480,
                 config: EngineConfig | None = None, recognizer=None, lexicon=None):
        self.predictor = predictor
        self.recognizer = recognizer
        self.lexicon = lexicon  # Hindi lexicon for the predictor path (ASL); recognisers carry their own
        self.cfg = config or EngineConfig()
        self.width, self.height = width, height
        self.segment_seq = 0
        self.recent: dict[int, dict] = {}  # segment_id -> {"candidates": [...], "t": ...} for choose / reject
        self.validate(self.cfg)
        self.reset()

    @property
    def available(self) -> bool:
        return self.predictor is not None or self.recognizer is not None

    def validate(self, cfg: EngineConfig) -> None:
        if cfg.mode == "spell" and (self.recognizer is None or not hasattr(self.recognizer, "spell")
                                    or getattr(self.recognizer, "speller", None) is None):
            raise ValueError("spelling mode is not available for this language")
        if cfg.mode == "sliding" and self.recognizer is not None and self.predictor is None \
                and getattr(self.recognizer, "classifier", True) is None:
            raise ValueError("sliding mode needs a closed-set classifier, which this language pack does not have")

    def configure(self, cfg: EngineConfig) -> None:
        self.validate(cfg)
        self.cfg = cfg
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
        self.spell_last_t = -1e18

    @property
    def threshold(self) -> float:
        if self.cfg.threshold is not None:
            return float(self.cfg.threshold)
        if self.predictor is None and self.recognizer is not None:
            # sliding mode decides on the classifier's own probabilities (class_probs): its own threshold, never
            # the fused accept threshold, which is on the fused-probability scale (docs/ISL_DESIGN.md 15.16)
            # Per policy when the pack fitted it for this classifier (docs 15.17), else the checkpoint's one value.
            if self.cfg.mode == "sliding":
                per_pol = getattr(self.recognizer, "classifier_threshold_for", None)
                cls_thr = per_pol(self.cfg.policy) if callable(per_pol) else None
                if cls_thr is None:
                    cls_thr = getattr(self.recognizer, "classifier_threshold", None)
                if cls_thr is not None:
                    return float(cls_thr)
            acc = getattr(getattr(self.recognizer, "fusion", None), "accept", None)
            if isinstance(acc, dict) and self.cfg.policy in acc:
                return float(acc[self.cfg.policy])
            return float(self.recognizer.recommended_threshold)
        return self.predictor.recommended_threshold if self.predictor else 0.6

    def set_size(self, width: int, height: int) -> None:
        w, h = int(width), int(height)
        if w <= 0 or h <= 0:  # the aspect correction in the features needs a real frame size
            raise ValueError(f"width and height must be positive, got {w}x{h}")
        self.width, self.height = w, h

    # --------------------------------------------------------------- helpers
    def _raw(self, frames: list[_Frame]) -> RawSequence:
        return RawSequence.from_frames([f.data for f in frames], self.width, self.height)

    @staticmethod
    def _crops(frames: list[_Frame], tta: bool) -> list[list[_Frame]]:
        crops = [frames]
        n = len(frames)
        if tta and n >= 12:
            k = max(1, n // 10)
            crops += [frames[k:], frames[:-k]]
        return crops

    def _classify(self, frames: list[_Frame], tta: bool) -> dict | None:
        if self.predictor is None:
            if self.recognizer is None:
                return None
            r = self.recognizer.class_probs([self._raw(c) for c in self._crops(frames, tta)])
            if r is None:
                return {"error": "no body detected or no classifier in this language pack"}
            probs, self._class_names, feat_ms, inf_ms = r
            avg = probs.mean(axis=0)
            top1 = int(avg.argmax())
            agree = int(sum(int(p.argmax()) == top1 for p in probs))
            return {"probs": avg, "agree": agree, "crops": len(probs), "feature_ms": feat_ms, "inference_ms": inf_ms}
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
        if self.predictor is None and self.recognizer is not None:
            return self._topk_concepts(probs)
        names = self.predictor.class_names
        idx = np.argsort(-probs)[: self.cfg.top_k]
        out = [{"class_index": int(i), "gloss": names[i], "prob": float(probs[i])} for i in idx]
        if self.lexicon is not None:  # additive: English + Hindi text for the same gloss
            for d in out:
                hi, rev = self.lexicon.text(d["gloss"])
                d["text"], d["hi_review"] = {"en": d["gloss"], "hi": hi}, rev
        return out

    def _topk_concepts(self, probs: np.ndarray) -> list[dict]:
        """Closed-set top-k of a pack classifier (sliding mode) as candidate dicts."""
        names = self._class_names
        vocab = getattr(self.recognizer, "vocab", None)
        out = []
        for i in np.argsort(-probs)[: self.cfg.top_k]:
            cid = names[i]
            en = vocab.english(cid) if vocab is not None else cid
            hi, rev = (None, None)
            if getattr(self.recognizer, "lexicon", None) is not None:
                hi, rev = self.recognizer.lexicon.text(cid)
            out.append({"concept_id": cid, "class_index": vocab.index_of(cid) if vocab is not None else int(i),
                        "gloss": en, "prob": float(probs[i]), "text": {"en": en, "hi": hi}, "hi_review": rev,
                        "source": "classifier", "variant_id": None, "region": None, "origin": None})
        return out

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
        if not math.isfinite(t):  # a NaN/Infinity timestamp would stop old frames from ever leaving the buffer
            raise ValueError(f"t must be a finite number of milliseconds, got {t}")
        f = _Frame(t, frame, hands, pose)
        self.buf.append(f)
        horizon = max(self.cfg.max_segment_ms, self.cfg.window_ms) + self.cfg.pre_roll_ms + 500
        while self.buf and t - self.buf[0].t > horizon:
            self.buf.popleft()
        if hands:
            self.last_hand_t = t

        msg: dict = {"type": "state", "t": t, "hands": hands, "pose": pose}
        if not self.available:
            self.status = "unavailable"
        elif self.cfg.mode == "segment":
            msg.update(self._push_segment(f))
        elif self.cfg.mode == "spell":
            msg.update(self._push_spell(f))
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
            if self.recognizer is not None and self.predictor is None:
                rec = self.recognizer.recognize([self._raw(self.segment)], top_k=c.top_k, policy=c.policy,
                                                use_my_signs=c.use_my_signs)
                if rec.candidates:
                    self.status = "analyzing"
                    out["provisional"] = {"top": [x.to_json() for x in rec.candidates], "feature_ms": rec.feature_ms,
                                          "inference_ms": rec.inference_ms}
                return out
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
        if self.recognizer is not None and self.predictor is None:
            rec = self.recognizer.recognize([self._raw(x) for x in self._crops(frames, True)], top_k=c.top_k,
                                            policy=c.policy, use_my_signs=c.use_my_signs)
            return self._emit_recognition(rec, frames, t)
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
            if "text" in top[0]:
                out["commit"]["text"], out["commit"]["hi_review"] = top[0]["text"], top[0]["hi_review"]
        else:
            self.status = "uncertain"
        self.last_prediction = res
        return out

    # ------------------------------------------------------- recogniser results
    def _emit_recognition(self, rec, frames: list[_Frame], t: float, cooldown: bool = True) -> dict:
        c = self.cfg
        decision, reason = rec.decision, rec.reason
        if decision == "accept" and cooldown and t - self.last_commit_t < c.cooldown_ms:
            decision, reason = "suggest", "cooldown"
        if not rec.candidates and decision != "abstain":
            decision, reason = "abstain", reason
        self.segment_seq += 1
        sid = self.segment_seq
        top = [x.to_json() for x in rec.candidates]
        res = {"segment_id": sid, "accepted": decision == "accept", "decision": decision, "reason": reason,
               "confidence": rec.confidence, "frames": len(frames),
               "duration_ms": (frames[-1].t - frames[0].t) if frames else 0.0,
               "agreement": f"{rec.agree}/{rec.crops}", "top": top, "candidates": top,
               "feature_ms": rec.feature_ms, "inference_ms": rec.inference_ms}
        if "spell" in rec.detail:
            res["spell"] = rec.detail["spell"]
        self.recent[sid] = {"candidates": top, "t": t, "committed": None}
        for old in [k for k in self.recent if k <= sid - 32]:
            self.recent.pop(old, None)
        out: dict = {"segment_result": res}
        if decision == "accept":
            self.last_commit_t = t
            self.last_commit_class = top[0]["class_index"]
            self.status = "recognized"
            out["commit"] = self._commit_from(top[0], sid, t, chosen=False)
            self.recent[sid]["committed"] = top[0]["concept_id"]
        else:
            self.status = "uncertain" if decision == "suggest" else "not_sure"
        self.last_prediction = res
        return out

    @staticmethod
    def _commit_from(cand: dict, sid: int, t: float, chosen: bool) -> dict:
        return {"gloss": cand["gloss"], "class_index": cand["class_index"], "score": cand["prob"], "t": t,
                "concept_id": cand["concept_id"], "text": cand.get("text"), "hi_review": cand.get("hi_review"),
                "source": cand.get("source"), "segment_id": sid, "chosen_by_user": chosen}

    def choose(self, segment_id: int, concept_id: str) -> dict:
        """The user picked a candidate from "did you mean". Returns the commit (``chosen_by_user: true``)."""
        seg = self.recent.get(int(segment_id))
        if seg is None:
            raise ValueError(f"unknown or expired segment_id {segment_id}")
        cand = next((x for x in seg["candidates"] if x["concept_id"] == concept_id), None)
        if cand is None:
            raise ValueError(f"{concept_id!r} was not offered for segment {segment_id}")
        commit = self._commit_from(cand, int(segment_id), seg["t"], chosen=True)
        if seg["committed"] == concept_id:  # a repeated tap / resend: the word is already in the text
            commit["duplicate"] = True
            return commit
        seg["committed"] = concept_id
        self.last_commit_t = seg["t"]
        return commit

    def reject(self, segment_id: int) -> dict:
        """The user said an accepted word was wrong. Returns what was rejected."""
        seg = self.recent.get(int(segment_id))
        if seg is None:
            raise ValueError(f"unknown or expired segment_id {segment_id}")
        was, seg["committed"] = seg["committed"], None
        return {"segment_id": int(segment_id), "rejected": was}

    # ------------------------------------------------------------- spell mode
    def _push_spell(self, f: _Frame) -> dict:
        c = self.cfg
        out: dict = {}
        if not self.active:
            self.hand_streak = self.hand_streak + 1 if f.hands else 0
            if self.hand_streak >= c.start_frames:
                self.active = True
                self.segment = [x for x in self.buf if x.t >= f.t - c.pre_roll_ms]
                self.status = "spelling"
            else:
                self.status = "waiting_for_hands" if f.pose else "no_body"
            return out
        self.segment.append(f)
        absent_for = f.t - (self.last_hand_t if self.last_hand_t is not None else f.t)
        dur = f.t - self.segment[0].t
        if absent_for >= c.spell_end_ms or dur >= c.max_spell_ms:
            frames = [x for x in self.segment if x.t <= (self.last_hand_t or f.t) + 120]
            self.active, self.segment, self.hand_streak = False, [], 0
            if len(frames) < c.min_frames:
                self.status = "waiting_for_hands"
                out["segment_result"] = {"accepted": False, "reason": "too_short", "frames": len(frames)}
                return out
            rec = self.recognizer.spell_recognition(self._raw(frames), top_k=c.top_k)
            out.update(self._emit_recognition(rec, frames, f.t, cooldown=False))
            return out
        self.status = "spelling"
        if f.t - self.spell_last_t >= max(c.interval_ms, 300.0) and len(self.segment) >= c.min_frames:
            self.spell_last_t = f.t
            sr = self.recognizer.spell(self._raw(self.segment), k=c.top_k)
            if sr is not None:
                out["spell"] = sr.to_json()
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
        if ok and self.predictor is None and self._sliding_confirm_only():
            # language pack whose sliding thresholds are not certified for this policy (docs/ISL_DESIGN.md 15.25):
            # provisional words only, never an auto-commit
            ok, reason = False, "confirm_only"
        out["provisional"] = {"top": top, "feature_ms": r["feature_ms"], "inference_ms": r["inference_ms"],
                              "streak": self.streak, "reason": reason}
        if ok:
            self.last_commit_t, self.last_commit_class = f.t, cls
            self.status = "recognized"
            out["commit"] = {"gloss": top[0]["gloss"], "class_index": cls, "score": top[0]["prob"], "t": f.t}
            if "concept_id" in top[0]:  # language-pack classifier: same extra fields as segment-mode commits
                out["commit"].update({k: top[0][k] for k in ("concept_id", "text", "hi_review", "source")},
                                     segment_id=None, chosen_by_user=False)
        else:
            self.status = "analyzing" if reason in ("unstable",) else "uncertain"
        return out

    def _sliding_confirm_only(self) -> bool:
        rule = getattr(self.recognizer, "sliding_rule", None)
        return callable(rule) and rule(self.cfg.policy) == "confirm_only"

    def flush(self) -> dict:
        """End of stream (offline video): finalize any open segment."""
        if self.cfg.mode == "segment" and self.active and self.segment:
            frames = list(self.segment)
            self.active, self.segment = False, []
            return self._finalize(frames, frames[-1].t)
        return {}

    def describe(self) -> dict:
        cfg = asdict(self.cfg)
        if self.recognizer is None:
            for k in RECOGNIZER_FIELDS:
                cfg.pop(k, None)
        out = {"config": cfg, "threshold": self.threshold}
        info = getattr(self.recognizer, "certification_info", None) if self.recognizer is not None else None
        if callable(info):  # language packs only: what the UI shows next to words (docs/ISL_DESIGN.md 15.25)
            out["certification"] = info(self.cfg.policy, self.cfg.mode)
        return out
