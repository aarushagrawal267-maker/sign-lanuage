"""Recogniser abstraction shared by every sign-language pack (docs/ISL_DESIGN.md section 4.3).

The streaming engine (``engine.py``) owns segmentation, crops, cooldown and the hand-ratio gates; a
``Recognizer`` turns the crops of one finished segment into ranked, calibrated candidates plus a three-way
decision: ``accept`` (show the word), ``suggest`` ("did you mean" top-5) or ``abstain`` ("not sure").

``ClassifierRecognizer`` wraps the existing ``Predictor`` (ASL and any classifier-only pack) and reproduces the
engine's historical threshold / margin / agreement rule exactly, so wrapping the ASL model changes nothing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from app.features.build import FeatureError, build_features
from app.features.spec import FeatureConfig
from app.landmarks.raw import RawSequence
from app.models.checkpoint import Predictor

DECISIONS = ("accept", "suggest", "abstain")


@dataclass
class Candidate:
    concept_id: str  # ASL: the gloss itself ("book"); ISL: "isl.pencil"
    class_index: int  # ASL: model class index; ISL: row in vocabulary.jsonl
    gloss: str  # English headword (keeps the ASL-compatible TopK shape)
    prob: float  # calibrated probability after fusion
    text: dict = field(default_factory=dict)  # {"en": "pencil", "hi": "पेंसिल" | None}
    hi_review: str | None = None  # reviewed | machine | disputed | missing | None
    source: str = "classifier"  # classifier | retrieval | classifier+retrieval | my_signs | spelling
    variant_id: str | None = None
    region: str | None = None
    origin: str | None = None  # "ISLRTC dictionary" | "INCLUDE" | "My signs" | ...

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Recognition:
    candidates: list[Candidate]
    decision: str  # accept | suggest | abstain
    reason: str  # ok | low_confidence | ambiguous | unstable | out_of_vocabulary | no_features | ...
    confidence: float
    agree: int
    crops: int
    feature_ms: float = 0.0
    inference_ms: float = 0.0
    detail: dict = field(default_factory=dict)  # per-stage timings, raw scores (debug / evaluation)

    def to_json(self) -> dict:
        return {"candidates": [c.to_json() for c in self.candidates], "decision": self.decision,
                "reason": self.reason, "confidence": self.confidence, "agree": self.agree, "crops": self.crops,
                "feature_ms": self.feature_ms, "inference_ms": self.inference_ms, "detail": self.detail}


@runtime_checkable
class Recognizer(Protocol):
    language: str
    feature_config: FeatureConfig
    recommended_threshold: float

    def recognize(self, crops: list[RawSequence], top_k: int = 5, policy: str = "balanced",
                  use_my_signs: bool = True) -> Recognition: ...

    def class_probs(self, crops: list[RawSequence]) -> tuple[np.ndarray, list[str], float, float] | None:
        """Closed-set probabilities (sliding mode / provisional display): (probs averaged over crops, class
        concept ids, feature_ms, inference_ms). None if the pack has no classifier."""


class ClassifierRecognizer:
    """A ``Predictor`` as a ``Recognizer``. Decision rule = the engine's historical one:
    accept iff p1 >= threshold and p1 - p2 >= min_margin and enough crops agree; otherwise ``suggest``."""

    def __init__(self, predictor: Predictor, language: str = "asl-wlasl100", threshold: float | None = None,
                 min_margin: float = 0.10):
        self.predictor = predictor
        self.language = language
        self.feature_config = predictor.feature_config
        self.recommended_threshold = float(threshold if threshold is not None else predictor.recommended_threshold)
        self.min_margin = min_margin
        self.capabilities = frozenset({"words"})

    def class_probs(self, crops: list[RawSequence]):
        t0 = time.perf_counter()
        try:
            feats = np.stack([build_features(c, self.feature_config) for c in crops])
        except FeatureError:
            return None
        feat_ms = (time.perf_counter() - t0) * 1000
        probs, inf_ms = self.predictor.predict(feats)
        return probs, list(self.predictor.class_names), feat_ms, inf_ms

    def recognize(self, crops: list[RawSequence], top_k: int = 5, policy: str = "balanced",
                  use_my_signs: bool = True, threshold: float | None = None) -> Recognition:
        r = self.class_probs(crops)
        if r is None:
            return Recognition([], "abstain", "no_features", 0.0, 0, len(crops))
        probs, names, feat_ms, inf_ms = r
        avg = probs.mean(axis=0)
        top1 = int(avg.argmax())
        agree = int(sum(int(p.argmax()) == top1 for p in probs))
        idx = np.argsort(-avg)[:top_k]
        cands = [Candidate(concept_id=names[i], class_index=int(i), gloss=names[i], prob=float(avg[i]),
                           text={"en": names[i]}, source="classifier") for i in idx]
        th = self.recommended_threshold if threshold is None else float(threshold)
        srt = np.sort(avg)[::-1]
        need = 2 if len(crops) >= 3 else 1
        if srt[0] < th:
            decision, reason = "suggest", "low_confidence"
        elif len(srt) > 1 and srt[0] - srt[1] < self.min_margin:
            decision, reason = "suggest", "ambiguous"
        elif agree < need:
            decision, reason = "suggest", "unstable"
        else:
            decision, reason = "accept", "ok"
        return Recognition(cands, decision, reason, float(srt[0]), agree, len(crops), feat_ms, inf_ms)
