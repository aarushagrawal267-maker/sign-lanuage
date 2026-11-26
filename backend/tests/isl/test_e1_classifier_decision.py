"""The classifier alone (E1, sliding mode) decides on its own calibrated scale, never on the fused thresholds.

Regression for the commercial-safe report of 2026-11-26: E1 compared raw (temperature 1) classifier probabilities
with the fused balanced threshold 0.96 (fitted on the softmax of classifier + retrieval + prior) and reported
"not sure" 0.95-1.0 next to top-1 0.72-0.96; the live engine's sliding mode used the same mismatched threshold
(docs/ISL_DESIGN.md 15.16).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.features.spec import FeatureConfig
from app.isl.models.fusion import FusionParams

FUSED = FusionParams(accept={"strict": 0.96, "balanced": 0.96, "lenient": 0.96}, fitted=True)


class _Clf:
    """Classifier stub: every query gets the same three-crop probabilities (T = 1)."""

    def __init__(self, probs, meta, names=("a", "b", "c"), fcfg=None):
        self.probs = np.asarray(probs, float)
        self.meta = {"split": "s", "class_names": list(names), **meta}
        self.class_names = list(names)
        self.feature_config = fcfg or FeatureConfig()

    def predict(self, feats):
        return self.probs.copy(), 1.0


def _clips(truths):
    return [SimpleNamespace(clip_id=f"c{i}", concept_id=t, group=f"g{i % 2}", category=None)
            for i, t in enumerate(truths)]


def _e1(monkeypatch, clf, truths):
    from app.isl.evaluation import runners

    monkeypatch.setattr(runners, "_features", lambda clip, fcfg: np.zeros((3, 4, 2), np.float32))
    rec = SimpleNamespace(classifier=clf, fusion=FUSED, feature_config=clf.feature_config)
    return runners.e1_closed_set(rec, _clips(truths), "E1b", "s", "primary")


def test_e1_uses_the_classifier_temperature_and_threshold_not_the_fused_one(monkeypatch):
    # raw p1 0.55 < fused 0.96 (and < 0.6); at the classifier's temperature 0.5 it is 0.729 >= its threshold 0.6
    clf = _Clf([[0.55, 0.30, 0.15]] * 3, {"temperature": 0.5, "recommended_threshold": 0.6})
    out = _e1(monkeypatch, clf, ["a", "a", "a", "b"])
    m = out["metrics"]
    assert out["decision"]["threshold"] == 0.6 and out["decision"]["temperature"] == 0.5
    assert out["decision"]["threshold_in_checkpoint"] is True
    assert m["coverage"] == 1.0 and m["not_sure_rate"] == 0.0  # the old code said 1.0 (nothing reaches 0.96)
    assert m["wrong_accept_rate"] == pytest.approx(0.25) and m["accepted_precision"] == pytest.approx(0.75)
    assert m["top1"] == pytest.approx(0.75)
    assert "0.96" not in out["decision_note"] and "0.60" in out["decision_note"]


def test_e1_not_sure_needs_threshold_margin_and_crop_agreement(monkeypatch):
    # below the classifier's own threshold -> not accepted, counted as not sure (no suggestion step)
    low = _Clf([[0.5, 0.3, 0.2]] * 3, {"temperature": 1.0, "recommended_threshold": 0.7})
    assert _e1(monkeypatch, low, ["a"])["metrics"]["not_sure_rate"] == 1.0
    # high mean probability but the crops disagree (1 of 3 agree with the mean's top-1)
    split = _Clf([[0.99, 0.005, 0.005], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]], {"temperature": 1.0,
                                                                           "recommended_threshold": 0.4})
    assert _e1(monkeypatch, split, ["a"])["metrics"]["coverage"] == 0.0
    # no fitted threshold in the checkpoint: the Predictor default 0.6, flagged as not fitted
    old = _Clf([[0.9, 0.05, 0.05]] * 3, {})
    out = _e1(monkeypatch, old, ["a"])
    assert out["decision"]["threshold"] == 0.6 and out["decision"]["threshold_in_checkpoint"] is False
    assert out["decision"]["temperature"] == 1.0 and out["metrics"]["coverage"] == 1.0


def test_report_shows_accepted_and_says_what_not_sure_means_per_protocol():
    from app.isl.evaluation.runners import markdown_report

    e1 = {"run": True, "n": 10, "n_groups": 2, "label": "E1b", "metrics": {
        "top1": 0.7, "top5": 0.9, "coverage": 0.71, "not_sure_rate": 0.29, "wrong_accept_rate": 0.1}}
    e3 = {"run": True, "n": 10, "n_groups": 2, "label": "E3", "metrics": {
        "top1": 0.7, "top5": 0.87, "coverage": 0.27, "not_sure_rate": 0.105, "wrong_accept_rate": 0.019}}
    doc = {"profile": "research", "label": "x", "created": "now", "smoke": False,
           "protocols": {"E1b": e1, "E3": e3, "E2": {"run": False, "reason": "no folds"}},
           "leakage_guards": {"pretraining_check": {"reason": "none"}}}
    md = markdown_report(doc)
    assert "| top-5 | accepted | not sure | wrong accept |" in md
    assert "| 0.900 | 0.710 | 0.290 | 0.100 |" in md and "| 0.870 | 0.270 | 0.105 | 0.019 |" in md
    assert "E1a/E1b/E1c = 1 - accepted" in md and "E2/E3 = abstain rate only" in md
    header = next(ln for ln in md.splitlines() if ln.startswith("| Protocol"))
    for ln in md.splitlines():
        if ln.startswith(("| E1b", "| E3", "| E2")):
            assert ln.count("|") == header.count("|"), ln


def _isl_recognizer(clf, fusion=FUSED):
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    return IslRecognizer("isl", clf.feature_config, Vocabulary([]), classifier=clf, fusion=fusion)


def test_isl_class_probs_are_temperature_scaled_and_sliding_mode_uses_the_classifier_threshold():
    from app.dataset.synthetic import make_sequence
    from app.inference.engine import EngineConfig, RecognitionEngine

    fcfg = FeatureConfig(canonical_hand=True)
    clf = _Clf([[0.55, 0.30, 0.15]], {"temperature": 0.5, "recommended_threshold": 0.6}, fcfg=fcfg)
    rec = _isl_recognizer(clf)
    probs, names, _, _ = rec.class_probs([make_sequence(1, 2, n_frames=30)])
    assert names == ["a", "b", "c"]
    assert probs[0, 0] == pytest.approx(0.55 ** 2 / (0.55 ** 2 + 0.3 ** 2 + 0.15 ** 2))
    assert rec.classifier_threshold == 0.6 and rec.recommended_threshold == 0.96
    sliding = RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding"))
    segment = RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="segment", policy="strict"))
    assert sliding.threshold == 0.6  # classifier scale (class_probs), not the fused 0.96
    assert segment.threshold == 0.96  # segment mode: fused decision, fused threshold
    ok, reason = sliding._decision(probs[0], True)
    assert ok and reason == "ok"
    # an explicit user threshold still wins
    assert RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding", threshold=0.8)).threshold == 0.8
