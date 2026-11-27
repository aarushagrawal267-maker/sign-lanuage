"""What the app needs for its default (docs/ISL_DESIGN.md 15.29): the certification of every policy in
``describe()``, sliding mode auto-commits only when sliding strict is certified, and dictionary-only words are
flagged so the app always asks about them."""

from __future__ import annotations

from app.inference.engine import POLICIES, EngineConfig, RecognitionEngine
from app.isl.models.fusion import FusionParams

from tests.isl.test_engine_recognizer import FakeRecognizer, _run
from tests.isl.test_sliding_replay_thresholds import NAMES, _certified_sliding_pack, _commits, _ScriptedClf


def _isl_rec(clf, fusion):
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    vocab = Vocabulary([{"concept_id": n, "headword": n, "display": {"en": n}} for n in NAMES])
    return IslRecognizer("isl", clf.feature_config, vocab, classifier=clf, fusion=fusion)


def _pack_with(clf, rules: dict):
    """A certified sliding pack whose per-policy sliding rules are ``rules`` (missing ones certified)."""
    p = _certified_sliding_pack(clf)
    fit = dict(p.classifier_fit)
    fit["policies"] = {k: {**v, "rule": rules.get(k, "certified")} for k, v in fit["policies"].items()}
    return p.with_(classifier_fit=fit)


def test_sliding_auto_commits_only_when_sliding_strict_is_certified():
    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    # balanced and lenient certified, strict not: sliding mode never auto-commits, under any policy
    rec = _isl_rec(clf, _pack_with(clf, {"strict": "confirm_only"}))
    for pol in POLICIES:
        assert rec.sliding_rule(pol) == "confirm_only"
    eng, commits, reasons = _commits(rec, clf, policy="balanced")
    assert commits == [] and "confirm_only" in reasons
    info = eng.describe()["certification"]
    assert info["rule"] == "confirm_only" and info["certified_precision"] is None
    assert "sliding strict" in info["reason"]
    # strict certified too: balanced may auto-commit (still with a confirm tap in the app)
    rec2 = _isl_rec(clf, _pack_with(clf, {}))
    assert rec2.sliding_rule("balanced") == "certified"
    _, commits2, _ = _commits(rec2, clf, policy="balanced")
    assert commits2 and commits2[0]["commit"]["needs_confirm"] is True
    # strict certified but lenient not: lenient stays confirm-only
    assert _isl_rec(clf, _pack_with(clf, {"lenient": "confirm_only"})).sliding_rule("lenient") == "confirm_only"
    # in-code parameters are not gated (unchanged)
    assert _isl_rec(clf, FusionParams(fitted=True)).sliding_rule("balanced") == "ungated"


def test_describe_gives_every_policy_for_segment_and_sliding_mode():
    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    rec = _isl_rec(clf, _pack_with(clf, {"lenient": "confirm_only"}))
    d = RecognitionEngine(None, recognizer=rec, config=EngineConfig(policy="strict")).describe()
    by = d["certification_by_policy"]
    assert set(by["segment"]) == set(by["sliding"]) == set(POLICIES)
    assert by["sliding"]["strict"]["rule"] == "certified" and by["sliding"]["strict"]["certified_precision"] == 0.97
    assert by["sliding"]["lenient"]["rule"] == "confirm_only"
    assert all(v["confirm_tap"] is True for m in by.values() for v in m.values())
    # the segment rows are the fusion's own (this pack has no segment certification: confirm-only)
    assert by["segment"]["strict"] == rec.fusion.certification_info("strict")
    assert d["certification"] == by["segment"]["strict"]


def test_describe_has_no_sliding_without_a_classifier_and_old_packs_are_confirm_only():
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    old = FusionParams.from_json({"accept": {"strict": 0.96, "balanced": 0.96, "lenient": 0.96}, "fitted": True})
    vocab = Vocabulary([{"concept_id": n, "headword": n, "display": {"en": n}} for n in NAMES])
    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    rec = IslRecognizer("isl", clf.feature_config, vocab, classifier=None, fusion=old)
    by = RecognitionEngine(None, recognizer=rec).describe()["certification_by_policy"]
    assert by["sliding"] is None
    assert {v["rule"] for v in by["segment"].values()} == {"confirm_only"}
    assert all(v["certified_precision"] is None for v in by["segment"].values())


def test_asl_describe_is_unchanged():
    eng = RecognitionEngine(None)
    d = eng.describe()
    assert set(d) == {"config", "threshold"}
    # a recogniser without certification (older fakes, spelling-only) gets no certification fields either
    assert "certification_by_policy" not in RecognitionEngine(None, recognizer=FakeRecognizer()).describe()


def test_open_vocabulary_words_are_flagged_on_results_and_commits():
    cert = {"rule": "certified", "certified_precision": 0.96, "confirm_tap": True}

    class OvRecognizer(FakeRecognizer):
        def __init__(self, decision, ov):
            super().__init__(decision)
            self.ov = ov

        def recognize(self, crops, top_k=5, policy="balanced", use_my_signs=True):
            r = super().recognize(crops, top_k, policy, use_my_signs)
            r.detail["certification"] = dict(cert)
            return r

        def is_open_vocabulary(self, key):
            return key in self.ov

    for ov, expect in (({"isl.pencil"}, True), (set(), False)):
        states = _run(RecognitionEngine(None, recognizer=OvRecognizer("accept", ov)))
        seg = next(s["segment_result"] for s in states if "decision" in s.get("segment_result", {}))
        commit = next(s["commit"] for s in states if "commit" in s)
        assert seg["open_vocabulary"] is expect and commit["open_vocabulary"] is expect
        assert commit["needs_confirm"] is True
    # without a certification (older recognisers) the messages are as before
    plain = _run(RecognitionEngine(None, recognizer=FakeRecognizer("accept")))
    assert not any("open_vocabulary" in s.get("segment_result", {}) or "open_vocabulary" in s.get("commit", {})
                   for s in plain)
