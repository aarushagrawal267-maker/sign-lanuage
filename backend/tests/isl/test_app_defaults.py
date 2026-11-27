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


def _cert_recognizer(decision="accept", ov=frozenset(), source=None):
    """A fake pack recogniser whose words carry a certification with a confirm tap (and, optionally, a source)."""
    cert = {"rule": "certified", "certified_precision": 0.96, "confirm_tap": True}

    class R(FakeRecognizer):
        def recognize(self, crops, top_k=5, policy="balanced", use_my_signs=True):
            r = super().recognize(crops, top_k, policy, use_my_signs)
            r.detail["certification"] = dict(cert)
            if source is not None:
                r.candidates[0].source = source
            return r

        def is_open_vocabulary(self, key):
            return key in ov

    return R(decision)


def test_my_signs_never_counts_as_dictionary_only():
    # the recogniser says "outside the classifier" for this word, but it is the user's own sign (docs 15.29)
    states = _run(RecognitionEngine(None, recognizer=_cert_recognizer(ov={"isl.pencil"}, source="my_signs")))
    seg = next(s["segment_result"] for s in states if "decision" in s.get("segment_result", {}))
    commit = next(s["commit"] for s in states if "commit" in s)
    assert seg["top"][0]["source"] == "my_signs"
    assert seg["open_vocabulary"] is False and commit["open_vocabulary"] is False
    assert commit["needs_confirm"] is True


def test_unconfirmed_words_are_kept_by_segment_id_or_time():
    from app.api.recognition import _Unconfirmed

    u = _Unconfirmed()
    u.add({"concept_id": "isl.a", "segment_id": 3, "t": 10.0}, None, "isl")
    u.add({"concept_id": "isl.b", "segment_id": None, "t": 1234.5}, None, "isl")
    assert u.pop(_Unconfirmed.key(segment_id=3))[0]["concept_id"] == "isl.a"
    assert u.pop(_Unconfirmed.key(segment_id=3)) is None  # a second tap logs nothing
    assert u.pop(_Unconfirmed.key(t=1234.5))[0]["concept_id"] == "isl.b"
    for i in range(_Unconfirmed.MAX + 5):
        u.add({"concept_id": f"isl.{i}", "segment_id": i, "t": float(i)}, None, "isl")
    assert len(u.items) == _Unconfirmed.MAX and u.pop(_Unconfirmed.key(segment_id=0)) is None
    import pytest

    for bad in ({}, {"t": float("nan")}, {"t": "x"}, {"segment_id": "x"}):
        with pytest.raises((ValueError, TypeError)):
            _Unconfirmed.key(**bad)


def test_record_commit_keeps_needs_confirm_in_the_detail(monkeypatch):
    import app.api.recognition as api

    rows = []
    monkeypatch.setattr(api, "_record", lambda *a: rows.append(a))
    c = {"gloss": "pencil", "class_index": 7, "score": 0.8, "t": 1.0, "concept_id": "isl.pencil", "text": {"en": "pencil"},
         "source": "classifier", "segment_id": 2, "chosen_by_user": False, "needs_confirm": True}
    api._record_commit(c, "video", None, "isl")
    api._record_commit(c, "live", None, "isl", "confirmed_by_user")
    assert rows[0][6]["needs_confirm"] is True and rows[0][6]["decision"] == "accept"
    assert rows[1][6]["decision"] == "confirmed_by_user"
    # ASL commits (no concept id) are stored exactly as before: no detail
    api._record_commit({"gloss": "book", "class_index": 2, "score": 0.9, "t": 1.0}, "live", 1, None)
    assert rows[2][6] is None


def test_websocket_logs_a_word_that_needs_a_tap_only_once_it_is_confirmed(monkeypatch):
    """docs/ISL_DESIGN.md 15.30: an accepted word with needs_confirm is not logged when it arrives; "confirm" logs it as
    confirmed_by_user (once); a word the user rejected or replaced by another guess is never logged as a word."""
    from fastapi.testclient import TestClient

    import app.api.recognition as api
    from app.dataset.synthetic import frames_for_stream, idle_frames, make_sequence
    from app.main import app

    rows = []
    monkeypatch.setattr(api, "_record", lambda *a: rows.append(a))

    def session(pack_id=None, width=640, height=480, engine_cfg=None):
        eng = RecognitionEngine(None, width, height, EngineConfig(cooldown_ms=0), recognizer=_cert_recognizer())
        return eng, {"name": "fake"}, None, {"id": "isl", "display_name": {"en": "ISL"}, "output_languages": ["en"],
                                              "capabilities": ["words"], "profile": "commercial-safe", "status": "ready",
                                              "sign_language": "ISL"}, None

    monkeypatch.setattr(api, "_language_session", session)
    monkeypatch.setattr(api, "_languages", lambda: [])

    def sign(ws, t0):
        commits, out = [], []
        frames = idle_frames(10, t0) + frames_for_stream(make_sequence(4, 2, n_frames=40), t0 + 400) + idle_frames(20, t0 + 2400)
        for f in frames:
            ws.send_json({"type": "frame", **f})
            st = ws.receive_json()
            out.append(st)
            if "commit" in st:
                commits.append(st["commit"])
        return commits

    with TestClient(app) as c, c.websocket_connect("/api/ws/recognize?language=isl") as ws:
        assert ws.receive_json()["type"] == "hello"
        (c1,) = sign(ws, 0)
        assert c1["needs_confirm"] is True and c1["segment_id"] is not None
        assert rows == []  # not logged on arrival
        ws.send_json({"type": "confirm", "segment_id": c1["segment_id"]})
        ack = ws.receive_json()
        assert ack == {"type": "confirm_ack", "segment_id": c1["segment_id"], "t": None, "confirmed": "isl.pencil"}
        assert len(rows) == 1 and rows[0][6]["decision"] == "confirmed_by_user" and rows[0][6]["needs_confirm"] is True
        ws.send_json({"type": "confirm", "segment_id": c1["segment_id"]})  # a double tap
        assert ws.receive_json()["confirmed"] is None and len(rows) == 1
        # rejected: only the feedback event, never the word
        (c2,) = sign(ws, 10_000)
        ws.send_json({"type": "reject", "segment_id": c2["segment_id"]})
        assert ws.receive_json()["type"] == "reject_ack"
        ws.send_json({"type": "confirm", "segment_id": c2["segment_id"]})
        assert ws.receive_json()["confirmed"] is None
        assert [r[6]["decision"] for r in rows] == ["confirmed_by_user", "rejected_by_user"]
        # another guess picked instead: only the chosen word is logged
        (c3,) = sign(ws, 20_000)
        ws.send_json({"type": "choose", "segment_id": c3["segment_id"], "concept_id": "isl.pen"})
        assert ws.receive_json()["type"] == "choice_ack"
        ws.send_json({"type": "confirm", "segment_id": c3["segment_id"]})
        assert ws.receive_json()["confirmed"] is None
        assert [r[6]["decision"] for r in rows] == ["confirmed_by_user", "rejected_by_user", "chosen"]
        assert rows[-1][6]["concept_id"] == "isl.pen"
        # ignored (never confirmed): nothing is logged; bad confirm messages are errors
        sign(ws, 30_000)
        ws.send_json({"type": "confirm"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "confirm", "segment_id": "x"})
        assert ws.receive_json()["type"] == "error"
    assert len(rows) == 3
