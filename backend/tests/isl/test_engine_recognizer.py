"""The streaming engine with a language-pack recogniser (accept / suggest / abstain, choose, cooldown, spell)."""

from __future__ import annotations

import pytest

from app.dataset.synthetic import frames_for_stream, idle_frames, make_sequence
from app.features.spec import FeatureConfig
from app.inference.engine import EngineConfig, RecognitionEngine
from app.inference.recognizer import Candidate, ClassifierRecognizer, Recognition


class FakeSpell:
    def __init__(self):
        self.letters, self.buffer, self.suggestions = [{"letter": "I", "prob": 0.9}], "I", []

    def to_json(self):
        return {"letters": self.letters, "buffer": self.buffer, "suggestions": self.suggestions}


class FakeRecognizer:
    language = "isl"
    feature_config = FeatureConfig(canonical_hand=True)
    recommended_threshold = 0.7
    classifier = None
    speller = object()

    def __init__(self, decision="accept"):
        self.decision = decision
        self.calls = []

    def _cands(self):
        return [Candidate("isl.pencil", 7, "pencil", 0.8, {"en": "pencil", "hi": "पेंसिल"}, "reviewed", "classifier+retrieval"),
                Candidate("isl.pen", 6, "pen", 0.1, {"en": "pen", "hi": None}, None, "retrieval")]

    def recognize(self, crops, top_k=5, policy="balanced", use_my_signs=True):
        self.calls.append((len(crops), policy, use_my_signs))
        reason = {"accept": "ok", "suggest": "ambiguous", "abstain": "out_of_vocabulary"}[self.decision]
        return Recognition(self._cands(), self.decision, reason, 0.8, 3, len(crops))

    def class_probs(self, crops):
        return None

    def spell(self, raw, k=5):
        return FakeSpell()

    def spell_recognition(self, raw, top_k=5):
        c = Candidate("isl.indore", 3, "Indore", 0.7, {"en": "Indore", "hi": "इंदौर"}, "machine", "spelling")
        return Recognition([c], "suggest", "spelling", 0.7, 1, 1, detail={"spell": FakeSpell().to_json()})


def _run(eng, labels=(4,), gap=12):
    t, states = 0.0, []
    for cls in labels:
        for f in idle_frames(gap, t):
            states.append(eng.push(f))
        t += gap * 40
        seq = make_sequence(cls, 2, n_frames=40)
        for f in frames_for_stream(seq, t):
            states.append(eng.push(f))
        t += len(seq) * 40
    for f in idle_frames(gap * 2, t):
        states.append(eng.push(f))
    return states


def test_accept_commits_with_bilingual_text():
    rec = FakeRecognizer("accept")
    eng = RecognitionEngine(None, recognizer=rec, config=EngineConfig(policy="strict", use_my_signs=False))
    states = _run(eng)
    res = [s["segment_result"] for s in states if "segment_result" in s and "decision" in s["segment_result"]]
    commits = [s["commit"] for s in states if "commit" in s]
    assert len(res) == 1 and res[0]["decision"] == "accept" and res[0]["segment_id"] == 1
    assert commits[0]["concept_id"] == "isl.pencil" and commits[0]["text"]["hi"] == "पेंसिल"
    assert commits[0]["chosen_by_user"] is False and commits[0]["segment_id"] == 1
    assert rec.calls[-1] == (3, "strict", False)  # three crops, engine policy and My-signs switch passed through
    assert any("provisional" in s for s in states)
    assert eng.describe()["config"]["policy"] == "strict"
    rej = eng.reject(1)
    assert rej == {"segment_id": 1, "rejected": "isl.pencil"}


def test_cooldown_turns_accept_into_suggest():
    rec = FakeRecognizer("accept")
    eng = RecognitionEngine(None, recognizer=rec, config=EngineConfig(cooldown_ms=10_000_000))
    states = _run(eng, labels=(1, 2))
    res = [s["segment_result"] for s in states if "segment_result" in s and "decision" in s["segment_result"]]
    assert [r["decision"] for r in res] == ["accept", "suggest"] and res[1]["reason"] == "cooldown"


@pytest.mark.parametrize("decision,status", [("suggest", "uncertain"), ("abstain", "not_sure")])
def test_suggest_and_abstain_never_commit(decision, status):
    eng = RecognitionEngine(None, recognizer=FakeRecognizer(decision))
    states = _run(eng)
    assert not any("commit" in s for s in states)
    seg = next(s for s in states if "segment_result" in s and "decision" in s["segment_result"])
    assert seg["segment_result"]["decision"] == decision and seg["status"] == status
    commit = eng.choose(seg["segment_result"]["segment_id"], "isl.pen")
    assert commit["chosen_by_user"] is True and commit["concept_id"] == "isl.pen" and commit["text"]["en"] == "pen"
    with pytest.raises(ValueError):
        eng.choose(999, "isl.pen")


def test_spell_mode_reports_letters_and_word():
    eng = RecognitionEngine(None, recognizer=FakeRecognizer(), config=EngineConfig(mode="spell", spell_end_ms=400))
    states = _run(eng)
    assert any("spell" in s for s in states) and "spelling" in {s["status"] for s in states}
    seg = next(s["segment_result"] for s in states if "segment_result" in s and "decision" in s["segment_result"])
    assert seg["candidates"][0]["source"] == "spelling" and seg["decision"] == "suggest" and "spell" in seg


def test_spell_and_policy_validation():
    with pytest.raises(ValueError):
        EngineConfig.from_dict({"policy": "risky"})
    with pytest.raises(ValueError):
        EngineConfig.from_dict({"use_my_signs": "yes"})
    with pytest.raises(ValueError):
        RecognitionEngine(None, config=EngineConfig(mode="spell"))  # ASL / no speller
    eng = RecognitionEngine(None, recognizer=FakeRecognizer())
    with pytest.raises(ValueError):
        eng.configure(EngineConfig(mode="sliding"))  # no closed-set classifier in this pack


def test_classifier_recognizer_matches_engine_rule(tmp_path):
    from app.models.checkpoint import Predictor, save_checkpoint
    from app.models.networks import build_model

    fcfg = FeatureConfig(seq_len=16)
    names = [f"g{i}" for i in range(6)]
    save_checkpoint(tmp_path / "m.pt", build_model("gru", fcfg.dim, 6, hidden=8, layers=1),
                    {"architecture": "gru", "model_kwargs": {"hidden": 8, "layers": 1}, "feature_config": fcfg.to_dict(),
                     "class_names": names, "recommended_threshold": 0.0})
    p = Predictor(tmp_path / "m.pt")
    r = ClassifierRecognizer(p, threshold=0.0, min_margin=0.0)
    rec = r.recognize([make_sequence(4, 7)] * 3)
    assert rec.decision == "accept" and len(rec.candidates) == 5 and rec.crops == 3 and rec.agree == 3
    assert rec.candidates[0].concept_id == rec.candidates[0].gloss and rec.candidates[0].gloss in names
    strict = ClassifierRecognizer(p, threshold=1.01).recognize([make_sequence(4, 7)])
    assert strict.decision == "suggest" and strict.reason == "low_confidence"


def test_repeated_choice_is_marked_duplicate():
    eng = RecognitionEngine(None, recognizer=FakeRecognizer("suggest"))
    seg = next(s for s in _run(eng) if "segment_result" in s and "decision" in s["segment_result"])
    sid = seg["segment_result"]["segment_id"]
    assert "duplicate" not in eng.choose(sid, "isl.pen")
    assert eng.choose(sid, "isl.pen")["duplicate"] is True  # double tap: the word is added once
    eng.reject(sid)
    assert "duplicate" not in eng.choose(sid, "isl.pen")  # chosen again after "wrong word"


def test_segment_results_and_commits_carry_the_certification():
    """Review of 15.25 (critic 15): every word message says which rule applied and the certified precision, and a
    word the app accepted by itself still needs a confirm tap; a word the user picked does not."""
    cert = {"rule": "certified", "certified_precision": 0.951, "worst_source": "include", "confirm_tap": True}

    class CertRecognizer(FakeRecognizer):
        def recognize(self, crops, top_k=5, policy="balanced", use_my_signs=True):
            r = super().recognize(crops, top_k, policy, use_my_signs)
            r.detail["certification"] = dict(cert)
            return r

    eng = RecognitionEngine(None, recognizer=CertRecognizer("accept"))
    states = _run(eng)
    seg = next(s["segment_result"] for s in states if "segment_result" in s and "decision" in s["segment_result"])
    commit = next(s["commit"] for s in states if "commit" in s)
    assert seg["certification"] == cert and seg["needs_confirm"] is True
    assert commit["certification"] == cert and commit["needs_confirm"] is True and commit["chosen_by_user"] is False
    chosen = eng.choose(seg["segment_id"], "isl.pen")
    assert chosen["needs_confirm"] is False and chosen["certification"] == cert
    # a recogniser without certification (spelling, older fakes): the messages are as before
    plain = RecognitionEngine(None, recognizer=FakeRecognizer("accept"))
    st = _run(plain)
    assert not any("needs_confirm" in s.get("segment_result", {}) or "needs_confirm" in s.get("commit", {})
                   for s in st)


# ------------------------------------------------------------------ QA plan 0.5 (docs/ISL_DESIGN.md 15.27)

def test_segment_mode_reports_the_threshold_the_pack_uses():
    """Segment mode with a language pack decides with the pack's fused accept threshold for the policy; a
    ``threshold`` in the config is not used there, so the engine must not report it. Sliding mode still uses it."""
    from app.isl.models.fusion import FusionParams

    rec = FakeRecognizer("accept")
    rec.fusion = FusionParams(accept={"strict": 0.93, "balanced": 0.8, "lenient": 0.6})
    eng = RecognitionEngine(None, recognizer=rec, config=EngineConfig(threshold=0.1, policy="strict"))
    assert eng.threshold == 0.93 and eng.describe()["threshold"] == 0.93
    assert all(s["threshold"] == 0.93 for s in _run(eng))
    eng.configure(EngineConfig(threshold=0.1, policy="lenient"))
    assert eng.threshold == 0.6
    rec.classifier = object()  # sliding mode needs a classifier
    eng.configure(EngineConfig(mode="sliding", threshold=0.1))
    assert eng.threshold == 0.1  # sliding mode's own decision reads cfg.threshold


class _SlidingRecognizer(FakeRecognizer):
    """A pack classifier whose two classes are missing from the vocabulary (vocabulary index -1 for both)."""

    classifier = object()
    classifier_threshold = 0.5

    class _Vocab:
        def english(self, cid):
            return cid

        def index_of(self, cid):
            return -1

    vocab = _Vocab()
    lexicon = None

    def __init__(self, script):
        super().__init__()
        self.script = list(script)  # top class per classified window
        self.i = 0

    def class_probs(self, crops):
        import numpy as np

        c = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        p = np.full((len(crops), 2), 0.02, np.float32)
        p[:, c] = 0.98
        return p, ["isl.extra-a", "isl.extra-b"], 0.0, 0.0


def _hands_stream(eng, n, t0=0.0):
    from app.dataset.synthetic import make_sequence

    seq = make_sequence(1, 2, n_frames=n)
    return [eng.push(f) for f in frames_for_stream(seq, t0)]


def test_sliding_streak_and_repeats_key_on_the_class_not_the_vocabulary_index():
    """Two classifier classes outside the vocabulary both had vocabulary index -1: the engine counted them as one
    streak and suppressed the second word as a "repeat" of the first."""
    rec = _SlidingRecognizer([0] * 4 + [1] * 30)
    cfg = EngineConfig(mode="sliding", agree_n=2, cooldown_ms=0, interval_ms=0, window_ms=400, min_frames=3,
                       ema_alpha=1.0)
    eng = RecognitionEngine(None, recognizer=rec, config=cfg)
    commits = [s["commit"] for s in _hands_stream(eng, 30) if "commit" in s]
    assert [c["concept_id"] for c in commits] == ["isl.extra-a", "isl.extra-b"]
    assert all(c["class_index"] == -1 for c in commits)  # the commit still carries the vocabulary index
    assert eng.streak_class == 1  # the classifier's class index


def test_classifier_policy_thresholds_are_cached_per_classifier_and_fusion(monkeypatch):
    """The engine asks for the sliding threshold on every frame; the fingerprint hashes the whole class list."""
    import app.isl.recognizer as R
    from app.isl.models.fusion import FusionParams
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    fcfg = FeatureConfig(canonical_hand=True)

    class _Clf:
        feature_config = fcfg
        class_names = ["isl.a", "isl.b"]
        meta = {"recommended_threshold": 0.4, "class_names": ["isl.a", "isl.b"]}

    calls = []
    real = R.classifier_policy_thresholds
    monkeypatch.setattr(R, "classifier_policy_thresholds", lambda c, f: calls.append((c, f)) or real(c, f))
    rec = IslRecognizer("isl", fcfg, Vocabulary.from_class_names(["isl.a", "isl.b"]), classifier=_Clf())
    for _ in range(50):
        assert rec.classifier_threshold_for("balanced") == 0.4
    assert len(calls) == 1
    rec.fusion = FusionParams()  # another fusion.json: computed again, once
    for _ in range(5):
        rec.classifier_threshold_for("strict")
    rec.classifier = _Clf()  # another classifier
    rec.classifier_threshold_for("strict")
    assert len(calls) == 3
