"""Hindi lexicon (curated core list, number rules, model post-processing, build precedence, tracked files),
the lexicon reader, the rule-based sentence builder and the Hindi fields on the ASL path. No network, no model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from app.isl.text import hindi_build as hb
from app.isl.text.lexicon import HI_MISSING, HindiLexicon
from app.isl.text.numbers_hi import hindi_number
from app.isl.text.sentence import build_both, build_english, build_hindi, build_sentence, make_token

REPO = Path(__file__).resolve().parents[3]
ISL_LEX = REPO / "models" / "packs" / "isl" / "lexicon_hi.jsonl"
ASL_LEX = REPO / "models" / "packs" / "asl-wlasl100" / "lexicon_hi.jsonl"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ------------------------------------------------------------------------------------------------ numbers

@pytest.mark.parametrize("n,hi", [(0, "शून्य"), (5, "पाँच"), (12, "बारह"), (21, "इक्कीस"), (49, "उनचास"), (99, "निन्यानवे"),
                                  (100, "एक सौ"), (500, "पाँच सौ"), (1999, "एक हज़ार नौ सौ निन्यानवे"),
                                  (2000, "दो हज़ार"), (10000, "दस हज़ार"), (100000, "एक लाख"),
                                  (250000, "दो लाख पचास हज़ार"), (10000000, "एक करोड़")])
def test_hindi_numbers(n, hi):
    assert hindi_number(n) == hi


# ------------------------------------------------------------------------------------------------ core list

def test_core_list_is_large_clean_and_covers_everyday_groups():
    core = hb.load_core()
    assert len(core) >= 500
    groups = {e.group for e in core.values()}
    assert {"greetings", "family", "food", "places", "numbers", "time", "colours", "emotions", "questions",
            "medical", "school", "verbs"} <= groups
    for k, e in core.items():
        assert DEVANAGARI.search(e.hindi), k
        assert e.pos in {"n", "v", "adj", "adv", "pron", "det", "wh", "neg", "time", "num", "conj", "prep", "intj",
                         "phrase"}, k
        assert all(DEVANAGARI.search(a) for a in e.alt), k
    for w, hi in {"hello": "नमस्ते", "mother": "माँ", "water": "पानी", "doctor": "डॉक्टर", "monday": "सोमवार",
                  "red": "लाल", "happy": "ख़ुश", "what": "क्या", "where": "कहाँ", "help": "मदद करना",
                  "indore": "इंदौर", "hospital": "अस्पताल"}.items():
        assert core[w].hindi == hi
    # sense-specific keys
    assert core["isl.second~time"].hindi == "सेकंड" and core["isl.second~number"].hindi == "दूसरा"


# ------------------------------------------------------------------------------------- model input/output

def test_mt_inputs_frame_the_sense():
    assert hb.mt_inputs({"concept_id": "isl.avoid", "headword": "avoid", "pos": "verb"}) == ["to avoid"]
    assert hb.mt_inputs({"concept_id": "isl.apple", "headword": "apple", "pos": "noun"}) == ["an apple"]
    assert hb.mt_inputs({"concept_id": "isl.bat~cricket", "headword": "bat", "display": {"en": "bat (cricket)"},
                         "domain": "cricket", "pos": "noun"}) == ["a bat (cricket)"]
    assert hb.mt_inputs({"concept_id": "isl.abhor-hate", "headword": "abhor", "kind": "synonym_list",
                         "synonyms": ["abhor", "hate"], "pos": "verb"}) == ["to abhor", "to hate"]
    assert hb.mt_inputs({"concept_id": "isl.p", "headword": "a penny saved is a penny earned",
                         "kind": "phrase"}) == ["A penny saved is a penny earned"]
    assert hb.mt_inputs({"concept_id": "isl.real-estate", "headword": "real estate agent",
                         "kind": "multiword"}) == ["real estate agent"]


@pytest.mark.parametrize("inp,out,clean", [
    ("to hate", "नफरत करने के लिए", "नफरत करना"),
    ("to go", "जाने के लिए", "जाना"),
    ("a watch", "एक घड़ी", "घड़ी"),
    ("second (number)", "दूसरा ( संख्या )", "दूसरा"),
    ("beautiful", "सुंदर । । । । ।", "सुंदर"),
    ("avoid", "बचने की कोशिश करें............", "बचने की कोशिश करें"),
    ("happy", "खुश हैं", "खुश"),
    ("real estate agent", "रियल एस्टेट एजेंट", "रियल एस्टेट एजेंट"),
    ("xyz", "xyz", ""),
    ("ATM", "ए. टी. एम.", "एटीएम"),
    ("a rhombus (math)", "ए रॉम्बस ( math )", "रॉम्बस"),
    ("delighted glad joyful pleased", "खुश खुश खुश खुश", "खुश"),
    ("sometimes", "कभी कभी", "कभी कभी"),
    ("T-Shirt", "टी - शर्ट", "टी-शर्ट"),
])
def test_clean_output(inp, out, clean):
    assert hb.clean_output(inp, out) == clean


def test_choose_majority_and_alternatives():
    assert hb.choose(["घृणा", "नफरत", "नफरत", ""]) == ("नफरत", ["घृणा"])
    assert hb.choose(["", ""]) == ("", [])
    assert hb.choose(["क", "ख"]) == ("क", ["ख"])  # tie: the first (the headword's) translation


# ------------------------------------------------------------------------------------------------ build

def _vocab():
    return [
        {"concept_id": "isl.mother", "headword": "mother", "display": {"en": "mother"}, "kind": "word", "pos": "noun"},
        {"concept_id": "isl.angry-enraged", "headword": "angry", "kind": "synonym_list",
         "synonyms": ["angry", "enraged"], "pos": "adjective"},
        {"concept_id": "isl.cold~for-food", "headword": "cold", "display": {"en": "cold (for food)"},
         "domain": "for food", "kind": "word"},
        {"concept_id": "isl.twelve", "headword": "twelve", "kind": "number", "number_value": 12},
        {"concept_id": "isl.avoid", "headword": "avoid", "display": {"en": "avoid"}, "kind": "word", "pos": "verb"},
        {"concept_id": "isl.zymurgy", "headword": "zymurgy", "display": {"en": "zymurgy"}, "kind": "word"},
        {"concept_id": "isl.dog", "headword": "dog", "display": {"en": "dog"}, "kind": "word", "pos": "noun"},
    ]


def test_build_rows_precedence_and_flags():
    core = {"mother": hb.CoreEntry("mother", "माँ", ["माता"], "n", "family"),
            "angry": hb.CoreEntry("angry", "ग़ुस्सा", [], "adj", "emotions"),
            "cold": hb.CoreEntry("cold", "ठंडा", [], "adj", "medical"),
            "dog": hb.CoreEntry("dog", "कुत्ता", [], "n", "animals")}
    cache = {"cold (for food)": "ठंडा ( भोजन के लिए )", "to avoid": "बचने के लिए"}
    existing = {"isl.dog": {"concept_id": "isl.dog", "hi": "श्वान", "review": "reviewed", "reviewer": "t1"}}
    rows, stats = hb.build_rows(_vocab(), core, cache, existing)
    by = {r["concept_id"]: r for r in rows}
    assert by["isl.mother"] == {"concept_id": "isl.mother", "english": "mother", "hindi": "माँ", "hi_alt": ["माता"],
                                "source": "manual", "reviewed": False, "review": "curated", "model": None}
    assert by["isl.angry-enraged"]["hindi"] == "ग़ुस्सा"  # through the synonym list
    assert by["isl.cold~for-food"]["source"] == "model" and by["isl.cold~for-food"]["hindi"] == "ठंडा"  # sense: no core
    assert by["isl.cold~for-food"]["model"] == hb.MODEL_ID and by["isl.cold~for-food"]["review"] == "machine"
    assert by["isl.twelve"]["hindi"] == "बारह" and by["isl.twelve"]["source"] == "rule"
    assert by["isl.avoid"]["hindi"] == "बचना"
    assert by["isl.dog"]["hindi"] == "श्वान" and by["isl.dog"]["reviewed"] is True  # a person's decision is kept
    assert "isl.zymurgy" not in by  # never invented
    assert stats["missing"] == 1 and stats["kept_reviewed"] == 1 and stats["rows"] == 6
    assert [r["concept_id"] for r in rows] == sorted(by)
    assert all(r["reviewed"] is (r["review"] == "reviewed") for r in rows)


def test_translate_missing_is_resumable(tmp_path):
    class Fake:
        def __init__(self):
            self.seen = []

        def translate(self, xs):
            self.seen += xs
            return [f"अनुवाद {x}" for x in xs]

    cache = tmp_path / "c.jsonl"
    f = Fake()
    assert hb.translate_missing(["b", "a", "a", "cc"], cache, f, batch=2, log=lambda m: None) == 3
    assert f.seen == ["a", "b", "cc"]  # deduplicated, shortest first
    f2 = Fake()
    assert hb.translate_missing(["a", "dd"], cache, f2, batch=2, log=lambda m: None) == 1 and f2.seen == ["dd"]
    assert hb.read_cache(cache)["dd"] == "अनुवाद dd"


def test_merge_local_folds_corrections(tmp_path):
    base, local = tmp_path / "lex.jsonl", tmp_path / "local.jsonl"
    hb.write_rows(base, [hb._row("isl.dog", "dog", "कुत्ता", [], "model", "machine", hb.MODEL_ID)])
    HindiLexicon(base, local).set_local("isl.dog", "कुत्ता", ["श्वान"], "reviewed", "teacher-1")
    assert hb.merge_local(base, local) == 1
    r = json.loads(base.read_text(encoding="utf-8"))
    assert r["review"] == "reviewed" and r["reviewed"] is True and r["source"] == "manual" and r["hi_alt"] == ["श्वान"]


def test_fetch_verification_matches_git_object_ids(tmp_path, monkeypatch):
    from app.isl.text import mt

    small = tmp_path / "config.json"
    small.write_bytes(b'{"a": 1}\n')
    big = tmp_path / "model.safetensors"
    big.write_bytes(b"\x00" * 1000)
    import hashlib

    sha = hashlib.sha256(big.read_bytes()).hexdigest()
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize 1000\n".encode()
    monkeypatch.setattr(mt, "OFFICIAL_OIDS", {"config.json": (mt._git_blob_sha1(small.read_bytes()), False),
                                              "model.safetensors": (mt._git_blob_sha1(pointer), True)})
    assert mt.verify_file(small, "config.json") and mt.verify_file(big, "model.safetensors")
    big.write_bytes(b"\x01" * 1000)
    assert not mt.verify_file(big, "model.safetensors")
    # git's own blob id of the empty file
    assert mt._git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


# ------------------------------------------------------------------------------------- tracked lexicon files

def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_tracked_isl_lexicon_is_consistent():
    rows = _rows(ISL_LEX)
    vocab = {json.loads(ln)["concept_id"]: json.loads(ln)
             for ln in (ISL_LEX.parent / "vocabulary.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()}
    ids = [r["concept_id"] for r in rows]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    assert set(ids) <= set(vocab)
    assert len(rows) >= 0.95 * len(vocab)  # coverage of the unified vocabulary
    for r in rows:
        assert set(r) >= {"concept_id", "english", "hindi", "hi_alt", "source", "reviewed", "review", "model"}
        assert r["source"] in ("manual", "rule", "model") and r["review"] in ("curated", "machine", "reviewed",
                                                                             "disputed")
        assert r["reviewed"] is (r["review"] == "reviewed")
        assert DEVANAGARI.search(r["hindi"]), r
        assert (r["model"] is not None) == (r["source"] == "model")
    by = {r["concept_id"]: r for r in rows}
    assert by["isl.hello"]["hindi"] == "नमस्ते" and by["isl.hello"]["source"] == "manual"
    assert by["isl.one-lakh"]["hindi"] == "एक लाख"
    assert sum(r["source"] == "manual" for r in rows) >= 500
    # every INCLUDE-linked (core, multi-signer) concept has Hindi
    include = [c for c, v in vocab.items() if any(x.get("source") == "include" for x in v.get("variants", []))]
    assert include and all(c in by for c in include)


def test_tracked_asl_lexicon_covers_wlasl100():
    names = json.loads((REPO / "models" / "wlasl100-gru-signer-holdout" / "model.json").read_text())["class_names"]
    rows = {r["concept_id"]: r for r in _rows(ASL_LEX)}
    assert set(rows) == set(names)
    assert rows["book"]["hindi"] == "किताब" and rows["thursday"]["hindi"] == "गुरुवार"


# ------------------------------------------------------------------------------------------- reader

def test_lexicon_reader_accepts_old_and_new_rows(tmp_path):
    base, local = tmp_path / "b.jsonl", tmp_path / "l.jsonl"
    base.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"concept_id": "isl.a", "hindi": "क", "source": "model", "review": "machine"},
        {"concept_id": "isl.b", "hi": "ख", "source": "manual", "review": "curated"},
        {"concept_id": "isl.c", "hindi": "ग", "review": "missing"}]) + "\nnot json\n", encoding="utf-8")
    lex = HindiLexicon(base, local)
    assert lex.text("isl.a") == ("क", "machine") and lex.text("isl.b") == ("ख", "curated")
    assert lex.get("isl.c") is None and lex.text("isl.zzz") == (None, None)
    row = lex.set_local("isl.a", " कख ", ["x"], "reviewed", "t1", english="a")
    assert row["hindi"] == row["hi"] == "कख" and row["reviewed"] is True
    assert HindiLexicon(base, local).text("isl.a") == ("कख", "reviewed")
    st = HindiLexicon(base, local).stats()
    assert st["entries"] == 2 and st["reviewed"] == 1 and st["local"] == 1
    with pytest.raises(ValueError):
        lex.set_local("isl.a", "x", review="bogus")
    assert HI_MISSING == "हिंदी उपलब्ध नहीं"


# ------------------------------------------------------------------------------------------- sentences

def _toks(*words):
    core = hb.load_core()
    return [make_token(f"isl.{w}", w, core[w].hindi if w in core else None) for w in words]


@pytest.mark.parametrize("words,en,hi", [
    (("your", "name", "what"), "What is your name?", "आपका नाम क्या है?"),
    (("you", "name", "what"), "What is your name?", "आपका नाम क्या है?"),
    (("you", "go", "where"), "Where do you go?", "आप कहाँ जाते हैं?"),
    (("i", "school", "go"), "I go to school.", "मैं विद्यालय जाता हूँ।"),
    (("i", "school", "go", "not"), "I do not go to school.", "मैं विद्यालय नहीं जाता।"),
    (("me", "happy"), "I am happy.", "मैं ख़ुश हूँ।"),
    (("i", "happy", "not"), "I am not happy.", "मैं ख़ुश नहीं हूँ।"),
    (("tomorrow", "i", "market", "go"), "Tomorrow I go to market.", "कल मैं बाज़ार जाता हूँ।"),
    (("mother", "food", "cook"), "Mother cooks food.", "माँ खाना पकाती है।"),
    (("water", "i", "want"), "I want water.", "मैं पानी चाहता हूँ।"),
    (("she", "sad", "why"), "Why is she sad?", "वह दुखी क्यों है?"),
    (("pizza", "i", "like"), "I like pizza.", "मैं पिज़्ज़ा पसंद करता हूँ।"),
    (("hello",), "Hello.", "नमस्ते।"),
    (("thank you",), "Thank you.", "धन्यवाद।"),
])
def test_sentence_examples(words, en, hi):
    toks = _toks(*words)
    e, h = build_english(toks), build_hindi(toks)
    assert (e.text, h.text) == (en, hi)
    assert e.approximate and h.approximate
    assert build_sentence(toks, "en").text == en and build_sentence(toks, "hi").text == hi


def test_sentence_rules_missing_hindi_and_labels():
    toks = [make_token("isl.i", "i", "मैं"), make_token("isl.zymurgy", "zymurgy", None)]
    out = build_both(toks)
    assert out["approximate"] is True and "Approximate" in out["label"]["en"] and "अनुमानित" in out["label"]["hi"]
    assert out["hi"]["missing_hi"] == ["isl.zymurgy"] and "[zymurgy]" in out["hi"]["text"]
    assert out["en"]["text"] == "I am zymurgy."
    q = build_both(_toks("you", "go", "where"))
    assert "moved the question word to the front" in q["en"]["rules"]
    assert any("masculine" in r for r in q["hi"]["rules"])
    assert build_both([])["en"]["text"] == ""
    with pytest.raises(ValueError):
        build_sentence(toks, "fr")


# ------------------------------------------------------------------------------- ASL path (WLASL100 glosses)

class _StubPredictor:
    def __init__(self, names):
        from app.features.spec import FeatureConfig

        self.class_names, self.feature_config, self.recommended_threshold = names, FeatureConfig(), 0.5

    def predict(self, feats):
        p = np.full((len(feats), len(self.class_names)), 0.05 / (len(self.class_names) - 1), dtype=np.float32)
        p[:, 0] = 0.95
        return p, 1.0


def test_asl_classifier_and_legacy_engine_carry_hindi():
    from app.dataset.synthetic import frames_for_stream, idle_frames, make_sequence
    from app.inference.engine import RecognitionEngine
    from app.inference.recognizer import ClassifierRecognizer

    lex = HindiLexicon(ASL_LEX, None)
    pred = _StubPredictor(["book", "drink", "computer"])
    rec = ClassifierRecognizer(pred, lexicon=lex).recognize([make_sequence(1, 2, n_frames=40)], top_k=2)
    assert rec.candidates[0].text == {"en": "book", "hi": "किताब"} and rec.candidates[0].hi_review == "curated"
    eng = RecognitionEngine(pred, lexicon=lex)
    t, states = 0.0, []
    for f in idle_frames(12, t):
        states.append(eng.push(f))
    t += 12 * 40
    for f in frames_for_stream(make_sequence(1, 2, n_frames=40), t):
        states.append(eng.push(f))
    t += 40 * 40
    for f in idle_frames(24, t):
        states.append(eng.push(f))
    commits = [s["commit"] for s in states if "commit" in s]
    assert commits and commits[0]["gloss"] == "book" and commits[0]["text"] == {"en": "book", "hi": "किताब"}
    seg = next(s["segment_result"] for s in states if "segment_result" in s and "top" in s["segment_result"])
    assert seg["top"][0]["text"]["hi"] == "किताब" and set(seg["top"][0]) >= {"class_index", "gloss", "prob"}
    # without a lexicon the legacy messages are exactly as before
    eng2 = RecognitionEngine(pred)
    assert "text" not in eng2._topk(np.array([0.9, 0.05, 0.05]))[0]
