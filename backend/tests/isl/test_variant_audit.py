"""Variant forms instead of forced merges (QA plan 1.3, critics 7 and 13, docs/ISL_DESIGN.md 15.36).

* the audit relinks an ISL500 word only when the reference classifier disagrees AND the dictionary (or the reverse
  direction) says the ISL500 clips are another sign form; signer shift (the dictionary agrees) is never relinked;
* the dictionary rule is calibrated on the words the classifier agrees on (signer shift on the same encoder), not on
  the in-sample INCLUDE rank; a rule with too many false alarms is marked not usable and the link step refuses it;
* the audit is a proposal in the profile's work folder; the link step reads only the approved copy from there (or a
  file named explicitly), checks the ISL500 split it used, freezes the relinks with the file's SHA-256 into
  link_settings.json, and later steps use that copy;
* a variant concept is an alias of its word: alias-scored E1b counts a variant prediction as correct, strict does not;
  word labels count only variants as their word (twins stay strict).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.sources import Clip
from app.isl.training import multisource as ms
from app.isl.training import variant_audit as va

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=2, lr=3e-3)
CLASSES = ["isl.hot", "isl.cat", "isl.dog", "isl.big"]
DICT = ["isl.hot", "isl.cat", "isl.dog", "isl.big", "isl.egg", "isl.fish"]


def _clip(cid: str, i: int, source: str = "isl500") -> Clip:
    return Clip(f"{source}/{cid}/{i}", source, cid.split(".")[1], cid, None, None, Path("/nonexistent.npz"),
                group=f"{source}-u{i % 4}")


def _onehot(k: int, d: int = len(DICT)) -> np.ndarray:
    v = np.full(d, 0.01, np.float32)
    v[k] = 1.0
    return v / np.linalg.norm(v)


class _Fake:
    """A fake reference: per clip id, a probability row over CLASSES and an embedding in the dictionary space."""

    def __init__(self):
        self.p: dict[str, np.ndarray] = {}
        self.e: dict[str, np.ndarray] = {}

    def add(self, clip: Clip, pred: str, emb: np.ndarray):
        row = np.full(len(CLASSES), 0.01, np.float32)
        row[CLASSES.index(pred)] = 1.0
        self.p[clip.clip_id], self.e[clip.clip_id] = row / row.sum(), emb
        return clip

    def score(self, clips):
        used = [c for c in clips if c.clip_id in self.p]
        P = np.stack([self.p[c.clip_id] for c in used]) if used else np.zeros((0, len(CLASSES)))
        E = np.stack([self.e[c.clip_id] for c in used]) if used else np.zeros((0, len(DICT)))
        return P, E, used


def _far(word: str) -> np.ndarray:
    """An embedding close to every dictionary row but ``word``'s (another sign form)."""
    v = np.array([-1.0 if c == word else 1.0 - 0.1 * i for i, c in enumerate(DICT)], np.float32)
    return v / np.linalg.norm(v)


P1 = va.AuditParams(top_k=1, dict_rank_floor=1, dict_include_max_rank=2, dict_calib_min_words=1)


def _dictionary() -> va.Dictionary:
    # one row per concept; row i points along axis i
    return va.Dictionary(np.stack([_onehot(i) for i in range(len(DICT))]), list(DICT))


def _toy(n: int = 24, shift_word_dict: bool = False):
    """ISL500: "hot" signed in another form (predicted dog, retrieves the egg row), "dog" by new signers (predicted
    cat, but its centroid still retrieves the dog row: signer shift), "cat" agrees, "big" has too few clips."""
    f = _Fake()
    isl, inc = [], []
    for i in range(n):
        isl.append(f.add(_clip("isl.hot", i), "isl.dog", _far("isl.hot")))
        isl.append(f.add(_clip("isl.dog", i), "isl.cat", _onehot(DICT.index("isl.dog"))))
        isl.append(f.add(_clip("isl.cat", i), "isl.cat", _onehot(DICT.index("isl.cat"))))
    for i in range(5):
        isl.append(f.add(_clip("isl.big", i), "isl.dog", _onehot(DICT.index("isl.fish"))))
    for w in ("isl.hot", "isl.dog", "isl.cat", "isl.big"):
        for i in range(6):
            inc.append(f.add(_clip(w, i, "include"), w, _onehot(DICT.index(w))))
    return f, isl, inc


def test_fake_classifier_gives_the_expected_relinks_with_evidence():
    f, isl, inc = _toy()
    models = va.AuditModels(CLASSES, f.score, _dictionary(), provenance={"classifier": "fake"})
    doc = va.run_audit(isl, inc, models, P1)
    assert doc["relinks"] == {"isl500": {"isl.hot": "isl.hot~isl500"}}
    hot = doc["classes"]["isl.hot"]
    assert hot["decision"] == "relink" and hot["variant_id"] == "isl.hot~isl500"
    assert hot["agreement_top1"] == 0.0 and hot["predicted_as"][0][0] == "isl.dog"
    d = hot["dictionary"]
    assert d["verdict"] == va.DIFFERENT and d["rank_include"] == 1 and d["rank_isl500"] > 1
    assert d["isl500_top3"][0] == "isl.cat" and d["rows"] == ["isl.hot"] and d["rank_isl500"] == len(DICT)
    assert doc["classes"]["isl.cat"]["decision"] == "keep" and "agrees" in doc["classes"]["isl.cat"]["reason"]
    assert doc["classes"]["isl.big"]["decision"] == "keep" and "fewer than" in doc["classes"]["isl.big"]["reason"]
    s = doc["summary"]
    assert s["words_judged"] == 3 and s["low_agreement"] == 2 and s["relinked"] == 1
    assert s["dictionary_false_alarms_on_agreeing_words"] == {"n": 0, "of": 1, "rate": 0.0, "method": "leave-one-out"}
    assert s["dictionary_rule_usable"] is True and s["dictionary_calibration"]["threshold"] == 1.0
    assert doc["part"] == "train" and doc["reference"] == {"classifier": "fake"}


def test_signer_shift_where_the_dictionary_agrees_is_not_relinked():
    f, isl, inc = _toy()
    doc = va.run_audit(isl, inc, va.AuditModels(CLASSES, f.score, _dictionary()), P1)
    dog = doc["classes"]["isl.dog"]
    assert dog["agreement_top1"] == 0.0 and dog["dictionary"]["verdict"] == va.NO_DIFFERENCE
    assert dog["decision"] == "keep" and "signer shift" in dog["reason"]
    assert "isl.dog" not in doc["relinks"]["isl500"] and "isl.hot" in doc["relinks"]["isl500"]
    # without any dictionary no word is relinked on the classifier alone
    nod = va.run_audit(isl, inc, va.AuditModels(CLASSES, f.score, None), va.AuditParams(top_k=1))
    assert nod["relinks"] == {"isl500": {}}
    assert all(e["decision"] == "keep" for e in nod["classes"].values())


def test_dictionary_verdict_rules():
    p = va.AuditParams()
    assert va.dictionary_verdict(3, 1, p, 5.0)[0] == va.NO_DIFFERENCE  # within the floor
    assert va.dictionary_verdict(36, 1, p, 40.0)[0] == va.NO_DIFFERENCE  # within the agreeing words' spread
    assert va.dictionary_verdict(41, 1, p, 40.0)[0] == va.DIFFERENT
    assert va.dictionary_verdict(500, 60, p, 40.0)[0] == va.INCONCLUSIVE  # INCLUDE does not reach the row either
    assert va.dictionary_verdict(None, None, p, 40.0)[0] == va.INCONCLUSIVE
    assert va.dictionary_verdict(500, 1, p, None)[0] == va.INCONCLUSIVE  # not calibrated


class _RankDictionary:
    """A dictionary whose rank is read off the centroid: embedding axis j -> rank RANKS[j] (INCLUDE clips: axis 0)."""

    def __init__(self, ranks: list[int]):
        self.ranks = ranks
        self.pos = {f"isl.w{i}": i for i in range(len(ranks))} | {"isl.gun": 0, "isl.mean": 0}

    def __len__(self):
        return 2000

    def rank(self, q, targets):
        return self.ranks[int(np.argmax(q))], [], None


def test_dictionary_rule_is_calibrated_on_agreeing_words_not_on_in_sample_include():
    """Review of b03eba0: the INCLUDE rank is in-sample (always 1); the yardstick is the ISL500 rank of the words the
    classifier agrees on. A low-agreement word inside that spread is signer shift (kept); far above it is relinked."""
    words = [f"isl.w{i}" for i in range(1, 31)]
    ranks = [1] + list(range(1, 31)) + [22, 278]  # axis 0: INCLUDE (rank 1); axes 1-30: agreeing words; 31: gun; 32
    names = words + ["isl.gun", "isl.mean"]
    dim = len(ranks)
    p_rows, e_rows = {}, {}
    isl, inc = [], []

    def axis(j):
        v = np.zeros(dim, np.float32)
        v[j] = 1.0
        return v

    for k, w in enumerate(names):
        agree = w.startswith("isl.w")
        for i in range(20):
            c = _clip(w, i)
            row = np.full(len(names), 0.01, np.float32)
            row[names.index(w) if agree else (k + 1) % len(names)] = 1.0
            p_rows[c.clip_id], e_rows[c.clip_id] = row, axis(k + 1)
            isl.append(c)
        for i in range(3):
            c = _clip(w, i, "include")
            p_rows[c.clip_id], e_rows[c.clip_id] = np.ones(len(names), np.float32), axis(0)
            inc.append(c)

    def score(clips):
        u = [c for c in clips if c.clip_id in p_rows]
        return np.stack([p_rows[c.clip_id] for c in u]), np.stack([e_rows[c.clip_id] for c in u]), u

    models = va.AuditModels(names, score, _RankDictionary(ranks))
    doc = va.run_audit(isl, inc, models, va.AuditParams(top_k=1))
    cal = doc["summary"]["dictionary_calibration"]
    assert cal["n_words"] == 30 and 28 < cal["threshold"] < 30 and cal["usable"] is True
    assert cal["false_alarms"]["rate"] <= 0.10 and doc["summary"]["dictionary_rule_usable"] is True
    gun, mean = doc["classes"]["isl.gun"], doc["classes"]["isl.mean"]
    assert gun["agreement_top1"] == 0.0 and gun["dictionary"]["rank_isl500"] == 22
    assert gun["dictionary"]["rank_include"] == 1  # in-sample: the old ratio rule would have said "another form"
    assert gun["dictionary"]["verdict"] == va.NO_DIFFERENCE and gun["decision"] == "keep"
    assert mean["dictionary"]["verdict"] == va.DIFFERENT and mean["decision"] == "relink"
    assert doc["relinks"] == {"isl500": {"isl.mean": "isl.mean~isl500"}}
    # a noisy rule (too many agreeing words above a strict percentile) is marked not usable
    noisy = va.run_audit(isl, inc, models, va.AuditParams(top_k=1, dict_calib_percentile=50, dict_rank_floor=1))
    assert noisy["summary"]["dictionary_rule_usable"] is False
    assert "false-alarm" in noisy["summary"]["dictionary_calibration"]["reason"]
    # too few agreeing words: not calibrated, every verdict inconclusive, nothing relinked
    few = va.run_audit(isl, inc, models, va.AuditParams(top_k=1, dict_calib_min_words=31))
    assert few["summary"]["dictionary_calibration"]["threshold"] is None and few["relinks"] == {"isl500": {}}
    assert few["classes"]["isl.mean"]["dictionary"]["verdict"] == va.INCONCLUSIVE


def test_the_reverse_direction_can_veto_or_decide():
    p = va.AuditParams(top_k=1)
    assert va.decide(0.1, 30, va.DIFFERENT, va.NO_DIFFERENCE, p)[0] == "keep"
    assert va.decide(0.1, 30, va.INCONCLUSIVE, va.DIFFERENT, p)[0] == "relink"
    assert va.decide(0.1, 30, va.INCONCLUSIVE, None, p)[0] == "keep"
    assert va.decide(0.1, 30, va.NO_DIFFERENCE, va.DIFFERENT, p)[0] == "keep"
    f, isl, inc = _toy()
    rev = _Fake()
    for c in inc:  # an ISL500-trained model: recognises INCLUDE "hot" (same form after all), not INCLUDE "dog"
        rev.add(c, "isl.hot" if c.concept_id == "isl.hot" else "isl.cat", np.zeros(len(DICT)))
    models = va.AuditModels(CLASSES, f.score, _dictionary(), CLASSES, rev.score)
    doc = va.run_audit(isl, inc, models, P1)
    hot = doc["classes"]["isl.hot"]
    assert hot["reverse"]["verdict"] == va.NO_DIFFERENCE and hot["decision"] == "keep"
    assert doc["relinks"] == {"isl500": {}}


def test_variant_links_file_is_profile_scoped_and_hashed(tmp_path):
    f, isl, inc = _toy()
    doc = va.run_audit(isl, inc, va.AuditModels(CLASSES, f.score, _dictionary()), P1)
    out = tmp_path / "research" / va.VARIANT_LINKS
    meta = va.write_variant_links(out, doc, profile="research", work=tmp_path / "research",
                                  extra={"parts_used": {"isl500": "train", "include": ["train", "val"]}})
    rel, m = va.load_variant_links(out, "research")
    assert rel == {"isl500": {"isl.hot": "isl.hot~isl500"}} and m["sha256"] == meta["sha256"] and m["n_relinks"] == 1
    assert m["parts_used"]["include"] == ["train", "val"] and m["dictionary_rule_usable"] is True
    # a rule marked not usable is refused unless accepted (and then recorded)
    noisy = tmp_path / "noisy.json"
    va.write_variant_links(noisy, {**doc, "summary": {**doc["summary"], "dictionary_rule_usable": False}},
                           profile="research", work=tmp_path / "research")
    with pytest.raises(va.VariantLinksError, match="not usable"):
        va.load_variant_links(noisy, "research")
    assert va.load_variant_links(noisy, "research", accept_noisy=True)[1]["accepted_noisy"] is True
    # an audit of the old, uncalibrated rule is refused
    old = tmp_path / "old.json"
    va.write_variant_links(old, {**doc, "audit_version": "isl-variant-audit-1"}, profile="research", work=tmp_path)
    with pytest.raises(va.VariantLinksError, match="audit version"):
        va.load_variant_links(old, "research")
    # the split check: every audited train signer must still be train
    rec = va.split_record(0, {"val_frac": 0.1}, {"u1", "u2", "u3"})
    assert va.check_split(rec, {"u1", "u2", "u3", "u4"})["ok"] is True
    bad_split = va.check_split(rec, {"u1", "u2"})
    assert bad_split["ok"] is False and bad_split["moved_groups"] == ["u3"]
    assert va.check_split(None, {"u1"})["ok"] is False
    with pytest.raises(va.VariantLinksError, match="profile"):
        va.load_variant_links(out, "commercial-safe")
    bad = json.loads(out.read_text())
    bad["relinks"]["isl500"]["isl.hot"] = "isl.cat"
    out.write_text(json.dumps(bad))
    with pytest.raises(va.VariantLinksError, match="not isl.hot~isl500"):
        va.load_variant_links(out, "research")


VOCAB = [{"concept_id": "isl.hot", "headword": "hot", "display": {"en": "hot"}, "kind": "word", "category": "x"},
         {"concept_id": "isl.cat", "headword": "cat", "display": {"en": "cat"}, "kind": "word"}]


def test_the_linker_relinks_only_that_source_and_word_and_freezes_the_settings(tmp_path):
    vmeta = {"file": "w/variant_links.json", "sha256": "ab" * 32}
    settings = ms.link_settings(False, {"isl500": {"isl.hot": "isl.hot~isl500"}}, vmeta)
    lk = ms.GlossLinker(VOCAB, **ms.linker_options(settings))
    clips = [Clip("isl500/a", "isl500", "Hot", None, None, None, Path("x")),
             Clip("isl500/b", "isl500", "Cat", None, None, None, Path("x")),
             Clip("isl500/c", "isl500", "x", "isl.hot", None, None, Path("x")),  # adapter-linked
             Clip("mendeley_emergency/d", "mendeley_emergency", "Hot", None, None, None, Path("x"))]
    out = lk.link_clips(clips)
    assert [c.concept_id for c in out] == ["isl.hot~isl500", "isl.cat", "isl.hot~isl500", "isl.hot"]
    assert out[0].quality["variant_of"] == "isl.hot" and out[2].quality["variant_of"] == "isl.hot"
    assert lk.alias_of == {"isl.hot~isl500": "isl.hot"} and lk.counts["variant_relinked:isl500"] == 2
    row = lk.extra_concepts()[0]
    assert row["concept_id"] == "isl.hot~isl500" and row["display"] == {"en": "hot"} and row["alias_of"] == "isl.hot"
    assert lk.expand({"isl.hot"}) >= {"isl.hot", "isl.hot~isl500"}  # held-out guard covers the variant
    d = ms.write_link_settings(tmp_path, settings, lk)
    assert d["variants"] == {"isl500": {"isl.hot": "isl.hot~isl500"}} and d["variant_links"]["sha256"] == "ab" * 32
    assert d["alias_of"] == {"isl.hot~isl500": "isl.hot"} and d["variant_relinked"] == 2 and d["aliases"] is False
    # a later step reads the frozen file and links the same way
    lk2 = ms.GlossLinker(VOCAB, **ms.linker_options(ms.read_link_settings(tmp_path)))
    assert [c.concept_id for c in lk2.link_clips(clips)] == [c.concept_id for c in out]
    # without variants nothing changes and no new key is written
    assert set(ms.link_settings(False)) == {"version", "strip_sign_numbers", "aliases", "alias_version"}
    assert "variants" not in ms.linker_options(ms.link_settings(True))
    # a variant id that already is a vocabulary concept is never used
    lk3 = ms.GlossLinker(VOCAB + [{"concept_id": "isl.hot~isl500", "headword": "hot!", "kind": "word"}],
                         variants={"isl500": {"isl.hot": "isl.hot~isl500"}})
    assert lk3.link_clips(clips[2:3])[0].concept_id == "isl.hot" and lk3.counts["variant_skipped_known"] == 1


def test_word_labels_count_a_variant_as_its_word_but_keep_twins_strict():
    """Review of b03eba0: a variant is the same word and the same output; word labels collapse only the variants
    (from the frozen link settings), so a twin prediction stays wrong."""
    from app.isl.evaluation.runners import e1_alias_rates, e1_like_for_like, group_hits, variant_word_map

    link = {"variants": {"isl500": {"isl.hot": "isl.hot~isl500"}},
            "alias_of": {"isl.hot~isl500": "isl.hot", "isl.woman-female": "isl.female"}}
    vmap = variant_word_map(link)
    assert vmap == {"isl.hot~isl500": "isl.hot"} and variant_word_map({}) == {}
    ids = ["isl.cat", "isl.hot", "isl.hot~isl500", "isl.female", "isl.woman-female"]
    P = np.array([[0.1, 0.3, 0.6, 0.0, 0.0],     # INCLUDE "hot" predicted as the ISL500 variant: right on word labels
                  [0.0, 0.0, 0.0, 0.3, 0.7],     # "female" predicted as its twin: stays wrong on word labels
                  [0.1, 0.7, 0.2, 0.0, 0.0],     # an ISL500 "hot~isl500" clip predicted as the word: right
                  [0.9, 0.1, 0.0, 0.0, 0.0]])    # "cat": right both ways
    y = np.array([1, 3, 2, 0])
    hit = (P.argmax(1) == y).astype(float)
    r = e1_alias_rates(P, y, ids, vmap, hit, np.ones(4), rule="word")
    assert r["top1_strict"] == 0.25 and r["top1"] == 0.75 and r["rule"] == "word"
    h1, _ = group_hits(P, y, ids, vmap)
    assert h1.tolist() == [True, False, True, True]
    # alias labels (twins too) would count the twin as right as well
    assert e1_alias_rates(P, y, ids, link["alias_of"] | vmap, hit, np.ones(4))["top1"] == 1.0
    # like for like on common clips: the word-label row next to the strict one
    used = [Clip(f"c{i}", "include", "x", ids[t], None, None, Path("x")) for i, t in enumerate(y)]
    ref = {"split": "s", "clip_ids": ["c0", "c1", "c2"], "hit1": "111", "hit5": "111"}
    lf = e1_like_for_like(P, y, ids, used, hit, np.ones(4), "s", ref, word_hits=group_hits(P, y, ids, vmap))
    assert lf["common_clips"]["top1"] == 0.0 and lf["common_clips_word_labels"]["top1"] == pytest.approx(2 / 3)


def test_alias_scored_e1b_counts_a_variant_prediction_strict_does_not():
    from app.isl.evaluation.runners import e1_alias_rates

    lk = ms.GlossLinker(VOCAB, variants={"isl500": {"isl.hot": "isl.hot~isl500"}})
    lk.link_clips([Clip("isl500/a", "isl500", "Hot", None, None, None, Path("x"))])
    ids = ["isl.cat", "isl.hot", "isl.hot~isl500"]
    P = np.array([[0.1, 0.3, 0.6],    # INCLUDE "hot" clip, the ISL500 form predicted: strict wrong, alias right
                  [0.7, 0.2, 0.1]])   # INCLUDE "cat": right both ways
    y = np.array([1, 0])
    hit = (P.argmax(1) == y).astype(float)
    r = e1_alias_rates(P, y, ids, lk.alias_of, hit, np.ones(2))
    assert r["top1_strict"] == 0.5 and r["top1"] == 1.0 and r["top1_only_alias"] == 1
    assert r["by_group"]["isl.hot"] == {"n": 1, "top1_only_alias": 1, "top1_only_strict": 0}


# ------------------------------------------------------------------------------------------------ CLI


class _Adapter:
    def __init__(self, name, clips):
        self.name, self.commercial, self.licence = name, False, "test"
        self._clips = clips

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        yield from self._clips


@pytest.fixture()
def world(tmp_path, monkeypatch):
    from tests.isl.isl_world import make_world
    from tests.isl.test_eval_runners import _extra_world

    w = make_world(tmp_path)
    extra = _extra_world(tmp_path, users=8, takes=3)
    ms.register_adapter("isl500", lambda p, v: _Adapter("isl500", extra))
    base = tmp_path / "art"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    common = ["--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1"]
    yield {**w, "base": base, "common": common, "root": ["--root", str(w["paths"].root)], "extra": extra}
    ms.register_adapter("isl500", None)


def test_cli_link_sources_reads_only_this_work_folders_audit_and_freezes_it(world):
    from app.isl import cli

    base, root = world["base"], world["root"]
    common = ["--profile", "research", *world["common"]]
    work = base / "research"
    work.mkdir(parents=True)
    train_groups = ms.source_train_groups(world["paths"], None, "research", "isl500")
    assert 0 < len(train_groups) < 8
    doc = {"kind": va.VARIANT_KIND, "audit_version": va.AUDIT_VERSION, "relinks": {"isl500": {"isl.cat": "isl.cat~isl500"}},
           "isl500_split": va.split_record(0, ms.SIGNER_SPLIT, train_groups),
           "parts_used": {"isl500": "train", "include": ["train"]}}
    # a proposal is never read by the link step itself
    va.write_variant_links(work / va.VARIANT_AUDIT, doc, profile="research", work=work)
    assert cli.main(root + ["link-sources", *common]) == 0
    assert "variants" not in json.loads((work / "link_settings.json").read_text())
    va.write_variant_links(work / va.VARIANT_LINKS, doc, profile="research", work=work)
    sha = va.file_sha256(work / va.VARIANT_LINKS)
    assert cli.main(root + ["link-sources", *common]) == 0
    s = json.loads((work / "link_settings.json").read_text())
    assert s["variants"] == {"isl500": {"isl.cat": "isl.cat~isl500"}} and s["variant_links"]["sha256"] == sha
    assert s["variant_links"]["split_check"]["ok"] is True
    assert s["variant_links"]["parts_used"] == {"isl500": "train", "include": ["train"]}
    assert s["alias_of"] == {"isl.cat~isl500": "isl.cat"} and s["variant_relinked"] == 8 * 3
    assert "isl.cat~isl500" in (work / "extra_concepts.jsonl").read_text()
    # a later step uses the frozen copy, even after the audit file changed
    (work / va.VARIANT_LINKS).write_text(json.dumps({**doc, "profile": "research", "relinks": {}}))
    args = cli.build_parser().parse_args(root + ["build-index", *common, "--all-sources"])
    _, data, _, _ = cli._model_ctx(world["paths"], args)
    isl = [c for c in data.extra if c.source == "isl500"]
    assert {c.concept_id for c in isl if c.raw_gloss == "Cat"} == {"isl.cat~isl500"}
    assert {c.concept_id for c in data.include if c.concept_id} >= {"isl.cat"}  # INCLUDE keeps its word
    assert data.alias_of == {"isl.cat~isl500": "isl.cat"}
    # --variant-links none: linked as before
    assert cli.main(root + ["link-sources", *common, "--variant-links", "none"]) == 0
    s = json.loads((work / "link_settings.json").read_text())
    assert "variants" not in s and s["alias_of"] == {}
    # an approved file copied from another work folder (e.g. seeded into a scratch run) is ignored in auto mode, and
    # used when named explicitly
    foreign = dict(doc)
    va.write_variant_links(work / va.VARIANT_LINKS, foreign, profile="research", work=base / "elsewhere")
    assert cli.main(root + ["link-sources", *common]) == 0
    assert "variants" not in json.loads((work / "link_settings.json").read_text())
    assert cli.main(root + ["link-sources", *common, "--variant-links", str(work / va.VARIANT_LINKS)]) == 0
    assert "variants" in json.loads((work / "link_settings.json").read_text())
    # another profile's file is refused, also when given explicitly
    other = base / "x.json"
    va.write_variant_links(other, doc, profile="commercial-safe", work=base / "commercial-safe")
    assert cli.main(root + ["link-sources", *common, "--variant-links", str(other)]) == 2
    assert cli.main(root + ["link-sources", *common, "--variant-links", str(base / "missing.json")]) == 2


def test_cli_link_sources_refuses_an_audit_made_under_another_split(world, monkeypatch):
    """Review of b03eba0: the audit records its ISL500 train signers; when the split moved one of them to val / test
    (e.g. QA plan 1.6's larger ISL500 val part), the link step refuses the file (exit 2) unless overridden, recorded."""
    from app.isl import cli

    base, root = world["base"], world["root"]
    common = ["--profile", "research", *world["common"]]
    work = base / "research"
    work.mkdir(parents=True)
    train_groups = ms.source_train_groups(world["paths"], None, "research", "isl500")
    doc = {"kind": va.VARIANT_KIND, "audit_version": va.AUDIT_VERSION, "relinks": {"isl500": {"isl.cat": "isl.cat~isl500"}},
           "isl500_split": va.split_record(0, ms.SIGNER_SPLIT, train_groups)}
    va.write_variant_links(work / va.VARIANT_LINKS, doc, profile="research", work=work)
    monkeypatch.setitem(ms.SIGNER_SPLIT, "val_frac", 0.27)  # 8 signers: 2 in val instead of 1
    assert ms.source_train_groups(world["paths"], None, "research", "isl500") < train_groups
    assert cli.main(root + ["link-sources", *common]) == 2
    assert not (work / "link_settings.json").exists()
    assert cli.main(root + ["link-sources", *common, "--allow-split-change"]) == 0
    sc = json.loads((work / "link_settings.json").read_text())["variant_links"]["split_check"]
    assert sc["ok"] is False and sc["override"] is True and len(sc["moved_groups"]) == 1
    # a hand-made file without the split record is refused as well
    va.write_variant_links(work / va.VARIANT_LINKS, {**doc, "isl500_split": None}, profile="research", work=work)
    monkeypatch.setitem(ms.SIGNER_SPLIT, "val_frac", 0.1)
    assert cli.main(root + ["link-sources", *common]) == 2


def test_cli_variant_audit_end_to_end_on_train_clips_only(world, tmp_path):
    from app.isl import cli
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.common import ISL_FEATURES, load_isl_data
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex
    from app.training.trainer import TrainConfig

    P = world["paths"]
    ref = tmp_path / "ref"
    vocab = VocabIndex.load(world["vocab"])
    data = load_isl_data(P, vocab, "commercial-safe", world["split"])
    cfg = TrainConfig(**TINY)
    train_classifier(data, ref / "classifier.pt", ISL_FEATURES, cfg, smoke=True, threads=1, use_extra=False)
    train_encoder(data, ref / "encoder.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2, smoke=True, threads=1,
                  batches_per_epoch=2)
    build_index(EncoderPredictor(ref / "encoder.pt"), data.islrtc, data.include_part("train"), ref / "index",
                "commercial-safe", split=data.split["name"])
    root, common = world["root"], world["common"]
    args = [*root, "variant-audit", "--profile", "research", *common, "--reference-work", str(ref), "--min-clips", "3"]
    assert cli.main(args) == 0
    assert cli.build_parser().parse_args([*args, "--top-k", "1"]).top_k == 1
    out = world["base"] / "research" / va.VARIANT_AUDIT  # a proposal, not the approved variant_links.json
    assert not (world["base"] / "research" / va.VARIANT_LINKS).exists()
    doc = json.loads(out.read_text())
    assert doc["kind"] == va.VARIANT_KIND and doc["profile"] == "research" and doc["part"] == "train"
    assert doc["parts_used"] == {"isl500": "train", "include": ["train"]}
    assert doc["isl500_split"]["seed"] == 0 and doc["isl500_split"]["params"] == ms.SIGNER_SPLIT
    assert doc["isl500_split"]["train_groups"] == sorted(ms.source_train_groups(P, None, "research", "isl500"))
    assert "dictionary_calibration" in doc["summary"]
    assert doc["reference"]["classifier_profile"] == "commercial-safe" and doc["reference"]["index_sha256"]
    assert doc["reference"]["dictionary_concepts"] >= 4
    # only ISL500 train clips were scored: every word's count is its train clips
    rdata = load_isl_data(P, vocab, "research", world["split"])
    ms.attach_all_sources(rdata, P, vocab, "research")
    n_train = {}
    for c in rdata.extra_part("train", ("isl500",)):
        n_train[c.concept_id] = n_train.get(c.concept_id, 0) + 1
    assert {w: e["n_clips"] for w, e in doc["classes"].items()} == n_train
    assert all("dictionary" in e and e["decision"] in ("keep", "relink") for e in doc["classes"].values())
    assert not list(Path(P.derived).glob("variant_links*"))  # never the shared data/isl/derived
    va.load_variant_links(out, "research")  # the link step can read it
    # refusals: a profile without ISL500, a reference trained on ISL500, an output under data/isl/derived
    cs = [*root, "variant-audit", "--profile", "commercial-safe", *common, "--reference-work", str(ref)]
    assert cli.main(cs) == 2
    assert cli.main([*args, "--out", str(Path(P.derived) / va.VARIANT_AUDIT)]) == 2
    import torch

    ck = torch.load(ref / "classifier.pt", weights_only=False)
    ck["meta"]["profile"] = "research"
    torch.save(ck, ref / "classifier.pt")
    assert cli.main(args) == 2


def test_pack_shows_the_words_hindi_for_a_variant(tmp_path):
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack
    from tests.isl.isl_world import make_world

    w = make_world(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    (work / "extra_concepts.jsonl").write_text(json.dumps({"concept_id": "isl.cat~isl500", "headword": "cat",
                                                           "display": {"en": "cat"}, "kind": "word",
                                                           "alias_of": "isl.cat"}) + "\n")
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "pack", smoke=True, vocab_path=w["vocab"])
    lex = [json.loads(x) for x in (Path(r["pack"]) / "lexicon_hi.jsonl").read_text().splitlines()]
    by = {x["concept_id"]: x for x in lex}
    if "isl.cat" not in by:
        pytest.skip("no Hindi row for isl.cat in models/packs/isl/lexicon_hi.jsonl")
    assert by["isl.cat~isl500"]["hindi"] == by["isl.cat"]["hindi"] and by["isl.cat~isl500"]["alias_of"] == "isl.cat"


def test_report_and_pack_trace_the_variant_links_and_warn_on_val_evidence(tmp_path):
    """Review of b03eba0: the audit's file, sha, relinks and evidence scope reach the evaluate report and the pack
    manifest; validation evidence is warned about."""
    from app.isl.evaluation.runners import _word_label_lines, variant_links_summary
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack
    from tests.isl.isl_world import make_world

    assert variant_links_summary({}) is None and variant_links_summary(ms.link_settings(True)) is None
    meta = {"file": "w/variant_links.json", "sha256": "ab" * 32, "n_relinks": 1, "audit_version": va.AUDIT_VERSION,
            "parts_used": {"isl500": "train", "include": ["train", "val"]}, "dictionary_false_alarm_rate": 0.05}
    settings = {**ms.link_settings(False, {"isl500": {"isl.cat": "isl.cat~isl500"}}, meta), "variant_relinked": 24}
    vl = variant_links_summary(settings)
    assert vl["sha256"] == "ab" * 32 and vl["n_relinks"] == 1 and vl["variant_concepts"] == ["isl.cat~isl500"]
    assert vl["relinked_clips"] == 24 and any("validation" in w for w in vl["warnings"])
    doc = {"aliases": {"used": True, "variant_links": vl},
           "protocols": {"E1b": {"word_labels": {"n": 4, "top1": 0.75, "top1_strict": 0.25, "top5": 1.0,
                                                 "top5_strict": 1.0}}}}
    md = "\n".join(_word_label_lines(doc))
    assert "warning: validation clips" in md and "E1b word labels" in md and "abababab" in md
    w = make_world(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    (work / "link_settings.json").write_text(json.dumps(settings))
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "pack", smoke=True, vocab_path=w["vocab"])
    man = json.loads((Path(r["pack"]) / "pack.json").read_text())
    assert man["variant_links"]["sha256"] == "ab" * 32 and man["variant_links"]["parts_used"]["include"][-1] == "val"
    # without variants the manifest has no such key
    (work / "link_settings.json").write_text(json.dumps(ms.link_settings(True)))
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "pack2", smoke=True, vocab_path=w["vocab"])
    assert "variant_links" not in json.loads((Path(r["pack"]) / "pack.json").read_text())


# ------------------------------------------------------------------------------------------------ driver

def test_driver_variant_links_setting(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import _calls, _run_driver

    for k in ("VARIANT_LINKS", "VARIANT_ACCEPT_NOISY", "VARIANT_ALLOW_SPLIT_CHANGE"):
        monkeypatch.delenv(k, raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [c for c in _calls(tmp_path) if "link-sources" in c]
    assert calls and not any("--variant-links" in c for c in calls)
    assert "variant_links=none" in r.stdout and "variant_links=none" in next(base.rglob("link.done")).read_text()
    work = base / "research"
    (work / "variant_links.json").write_text("{}")
    sha = va.file_sha256(work / "variant_links.json")
    # --only link with another variant state than the folder's frozen one: refused (use --from link)
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert r.returncode == 2 and "--from link" in r.stderr
    # the folder was linked with this file: --only link may run again
    (work / "link_settings.json").write_text(json.dumps(
        {"variants": {"isl500": {}}, "variant_links": {"file": "x", "sha256": sha, "parts_used": {"isl500": "train"}}},
        indent=1))
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "variant_links=sha256:" in next(base.rglob("link.done")).read_text()
    monkeypatch.setenv("VARIANT_ACCEPT_NOISY", "1")
    (tmp_path / "fake_py.log").unlink(missing_ok=True)
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert any("--accept-noisy" in c for c in _calls(tmp_path) if "link-sources" in c)
    monkeypatch.delenv("VARIANT_ACCEPT_NOISY")
    monkeypatch.setenv("VARIANT_LINKS", "none")
    (work / "link_settings.json").unlink()
    (tmp_path / "fake_py.log").unlink(missing_ok=True)
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert any("--variant-links none" in c for c in _calls(tmp_path)) and "variant_links=off" in r.stdout
    monkeypatch.setenv("VARIANT_LINKS", "maybe")
    bad = _run_driver(tmp_path, "--only", "link", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "VARIANT_LINKS" in bad.stderr
    monkeypatch.delenv("VARIANT_LINKS")
    (work / "link_settings.json").write_text(json.dumps({"variants": {"isl500": {}}}))
    dry = _run_driver(tmp_path, "--only", "fusion", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert "variant_links=(link step not run; folder: on)" in dry.stdout
    assert "variant links: VARIANT_LINKS=auto" in dry.stdout


def test_driver_never_seeds_the_audit_file_into_a_scratch_run(tmp_path, monkeypatch):
    """Review of b03eba0 (critic 7): a scratch run seeds link's outputs (the frozen link_settings.json) but never the
    real folder's variant_links.json, so a later scratch --from link links without variants."""
    from tests.isl.test_phase2_guards import _old_run, _run_driver

    monkeypatch.delenv("VARIANT_LINKS", raising=False)
    isl, seed, scratch = tmp_path / "isl", tmp_path / "seed", tmp_path / "scratch"
    real = seed / "research"
    _old_run(real, tmp_path / "unused_stamps")
    (real / "link_settings.json").write_text(json.dumps({"variants": {"isl500": {"isl.hot": "isl.hot~isl500"}}}))
    (real / "variant_links.json").write_text("{}")
    r = _run_driver(tmp_path, "--from", "bench", work_base=scratch, isl_root=isl, seed_base=seed, profile="research")
    assert r.returncode == 0, r.stdout + r.stderr
    work = scratch / "research"
    assert (work / "link_settings.json").exists() and not (work / "variant_links.json").exists()
    dry = _run_driver(tmp_path, "--from", "link", "--dry-run", work_base=scratch, isl_root=isl, seed_base=seed,
                      profile="research")
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert "variant_links=none" in dry.stdout and "would run link" in dry.stdout


# ------------------------------------------------------------------------------------------------ evaluate --all
from tests.isl.test_eval_runners import phase2  # noqa: E402,F401  (the toy phase-2 world: module fixture)


def test_evaluate_all_reports_word_labels_with_variant_links(phase2, tmp_path):
    """Review of b03eba0: with frozen variant links, E1b / E1c / E3 get ``word_labels`` (a variant counted as its
    word) next to strict and alias labels, per source too, the like-for-like common clips get a word-label row, and
    the report names the audit file. Without variants nothing new is written."""
    import dataclasses
    import shutil

    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training.common import ISL_FEATURES

    w = phase2["world"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work)
    kw = dict(profile="research", smoke=True, vocab_path=w["vocab"], paths=w["paths"], e4_ks=(1,), latency_n=1)
    plain = evaluate_all(work, phase2["data"], ISL_FEATURES, out_dirs=[tmp_path / "ref"], **kw)
    assert "word_labels" not in plain["protocols"]["E1b"] and "variant_links" not in plain["aliases"]
    # a toy variant: isl.book stands for the ISL500 form of isl.apple
    link = {**ms.link_settings(False, {"isl500": {"isl.apple": "isl.book"}},
                               {"file": "v.json", "sha256": "cd" * 32, "n_relinks": 1,
                                "parts_used": {"isl500": "train", "include": ["train"]}}),
            "alias_of": {"isl.book": "isl.apple"}, "variant_relinked": 3}
    data = dataclasses.replace(phase2["data"], link=link)
    doc = evaluate_all(work, data, ISL_FEATURES, out_dirs=[], compare_to=tmp_path / "ref" / "report.json", **kw)
    assert doc["aliases"]["variant_links"]["sha256"] == "cd" * 32 and "warnings" not in doc["aliases"]["variant_links"]
    for k in ("E1b", "E3"):
        p = doc["protocols"][k]
        r = p["word_labels"]
        assert r["top1"] >= r["top1_strict"] - 1e-12 and r["top1_by_source"]
        assert p["metrics"]["top1"] == pytest.approx(r["top1_strict"])  # metrics stay strict
    lf = doc["protocols"]["E1b"]["like_for_like"]
    assert lf["common_clips_word_labels"]["n"] == lf["common_clips"]["n"]
    assert lf["common_clips_word_labels"]["top1"] >= lf["common_clips"]["top1"] - 1e-12
    md = (work / "eval" / "report.md").read_text()
    assert "Variant links" in md and "E1b word labels" in md
