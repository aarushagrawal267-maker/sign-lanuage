"""Variant forms instead of forced merges (QA plan 1.3, critics 7 and 13, docs/ISL_DESIGN.md 15.36).

* the audit relinks an ISL500 word only when the reference classifier disagrees AND the dictionary (or the reverse
  direction) says the ISL500 clips are another sign form; signer shift (the dictionary agrees) is never relinked;
* the audit file is scoped to the profile's work folder; the link step reads it only from there (or a file named
  explicitly), freezes the relinks with the file's SHA-256 into link_settings.json, and later steps use that copy;
* a variant concept is an alias of its word: alias-scored E1b counts a variant prediction as correct, strict does not.
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


P1 = va.AuditParams(top_k=1, dict_rank_floor=1, dict_include_max_rank=2)


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
    assert s["dictionary_false_alarms_on_agreeing_words"] == {"n": 0, "of": 1, "rate": 0.0}
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
    assert va.dictionary_verdict(3, 1, p)[0] == va.NO_DIFFERENCE  # within the floor
    assert va.dictionary_verdict(36, 12, p)[0] == va.NO_DIFFERENCE  # within 3 x INCLUDE's rank
    assert va.dictionary_verdict(37, 12, p)[0] == va.DIFFERENT and va.dictionary_verdict(200, 4, p)[0] == va.DIFFERENT
    assert va.dictionary_verdict(500, 60, p)[0] == va.INCONCLUSIVE  # INCLUDE does not retrieve it either
    assert va.dictionary_verdict(None, None, p)[0] == va.INCONCLUSIVE


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
    meta = va.write_variant_links(out, doc, profile="research", work=tmp_path / "research")
    rel, m = va.load_variant_links(out, "research")
    assert rel == {"isl500": {"isl.hot": "isl.hot~isl500"}} and m["sha256"] == meta["sha256"] and m["n_relinks"] == 1
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
    doc = {"kind": va.VARIANT_KIND, "audit_version": va.AUDIT_VERSION, "relinks": {"isl500": {"isl.cat": "isl.cat~isl500"}}}
    va.write_variant_links(work / va.VARIANT_LINKS, doc, profile="research", work=work)
    sha = va.file_sha256(work / va.VARIANT_LINKS)
    assert cli.main(root + ["link-sources", *common]) == 0
    s = json.loads((work / "link_settings.json").read_text())
    assert s["variants"] == {"isl500": {"isl.cat": "isl.cat~isl500"}} and s["variant_links"]["sha256"] == sha
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
    # another profile's file is refused, also when given explicitly
    other = base / "x.json"
    va.write_variant_links(other, doc, profile="commercial-safe", work=base / "commercial-safe")
    assert cli.main(root + ["link-sources", *common, "--variant-links", str(other)]) == 2
    assert cli.main(root + ["link-sources", *common, "--variant-links", str(base / "missing.json")]) == 2


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
    out = world["base"] / "research" / va.VARIANT_LINKS
    doc = json.loads(out.read_text())
    assert doc["kind"] == va.VARIANT_KIND and doc["profile"] == "research" and doc["part"] == "train"
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
    assert cli.main([*args, "--out", str(Path(P.derived) / "variant_links.json")]) == 2
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


# ------------------------------------------------------------------------------------------------ driver

def test_driver_variant_links_setting(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import _calls, _run_driver

    monkeypatch.delenv("VARIANT_LINKS", raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [c for c in _calls(tmp_path) if "link-sources" in c]
    assert calls and not any("--variant-links" in c for c in calls)
    assert "variant_links=none" in r.stdout and "variant_links=none" in next(base.rglob("link.done")).read_text()
    work = base / "research"
    (work / "variant_links.json").write_text("{}")
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert "variant_links=sha256:" in next(base.rglob("link.done")).read_text()
    monkeypatch.setenv("VARIANT_LINKS", "none")
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
