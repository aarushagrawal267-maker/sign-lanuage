"""Phase-2 evaluation and multi-source data (no network, synthetic data only): fusion policies are always
complete, gloss linking / dedupe / licence guards, signer-held-out splits, and ``evaluate --all`` end to end on a
tiny world with an extra signer-labelled source (E1b, E1c, E2 held-out concepts with its leakage guard, E3, E4)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.dataset.synthetic import make_sequence
from app.isl.data.sources import Clip
from app.isl.models.calibration import fit_thresholds
from app.isl.models.fusion import POLICIES, FusionInputs, FusionParams, decide, fuse, normalize_accept
from app.isl.training import multisource as ms
from tests.isl.isl_world import CORE, make_world

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=2, lr=3e-3)


# --------------------------------------------------------------------------------------------- fusion policies

def test_fusion_params_always_have_all_three_policies():
    for acc in ({}, {"balanced": 0.8}, {"strict": 0.9}, {"lenient": 0.4, "bogus": 0.1}, None):
        p = FusionParams(accept=acc) if acc is not None else FusionParams.from_json({"accept": None})
        assert set(p.accept) == set(POLICIES)
        assert p.accept["strict"] >= p.accept["balanced"] >= p.accept["lenient"]
        for pol in POLICIES:
            decide(np.array([0.9, 0.05]), 3, 3, p, pol)  # never raises for a known policy
    with pytest.raises(ValueError):
        decide(np.array([0.9]), 3, 3, FusionParams(), "nonsense")
    assert normalize_accept({"balanced": 0.8}) == {"strict": 0.8, "balanced": 0.8, "lenient": 0.8}
    # a fusion.json written with only one policy loads complete
    p = FusionParams.from_json({"accept": {"balanced": 0.6}, "fitted": True})
    assert set(p.accept) == set(POLICIES) and p.with_(margin=0.2).accept == p.accept


def test_fit_thresholds_writes_every_policy_even_with_partial_targets():
    rng = np.random.default_rng(0)
    inputs, truths = [], []
    for i in range(60):
        s = rng.normal(0, 0.1, (3, 5)).astype(np.float32)
        t = i % 5
        s[:, t] += 0.6 if i % 4 else 0.0
        inputs.append(FusionInputs(ret=s, ret_ids=[f"c{j}" for j in range(5)]))
        truths.append(f"c{t}")
    res = [fuse(x, FusionParams()) for x in inputs]
    p, rep = fit_thresholds(res, truths, FusionParams(), targets={"balanced": 0.5})
    assert set(p.accept) == set(POLICIES) and set(p.achieved) == set(POLICIES)
    assert all(f"{pol}_target_reached" in rep for pol in POLICIES)


# --------------------------------------------------------------------------------------------- linking, dedupe

def _entries():
    return [{"concept_id": "isl.big-large", "headword": "big large", "synonyms": ["big", "large"], "kind": "synonym_list"},
            {"concept_id": "isl.range", "headword": "range", "kind": "word"},
            {"concept_id": "isl.range~math", "headword": "range", "kind": "word"},
            {"concept_id": "isl.tea", "headword": "tea", "kind": "word"}]


def test_gloss_linker_exact_synonym_new_and_expand():
    lk = ms.GlossLinker(_entries())
    assert lk.link("Range") == ("isl.range", "exact")  # plain concept preferred over the domain one
    assert lk.link("LARGE") == ("isl.big-large", "synonym")
    cid, how = lk.link("Auto Rickshaw", "wslp")
    assert how == "new" and cid == "isl.auto-rickshaw"
    assert lk.link("auto-rickshaw", "isl500")[0] == cid  # same new concept from another source
    assert lk.extra_concepts()[0]["sources"] == ["wslp", "isl500"]
    assert {"isl.range", "isl.range~math"} <= lk.expand({"isl.range"})
    assert "isl.big-large" in lk.expand({"isl.big-large"}) and "isl.tea" not in lk.expand({"isl.range"})


def _clip(cid, src="wslp", group=None, path=Path("/nonexistent"), concept=None, quality=None):
    return Clip(cid, src, "w", concept, None, None, path, group=group, quality=quality or {})


def test_dedup_index_and_drop_leaking(tmp_path):
    a = tmp_path / "a.npz"
    a.write_bytes(b"same")
    b = tmp_path / "b.npz"
    b.write_bytes(b"same")
    c = tmp_path / "c.npz"
    c.write_bytes(b"other")
    dup = tmp_path / "dups.jsonl"
    dup.write_text(json.dumps({"clips": ["x", "y"]}) + "\n")
    clips = [_clip("a", path=a), _clip("b", "cislr", path=b), _clip("c", path=c),
             _clip("v1", quality={"sha256": "h"}), _clip("v2", "cislr", quality={"sha256": "h"}), _clip("x"), _clip("y")]
    d = ms.DedupIndex.build(clips, dup)
    assert d.group("a") == d.group("b") != d.group("c")
    assert d.group("v1") == d.group("v2") and d.group("x") == d.group("y")
    keep, n = ms.drop_leaking([clips[0], clips[2], clips[5]], [clips[1], clips[6]], d)
    assert [k.clip_id for k in keep] == ["c"] and n == 2


def test_signer_split_holds_out_whole_signers():
    clips = [_clip(f"isl500/u{u}/{i}", "isl500", group=f"isl500-user{u:03d}") for u in range(10) for i in range(4)]
    clips += [_clip(f"wslp/{i}", "wslp", group=None) for i in range(5)]
    a = ms.signer_split(clips)
    by = {}
    for c in clips:
        by.setdefault(c.group, set()).add(a[c.clip_id])
    assert all(len(v) == 1 for g, v in by.items() if g)  # a signer is never split across parts
    parts = [a[c.clip_id] for c in clips if c.source == "isl500"]
    assert {"train", "val", "test"} <= set(parts)
    assert all(a[c.clip_id] == "train" for c in clips if c.source == "wslp")


class _FakeAdapter:
    def __init__(self, name, commercial, clips=()):
        self.name, self.commercial, self.licence = name, commercial, "test"
        self._clips = list(clips)

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        yield from self._clips


def test_licence_guard_refuses_research_sources_in_commercial_profile(tmp_path):
    from app.isl.paths import isl_paths

    P = isl_paths(tmp_path)
    ms.register_adapter("isl500", lambda p, v: _FakeAdapter("isl500", False))
    ms.register_adapter("mendeley_agri", lambda p, v: _FakeAdapter("mendeley_agri", False))
    try:
        with pytest.raises(ms.LicenceError):
            ms.open_adapter("isl500", P, profile="commercial-safe")  # not in the allow-list
        with pytest.raises(ms.LicenceError):
            ms.open_adapter("mendeley_agri", P, profile="commercial-safe")  # allowed name, but adapter says NC
        assert ms.open_adapter("isl500", P, profile="research") is not None
    finally:
        ms.register_adapter("isl500", None)
        ms.register_adapter("mendeley_agri", None)
    assert "isl500" not in ms.WORD_SOURCES["commercial-safe"] and "glosl" not in ms.PRETRAIN_SOURCES["commercial-safe"]


# --------------------------------------------------------------------------------------------- end to end

def _extra_world(root: Path, users: int = 8, takes: int = 3) -> list[Clip]:
    """An ISL500-like source: CORE words x users x takes, real signer ids, glosses only (no concept ids)."""
    d = root / "isl500"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for ci, w in enumerate(CORE):
        for u in range(users):
            for t in range(takes):
                raw = make_sequence(ci, signer=200 + u, n_frames=40, fps=12.0, n_classes=6,
                                    rng=np.random.default_rng(1000 * ci + 10 * u + t))
                p = d / f"{w}_{u}_{t}.npz"
                raw.save(p)
                out.append(Clip(f"isl500/user{u:03d}/{w}_{t}", "isl500", w.capitalize(), None, None, None, p,
                                group=f"isl500-user{u:03d}", quality={"n_frames": 40}))
    return out


@pytest.fixture(scope="module")
def phase2(tmp_path_factory):
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.common import ISL_FEATURES, load_isl_data
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import load_recognizer
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex
    from app.training.trainer import TrainConfig

    root = tmp_path_factory.mktemp("phase2")
    w = make_world(root)
    extra = _extra_world(root)
    ms.register_adapter("isl500", lambda p, v: _FakeAdapter("isl500", False, extra))
    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(w["paths"], vocab, "research", w["split"])
    rep = ms.attach_all_sources(data, w["paths"], vocab, "research", hash_files=True)
    work = root / "work"
    work.mkdir()
    cfg = TrainConfig(**TINY)
    train_classifier(data, work / "classifier.pt", ISL_FEATURES, cfg, smoke=True, threads=1, use_extra=True)
    train_encoder(data, work / "encoder.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2, smoke=True, threads=1,
                  batches_per_epoch=3)
    build_index(EncoderPredictor(work / "encoder.pt"), data.islrtc, data.include_part("train"), work / "index",
                "research", extra=data.extra_part("train"), split=data.split["name"])
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="research")
    params, _ = fit_fusion(rec, data.labelled_part("val"), "research")
    params.save(work / "fusion.json")
    # held-out-concept protocol: fold 0 holds out "apple"
    P = w["paths"]
    folds = [{"fold": 0, "held_out_concepts": ["isl.apple"]}, {"fold": 1, "held_out_concepts": ["isl.book"]}]
    (P.splits / "oneshot_concepts.json").write_text(json.dumps(
        {"concepts": [{"concept_id": f"isl.{c}", "usable": True} for c in CORE], "protocol": {"k": 2, "folds": folds}}))
    (work / "oneshot" / "fold0").mkdir(parents=True)
    held = ms.GlossLinker.from_vocab(vocab).expand({"isl.apple"})
    train_encoder(data, work / "oneshot" / "fold0" / "encoder.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2,
                  smoke=True, threads=1, batches_per_epoch=3, exclude_concepts=held, version="isl-enc-1-oneshot0")
    yield {"world": w, "data": data, "work": work, "rep": rep, "vocab": vocab}
    ms.register_adapter("isl500", None)


def test_attach_all_sources_links_and_splits(phase2):
    data, rep = phase2["data"], phase2["rep"]
    assert rep.links["exact"] == len(CORE) * 8 * 3  # every ISL500 gloss linked to the existing concept
    assert {c.concept_id for c in data.extra} == {f"isl.{w}" for w in CORE}
    parts = rep.parts["isl500"]
    assert parts["test"] and parts["val"] and parts["train"]
    held_groups = {c.group for c in data.extra_part("test")}
    assert not held_groups & {c.group for c in data.extra_part("train")}
    assert data.summary()["extra_sources"]["isl500"] == parts


def test_encoder_exclusion_removes_every_source_of_held_out_concepts(phase2):
    from app.isl.training.train_encoder import encoder_training_clips

    clips = encoder_training_clips(phase2["data"], {"isl.apple"})
    assert clips and all(c.concept_id != "isl.apple" for c in clips)
    assert {c.source for c in clips} >= {"include", "islrtc", "isl500"}


def test_evaluate_all_writes_report_with_rates_and_groups(phase2, tmp_path):
    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training.common import ISL_FEATURES

    w, data, work = phase2["world"], phase2["data"], phase2["work"]
    out = tmp_path / "pack" / "eval"
    doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[out], e4_ks=(1, 2), latency_n=2)
    assert (out / "report.json").exists() and (out / "report.md").exists() and (work / "eval" / "report.json").exists()
    p = doc["protocols"]
    assert "NOT the accuracy" in doc["label"]
    for k in ("E1b", "E1c", "E3"):
        assert p[k]["run"], (k, p[k])
        m = p[k]["metrics"]
        assert m["top1"] is not None and m["not_sure_rate"] is not None and m["wrong_accept_rate"] is not None
    assert p["E1c"]["n_groups"] >= 1 and "signer" in p["E1c"]["label"]
    assert set(p["E3"]["policies"]) == set(POLICIES)
    # E2: only fold 0 has an encoder; queries = val/test INCLUDE clips of the held-out concept (never pre-trained on)
    assert p["E2"]["run"] and len(p["E2"]["folds"]) == 1 and p["E2"]["n"] == 2
    f0 = p["E2"]["folds"][0]
    assert f0["full"]["metrics"]["not_sure_rate"] is not None and "seen_class" in f0
    assert p["E4"]["run"] and p["E4"]["by_k"]["k1"]["n"] > 0
    assert p["E4"]["by_k"]["k1"]["user_sign_false_accept_rate"] is not None
    assert p["E1a"]["run"] is False and p["E5"]["run"] is False
    assert "| E2 |" in (out / "report.md").read_text()
    # E2 says that unlabelled pre-training saw the dictionary clips it retrieves against
    assert "WITHOUT labels" in p["E2"]["label"] and "reference clips E2 retrieves against" in p["E2"]["label"]
    # no pre-training in this build: the guard says so instead of claiming True
    assert doc["leakage_guards"]["pretraining_excludes_eval_clips"] is None
    # the index guard covered the other sources too: their centroids list their clips (docs 15.22)
    ig = doc["leakage_guards"]["index"]
    assert ig["training_parts_only"] is True and ig["train_clip_ids_checked"] and ig["eval_clip_ids_checked"] > 0
    from app.isl.models.index import EmbeddingIndex

    rows = EmbeddingIndex.load(work / "index").rows
    multi = [r for r in rows if r["kind"] in ("centroid", "signer_centroid")]
    assert multi and all(len(r["clips"]) == r["n_clips"] for r in multi)
    assert {i for r in rows for i in r.get("clips", [r.get("clip")]) if i} <= (
        {c.clip_id for c in data.islrtc} | {c.clip_id for c in data.include_part("train")}
        | {c.clip_id for c in data.extra_part("train")})
    assert "Leakage guards" in (out / "report.md").read_text()
    # every interval either exists with its group count or has a reason
    for k in ("E1b", "E1c"):
        for m, ci in p[k]["intervals"].items():
            assert ci is not None or p[k]["interval_reasons"][m], (k, m)
            assert p[k]["interval_groups"][m] >= (5 if ci else 0)


def test_e2_refuses_an_encoder_that_saw_held_out_concepts(phase2):
    from app.isl.evaluation.runners import LeakageError, e2_oneshot_fold
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.common import ISL_FEATURES

    data, work = phase2["data"], phase2["work"]
    enc = EncoderPredictor(work / "encoder.pt")  # trained on every concept
    with pytest.raises(LeakageError):
        e2_oneshot_fold(enc, {"fold": 0, "held_out_concepts": ["isl.apple"]}, data, data.linker, ISL_FEATURES,
                        FusionParams(), set(), [])


def test_e1_refuses_a_classifier_from_another_split():
    from app.isl.evaluation.runners import LeakageError, check_classifier_split

    check_classifier_split({"split": "a"}, "a")
    with pytest.raises(LeakageError):
        check_classifier_split({"split": "include-official"}, "include-session-cv5-fold0")


def test_build_pack_refuses_non_commercial_sources_in_commercial_profile(phase2, tmp_path):
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack

    work = tmp_path / "w"
    work.mkdir()
    (work / "sources.json").write_text(json.dumps({"isl500": {"available": True, "licence": "research-only",
                                                              "commercial": False}}))
    with pytest.raises(ValueError, match="commercial-safe"):
        build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack", smoke=True,
                   vocab_path=phase2["world"]["vocab"])
    # research pack lists the extra source and appends concepts only other sources know
    (work / "extra_concepts.jsonl").write_text(json.dumps({"concept_id": "isl.zz-new", "headword": "zz new",
                                                           "display": {"en": "zz new"}, "kind": "word"}) + "\n")
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "pack", smoke=True, vocab_path=phase2["world"]["vocab"])
    man = json.loads((Path(r["pack"]) / "pack.json").read_text())
    assert any(x["source"] == "isl500" for x in man["licences"])
    assert "isl.zz-new" in (Path(r["pack"]) / "vocabulary.jsonl").read_text()


def test_labelled_part_combines_sources(phase2):
    data = phase2["data"]
    val = data.labelled_part("val")
    assert {c.source for c in val} == {"include", "isl500"}
    assert replace(val[0]).concept_id


def test_e1b_cross_validation_pools_held_out_signers(phase2, tmp_path):
    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training.common import ISL_FEATURES, load_isl_data
    from app.isl.training.train_classifier import train_classifier
    from app.training.trainer import TrainConfig

    w, data, work = phase2["world"], phase2["data"], phase2["work"]
    P = w["paths"]
    base = json.loads((P.splits / "test-split.json").read_text())
    # a second fold: signer 0 is the test signer, signer 5 trains
    assign = {}
    for cid, part in base["assign"].items():
        s = int(cid.rsplit("_", 1)[1]) % 10
        assign[cid] = "test" if s == 0 else ("val" if s == 4 else "train")
    (P.splits / "test-split-b.json").write_text(json.dumps({**base, "name": "test-split-b", "assign": assign}))
    d2 = load_isl_data(P, phase2["vocab"], "research", "test-split-b")
    (work / "cv" / "fold1").mkdir(parents=True, exist_ok=True)
    train_classifier(d2, work / "cv" / "fold1" / "classifier.pt", ISL_FEATURES, TrainConfig(**TINY), smoke=True,
                     threads=1, use_extra=False)
    try:
        doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                           paths=P, out_dirs=[], e4_ks=(1,), latency_n=1)
    finally:
        import shutil

        shutil.rmtree(work / "cv")
    cv = doc["protocols"]["E1b"]["cv"]
    assert cv["folds"] == 2 and cv["pooled_n"] == 8 and cv["pooled_groups"] >= 2 and cv["pooled_top1_interval"]
    assert "_hits" not in doc["protocols"]["E1b"]
    assert "E1b cross-validation" in (work / "eval" / "report.md").read_text()


def test_e4_runs_with_both_user_sign_classifier_modes(phase2):
    from app.isl.evaluation.runners import e4_my_signs
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.fit_fusion import choose_user_mode
    from app.isl.training.pipeline import load_recognizer

    w, data, work = phase2["world"], phase2["data"], phase2["work"]
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="research")
    clips = data.extra_part("train")[:60]  # several real signer groups with 3 takes per word
    cache: dict = {}
    out = {m: e4_my_signs(rec, clips, ks=(1,), params=rec.fusion.with_(user_cls_mode=m), cache=cache)
           for m in ("penalty", "neutral")}
    assert out["penalty"]["run"] and out["neutral"]["user_cls_mode"] == "neutral"
    assert out["penalty"]["by_k"]["k1"]["n"] == out["neutral"]["by_k"]["k1"]["n"]  # same tests, shared evidence
    p, rep = choose_user_mode(rec, clips, rec.fusion, ks=(1,))
    assert rep["chosen"] in ("penalty", "neutral") and p.user_cls_mode == rep["chosen"]


def test_e3_and_e4_are_refused_with_a_classifier_from_another_split(phase2, tmp_path):
    """Review fix: E1b refused a classifier trained on another split, but E3 / E4 used the same classifier and
    still reported numbers."""
    import shutil

    import torch

    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training.common import ISL_FEATURES

    w, data = phase2["world"], phase2["data"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work, ignore=shutil.ignore_patterns("eval", "oneshot"))
    ck = torch.load(work / "classifier.pt", map_location="cpu", weights_only=False)
    ck["meta"]["split"] = "some-other-split"
    torch.save(ck, work / "classifier.pt")
    doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[], e4_ks=(1,), latency_n=1)
    p = doc["protocols"]
    for k in ("E1b", "E3", "E4"):
        assert p[k]["run"] is False and p[k]["reason"].startswith("refused:"), (k, p[k])
        assert "some-other-split" in p[k]["leakage"]
    assert doc["leakage_guards"]["classifier_refused"]
    assert "REFUSED" in (work / "eval" / "report.md").read_text()
