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
    # which clips and classes E1b measured (docs 15.27): a later report can compare on the same clips
    e1 = p["E1b"]
    assert len(e1["clip_ids"]) == e1["n"] == len(e1["per_clip"]["hit1"]) and e1["split"] == data.split["name"]
    assert set(e1["class_ids"]) >= {c.concept_id for c in data.include_part("test")} and "class_ids" not in p["E1c"]
    assert set(p["E3"]["policies"]) == set(POLICIES)
    # E2: only fold 0 has an encoder; queries = val/test INCLUDE clips of the held-out concept (never pre-trained on)
    assert p["E2"]["run"] and len(p["E2"]["folds"]) == 1 and p["E2"]["n"] == 2
    f0 = p["E2"]["folds"][0]
    assert f0["full"]["metrics"]["not_sure_rate"] is not None and "seen_class" in f0
    # E4 headline = INCLUDE test signers only (one take per word in this world: not run); the pooled population
    # with the ISL500-like signers runs (docs 15.27)
    assert "INCLUDE" in p["E4"]["population"] and p["E4"]["run"] is False and "repeated takes" in p["E4"]["reason"]
    e4 = p["E4"]["all_sources"]
    assert e4["run"] and e4["by_k"]["k1"]["n"] > 0
    assert e4["by_k"]["k1"]["user_sign_false_accept_rate"] is not None
    assert e4["by_k"]["k1"]["user_sign_false_accept_rate_threshold_rule"] is not None
    assert "FALSE ACCEPT" in (out / "report.md").read_text()
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


def test_fit_fusion_evidence_feeds_the_my_signs_choice(phase2):
    """docs 15.23: fit-fusion's own (compact) evidence fills the E4 cache, so choose_user_mode computes nothing again
    and reports exactly what a fresh evidence pass gives."""
    from app.isl.models.calibration import weight_grid
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.fit_fusion import choose_user_mode, collect_evidence
    from app.isl.training.pipeline import load_recognizer

    w, data, work = phase2["world"], phase2["data"], phase2["work"]
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="research")
    clips = data.extra_part("train")[:60]
    cache: dict = {}
    taus = weight_grid(True, True, rec.fusion)["tau_r"] + [rec.fusion.tau_r]
    inputs, _, used = collect_evidence(rec, clips, compact={"taus": taus, "top_n": rec.fusion.top_n}, e4_cache=cache)
    assert set(cache) == {c.clip_id for c in used} and all(i.compact for i in inputs)
    c0 = used[0]
    from app.isl.evaluation.runners import _features

    assert np.allclose(cache[c0.clip_id][1], rec.encoder.embed(_features(c0, ISL_FEATURES))[0], atol=1e-6)
    calls = {"n": 0}
    real = rec.evidence

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    rec.evidence = counted
    p1, r1 = choose_user_mode(rec, clips, rec.fusion, ks=(1,), cache=cache)
    assert calls["n"] == 0
    p2, r2 = choose_user_mode(rec, clips, rec.fusion, ks=(1,))
    assert calls["n"] > 0 and r1 == r2 and p1 == p2


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


def _repeat_takes(root: Path, users: int = 3, takes: int = 5) -> list[Clip]:
    """Held-out signers with ``takes`` repeated takes of every CORE word (the app enrols 3 to 5)."""
    d = root / "e4_takes"
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for ci, w in enumerate(CORE):
        for u in range(users):
            for t in range(takes):
                raw = make_sequence(ci, signer=300 + u, n_frames=40, fps=12.0, n_classes=6,
                                    rng=np.random.default_rng(7000 + 100 * ci + 10 * u + t))
                p = d / f"{w}_{u}_{t}.npz"
                raw.save(p)
                out.append(Clip(f"e4/user{u}/{w}_{t}", "isl500", w, f"isl.{w}", None, None, p, group=f"e4-user{u}",
                                quality={"n_frames": 40}))
    return out


@pytest.mark.parametrize("gated", [True, False])
def test_e4_matches_the_recognizer_clip_for_clip(phase2, tmp_path, gated):
    """QA plan 0.5 (docs 15.27): E4 takes the app's decisions. Every E4 test take, enrolled into a real MySignsStore
    with the same takes, gets the same decision and the same top word from ``IslRecognizer.recognize`` (crop
    agreement on the override, confirm-only demotion, a user sign on top without the override is a suggestion)."""
    from app.isl.evaluation.runners import e4_my_signs
    from app.isl.models.fusion import USER_PREFIX
    from app.isl.mysigns.store import MySign, MySignsStore
    from app.isl.training.common import ISL_FEATURES, engine_crops
    from app.isl.training.pipeline import load_recognizer

    w, work = phase2["world"], phase2["work"]
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="research")
    # a user sign wins the override whenever it beats its theta and agrees across crops (the toy dictionary is
    # otherwise too close to the held-out signers' takes for the user-sign rules to show up)
    rec.fusion = rec.fusion.with_(user_margin=-1.0, user_bonus=3.0)
    if not gated:  # in-code parameters: no certification gate, so the override can really accept
        rec.fusion = rec.fusion.with_(certification=None)
    clips = _repeat_takes(tmp_path)
    by_id = {c.clip_id: c for c in clips}
    trace: list = []
    res = e4_my_signs(rec, clips, ks=(3,), trace=trace)
    assert res["run"] and res["ks"] == [3] and trace and res["by_k"]["k3"]["n"] > 0
    stores: dict = {}
    seen = {"decisions": set(), "user_top": 0}
    for i, t in enumerate(trace):
        key = (t["k"], t["group"])
        if key not in stores:  # one store per (k, signer): the words that signer enrolled, with those takes
            st = MySignsStore(tmp_path / f"store{i}", "isl")
            names = {}
            for word, ids in sorted(t["enrolment"].items()):
                s = st.save_sign(MySign(sign_id=f"00000000-0000-0000-0000-{len(names):012d}", pack="isl",
                                        concept_id=None, text_en=word), [by_id[x].load() for x in ids])
                names[f"{USER_PREFIX}{s.sign_id}"] = f"{USER_PREFIX}{word}"
            stores[key] = (st, names)
        st, names = stores[key]
        rec.my_signs = st
        r = rec.recognize(engine_crops(by_id[t["clip_id"]].load()), policy="balanced")
        top = r.candidates[0].concept_id if r.candidates else ""
        assert (r.decision, names.get(top, top)) == (t["decision"], t["top"]), t
        seen["decisions"].add(t["decision"])
        seen["user_top"] += t["top"].startswith(USER_PREFIX)
    rec.my_signs = None
    assert seen["user_top"] > 0  # the user-sign rules were exercised
    if gated:  # the phase-2 world's fusion.json is a pack record: confirm-only policies never accept
        assert rec.fusion.gated and (rec.fusion.policy_rule("balanced") == "certified" or "accept" not in seen["decisions"])
        k3 = res["by_k"]["k3"]
        assert k3["user_sign_false_accept_rate_threshold_rule"] >= k3["user_sign_false_accept_rate"]
    else:
        assert "accept" in seen["decisions"]


def test_user_sign_decision_crop_agreement_and_demotions():
    """The rules E4 used to skip (QA plan 0.5): the override needs the user sign to be the best one in >= 2 of 3
    crops; a user sign on top of the fused list without the override is only a suggestion; a confirm-only policy
    turns the override into a suggestion."""
    from app.isl.models.fusion import USER_PREFIX
    from app.isl.recognizer import user_sign_decision

    ids = [f"{USER_PREFIX}a", f"{USER_PREFIX}b"]
    ret = np.full((3, 2), 0.1, np.float32)
    p = FusionParams(accept={"strict": 0.3, "balanced": 0.3, "lenient": 0.3}, margin=0.0, user_margin=0.05,
                     user_bonus=5.0)

    def run(user, params=p, theta=0.7):
        inp = FusionInputs(ret=ret, ret_ids=["isl.x", "isl.y"], user=np.asarray(user, np.float32), user_ids=ids)
        r = fuse(inp, params)
        return user_sign_decision(r, inp, 3, params, "balanced", lambda k: theta)[:3]

    # a: best by mean (0.863) but the best user sign in only 1 of 3 crops -> no override (the old E4 accepted it)
    dec, why, keys = run([[0.99, 0.5], [0.80, 0.85], [0.80, 0.85]])
    assert keys[0] == ids[0] and dec != "accept" and why != "my_signs"
    # every crop agrees, the fused rule accepts a, but a is below its own theta: no override -> only a suggestion
    dec, why, keys = run([[0.99, 0.5], [0.90, 0.5], [0.90, 0.5]], theta=0.999)
    assert keys[0] == ids[0] and (dec, why) == ("suggest", "low_confidence")
    # a in every crop -> override
    dec, why, keys = run([[0.99, 0.5], [0.90, 0.85], [0.90, 0.85]])
    assert (dec, why, keys[0]) == ("accept", "my_signs", ids[0])
    # the same under an uncertified pack record: the top "did you mean", never an accept
    dec, why, keys = run([[0.99, 0.5], [0.90, 0.85], [0.90, 0.85]], p.with_(certification={}))
    assert (dec, why, keys[0]) == ("suggest", "confirm_only", ids[0])


def test_measuring_recogniser_sees_the_pack_vocabulary_with_extra_concepts(tmp_path):
    """QA plan 0.5 (docs 15.27): concepts only other sources know (the link step's extra_concepts.jsonl) are appended
    to the pack's vocabulary by build_pack; fit-fusion and evaluation must load the same vocabulary (prior, index)."""
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import extra_concept_rows, load_recognizer

    vocab = tmp_path / "vocabulary.jsonl"
    vocab.write_text("".join(json.dumps({"concept_id": c, "headword": c, "kind": "word", "in_classifier": True}) + "\n"
                             for c in ("isl.a", "isl.b")))
    work = tmp_path / "work"
    work.mkdir()
    rec = load_recognizer(work, ISL_FEATURES, vocab)  # no extra_concepts.jsonl: unchanged
    assert len(rec.vocab) == 2
    (work / "extra_concepts.jsonl").write_text(
        json.dumps({"concept_id": "isl.b", "headword": "b", "kind": "word"}) + "\n"  # already known: skipped
        + "not json\n"
        + json.dumps({"concept_id": "isl.new-1", "headword": "new(1)", "kind": "word", "in_classifier": False}) + "\n")
    assert [r["concept_id"] for r in extra_concept_rows(work, {"isl.a", "isl.b"})] == ["isl.new-1"]
    rec = load_recognizer(work, ISL_FEATURES, vocab)
    assert len(rec.vocab) == 3 and rec.vocab.index_of("isl.new-1") == 2
    assert rec.prior["isl.new-1"] == 0.5  # was missing (0) while measuring, 0.5 in the app
    rec = load_recognizer(work, ISL_FEATURES, vocab)
    assert len(rec.vocab) == 3  # nothing written to disk: loading twice does not append twice
    assert len(vocab.read_text().splitlines()) == 2


def test_e1b_like_for_like_against_a_reference_report(phase2, tmp_path):
    """QA plan 0.5 / critic 2 (docs 15.27): E1b of two packs differ in classes and clips (research 852 clips / 168
    classes vs commercial-safe 792 / 160). With a reference report, E1b also reports top-1 / top-5 masked to the
    reference's classes and on the clips both evaluated, with the reference's own numbers on those clips."""
    import shutil

    from app.isl.evaluation.runners import _like_for_like_lines, e1_closed_set, load_e1_reference
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import load_recognizer

    w, data, work = phase2["world"], phase2["data"], phase2["work"]
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="research")
    test = data.include_part("test")
    split = data.split["name"]
    own = e1_closed_set(rec, test, "E1b", split, "primary")
    keep_cls = sorted(own["class_ids"])[:2]
    keep_clips = own["clip_ids"][1:]  # the reference evaluated all but the first clip
    ref_hits = "".join("1" if i % 2 else "0" for i in range(len(keep_clips)))
    ref_doc = {"profile": "commercial-safe", "leakage_guards": {"split": split},
               "protocols": {"E1b": {"n": len(keep_clips), "split": split, "metrics": {"top1": 0.5, "top5": 0.9},
                                     "clip_ids": keep_clips, "class_ids": keep_cls,
                                     "per_clip": {"hit1": ref_hits, "hit5": "1" * len(keep_clips)}}}}
    rp = tmp_path / "ref" / "eval" / "report.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(ref_doc))
    r = e1_closed_set(rec, test, "E1b", split, "primary", reference=load_e1_reference(rp))
    lf = r["like_for_like"]
    assert lf["run"] and lf["reference"]["profile"] == "commercial-safe"
    truth = {c.clip_id: c.concept_id for c in test}
    in_cls = [c for c in own["clip_ids"] if truth[c] in keep_cls]
    assert lf["reference_classes"]["n"] == len(in_cls) and lf["reference_classes"]["n_shared_classes"] == 2
    own_hit = dict(zip(own["clip_ids"], own["per_clip"]["hit1"]))
    # masking can only help: a clip right among all classes is right among the reference's classes
    assert lf["reference_classes"]["top1"] >= np.mean([own_hit[c] == "1" for c in in_cls])
    cc = lf["common_clips"]
    assert cc["n"] == len(keep_clips) and abs(cc["reference_top1"] - ref_hits.count("1") / len(keep_clips)) < 1e-9
    assert abs(cc["top1"] - np.mean([own_hit[c] == "1" for c in keep_clips])) < 1e-9
    assert cc["discordant_top1"]["only_this"] + cc["discordant_top1"]["only_reference"] <= cc["n"]
    both = lf["reference_classes_common_clips"]
    assert both["n"] == len([c for c in keep_clips if truth[c] in keep_cls]) and "reference_top1" in both
    assert any("like for like" in x for x in _like_for_like_lines(lf))
    # another split: no comparison
    ref_doc["protocols"]["E1b"]["split"] = "other-split"
    rp.write_text(json.dumps(ref_doc))
    lf = e1_closed_set(rec, test, "E1b", split, "primary", reference=load_e1_reference(rp))["like_for_like"]
    assert lf["run"] is False and "other-split" in lf["reason"]
    # a report written before docs 15.27 (no clip / class ids): class ids from the classifier next to it
    old = {"profile": "commercial-safe", "leakage_guards": {"split": split},
           "protocols": {"E1b": {"n": 5, "metrics": {"top1": 0.7}}}}
    rp.write_text(json.dumps(old))
    shutil.copy2(work / "classifier.pt", rp.parent.parent / "classifier.pt")
    ref = load_e1_reference(rp)
    assert ref["class_ids"] == own["class_ids"] and any("no clip ids" in n for n in ref["notes"])
    lf = e1_closed_set(rec, test, "E1b", split, "primary", reference=ref)["like_for_like"]
    assert lf["run"] and lf["reference_classes"]["top1"] == own["metrics"]["top1"]  # same classes: nothing masked
    assert lf["common_clips"]["run"] is False
