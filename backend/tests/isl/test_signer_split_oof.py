"""Signer split v2 and out-of-fold INCLUDE calibration (docs/ISL_DESIGN.md 15.42; QA plan 1.6, critics 8 and 10).

* v2 gives ISL500 9 train / 4 val / 2 test signers with the SAME test signers as v1 (same seed), is opt-in through the
  link settings, enters the resume signatures only when on, and a v2 encoder refuses a warm start whose pre-training
  saw a v2 validation clip.
* ``fit-fusion --oof-calibration`` certifies INCLUDE on the test parts of the other cross-validation folds, each
  scored by its own fold classifier against a leave-fold-out gallery; no clip of the primary test part is ever used.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.sources import Clip
from app.isl.training import multisource as ms
from app.isl.training import oof_calibration as oc

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=2, lr=3e-3)


def _clip(cid, src, group, concept=None):
    return Clip(cid, src, "w", concept, None, None, Path("/nonexistent"), group=group)


def _isl500(n_users=15, per=3):
    return [_clip(f"isl500/u{u}/{i}", "isl500", f"isl500-user{u:03d}") for u in range(1, n_users + 1)
            for i in range(per)]


def _parts(clips, assign):
    out: dict[str, set] = {}
    for c in clips:
        out.setdefault(assign[c.clip_id], set()).add(c.group)
    return out


# ------------------------------------------------------------------------------------------------ signer split v2

def test_v2_keeps_the_test_signers_and_has_four_validation_signers():
    clips = _isl500()
    clips += [_clip(f"mend/{p}/{i}", "mendeley_emergency", f"emergency-p{p:03d}") for p in range(1, 27)
              for i in range(2)]
    v1 = ms.signer_split(clips, seed=0, **ms.signer_split_params("v1"))
    v2 = ms.signer_split(clips, seed=0, **ms.signer_split_params("v2"))
    isl = [c for c in clips if c.source == "isl500"]
    p1, p2 = _parts(isl, v1), _parts(isl, v2)
    assert p1["test"] == p2["test"] == {"isl500-user005", "isl500-user015"}  # the real names (seed 0, verified)
    assert len(p1["val"]) == 2 and len(p1["train"]) == 11
    assert len(p2["val"]) == 4 and len(p2["train"]) == 9
    assert p1["val"] < p2["val"]  # the v1 validation signers stay validation
    moved = p2["val"] - p1["val"]
    assert moved == {"isl500-user009", "isl500-user013"} and moved <= p1["train"]  # critic 10: train -> val
    # every other source is split exactly as before
    assert all(v1[c.clip_id] == v2[c.clip_id] for c in clips if c.source != "isl500")
    # the default arguments are v1 (every caller before 15.42)
    assert ms.signer_split(clips, seed=0) == v1
    with pytest.raises(ValueError):
        ms.signer_split_params("v9")


def test_link_settings_are_unchanged_for_v1_and_record_v2():
    assert ms.link_settings(True) == ms.link_settings(True, signer_split="v1")
    assert "signer_split" not in ms.link_settings(False)
    s = ms.link_settings(False, signer_split="v2")
    assert s["signer_split"] == "v2" and s["signer_split_params"]["val_frac"]["isl500"] == 0.27
    assert ms.signer_split_version(s) == "v2" and ms.signer_split_version({}) == "v1"
    assert ms.signer_split_version(None) == "v1"
    with pytest.raises(ValueError):
        ms.link_settings(False, signer_split="v3")


def test_resume_signature_and_summary_name_the_split_only_when_not_v1():
    from types import SimpleNamespace

    from app.isl.training.common import signer_split_signature

    assert signer_split_signature(SimpleNamespace(link={})) == {}
    assert signer_split_signature(SimpleNamespace(link={"signer_split": "v1"})) == {}
    assert signer_split_signature(SimpleNamespace(link={"signer_split": "v2"})) == {"signer_split": "v2"}


class _Fake:
    def __init__(self, name, clips):
        self.name, self.commercial, self.licence = name, False, "test"
        self._clips = list(clips)

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        return iter(self._clips)


@pytest.fixture
def isl500_world(tmp_path):
    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex
    from tests.isl.isl_world import make_world

    w = make_world(tmp_path)
    extra = [Clip(f"isl500/u{u}/{i}", "isl500", "Apple", None, None, None, Path("/nonexistent"),
                  group=f"isl500-user{u:03d}") for u in range(1, 16) for i in range(2)]
    ms.register_adapter("isl500", lambda p, v: _Fake("isl500", extra))
    vocab = VocabIndex.load(w["vocab"])
    try:
        yield w, vocab, lambda: load_isl_data(w["paths"], vocab, "research", w["split"])
    finally:
        ms.register_adapter("isl500", None)


def test_attach_all_sources_and_the_pretraining_guards_follow_the_link_settings(isl500_world, tmp_path, monkeypatch,
                                                                                   capsys):
    from app.isl import cli
    from app.isl.evaluation.runners import pretraining_guard

    w, vocab, load = isl500_world
    d1, d2 = load(), load()
    ms.attach_all_sources(d1, w["paths"], vocab, "research", hash_files=False, link=ms.link_settings(False))
    ms.attach_all_sources(d2, w["paths"], vocab, "research", hash_files=False,
                          link=ms.link_settings(False, signer_split="v2"))
    d2.link = ms.link_settings(False, signer_split="v2")
    g1 = _parts(d1.extra, d1.extra_assign)
    g2 = _parts(d2.extra, d2.extra_assign)
    assert g1["test"] == g2["test"] and len(g2["val"]) == 4 and len(g2["train"]) == 9
    assert d2.summary()["signer_split"] == "v2" and "signer_split" not in d1.summary()
    # a pre-training of the v1 split saw the two signers v2 moves to validation: the report's guard says so ...
    work = tmp_path / "work"
    work.mkdir()
    pre = sorted(c.clip_id for c in d1.extra if d1.extra_assign[c.clip_id] == "train")
    ms.write_pretrain_list(work, [c for c in d1.extra if c.clip_id in set(pre)])
    (work / "pretrained_encoder.pt").write_bytes(b"x")
    assert pretraining_guard(work, d1)["excludes_eval_clips"] is True
    g = pretraining_guard(work, d2)
    assert g["excludes_eval_clips"] is False and g["overlap"] == 4  # 2 signers x 2 clips
    # ... and train-encoder / train-oneshot of a v2 folder refuse that warm start (v1 is not checked, as before)
    args = cli.build_parser().parse_args(["train-encoder", "--init", "auto"])
    assert cli._pretrain_split_guard("train-encoder", args, work, d1) is True
    assert cli._pretrain_split_guard("train-encoder", args, work, d2) is False
    none = cli.build_parser().parse_args(["train-encoder", "--init", "none"])
    assert cli._pretrain_split_guard("train-encoder", none, work, d2) is True
    # an explicit --init of ANOTHER folder's pretrained_encoder.pt is checked against the list next to it (15.43)
    other = tmp_path / "other"
    other.mkdir()
    (other / "pretrained_encoder.pt").write_bytes(b"x")
    ms.write_pretrain_list(other, [c for c in d1.extra if c.clip_id in set(pre)])
    ext = cli.build_parser().parse_args(["train-encoder", "--init", str(other / "pretrained_encoder.pt")])
    assert cli._pretrain_split_guard("train-encoder", ext, work, d2) is False
    assert cli._pretrain_split_guard("train-encoder", ext, work, d1) is True
    # ... and one that cannot be checked (no list next to it) passes with a warning
    (other / ms.PRETRAIN_LIST).unlink()
    capsys.readouterr()
    assert cli._pretrain_split_guard("train-encoder", ext, work, d2) is True
    assert "not checked" in capsys.readouterr().err
    # a pre-training on the v2 split passes
    ms.write_pretrain_list(work, [c for c in d2.extra if d2.extra_assign[c.clip_id] == "train"])
    assert cli._pretrain_split_guard("train-encoder", args, work, d2) is True


def test_a_v1_variant_audit_is_refused_by_a_v2_link_step(isl500_world):
    from app.isl.training import variant_audit as va

    w, vocab, _ = isl500_world
    t1 = ms.source_train_groups(w["paths"], vocab, "research", "isl500", seed=0)
    t2 = ms.source_train_groups(w["paths"], vocab, "research", "isl500", seed=0, version="v2")
    rec = va.split_record(0, ms.signer_split_params("v1"), t1)
    assert va.check_split(rec, t1)["ok"]
    chk = va.check_split(rec, t2)
    assert not chk["ok"] and chk["moved_groups"] == ["isl500-user009", "isl500-user013"]


def test_cli_link_sources_freezes_v2(isl500_world, tmp_path, monkeypatch):
    from app.isl import cli

    w, _, _ = isl500_world
    base = tmp_path / "art"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    common = ["--profile", "research", "--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1"]
    root = ["--root", str(w["paths"].root)]
    assert cli.main(root + ["link-sources", *common]) == 0
    assert "signer_split" not in json.loads((base / "research" / "link_settings.json").read_text())
    assert cli.main(root + ["link-sources", *common, "--signer-split", "v2"]) == 0
    s = json.loads((base / "research" / "link_settings.json").read_text())
    assert s["signer_split"] == "v2"
    assert json.loads((base / "research" / "multisource.json").read_text())["parts"]["isl500"]["val"] == 8
    # a later step reads it without any flag
    args = cli.build_parser().parse_args(root + ["build-index", *common, "--all-sources"])
    _, data, _, _ = cli._model_ctx(w["paths"], args, record=False)
    assert data.signer_split == "v2" and len(_parts(data.extra, data.extra_assign)["val"]) == 4


def test_driver_passes_the_signer_split_and_oof_flags(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import STEPS_DONE, _calls, _run_driver

    for k in ("SIGNER_SPLIT", "FUSION_OOF", "FUSION_OOF_KEEP_IN_SAMPLE"):
        monkeypatch.delenv(k, raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"

    def calls(profile, step):
        (tmp_path / "fake_py.log").unlink(missing_ok=True)
        r = _run_driver(tmp_path, "--only", step, work_base=base, isl_root=isl, profile=profile)
        assert r.returncode == 0, r.stdout + r.stderr
        return [c for c in _calls(tmp_path) if (" link-sources " if step == "link" else " fit-fusion ") in c], r

    c, r = calls("research", "link")
    assert c and all("--signer-split v2" in x for x in c)
    assert "signer_split=v2" in next(base.rglob("research/**/link.done")).read_text()
    c, _ = calls("commercial-safe", "link")
    assert c and not any("--signer-split" in x for x in c)
    c, _ = calls("research", "fusion")
    assert c and not any("--oof" in x for x in c)  # opt-in
    monkeypatch.setenv("FUSION_OOF", "1")
    c, _ = calls("commercial-safe", "fusion")
    assert c and all("--oof-calibration" in x and "--oof-keep-in-sample" not in x for x in c)
    monkeypatch.setenv("FUSION_OOF_KEEP_IN_SAMPLE", "1")
    c, _ = calls("research", "fusion")
    assert c and all("--oof-calibration --oof-keep-in-sample" in x for x in c)
    monkeypatch.setenv("SIGNER_SPLIT", "v1")
    c, _ = calls("research", "link")
    assert c and not any("--signer-split" in x for x in c)
    for k, v in (("SIGNER_SPLIT", "v3"), ("FUSION_OOF", "yes")):
        monkeypatch.setenv(k, v)
        bad = _run_driver(tmp_path, "--only", "link", "--dry-run", work_base=base, isl_root=isl, profile="research")
        assert bad.returncode == 2 and k in bad.stderr
        monkeypatch.delenv(k)
    # FUSION_OOF=1 passes the folds the cv step trains, and is refused at start when cv is skipped (docs 15.43)
    monkeypatch.setenv("FUSION_OOF", "1")
    c, _ = calls("research", "fusion")
    assert c and all(x.rstrip().endswith("--oof-folds 1 2 3 4") for x in c)
    monkeypatch.setenv("CV_FOLDS", "1 3")
    c, _ = calls("research", "fusion")
    assert c and all(x.rstrip().endswith("--oof-folds 1 3") for x in c)
    monkeypatch.setenv("CV_FOLDS", "")
    bad = _run_driver(tmp_path, "--only", "fusion", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "CV_FOLDS is empty" in bad.stderr
    monkeypatch.setenv("CV_FOLDS", "1;rm")
    bad = _run_driver(tmp_path, "--only", "fusion", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "CV_FOLDS" in bad.stderr
    monkeypatch.delenv("CV_FOLDS")
    # smoke trains cv fold 1 only (dry run: a smoke run's work folder is the real artifacts/isl/phase2-smoke)
    smoke = _run_driver(tmp_path, "--smoke", "--only", "fusion", "--dry-run", work_base=base, isl_root=isl,
                        profile="research")
    assert smoke.returncode == 0 and "--oof-keep-in-sample --oof-folds 1)" in smoke.stdout
    # a resume over a fusion stamp of the other setting says so (it never silently keeps the other fusion.json)
    c, _ = calls("research", "fusion")
    stamp = next(base.rglob("research/**/fusion.done"))
    assert " oof=1" in stamp.read_text()
    for s in STEPS_DONE:  # every step done, so a resume skips fusion
        if s != "fusion":
            (stamp.parent / f"{s}.done").write_text("old\n")
    stamp.write_text("2026-11-27T00:00:00Z 10s\n")

    def dry():
        r = _run_driver(tmp_path, "--dry-run", "--from", "trim_eval", work_base=base, isl_root=isl,
                        profile="research", seed_base=tmp_path / "no_seed")
        assert r.returncode == 0, r.stdout + r.stderr
        assert "would skip fusion" in r.stdout
        return r.stdout

    assert "warning: fusion stamp settings differ from the current ones (stamp oof=0, FUSION_OOF=1)" in dry()
    monkeypatch.delenv("FUSION_OOF")
    assert "warning: fusion stamp" not in dry()
    stamp.write_text("2026-11-27T00:00:00Z 10s oof=1\n")
    assert "warning: fusion stamp settings differ from the current ones (stamp oof=1, FUSION_OOF=0)" in dry()
    for s in STEPS_DONE:
        (stamp.parent / f"{s}.done").unlink(missing_ok=True)
    monkeypatch.setenv("FUSION_OOF_KEEP_IN_SAMPLE", "0")
    # a folder linked with v1 and a run that does not link: the start line says the folder's split
    monkeypatch.setenv("SIGNER_SPLIT", "v2")
    dry = _run_driver(tmp_path, "--only", "fusion", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert "signer_split=v2 (link step not run; folder: v1" in dry.stdout and "warning: SIGNER_SPLIT=v2" in dry.stdout


# ------------------------------------------------------------------------------------------------ out-of-fold calibration

def _signer(cid: str) -> int:
    return int(cid.rsplit("_", 1)[1]) % 10


FOLD_TEST = {0: {5}, 1: {4}, 2: {2, 3}}  # fold 1's test signer is fold 0's validation one; fold 2's are fold 0 train
FOLD_VAL = {0: {4}, 1: {3}, 2: {1}}


@pytest.fixture(scope="module")
def oof_world(tmp_path_factory):
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.common import ISL_FEATURES, load_isl_data
    from app.isl.training.pipeline import load_recognizer
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex
    from app.training.trainer import TrainConfig
    from tests.isl.isl_world import make_world

    root = tmp_path_factory.mktemp("oof")
    w = make_world(root)
    P = w["paths"]
    base = json.loads((P.splits / "test-split.json").read_text())
    ids = sorted(base["assign"])
    # pseudo-signer groups (two of them) and sessions (one per signer), as the real include_groups.json
    P.include_groups.write_text(json.dumps({"clips": {i: {"session": f"s{_signer(i)}", "pseudo_signer":
                                                           f"ps{_signer(i) % 2}"} for i in ids}}))
    for k in range(3):
        a = {i: "test" if _signer(i) in FOLD_TEST[k] else ("val" if _signer(i) in FOLD_VAL[k] else "train")
             for i in ids}
        (P.splits / f"toy-cv5-fold{k}.json").write_text(json.dumps({**base, "name": f"toy-cv5-fold{k}", "assign": a}))
    vocab = VocabIndex.load(w["vocab"])
    prof = "commercial-safe"
    d0 = load_isl_data(P, vocab, prof, "toy-cv5-fold0")
    work = root / "work"
    cfg = TrainConfig(**TINY)
    train_classifier(d0, work / "classifier.pt", ISL_FEATURES, cfg, smoke=True, threads=1)
    for k in (1, 2):
        dk = load_isl_data(P, vocab, prof, f"toy-cv5-fold{k}")
        train_classifier(dk, work / "cv" / f"fold{k}" / "classifier.pt", ISL_FEATURES, cfg, smoke=True, threads=1)
    train_encoder(d0, work / "encoder.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2, smoke=True, threads=1,
                  batches_per_epoch=2)
    build_index(EncoderPredictor(work / "encoder.pt"), d0.islrtc, d0.include_part("train"), work / "index", prof,
                split=d0.split["name"])
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile=prof)
    return {"w": w, "P": P, "data": d0, "work": work, "rec": rec, "prof": prof}


def test_oof_parts_are_the_other_folds_test_parts_and_never_the_primary_test(oof_world):
    from app.isl.models.index import LeakageError

    d, P, work = oof_world["data"], oof_world["P"], oof_world["work"]
    parts = oc.oof_parts(P, d, work)
    assert [p.fold for p in parts] == [1, 2]
    test0 = {c.clip_id for c in d.include if d.assign.get(c.clip_id) == "test"}
    assert test0 and not any(c.clip_id in test0 for p in parts for c in p.clips)
    assert {_signer(c.clip_id) for c in parts[0].clips} == {4} and parts[0].encoder_in_sample == 0.0
    assert {_signer(c.clip_id) for c in parts[1].clips} == {2, 3} and parts[1].encoder_in_sample == 1.0
    assert oc.check_oof_clips(parts, d)["in_primary_test"] == 0
    with pytest.raises(oc.OofError, match="primary"):
        oc.oof_parts(P, d, work, folds=[0])
    with pytest.raises(oc.OofError, match="not found"):
        oc.oof_parts(P, d, work, folds=[1, 3])
    f3 = P.splits / "toy-cv5-fold3.json"
    f3.write_text((P.splits / "toy-cv5-fold1.json").read_text())
    try:
        with pytest.raises(oc.OofError, match="no cross-validation classifier"):
            oc.oof_parts(P, d, work)  # every fold file but the primary one: fold 3 has no classifier
    finally:
        f3.unlink()
    # a fold whose test part holds a fold-0 test clip is refused before any work
    bad = oc.OofPart(9, "x", work / "classifier.pt", [c for c in d.include if c.clip_id in test0][:1])
    with pytest.raises(LeakageError, match="test part"):
        oc.check_oof_clips(parts + [bad], d)

    # ... and so is a duplicate of a test clip (data.dedup)
    class _Dup:
        parent = {"x": "x"}

        def group(self, i):
            return "g" if i in test0 or i == parts[0].clips[0].clip_id else i

    d.dedup = _Dup()
    try:
        with pytest.raises(LeakageError, match="duplicate"):
            oc.check_oof_clips(parts, d)
    finally:
        d.dedup = None


def test_leave_fold_out_gallery_has_no_row_of_the_fold(oof_world):
    rec, d, P, work = oof_world["rec"], oof_world["data"], oof_world["P"], oof_world["work"]
    idx = rec.index
    by_id = {c.clip_id: c for c in d.include}
    need = sorted({x for r in idx.rows if r.get("source") == "include" for x in oc._row_ids(r)})
    emb = oc.include_embeddings(rec.encoder, [by_id[x] for x in need])
    assert oc.check_centroids(idx, emb)["checked"] == 4  # the stored centroids are rebuilt exactly
    p2 = oc.oof_parts(P, d, work, folds=[2])[0]
    g, rep = oc.fold_gallery(idx, oc.banned_ids(p2, idx, d), emb)
    fold = {c.clip_id for c in p2.clips}
    assert fold & idx.row_clip_ids() and not (fold & g.row_clip_ids())
    assert rep["include_centroids_recomputed"] == 4 and rep["rows"] == len(idx.rows)
    # every INCLUDE centroid now averages the other two training signers only
    cents = [r for r in g.rows if r["source"] == "include"]
    assert all(r["n_clips"] == 2 and r.get("oof_recomputed") for r in cents)
    # fold 1's clips (fold 0 validation) are not in the index: the gallery is the index itself
    p1 = oc.oof_parts(P, d, work, folds=[1])[0]
    g1, rep1 = oc.fold_gallery(idx, oc.banned_ids(p1, idx, d), emb)
    assert rep1 == {"rows": len(idx.rows), "rows_before": len(idx.rows),
                    "include_support": {"centroids": 4, "mean_clips": 4.0}}
    # the leave-fold-out gallery has less support per INCLUDE centroid (docs 15.43: why the screen shares a gallery)
    assert rep["include_support"] == {"centroids": 4, "mean_clips": 2.0}
    # a centroid that cannot be rebuilt (another encoder's embeddings) is refused
    bad = {k: np.random.default_rng(0).normal(size=v.shape).astype(np.float32) for k, v in emb.items()}
    with pytest.raises(oc.OofError, match="cannot be rebuilt"):
        oc.check_centroids(idx, bad)


def test_fold_classifier_of_other_settings_is_not_exchangeable(oof_world):
    from app.models.checkpoint import Predictor

    work = oof_world["work"]
    main = Predictor(work / "classifier.pt").meta
    f1 = Predictor(work / "cv" / "fold1" / "classifier.pt").meta
    assert oc.check_fold_classifier(main, f1, "toy-cv5-fold1") == []
    assert oc.check_fold_classifier(main, f1, "toy-cv5-fold2")[0].startswith("split")
    other = {**f1, "model_kwargs": {**f1["model_kwargs"], "hidden": 999}}
    assert any(x.startswith("model_kwargs") for x in oc.check_fold_classifier(main, other, "toy-cv5-fold1"))
    other = {**f1, "data": {**(f1.get("data") or {}), "signer_split": "v2"}}
    assert any(x.startswith("signer_split") for x in oc.check_fold_classifier(main, other, "toy-cv5-fold1"))


def test_fit_fusion_certifies_include_on_out_of_fold_clips_with_their_source_and_group(oof_world, monkeypatch):
    from app.isl.models import calibration as cal
    from app.isl.training import fit_fusion as ff

    rec, d, P, work = oof_world["rec"], oof_world["data"], oof_world["P"], oof_world["work"]
    seen = {}
    real = ff.fit_thresholds

    def spy(results, truths, params, **kw):
        seen.update(n=len(results), sources=list(kw["sources"]), groups=list(kw["groups"]), truths=list(truths))
        return real(results, truths, params, **kw)

    monkeypatch.setattr(ff, "fit_thresholds", spy)
    val = d.labelled_part("val")
    base, brep = ff.fit_fusion(rec, val, oof_world["prof"], data=d)
    assert "oof_calibration" not in brep and "calibration_sets" not in (base.certification or {})
    assert seen["n"] == brep["val_queries"] == len(val)
    assert Counter(seen["sources"]) == Counter(c.source for c in val)
    assert base.certification["selection_note"].startswith("certified on the same validation clips")

    parts = oc.oof_parts(P, d, work)
    oof = oc.OofCalibration(parts, d, sessions=oc.load_sessions(P), keep_in_sample=True, n_boot=50)
    params, rep = ff.fit_fusion(rec, val, oof_world["prof"], data=d, oof=oof)
    orep = rep["oof_calibration"]
    oof_clips = [c for p in parts for c in p.clips]
    assert orep["folds_used"] == [1, 2] and orep["clips_used"] == len(oof_clips)
    # fold 1's test part IS the primary validation INCLUDE part: replaced, never counted twice
    assert orep["primary_val_clips_replaced"] == sum(1 for c in val if c.source == "include")
    assert seen["n"] == len(val) - orep["primary_val_clips_replaced"] + len(oof_clips)
    assert Counter(seen["sources"]) == Counter({"include": len(oof_clips)})
    # every out-of-fold clip certifies under source include with ITS OWN signer group, in order
    assert seen["groups"] == [c.group for c in oof_clips] and set(seen["groups"]) == {"ps0", "ps1"}
    assert seen["truths"] == [c.concept_id for c in oof_clips]
    assert orep["certification_by_source"] == {"include": len(oof_clips)}
    assert orep["certification_groups_by_source"] == {"include": 2}
    cs = params.certification["calibration_sets"]
    assert cs["clips"] == len(oof_clips) and cs["by_source"] == {"include": len(oof_clips)}
    assert "out-of-fold" in params.certification["selection_note"] and "exchangeable" in cs["assumption"]
    assert params.certification["calibration_clips"] == len(oof_clips)
    chk = orep["in_sample_check"]
    assert chk["testable"] and chk["n_in"] == 8 and chk["n_out"] == 4 and len(chk["did_retrieval_ci95"]) == 2
    # both arms on the same gallery: fold 1's clips were also scored on fold 2's leave-fold-out gallery (docs 15.43)
    assert chk["gallery_matched"] and set(chk["per_fold"]) == {"2"}
    assert chk["gallery_support"]["1"] == {"arm": "out", "centroids": 4, "mean_clips": 4.0, "centroids_dropped": 0}
    assert chk["gallery_support"]["2"]["arm"] == "in" and chk["gallery_support"]["2"]["mean_clips"] == 2.0
    assert orep["parts"][1]["out_of_sample_on_this_gallery"]["n"] == 4
    assert all(p["dropped_not_a_fold_class"] == 0 for p in orep["parts"])
    assert [p["fold"] for p in orep["parts"]] == [1, 2] and orep["parts"][1]["gallery"][
        "include_centroids_recomputed"] == 4
    # the weights and temperature are the primary validation fit, unchanged by the out-of-fold clips
    for k in ("alpha", "beta", "gamma", "tau_r", "t_c"):
        assert getattr(params, k) == getattr(base, k)
    assert isinstance(params.certification, dict) and cal.CERT_VERSION == params.certification["version"]


def test_in_sample_screen_drops_the_folds_the_encoder_trained_on():
    def rows(n, fold, ins, r, c):
        return [{"fold": fold, "in_sample": ins, "r_hit": i < r * n, "c_hit": i < c * n, "f_hit": i < r * n,
                 "cluster": f"{fold}-{i % 5}", "group": "g", "result": None, "truth": "x", "source": "include"}
                for i in range(n)]

    same = rows(100, 1, False, 0.5, 0.6) + rows(100, 2, True, 0.51, 0.6)
    chk = oc.encoder_in_sample_check(same, 0.03, n_boot=100)
    assert chk["testable"] and chk["passed"] and abs(chk["did_retrieval"] - 0.01) < 1e-9
    memo = rows(100, 1, False, 0.5, 0.6) + rows(100, 2, True, 0.8, 0.6)  # encoder memorised fold 2
    chk = oc.encoder_in_sample_check(memo, 0.03, n_boot=100)
    assert chk["passed"] is False and abs(chk["did_retrieval"] - 0.3) < 1e-9
    assert chk["did_retrieval_ci95"][0] > 0.03
    # an easier fold for every model is not an encoder effect: the classifier control removes it
    easy = rows(100, 1, False, 0.5, 0.6) + rows(100, 2, True, 0.8, 0.9)
    assert oc.encoder_in_sample_check(easy, 0.03, n_boot=50)["passed"] is True
    o = oc.OofCalibration([], None, n_boot=50)
    used, rep = oc.select_rows({"rows": memo, "parts": []}, o)
    assert rep["folds_used"] == [1] and rep["folds_dropped"] == [2] and len(used) == 100
    assert "screen failed" in rep["dropped_reason"]
    o.keep_in_sample = True
    used, rep = oc.select_rows({"rows": memo, "parts": []}, o)
    assert rep["folds_used"] == [1, 2] and not rep["folds_dropped"]
    # nothing out of sample: not testable, the in-sample folds are not used
    used, rep = oc.select_rows({"rows": rows(50, 2, True, 0.5, 0.5), "parts": []}, oc.OofCalibration([], None))
    assert rep["in_sample_check"]["testable"] is False and used == [] and rep["folds_dropped"] == [2]


# ------------------------------------------------------------------------------------------------ review of 15.42 (15.43)

def _fold_file(P, k, test, val):
    base = json.loads((P.splits / "test-split.json").read_text())
    a = {i: "test" if _signer(i) in test else ("val" if _signer(i) in val else "train") for i in base["assign"]}
    f = P.splits / f"toy-cv5-fold{k}.json"
    f.write_text(json.dumps({**base, "name": f"toy-cv5-fold{k}", "assign": a}))
    return f


def test_a_fold_classifier_selected_on_the_primary_test_is_skipped_and_overlaps_are_reported(oof_world):
    import shutil

    d, P, work = oof_world["data"], oof_world["P"], oof_world["work"]
    n_test = sum(1 for p in d.assign.values() if p == "test")
    # as grouped_kfold's last fold: its validation part IS fold 0's test part (the E1b test)
    f3 = _fold_file(P, 3, test={1}, val=FOLD_TEST[0])
    ck3 = work / "cv" / "fold3" / "classifier.pt"
    try:
        skipped: list = []
        parts = oc.oof_parts(P, d, work, skipped=skipped)  # skipped before its (missing) classifier is looked for
        assert [p.fold for p in parts] == [1, 2]
        assert skipped == [{"fold": 3, "split": "toy-cv5-fold3", "selected_on_primary_test_clips": n_test,
                            "trained_on_primary_test_clips": 0, "reason": skipped[0]["reason"]}]
        assert "E1b test" in skipped[0]["reason"]
        # folds 1 and 2 TRAIN on fold 0's test clips (fold 0 is in their train part) but never selected on them
        assert all(p.trained_on_primary_test == n_test and p.selected_on_primary_test == 0 for p in parts)
        with pytest.raises(oc.OofError, match="selected on the E1b test"):
            oc.oof_parts(P, d, work, folds=[3])  # asked for explicitly: still refused, nothing left
        ck3.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "cv" / "fold1" / "classifier.pt", ck3)
        p3 = oc.oof_parts(P, d, work, folds=[3], allow_test_selected=True)[0]  # the explicit opt-in
        assert p3.selected_on_primary_test == n_test and p3.trained_on_primary_test == 0
        # duplicates of a test clip count too (data.dedup)
        extra = next(i for i, p in json.loads(f3.read_text())["assign"].items() if p == "train")
        test0 = {i for i, p in d.assign.items() if p == "test"}

        class _Dup:
            parent = {"x": "x"}

            def group(self, i):
                return "g" if i in test0 or i == extra else i

        d.dedup = _Dup()
        try:
            p3 = oc.oof_parts(P, d, work, folds=[3], allow_test_selected=True)[0]
            assert p3.trained_on_primary_test == 1
        finally:
            d.dedup = None
    finally:
        f3.unlink()
        shutil.rmtree(ck3.parent, ignore_errors=True)


def _screen_rows(n, fold, ins, r, c, r_g=None):
    out = []
    for i in range(n):
        x = {"fold": fold, "in_sample": ins, "r_hit": i < r * n, "c_hit": i < c * n, "f_hit": i < r * n,
             "cluster": f"{fold}-{i % 10}", "group": "g", "result": None, "truth": "x", "source": "include"}
        if r_g is not None:
            x["r_hit_g"] = {j: i < v * n for j, v in r_g.items()}
            x["f_hit_g"] = dict(x["r_hit_g"])
        out.append(x)
    return out


def test_in_sample_screen_compares_both_arms_on_the_same_gallery():
    # no memorisation: a leave-fold-out gallery has fewer clips per centroid, so retrieval drops from 0.8 to 0.7 for
    # EVERY clip. The old statistic (each arm on its own gallery) reads that as -0.1 and would hide memorisation.
    out = _screen_rows(200, 1, False, 0.8, 0.6, r_g={2: 0.7, 3: 0.7})
    fair = out + _screen_rows(200, 2, True, 0.7, 0.6) + _screen_rows(200, 3, True, 0.7, 0.6)
    chk = oc.encoder_in_sample_check(fair, 0.03, n_boot=100)
    assert chk["gallery_matched"] and abs(chk["did_retrieval"]) < 1e-9 and chk["passed"]
    assert abs(chk["did_retrieval_unmatched"] + 0.1) < 1e-9
    assert chk["per_fold"]["2"]["out_retrieval_top1_on_this_gallery"] == 0.7
    lo, hi = chk["did_retrieval_ci95"]
    assert lo <= 0 <= hi
    # the encoder memorised the in-sample folds (+0.12 on the shrunk gallery): the old statistic missed it (0.02),
    # the gallery-matched one catches it
    memo = out + _screen_rows(200, 2, True, 0.82, 0.6) + _screen_rows(200, 3, True, 0.82, 0.6)
    chk = oc.encoder_in_sample_check(memo, 0.03, n_boot=100)
    assert abs(chk["did_retrieval_unmatched"] - 0.02) < 1e-9 and chk["passed"] is False
    assert abs(chk["did_retrieval"] - 0.12) < 1e-9 and chk["did_retrieval_ci95"][0] > 0.03
    # per fold: each in-sample fold against its own gallery's out-of-sample rate, pooled by fold size
    mixed = (_screen_rows(100, 1, False, 0.8, 0.6, r_g={2: 0.7, 3: 0.5}) + _screen_rows(100, 2, True, 0.7, 0.6)
             + _screen_rows(300, 3, True, 0.6, 0.6))
    chk = oc.encoder_in_sample_check(mixed, 0.03, n_boot=20)
    assert chk["per_fold"]["2"]["did_retrieval"] == 0.0 and chk["per_fold"]["3"]["did_retrieval"] == 0.1
    assert abs(chk["did_retrieval"] - 0.075) < 1e-9


def test_cli_fit_fusion_oof_calibration(oof_world, tmp_path, monkeypatch, capsys):
    import shutil

    from app.isl import cli
    from app.isl.training import pipeline

    w, P, src = oof_world["w"], oof_world["P"], oof_world["work"]
    base = tmp_path / "art"
    work = base / "commercial-safe"
    work.mkdir(parents=True)
    for f in ("encoder.pt", "classifier.pt"):
        shutil.copy2(src / f, work / f)
    shutil.copytree(src / "index", work / "index")
    shutil.copytree(src / "cv", work / "cv")
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    root = ["--root", str(P.root)]

    def run(*extra, split="toy-cv5-fold0"):
        return cli.main(root + ["fit-fusion", "--profile", "commercial-safe", "--vocab", str(w["vocab"]), "--split",
                                split, "--threads", "1", "--oof-calibration", *extra])

    real = pipeline.load_recognizer

    def refuse(*a, **k):
        raise AssertionError("the out-of-fold checks run before the recogniser is loaded")

    monkeypatch.setattr(pipeline, "load_recognizer", refuse)
    assert run(split="test-split") == 2  # not a fold of a cross-validation split
    assert "not a fold" in capsys.readouterr().err
    shutil.move(work / "cv" / "fold1", tmp_path / "fold1")
    assert run() == 2
    assert "no cross-validation classifier" in capsys.readouterr().err
    shutil.move(tmp_path / "fold1", work / "cv" / "fold1")
    f3 = _fold_file(P, 3, test=FOLD_TEST[0], val={1})  # a "fold" whose test part is fold 0's: the leak check
    (work / "cv" / "fold3").mkdir()
    shutil.copy2(work / "cv" / "fold1" / "classifier.pt", work / "cv" / "fold3" / "classifier.pt")
    try:
        assert run("--oof-folds", "3") == 2
        assert "test part" in capsys.readouterr().err
    finally:
        f3.unlink()
        shutil.rmtree(work / "cv" / "fold3")
    assert not (work / "fusion.json").exists()
    monkeypatch.setattr(pipeline, "load_recognizer", real)
    capsys.readouterr()
    assert run("--oof-folds", "1", "2") == 0
    out = capsys.readouterr().out
    assert "oof_calibration" in out
    cert = json.loads((work / "fusion.json").read_text())["certification"]
    cs = cert["calibration_sets"]
    assert cs["requested_folds"] == [1, 2] and cs["keep_in_sample"] is False and cs["max_encoder_gain"] == 0.03
    assert set(cs["fold_classifiers"]) == {"1", "2"} and len(cs["fold_classifiers"]["1"]["classifier_sha256"]) == 64
    assert cs["fold_classifiers"]["1"]["trained_on_primary_test_clips"] > 0  # disclosed (docs 15.43)
    assert "TRAINED on E1b test clips" in cert["selection_note"]
    if cs["folds_used"] == [1]:  # the screen dropped fold 2: only the primary validation clips certify
        assert "primary validation clips" in cert["selection_note"]
    else:
        assert "let through" in cert["selection_note"]


def test_selection_note_says_what_the_include_clips_are():
    from app.isl.training.fit_fusion import _oof_certification

    parts = [{"fold": 1, "encoder_in_sample": 0.0, "classifier_sha256": "a" * 64, "trained_on_primary_test_clips": 5,
              "selected_on_primary_test_clips": 0},
             {"fold": 2, "encoder_in_sample": 1.0, "classifier_sha256": "b" * 64, "trained_on_primary_test_clips": 5,
              "selected_on_primary_test_clips": 0}]
    orep = {"version": "oof-1", "folds_used": [1], "folds_dropped": [2], "parts": parts, "keep_in_sample": False,
            "max_encoder_gain": 0.03, "requested_folds": None, "in_sample_check": {"passed": False}}
    c = _oof_certification({"version": 1}, orep, 10)
    assert c["selection_note"].startswith("INCLUDE certified on the clips of folds [1], which are the primary "
                                          "validation clips")
    assert c["calibration_sets"]["fold_classifiers"]["2"]["classifier_sha256"] == "b" * 64
    c = _oof_certification({"version": 1}, {**orep, "folds_used": [1, 2], "folds_dropped": []}, 10)
    assert "the encoder trained on folds [2]" in c["selection_note"]
    assert _oof_certification({}, orep, 0) == {}


def test_oof_rows_of_a_word_the_fold_classifier_lacks_are_dropped(oof_world, monkeypatch):
    from app.models import checkpoint

    rec, d, P, work = oof_world["rec"], oof_world["data"], oof_world["P"], oof_world["work"]
    parts = oc.oof_parts(P, d, work, folds=[1])
    word = parts[0].clips[0].concept_id
    real = checkpoint.Predictor

    class Lacking(real):  # fold 1's classifier without ``word`` (as fold 2 of the real data lacks 2 primary classes)
        def __init__(self, path, *a, **k):
            super().__init__(path, *a, **k)
            if "cv" in str(path):
                self.class_names = [n if n != word else "zz_other" for n in self.class_names]

    monkeypatch.setattr(checkpoint, "Predictor", Lacking)
    from app.isl.training.fit_fusion import FusionParams

    ev = oc.oof_evidence(rec, oc.OofCalibration(parts, d), FusionParams(profile=oof_world["prof"]))
    n_word = sum(1 for c in parts[0].clips if c.concept_id == word)
    rp = ev["parts"][0]
    assert rp["primary_classes_missing"] == 1 and rp["dropped_not_a_fold_class"] == n_word
    assert not any(x["truth"] == word for x in ev["rows"])
