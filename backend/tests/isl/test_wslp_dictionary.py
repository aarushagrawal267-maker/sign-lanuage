"""WSLP is a second view of the ISLRTC dictionary, not a signer (QA plan 1.1, docs/ISL_DESIGN.md 15.32).

* classifier classes: ISLRTC, WSLP and CISLR count as ONE signer group with ``dictionary_group``;
* encoder supervision: ``exclude_sources`` drops WSLP; the class space is in the checkpoint meta;
* index: WSLP rows of concepts that have an ISLRTC row are dropped (``--wslp-twins``), the leak-guard digest is
  unchanged;
* evaluation never pools classifiers of different class spaces; the driver passes the options for research only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dataset.synthetic import make_sequence
from app.isl.data.sources import Clip
from app.isl.training import common
from app.isl.training.build_index import drop_dictionary_twins, twin_sources_for
from app.isl.training.train_classifier import classifier_classes
from app.isl.training.train_encoder import encoder_excluded_sources, encoder_training_clips
from tests.isl.isl_world import CORE, make_world

TINY = dict(hidden=16, layers=1, epochs=1, batch_size=8, patience=2, lr=3e-3)


def _c(cid, src, concept, group=None):
    return Clip(cid, src, concept or "", concept, None, None, Path("/nonexistent"), group=group)


def _toy_data(extra: list[Clip], islrtc: list[Clip], include: list[Clip] = ()) -> common.IslData:
    include = list(include)
    return common.IslData(include=include, islrtc=islrtc, assign={c.clip_id: "train" for c in include},
                          split={"name": "toy"}, profile="research", extra=extra,
                          extra_assign={c.clip_id: "train" for c in extra})


# ------------------------------------------------------------------------------------------ classes and groups

def test_one_wslp_clip_and_one_islrtc_row_give_no_class():
    """The plan's toy world: a concept seen only as one WSLP clip plus its ISLRTC row was a class (two "signers");
    with the dictionary group it is one signer and no class. A real second signer (ISL500) still makes a class."""
    islrtc = [_c("islrtc/A", "islrtc", "isl.a", "islrtc-signer-1"), _c("islrtc/B", "islrtc", "isl.b", "islrtc-signer-1"),
              _c("islrtc/C", "islrtc", "isl.c", "islrtc-signer-1")]
    extra = [_c("wslp/a", "wslp", "isl.a"),  # dictionary twin only
             _c("isl500/u1/b", "isl500", "isl.b", "isl500-user001"),  # a real second signer
             _c("wslp/c", "wslp", "isl.c"), _c("cislr/c", "cislr", "isl.c")]  # three views of one dictionary
    data = _toy_data(extra, islrtc)
    old = classifier_classes(data, 2, 2, True)
    new = classifier_classes(data, 2, 2, True, dictionary_group=True)
    assert old == ["isl.a", "isl.b", "isl.c"]  # before 15.32: every WSLP / CISLR clip was its own signer
    assert new == ["isl.b"]
    g = common.multi_signer_counts(islrtc + extra, dictionary_group=True)
    assert g["isl.a"] == {"dictionary"} and g["isl.c"] == {"dictionary"}
    assert g["isl.b"] == {"dictionary", "isl500:isl500-user001"}
    assert common.multi_signer_counts(islrtc + extra)["isl.c"] == {"islrtc:islrtc-signer-1", "wslp:wslp/c",
                                                                   "cislr:cislr/c"}


def test_dictionary_group_is_a_noop_without_dictionary_views():
    """commercial-safe has no WSLP / CISLR: the grouping changes no class there."""
    islrtc = [_c("islrtc/A", "islrtc", "isl.a", "islrtc-signer-1")]
    extra = [_c("mendeley/a1", "mendeley_emergency", "isl.a", "m1"), _c("mendeley/a2", "mendeley_emergency", "isl.a", "m2")]
    data = _toy_data(extra, islrtc)
    assert classifier_classes(data, 2, 2, True) == classifier_classes(data, 2, 2, True, dictionary_group=True) == ["isl.a"]


def test_encoder_training_clips_leave_out_wslp_only_when_asked():
    islrtc = [_c("islrtc/A", "islrtc", "isl.a", "islrtc-signer-1")]
    extra = [_c("wslp/a", "wslp", "isl.a"), _c("isl500/u1/a", "isl500", "isl.a", "isl500-user001")]
    data = _toy_data(extra, islrtc)
    assert {c.source for c in encoder_training_clips(data, set())} == {"islrtc", "wslp", "isl500"}
    assert {c.source for c in encoder_training_clips(data, set(), exclude_sources=("wslp",))} == {"islrtc", "isl500"}
    assert encoder_excluded_sources({}) == [] and encoder_excluded_sources(
        {"class_space": {"excluded_sources": ["wslp"]}}) == ["wslp"]


def test_class_space_records_and_mismatch():
    assert common.class_space_of(None) == {"signer_grouping": "source", "excluded_sources": []}
    old = {"class_names": ["x"]}  # a checkpoint written before 15.32
    new = {"class_space": {"signer_grouping": "dictionary"}}
    assert common.classifier_grouping_mismatch(old, {"class_space": {"signer_grouping": "source"}}) is None
    assert common.classifier_grouping_mismatch(new, old) == {"signer_grouping": "source",
                                                             "main_signer_grouping": "dictionary"}


# ------------------------------------------------------------------------------------------------------ index

def test_drop_dictionary_twins_and_modes():
    extra = [_c("wslp/a", "wslp", "isl.a"), _c("wslp/z", "wslp", "isl.z"), _c("isl500/u1/a", "isl500", "isl.a", "u1")]
    keep, rep = drop_dictionary_twins(extra, {"isl.a"}, ("wslp",))
    assert [c.clip_id for c in keep] == ["wslp/z", "isl500/u1/a"]
    assert rep["dropped"] == 1 and rep["dropped_per_source"] == {"wslp": 1} and rep["kept_per_source"] == {"wslp": 1}
    assert drop_dictionary_twins(extra, {"isl.a"}, ()) == (extra, {})
    no = {"encoder_version": "isl-enc-1"}
    yes = {"class_space": {"excluded_sources": ["wslp"]}}
    assert twin_sources_for("keep", yes) == () and twin_sources_for("drop", no) == ("wslp",)
    assert twin_sources_for("auto", no) == () and twin_sources_for("auto", yes) == ("wslp",)
    with pytest.raises(ValueError):
        twin_sources_for("sometimes", no)


@pytest.fixture(scope="module")
def wslp_world(tmp_path_factory):
    """The toy world plus WSLP: a twin of every ISLRTC word (a slightly shifted copy of its dictionary clip) and one
    WSLP-only word."""
    from app.isl.training.common import load_isl_data
    from app.isl.training.multisource import GlossLinker
    from app.isl.vocab.build import VocabIndex

    root = tmp_path_factory.mktemp("wslp")
    w = make_world(root)
    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(w["paths"], vocab, "research", w["split"])
    data.linker = GlossLinker.from_vocab(vocab)
    d = root / "wslp"
    d.mkdir()
    wslp = []
    for c in data.islrtc:
        raw = c.load()
        raw.pose[..., :2] += 0.003  # a second view: the same video, re-encoded
        p = d / (Path(c.path).stem + ".npz")
        raw.save(p)
        wslp.append(Clip(f"wslp/{Path(c.path).stem}", "wslp", c.raw_gloss, c.concept_id, None, None, p,
                         quality={"n_frames": len(raw)}))
    raw = make_sequence(7, signer=60, n_frames=40, fps=12.0, n_classes=8)
    raw.save(d / "zebra.npz")
    wslp.append(Clip("wslp/zebra", "wslp", "zebra", "isl.zebra", None, None, d / "zebra.npz",
                     quality={"n_frames": 40}))
    data.extra = wslp
    data.extra_assign = {c.clip_id: "train" for c in wslp}
    return {"world": w, "data": data, "root": root}


def test_encoder_without_wslp_and_index_without_twins(wslp_world):
    """Encoder supervision without WSLP (meta says so), and an index with no WSLP twin of an ISLRTC row; the
    WSLP-only word keeps its row and the leak guard's clip digest still matches the data."""
    import torch

    from app.isl.models.encoder import EncoderPredictor
    from app.isl.models.index import EmbeddingIndex, clip_ids_sha256
    from app.isl.training.build_index import build_index, index_training_clip_ids
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.train_encoder import train_encoder
    from app.training.trainer import TrainConfig

    data, root = wslp_world["data"], wslp_world["root"]
    work = root / "work"
    work.mkdir()
    res = train_encoder(data, work / "encoder.pt", ISL_FEATURES, TrainConfig(**TINY), emb_dim=16, p=4, k=2,
                        smoke=True, threads=1, batches_per_epoch=2, exclude_sources=("wslp",))
    assert "wslp" not in res["train_sources"] and res["class_space"]["excluded_sources"] == ["wslp"]
    meta = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)["meta"]
    assert meta["class_space"]["excluded_sources"] == ["wslp"] and "isl.zebra" not in str(meta["data"])
    enc = EncoderPredictor(work / "encoder.pt")
    twins = twin_sources_for("auto", enc.meta)
    assert twins == ("wslp",)
    rep = build_index(enc, data.islrtc, data.include_part("train"), work / "index", "research",
                      extra=data.extra_part("train"), split=data.split["name"], twin_sources=twins)
    idx = EmbeddingIndex.load(work / "index")
    isl_concepts = {r["concept_id"] for r in idx.rows if r["source"] == "islrtc"}
    wrows = [r for r in idx.rows if r["source"] == "wslp"]
    assert not [r for r in wrows if r["concept_id"] in isl_concepts]  # no WSLP twin rows
    assert [r["concept_id"] for r in wrows] == ["isl.zebra"]  # a WSLP-only word is still served by retrieval
    assert idx.meta.row_filters["wslp_twins"]["dropped"] == len(data.islrtc) == rep["row_filters"]["wslp_twins"]["dropped"]
    # the digest covers every clip given to the build, so the leak guard still matches the data
    assert idx.meta.train_clip_ids_sha256 == clip_ids_sha256(index_training_clip_ids(data))
    # keep (the default) builds exactly as before: every WSLP clip has a row, no filter recorded
    build_index(enc, data.islrtc, data.include_part("train"), work / "index_keep", "research",
                extra=data.extra_part("train"), split=data.split["name"])
    keep = EmbeddingIndex.load(work / "index_keep")
    assert sum(r["source"] == "wslp" for r in keep.rows) == len(data.extra) and keep.meta.row_filters == {}


def test_classifier_meta_class_space_and_resume_signature(wslp_world, tmp_path):
    """The classifier records its class space; a state written with the dictionary group is never resumed without
    it (and the class lists differ: the WSLP twins are classes only with the old grouping)."""
    import torch

    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_classifier import train_classifier
    from app.training.trainer import TrainConfig

    data = wslp_world["data"]
    old = classifier_classes(data, 2, 2, True)
    new = classifier_classes(data, 2, 2, True, dictionary_group=True)
    assert set(old) - set(new) == {"isl.egg", "isl.fish"}  # dictionary-only words: ISLRTC row + its WSLP twin
    assert set(new) == {f"isl.{w}" for w in CORE}
    out = tmp_path / "classifier.pt"
    res = train_classifier(data, out, ISL_FEATURES, TrainConfig(**TINY), smoke=True, threads=1, use_extra=True,
                           steps_per_epoch=2, dictionary_group=True)
    assert res["classes"] == len(new) and res["class_space"]["signer_grouping"] == "dictionary"
    meta = torch.load(out, map_location="cpu", weights_only=False)["meta"]
    assert meta["class_space"]["signer_grouping"] == "dictionary" and meta["class_space"]["n_classes"] == len(new)
    with pytest.raises(StateMismatch):
        train_classifier(data, out, ISL_FEATURES, TrainConfig(**TINY), smoke=True, threads=1, use_extra=True,
                         steps_per_epoch=2, resume=True)


# ----------------------------------------------------------------------------------------------------- driver

def test_driver_passes_the_wslp_options_for_research_only(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import _calls, _run_driver

    for k in ("CLS_DICTIONARY_GROUP", "ENC_EXCLUDE_SOURCES", "INDEX_WSLP_TWINS"):
        monkeypatch.delenv(k, raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"

    def calls_of(step, profile):
        f = tmp_path / "fake_py.log"
        if f.exists():
            f.unlink()
        r = _run_driver(tmp_path, "--only", step, work_base=base, isl_root=isl, profile=profile)
        assert r.returncode == 0, r.stdout + r.stderr
        return [c for c in _calls(tmp_path) if " readiness " not in f" {c} "]

    for step, flag in (("classifier", "--dictionary-group"), ("encoder", "--exclude-sources wslp"),
                       ("oneshot", "--exclude-sources wslp"), ("index", "--wslp-twins auto"),
                       ("index_final", "--wslp-twins auto")):
        assert any(flag in c for c in calls_of(step, "research")), (step, flag)
        cs = calls_of(step, "commercial-safe")
        assert cs and not any(x in c for c in cs for x in ("--dictionary-group", "--exclude-sources", "--wslp-twins"))
    monkeypatch.setenv("CLS_DICTIONARY_GROUP", "0")
    monkeypatch.setenv("ENC_EXCLUDE_SOURCES", "")
    monkeypatch.setenv("INDEX_WSLP_TWINS", "drop")
    assert not any("--dictionary-group" in c for c in calls_of("classifier", "research"))
    assert not any("--exclude-sources" in c for c in calls_of("encoder", "research"))
    assert any("--wslp-twins drop" in c for c in calls_of("index", "research"))
    monkeypatch.setenv("INDEX_WSLP_TWINS", "sometimes")
    bad = _run_driver(tmp_path, "--only", "index", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "INDEX_WSLP_TWINS" in bad.stderr


def test_cli_parses_the_new_options():
    from app.isl.cli import _exclude_sources, build_parser

    p = build_parser()
    a = p.parse_args(["train-classifier", "--dictionary-group"])
    assert a.dictionary_group is True and p.parse_args(["train-classifier"]).dictionary_group is False
    assert _exclude_sources(p.parse_args(["train-encoder", "--exclude-sources", "wslp, cislr"])) == ("cislr", "wslp")
    assert _exclude_sources(p.parse_args(["train-oneshot"])) == ()
    assert p.parse_args(["build-index"]).wslp_twins == "keep"
    assert p.parse_args(["build-index", "--wslp-twins", "auto"]).wslp_twins == "auto"


# --------------------------------------------------------------------------- review fixes (docs 15.33)

def test_exclude_sources_refuses_unknown_and_core_sources():
    """A typo (``wlsp``) would silently train WSLP in; a core source would leave the validation gallery / queries
    without their training list. Both are refused: ValueError in code, argparse exit 2 on the command line."""
    from app.isl.cli import build_parser
    from app.isl.training.train_encoder import check_exclude_sources

    assert check_exclude_sources(["wslp", "cislr", "wslp"]) == ["cislr", "wslp"] and check_exclude_sources(()) == []
    for bad, msg in ((["wlsp"], "unknown"), (["islrtc"], "core"), (["include", "wslp"], "core")):
        with pytest.raises(ValueError, match=msg):
            check_exclude_sources(bad)
    p = build_parser()
    for cmd in ("train-encoder", "train-oneshot"):
        for v in ("wlsp", "islrtc", "wslp,include"):
            with pytest.raises(SystemExit) as e:
                p.parse_args([cmd, "--exclude-sources", v])
            assert e.value.code == 2


def test_default_arguments_add_no_new_resume_keys(wslp_world, monkeypatch, tmp_path):
    """Backward compatibility (docs 15.19): with no new option, LoopConfig.extra has neither ``dictionary_group`` nor
    ``exclude_sources``, so a training state written in Batch 0 still resumes; with the options it has them."""
    from app.isl.training import train_classifier as tc
    from app.isl.training import train_encoder as te
    from app.isl.training.common import ISL_FEATURES
    from app.training.trainer import TrainConfig

    class Captured(Exception):
        pass

    seen = []

    def fake_loop(model, batches, loss_fn, cfg, state_path, **kw):
        seen.append(cfg.extra)
        raise Captured

    monkeypatch.setattr(tc, "run_loop", fake_loop)
    monkeypatch.setattr(te, "run_loop", fake_loop)
    data = wslp_world["data"]
    cfg = TrainConfig(**TINY)
    runs = [lambda **kw: tc.train_classifier(data, tmp_path / "c.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                                             use_extra=True, steps_per_epoch=2, **kw),
            lambda **kw: te.train_encoder(data, tmp_path / "e.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2, smoke=True,
                                          threads=1, batches_per_epoch=2, **kw)]
    for run, on in zip(runs, ({"dictionary_group": True}, {"exclude_sources": ("wslp",)})):
        with pytest.raises(Captured):
            run()
        with pytest.raises(Captured):
            run(**on)
    cls_default, cls_on, enc_default, enc_on = seen
    for extra in (cls_default, enc_default):
        assert "dictionary_group" not in extra and "exclude_sources" not in extra
    assert cls_on["dictionary_group"] is True and enc_on["exclude_sources"] == ["wslp"]
    # E1a (no --all-sources): the grouping changes nothing, so it is neither in the signature nor in the meta
    with pytest.raises(Captured):
        tc.train_classifier(data, tmp_path / "c.pt", ISL_FEATURES, cfg, smoke=True, threads=1, use_extra=False,
                            steps_per_epoch=2, dictionary_group=True)
    assert "dictionary_group" not in seen[-1]


def test_e1a_records_the_effective_signer_grouping(wslp_world, tmp_path):
    """--dictionary-group without --all-sources (the driver's E1a) records "source": the classes are INCLUDE's own."""
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.train_classifier import train_classifier
    from app.training.trainer import TrainConfig

    res = train_classifier(wslp_world["data"], tmp_path / "e1a.pt", ISL_FEATURES, TrainConfig(**TINY), smoke=True,
                           threads=1, use_extra=False, steps_per_epoch=2, dictionary_group=True)
    assert res["class_space"]["signer_grouping"] == "source" and res["class_space"]["use_extra"] is False


def test_encoder_state_without_exclusion_is_not_resumed_with_it(wslp_world, tmp_path):
    """The other direction of the resume signature: a state trained WITH WSLP is never continued without it."""
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_encoder import train_encoder
    from app.training.trainer import TrainConfig

    data, out = wslp_world["data"], tmp_path / "encoder.pt"
    kw = dict(emb_dim=16, p=4, k=2, smoke=True, threads=1, batches_per_epoch=2)
    train_encoder(data, out, ISL_FEATURES, TrainConfig(**TINY), exclude_sources=(), **kw)
    with pytest.raises(StateMismatch):
        train_encoder(data, out, ISL_FEATURES, TrainConfig(**TINY), exclude_sources=("wslp",), resume=True, **kw)


def test_fusion_fit_space_and_mismatch():
    old_cls = {"class_names": ["isl.a", "isl.b"]}  # written before 15.32
    new_cls = {"class_names": ["isl.a"], "class_space": {"signer_grouping": "dictionary"}}
    enc = {"class_space": {"excluded_sources": ["wslp"]}}
    a = common.fusion_fit_space(old_cls, enc, {"wslp_twins": {"dropped": 3}})
    assert a == {"classifier": {"signer_grouping": "source", "n_classes": 2,
                                "classes_sha256": a["classifier"]["classes_sha256"]},
                 "encoder": {"excluded_sources": ["wslp"]}, "index_row_filters": ["wslp_twins"]}
    # index_final drops the same kind of rows from more clips: the counts are not compared
    assert common.fusion_space_mismatch(a, common.fusion_fit_space(old_cls, enc, {"wslp_twins": {"dropped": 9}})) == []
    assert common.fusion_space_mismatch(a, common.fusion_fit_space(new_cls, {}, {})) == [
        "classifier", "encoder", "index_row_filters"]
    assert common.fusion_space_mismatch({}, a) == []  # a fusion.json fitted before 15.33 records nothing
    assert common.fusion_fit_space(None, None, None) == {}


def test_driver_cv_e1a_and_the_run_log_carry_the_settings(tmp_path, monkeypatch):
    """cv and e1a get --dictionary-group too (research); the start line of the run log and the step stamps name the
    grouping, the encoder exclusion and the twin mode; include / islrtc are refused before anything runs."""
    from tests.isl.test_phase2_guards import _calls, _run_driver

    for k in ("CLS_DICTIONARY_GROUP", "ENC_EXCLUDE_SOURCES", "INDEX_WSLP_TWINS"):
        monkeypatch.delenv(k, raising=False)
    sroot = tmp_path / "sroot"
    (sroot / "splits").mkdir(parents=True)
    (sroot / "splits" / "splits_report.json").write_text('{"primary_split": "toy"}')
    (sroot / "splits" / "include-official.json").write_text("{}")
    monkeypatch.setenv("SROOT", str(sroot))  # the cv / e1a split files of a run without --smoke (never data/isl)
    py = tmp_path / "py_c"  # the fake python, but `-c` snippets (the cv step's split reader) really run
    py.write_text('#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exec python3 "$@"; fi\n'
                  'echo "$*" >> "$FAKE_PY_LOG"\nexit 0\n')
    py.chmod(0o755)
    isl, base = tmp_path / "isl", tmp_path / "art"
    for step, split in (("cv", "--split toy-fold1"), ("e1a", "--split include-official")):
        f = tmp_path / "fake_py.log"
        if f.exists():
            f.unlink()
        r = _run_driver(tmp_path, "--only", step, work_base=base, isl_root=isl, profile="research", py=py)
        assert r.returncode == 0, r.stdout + r.stderr
        calls = [c for c in _calls(tmp_path) if "train-classifier" in c]
        assert calls and all("--dictionary-group" in c for c in calls) and any(split in c for c in calls), (step, calls)
        assert "dict_group=1 enc_exclude=wslp wslp_twins=auto" in r.stdout
        stamp = next(base.rglob(f"{step}.done")).read_text()
        assert "dict_group=1" in stamp, stamp
    r = _run_driver(tmp_path, "--only", "encoder", work_base=base, isl_root=isl, profile="research", py=py)
    assert "enc_exclude=wslp" in next(base.rglob("encoder.done")).read_text()
    monkeypatch.setenv("ENC_EXCLUDE_SOURCES", "wslp,islrtc")
    bad = _run_driver(tmp_path, "--only", "encoder", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "core sources" in bad.stderr
