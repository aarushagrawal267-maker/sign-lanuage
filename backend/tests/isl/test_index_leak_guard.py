"""Leak guard of the retrieval index (docs/ISL_DESIGN.md 15.21, QA fix 0.2 / RC8).

The training-only ``work/index`` is what fit-fusion and evaluation read; the eval-including ``work/index_final`` is
only shipped in the pack. Fitting / evaluating on an index that contains evaluation clips, that does not say whether
it does (an older index file), or that was embedded with another encoder file is refused with ``LeakageError``.
15.22: the index also records its split and a digest of the clips it was built from, and every centroid row lists its
clips, so a changed split assignment or another split is refused too."""

from __future__ import annotations

import json
import os
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from app.isl.models.index import (EmbeddingIndex, IndexEncoderMismatch, IndexMeta, IndexMissingError,
                                  IndexVersionMismatch, LeakageError, clip_ids_sha256, file_sha256)
from app.isl.training.common import ISL_FEATURES, load_isl_data
from app.training.trainer import TrainConfig
from tests.isl.isl_world import make_world

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=2, lr=3e-3)


def _rows(n=4):
    return [{"variant_id": f"v{i}", "concept_id": f"isl.c{i % 2}", "source": "islrtc", "kind": "first_sign",
             "clip": f"islrtc/{i}"} for i in range(n)]


TRAIN_IDS = ["include/t1", "include/t2"]


def _save(folder: Path, contains_eval=None, encoder_sha256="", rows=None, split="s1",
          train_ids=TRAIN_IDS) -> EmbeddingIndex:
    rows = rows or _rows()
    emb = np.random.default_rng(0).normal(size=(len(rows), 8)).astype(np.float32)
    idx = EmbeddingIndex.build(emb, rows, "enc-x", {"seq_len": 48}, "commercial-safe", contains_eval=contains_eval,
                               encoder_sha256=encoder_sha256, split=split,
                               train_clip_ids_sha256=clip_ids_sha256(train_ids) if train_ids is not None else "")
    idx.save(folder)
    return idx


def _enc_file(tmp_path, content=b"encoder weights A") -> Path:
    p = tmp_path / "encoder.pt"
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------------------------- unit: meta + load

def test_old_format_index_loads_with_unknown_fields(tmp_path):
    """An index written before contains_eval / encoder_sha256 existed (old meta.json) still loads."""
    _save(tmp_path / "index")
    meta = json.loads((tmp_path / "index" / "meta.json").read_text())
    for k in ("contains_eval", "encoder_sha256"):
        meta.pop(k)
    (tmp_path / "index" / "meta.json").write_text(json.dumps(meta))
    idx = EmbeddingIndex.load(tmp_path / "index", expect_encoder_version="enc-x",
                              encoder_path=_enc_file(tmp_path))  # no sha recorded: nothing to compare (app path)
    assert idx.meta.contains_eval is None and idx.meta.encoder_sha256 == "" and len(idx) == 4
    with pytest.raises(LeakageError, match="does not record whether"):  # but never used to fit / evaluate
        idx.check_for_evaluation()
    # a meta.json written by a newer version with an extra field also loads
    (tmp_path / "index" / "meta.json").write_text(json.dumps({**meta, "future_field": 1}))
    assert EmbeddingIndex.load(tmp_path / "index").meta.contains_eval is None


def test_save_records_fields_and_replaces_folder_atomically(tmp_path):
    enc = _enc_file(tmp_path)
    _save(tmp_path / "index", contains_eval=True, encoder_sha256="x" * 64)
    _save(tmp_path / "index", contains_eval=False, encoder_sha256=file_sha256(enc), rows=_rows(6))
    m = IndexMeta.from_json(json.loads((tmp_path / "index" / "meta.json").read_text()))
    assert m.contains_eval is False and m.encoder_sha256 == file_sha256(enc) and m.n == 6
    assert sorted(p.name for p in tmp_path.iterdir()) == ["encoder.pt", "index"]  # no .tmp / .old left


def test_load_with_encoder_path_refuses_another_encoder_file(tmp_path):
    enc_a = _enc_file(tmp_path)
    _save(tmp_path / "index", contains_eval=False, encoder_sha256=file_sha256(enc_a))
    assert EmbeddingIndex.load(tmp_path / "index", encoder_path=enc_a).meta.encoder_sha256 == file_sha256(enc_a)
    enc_b = tmp_path / "b" / "encoder.pt"
    enc_b.parent.mkdir()
    enc_b.write_bytes(b"encoder weights B (retrained, same version string)")
    with pytest.raises(IndexEncoderMismatch) as ei:
        EmbeddingIndex.load(tmp_path / "index", expect_encoder_version="enc-x", encoder_path=enc_b)
    assert isinstance(ei.value, IndexVersionMismatch)  # callers that catch the version mismatch also catch this


def test_check_for_evaluation_rules(tmp_path):
    enc = _enc_file(tmp_path)
    sha = file_sha256(enc)
    ok = _save(tmp_path / "ok", contains_eval=False, encoder_sha256=sha)
    rep = ok.check_for_evaluation(enc, eval_clip_ids=["include/x"])
    assert rep["contains_eval"] is False and rep["encoder_sha256_checked"] and rep["eval_clip_ids_checked"] == 1
    with pytest.raises(LeakageError, match="contains validation / test"):
        _save(tmp_path / "final", contains_eval=True, encoder_sha256=sha).check_for_evaluation(enc)
    with pytest.raises(LeakageError, match="another encoder file"):
        _save(tmp_path / "stale", contains_eval=False, encoder_sha256="0" * 64).check_for_evaluation(enc)
    with pytest.raises(LeakageError, match="SHA-256"):
        _save(tmp_path / "nosha", contains_eval=False).check_for_evaluation(enc)
    with pytest.raises(LeakageError, match="evaluation clips"):  # a row of a clip being measured
        ok.check_for_evaluation(enc, eval_clip_ids=["islrtc/2"])


def test_check_for_evaluation_split_and_training_clips(tmp_path):
    """docs 15.22: the index records its split and which clips it was built from; centroids list their clips."""
    enc = _enc_file(tmp_path)
    sha = file_sha256(enc)
    rows = _rows() + [{"variant_id": "v9", "concept_id": "isl.c0", "source": "include", "kind": "centroid",
                       "n_clips": 2, "clips": ["include/t1", "include/t2"]}]
    ok = _save(tmp_path / "ok", contains_eval=False, encoder_sha256=sha, rows=rows)
    rep = ok.check_for_evaluation(enc, eval_clip_ids=["include/v1"], split="s1", train_clip_ids=TRAIN_IDS[::-1])
    assert rep["split"] == "s1" and rep["split_checked"] and rep["train_clip_ids_checked"]
    assert ok.row_clip_ids() >= {"include/t1", "include/t2", "islrtc/0"}
    with pytest.raises(LeakageError, match="evaluation clips"):  # a centroid built from a clip being measured
        ok.check_for_evaluation(enc, eval_clip_ids=["include/t2"])
    with pytest.raises(LeakageError, match="split 'other'"):  # fit-fusion / evaluate --split <other>
        ok.check_for_evaluation(enc, split="other")
    with pytest.raises(LeakageError, match="other training clips"):  # the assignment changed since the build
        ok.check_for_evaluation(enc, split="s1", train_clip_ids=["include/t1"])
    # an index written before 15.22 (contains_eval False, but no split / clip digest): refused
    for kw in ({"split": ""}, {"train_ids": None}):
        with pytest.raises(LeakageError, match="does not record its split"):
            _save(tmp_path / "pre", contains_eval=False, encoder_sha256=sha, **kw).check_for_evaluation(enc)


def test_check_index_for_fitting_needs_the_encoder_file(tmp_path):
    from app.isl.training.fit_fusion import check_index_for_fitting

    enc = _enc_file(tmp_path)
    idx = _save(tmp_path / "index", contains_eval=False, encoder_sha256=file_sha256(enc))
    assert check_index_for_fitting(types.SimpleNamespace(index=None, encoder=None), []) is None
    with pytest.raises(LeakageError, match="no encoder file path"):
        check_index_for_fitting(types.SimpleNamespace(index=idx, encoder=None), [])
    with pytest.raises(LeakageError, match="no encoder file path"):
        check_index_for_fitting(types.SimpleNamespace(index=idx, encoder=types.SimpleNamespace()), [])
    assert check_index_for_fitting(types.SimpleNamespace(index=idx, encoder=types.SimpleNamespace(path=enc)),
                                   [])["encoder_sha256_checked"]


def test_fit_fusion_refuses_an_index_with_eval_clips(tmp_path):
    from app.isl.training.fit_fusion import fit_fusion

    enc = _enc_file(tmp_path)
    idx = _save(tmp_path / "index_final", contains_eval=True, encoder_sha256=file_sha256(enc))
    rec = types.SimpleNamespace(index=idx, encoder=types.SimpleNamespace(path=enc), classifier=None)
    with pytest.raises(LeakageError):
        fit_fusion(rec, [], "commercial-safe")
    # encoder sha mismatch: refused too
    idx2 = _save(tmp_path / "index", contains_eval=False, encoder_sha256="f" * 64)
    with pytest.raises(LeakageError, match="another encoder file"):
        fit_fusion(types.SimpleNamespace(index=idx2, encoder=types.SimpleNamespace(path=enc), classifier=None), [],
                   "commercial-safe")


# --------------------------------------------------------------------------------------------- real components

@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex

    root = tmp_path_factory.mktemp("leakworld")
    w = make_world(root)
    data = load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "commercial-safe", w["split"])
    work = root / "work" / "commercial-safe"
    work.mkdir(parents=True)
    train_classifier(data, work / "classifier.pt", ISL_FEATURES, TrainConfig(**TINY), smoke=True, threads=1)
    train_encoder(data, work / "encoder.pt", ISL_FEATURES, TrainConfig(**TINY), emb_dim=16, p=4, k=2, smoke=True,
                  threads=1, batches_per_epoch=2)
    enc = EncoderPredictor(work / "encoder.pt")
    split = data.split["name"]
    r_train = build_index(enc, data.islrtc, data.include_part("train"), work / "index", "commercial-safe", split=split)
    r_final = build_index(enc, data.islrtc, [c for c in data.include if c.concept_id], work / "index_final",
                          "commercial-safe", contains_eval=True, split=split)
    return {"w": w, "data": data, "work": work, "r_train": r_train, "r_final": r_final}


def _copy_work(world, tmp_path) -> Path:
    work = tmp_path / "commercial-safe"
    shutil.copytree(world["work"], work)
    return work


def test_build_index_records_contains_eval_and_encoder_sha(world):
    work = world["work"]
    sha = file_sha256(work / "encoder.pt")
    assert world["r_train"]["contains_eval"] is False and world["r_train"]["encoder_sha256"] == sha
    assert world["r_final"]["contains_eval"] is True and world["r_final"]["encoder_sha256"] == sha
    assert world["r_final"]["centroids"] >= world["r_train"]["centroids"]
    # 15.22: split + digest of the training clips; every INCLUDE centroid lists its clips
    from app.isl.training.build_index import index_training_clip_ids

    data = world["data"]
    assert world["r_train"]["split"] == data.split["name"]
    assert world["r_train"]["train_clip_ids_sha256"] == clip_ids_sha256(index_training_clip_ids(data))
    idx = EmbeddingIndex.load(work / "index")
    cents = [r for r in idx.rows if r["kind"] == "centroid"]
    assert cents and all(len(r["clips"]) == r["n_clips"] for r in cents)
    assert {i for r in cents for i in r["clips"]} <= {c.clip_id for c in data.include_part("train")}


def _moved(data, clip_id: str, part: str = "val"):
    """A copy of ``data`` whose split assignment puts one clip into another part (as a changed split would)."""
    import copy

    d = copy.copy(data)
    d.assign = {**data.assign, clip_id: part}
    return d


def test_fit_fusion_refuses_an_index_built_under_another_assignment(world):
    """The review's case (a): the split assignment changes (e.g. a new val_frac) and fusion is re-fitted without
    rebuilding the index. A clip that was a training clip (inside an INCLUDE centroid) is now a validation clip."""
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import load_recognizer

    data = world["data"]
    rec = load_recognizer(world["work"], ISL_FEATURES, world["w"]["vocab"], profile="commercial-safe")
    moved = data.include_part("train")[0].clip_id
    d2 = _moved(data, moved)
    # the centroid lists the moved clip: refused by the row check alone (no data given) ...
    with pytest.raises(LeakageError, match="evaluation clips"):
        fit_fusion(rec, d2.labelled_part("val"), "commercial-safe")
    # ... and by the training-clip digest with the data
    with pytest.raises(LeakageError, match="other training clips"):
        fit_fusion(rec, d2.labelled_part("val"), "commercial-safe", data=d2)
    # case (b): fit-fusion --split <other> against the primary split's index
    import copy

    d3 = copy.copy(data)
    d3.split = {**data.split, "name": "another-split"}
    with pytest.raises(LeakageError, match="split 'another-split'"):
        fit_fusion(rec, d3.labelled_part("val"), "commercial-safe", data=d3)
    # the unchanged data passes and the report carries the guard summary
    _, rep = fit_fusion(rec, data.labelled_part("val"), "commercial-safe", data=data)
    assert rep["index_guard"]["split_checked"] and rep["index_guard"]["train_clip_ids_checked"]


def test_load_recognizer_refuses_a_missing_training_index_next_to_another(world, tmp_path):
    """No silent classifier-only fit / evaluation when work/index is gone but index_final or a half-swapped save is
    there (docs 15.22)."""
    from app.isl.training.pipeline import load_recognizer

    work = _copy_work(world, tmp_path)
    vocab = world["w"]["vocab"]
    (work / "index").replace(work / "index.old")  # killed between the two replace() calls of save()
    with pytest.raises(IndexMissingError, match="index.old"):
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")
    shutil.rmtree(work / "index.old")
    with pytest.raises(IndexMissingError, match="index_final"):
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")
    assert load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe", for_app=True).index is not None
    assert load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe", with_index=False).index is None
    shutil.rmtree(work / "index_final")  # no index at all: a classifier-only recogniser, as before
    assert load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe").index is None


def test_evaluate_all_records_the_index_guard(world, tmp_path):
    from app.isl.evaluation.runners import evaluate_all

    work = _copy_work(world, tmp_path)
    w = world["w"]
    doc = evaluate_all(work, world["data"], ISL_FEATURES, profile="commercial-safe", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[], e4_ks=(1,), latency_n=1)
    g = doc["leakage_guards"]["index"]
    assert g["dir"] == "index" and g["training_parts_only"] is True and g["contains_eval"] is False
    assert g["encoder_sha256_checked"] and g["split_checked"] and g["train_clip_ids_checked"]
    assert g["eval_clip_ids_checked"] > 0 and g["split"] == world["data"].split["name"]


def test_load_recognizer_uses_training_index_for_evaluation_and_final_index_for_the_app(world):
    from app.isl.training.pipeline import load_recognizer

    work = world["work"]
    rec = load_recognizer(work, ISL_FEATURES, world["w"]["vocab"], profile="commercial-safe")
    assert rec.index.meta.contains_eval is False
    app = load_recognizer(work, ISL_FEATURES, world["w"]["vocab"], profile="commercial-safe", for_app=True)
    assert app.index.meta.contains_eval is True


def test_evaluation_refuses_eval_including_old_or_stale_index(world, tmp_path):
    from app.isl.training.pipeline import evaluate, load_recognizer

    work = _copy_work(world, tmp_path)
    vocab = world["w"]["vocab"]
    # 1. the old driver: index_final written over work/index
    shutil.rmtree(work / "index")
    shutil.copytree(work / "index_final", work / "index")
    with pytest.raises(LeakageError, match="contains validation / test"):
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")
    with pytest.raises(LeakageError):
        evaluate(work, world["data"], ISL_FEATURES, smoke=True, vocab_path=vocab)
    # 2. an index file of the old format (no contains_eval): unknown = refused
    meta = json.loads((work / "index" / "meta.json").read_text())
    for k in ("contains_eval", "encoder_sha256"):
        meta.pop(k)
    (work / "index" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(LeakageError, match="does not record whether"):
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")
    # 3. a clean index, but the encoder was retrained afterwards (same ENCODER_VERSION, other weights)
    shutil.rmtree(work / "index")
    shutil.copytree(world["work"] / "index", work / "index")
    load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")  # clean: loads
    import torch

    ck = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)
    ck["meta"]["retrained"] = True
    torch.save(ck, work / "encoder.pt")
    with pytest.raises(LeakageError, match="another encoder file"):
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe")
    with pytest.raises(IndexEncoderMismatch):  # the app path checks the sha too
        load_recognizer(work, ISL_FEATURES, vocab, profile="commercial-safe", for_app=True)


def test_pack_ships_index_final_and_refuses_a_stale_one(world, tmp_path):
    from app.isl.training.pipeline import build_pack

    work = _copy_work(world, tmp_path)
    r = build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack", smoke=True, vocab_path=world["w"]["vocab"])
    pack = Path(r["pack"])
    pm = json.loads((pack / "index" / "meta.json").read_text())
    assert pm["contains_eval"] is True and pm == json.loads((work / "index_final" / "meta.json").read_text())
    man = json.loads((pack / "pack.json").read_text())
    assert man["components"]["index"]["built_from"] == "index_final"
    assert man["components"]["index"]["contains_eval"] is True
    assert man["components"]["index"]["encoder_sha256"] == file_sha256(pack / "encoder.pt")
    # the app's loader accepts it (same encoder file)
    EmbeddingIndex.load(pack / "index", expect_encoder_version=pm["encoder_version"], encoder_path=pack / "encoder.pt")
    # without index_final (an older work folder): the training-only index is shipped
    shutil.rmtree(work / "index_final")
    r2 = build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack2", smoke=True,
                    vocab_path=world["w"]["vocab"])
    assert json.loads((Path(r2["pack"]) / "pack.json").read_text())["components"]["index"]["built_from"] == "index"
    # an index_final older than index (index rebuilt without index_final): not shipped, and the pack says why
    shutil.copytree(world["work"] / "index_final", work / "index_final")
    m = json.loads((work / "index" / "meta.json").read_text())
    (work / "index" / "meta.json").write_text(json.dumps({**m, "created": "2099-01-01T00:00:00Z"}))
    r3 = build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack2b", smoke=True,
                    vocab_path=world["w"]["vocab"])
    ci = json.loads((Path(r3["pack"]) / "pack.json").read_text())["components"]["index"]
    assert ci["built_from"] == "index" and ci["contains_eval"] is False and "older than index" in ci["note"]
    from app.isl.training.pipeline import load_recognizer

    app = load_recognizer(work, ISL_FEATURES, world["w"]["vocab"], profile="commercial-safe", for_app=True)
    assert app.index.meta.contains_eval is False  # the app path makes the same choice
    shutil.rmtree(work / "index_final")
    (work / "index" / "meta.json").write_text(json.dumps(m))
    # an index_final embedded with an older encoder file is never shipped
    shutil.copytree(world["work"] / "index_final", work / "index_final")
    (work / "encoder.pt").write_bytes((work / "encoder.pt").read_bytes() + b"\0")
    with pytest.raises(IndexEncoderMismatch):
        build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack3", smoke=True,
                   vocab_path=world["w"]["vocab"])


def test_old_pack_without_new_index_fields_loads_in_the_app_loader(world, tmp_path):
    """A pack built before this change (index meta without contains_eval / encoder_sha256) still loads."""
    pack = tmp_path / "pack"
    pack.mkdir()
    shutil.copy2(world["work"] / "encoder.pt", pack / "encoder.pt")
    shutil.copytree(world["work"] / "index_final", pack / "index")
    meta = json.loads((pack / "index" / "meta.json").read_text())
    for k in ("contains_eval", "encoder_sha256"):
        meta.pop(k)
    (pack / "index" / "meta.json").write_text(json.dumps(meta))
    (pack / "encoder.pt").write_bytes((pack / "encoder.pt").read_bytes() + b"\0")  # nothing recorded to compare
    idx = EmbeddingIndex.load(pack / "index", expect_encoder_version=meta["encoder_version"],
                              encoder_path=pack / "encoder.pt")
    assert idx.meta.contains_eval is None


# --------------------------------------------------------------------------------------------- CLI + driver

def test_cli_build_index_writes_index_final_and_fit_fusion_refuses_it(world, tmp_path, monkeypatch):
    from app.isl import cli

    base = tmp_path / "art"
    work = base / "commercial-safe"
    work.mkdir(parents=True)
    shutil.copy2(world["work"] / "encoder.pt", work / "encoder.pt")
    shutil.copy2(world["work"] / "classifier.pt", work / "classifier.pt")
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    w = world["w"]
    common = ["--profile", "commercial-safe", "--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1"]
    root = ["--root", str(w["paths"].root)]
    # --all-include never over the training-only index
    assert cli.main(root + ["build-index", *common, "--all-include", "--out", str(work / "index")]) == 2
    assert not (work / "index").exists()
    assert cli.main(root + ["build-index", *common, "--all-include"]) == 0
    assert json.loads((work / "index_final" / "meta.json").read_text())["contains_eval"] is True
    assert not (work / "index").exists()
    assert cli.main(root + ["build-index", *common]) == 0
    assert json.loads((work / "index" / "meta.json").read_text())["contains_eval"] is False
    # the old driver's mistake: the final index in work/index -> fit-fusion is refused (exit 6), nothing written
    shutil.rmtree(work / "index")
    shutil.copytree(work / "index_final", work / "index")
    assert cli.main(root + ["fit-fusion", *common]) == cli.EX_LEAKAGE
    assert not (work / "fusion.json").exists()
    # evaluate --all is refused the same way (exit 6) and writes no report
    out = tmp_path / "packout"
    assert cli.main(root + ["evaluate", *common, "--all", "--smoke", "--out", str(out)]) == cli.EX_LEAKAGE
    assert not (work / "eval" / "report.json").exists() and not (out / "eval" / "report.json").exists()
    # so is the phase-1 evaluate
    assert cli.main(root + ["evaluate", *common, "--smoke"]) == cli.EX_LEAKAGE
    # an index built for the primary split, used with --split <other>: refused (docs 15.22)
    shutil.rmtree(work / "index")
    assert cli.main(root + ["build-index", *common]) == 0
    assert json.loads((work / "index" / "meta.json").read_text())["split"] == w["split"]
    sp = json.loads((w["paths"].splits / f"{w['split']}.json").read_text())
    other = w["paths"].splits / "test-split-b.json"  # same assignment, another name: refused all the same
    other.write_text(json.dumps({**sp, "name": "test-split-b"}))
    try:
        common2 = [a if a != w["split"] else "test-split-b" for a in common]
        assert cli.main(root + ["fit-fusion", *common2]) == cli.EX_LEAKAGE
        assert not (work / "fusion.json").exists()
        assert cli.main(root + ["fit-fusion", *common]) == 0  # its own split: fine
    finally:
        other.unlink()


def test_driver_index_steps_write_separate_folders(tmp_path):
    from tests.isl.test_phase2_guards import _calls, _old_run, _run_driver

    isl = tmp_path / "isl"
    work = tmp_path / "art" / "commercial-safe"
    _old_run(work, work / "driver" / "phase2_commercial-safe.stamps")
    # the 0.6 re-run: the train-only index is rebuilt before fusion, the final one after evaluate
    r = _run_driver(tmp_path, "--force", "index,fusion,trim_eval,evaluate,index_final,pack", "--dry-run",
                    work_base=tmp_path / "art", isl_root=isl)
    assert r.returncode == 0, r.stdout + r.stderr
    order = [ln.split()[2] for ln in r.stdout.splitlines() if ln.startswith("would run ")]
    assert order == ["index", "fusion", "trim_eval", "evaluate", "index_final", "pack"]
    for step in ("index", "index_final"):
        r = _run_driver(tmp_path, "--only", step, work_base=tmp_path / "art", isl_root=isl)
        assert r.returncode == 0, r.stdout + r.stderr
    # a rebuilt index forgets the later steps that used it or ship next to it (docs 15.22)
    stamps = work / "driver" / "phase2_commercial-safe.stamps"
    assert not any((stamps / f"{s}.done").exists() for s in ("fusion", "evaluate", "pack"))
    assert (stamps / "index.done").exists() and (stamps / "index_final.done").exists()  # re-run by --only after it
    assert (stamps / "trim_eval.done").exists()
    calls = [c for c in _calls(tmp_path) if " build-index " in c]
    assert len(calls) == 2
    assert calls[0].endswith(f"--out {work / 'index'}") and "--all-include" not in calls[0]
    assert "--all-include" in calls[1] and calls[1].endswith(f"--out {work / 'index_final'}")
    assert os.access(Path(__file__).parents[3] / "scripts" / "phase2_train.sh", os.X_OK)
