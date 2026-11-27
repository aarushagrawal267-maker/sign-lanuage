"""Leak guard of the retrieval index (docs/ISL_DESIGN.md 15.21, QA fix 0.2 / RC8).

The training-only ``work/index`` is what fit-fusion and evaluation read; the eval-including ``work/index_final`` is
only shipped in the pack. Fitting / evaluating on an index that contains evaluation clips, that does not say whether
it does (an older index file), or that was embedded with another encoder file is refused with ``LeakageError``."""

from __future__ import annotations

import json
import os
import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from app.isl.models.index import (EmbeddingIndex, IndexEncoderMismatch, IndexMeta, IndexVersionMismatch,
                                  LeakageError, file_sha256)
from app.isl.training.common import ISL_FEATURES, load_isl_data
from app.training.trainer import TrainConfig
from tests.isl.isl_world import make_world

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=2, lr=3e-3)


def _rows(n=4):
    return [{"variant_id": f"v{i}", "concept_id": f"isl.c{i % 2}", "source": "islrtc", "kind": "first_sign",
             "clip": f"islrtc/{i}"} for i in range(n)]


def _save(folder: Path, contains_eval=None, encoder_sha256="", rows=None) -> EmbeddingIndex:
    rows = rows or _rows()
    emb = np.random.default_rng(0).normal(size=(len(rows), 8)).astype(np.float32)
    idx = EmbeddingIndex.build(emb, rows, "enc-x", {"seq_len": 48}, "commercial-safe", contains_eval=contains_eval,
                               encoder_sha256=encoder_sha256)
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
    r_train = build_index(enc, data.islrtc, data.include_part("train"), work / "index", "commercial-safe")
    r_final = build_index(enc, data.islrtc, [c for c in data.include if c.concept_id], work / "index_final",
                          "commercial-safe", contains_eval=True)
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
    calls = [c for c in _calls(tmp_path) if " build-index " in c]
    assert len(calls) == 2
    assert calls[0].endswith(f"--out {work / 'index'}") and "--all-include" not in calls[0]
    assert "--all-include" in calls[1] and calls[1].endswith(f"--out {work / 'index_final'}")
    assert os.access(Path(__file__).parents[3] / "scripts" / "phase2_train.sh", os.X_OK)
