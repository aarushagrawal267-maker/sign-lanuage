"""Encoder warm start from another work folder (driver ENC_INIT, train-encoder --init, train-oneshot --init-dir;
docs/ISL_DESIGN.md 15.46) on the toy world."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.isl.isl_world import CORE
from tests.isl.test_eval_runners import phase2  # noqa: F401 (module fixture: the trained toy work folder)

TRAIN = ["--hidden", "16", "--layers", "1", "--emb-dim", "16", "--epochs", "1", "--batches", "1", "--p", "4",
         "--batch-size", "8"]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------- driver

def _driver(tmp_path, monkeypatch, profile, *args, enc_init=None, seed_base=None):
    from tests.isl.test_phase2_guards import _run_driver

    for k in ("ENC_BRIDGE_P", "ENC_SOURCE_WEIGHTS", "ENC_VAL", "ENC_STEPS", "ONESHOT_STEPS", "ENC_BAR", "ENC_INIT",
              "ONESHOT_FOLDS"):
        monkeypatch.delenv(k, raising=False)
    if enc_init is not None:
        monkeypatch.setenv("ENC_INIT", str(enc_init))
    return _run_driver(tmp_path, *args, work_base=tmp_path / "art", isl_root=tmp_path / "isl", profile=profile,
                       seed_base=seed_base)


def _stamps(tmp_path, profile="research"):
    from tests.isl.test_phase2_guards import STEPS_DONE

    st = tmp_path / "art" / profile / "driver" / f"phase2_{profile}.stamps"
    st.mkdir(parents=True, exist_ok=True)
    for s in STEPS_DONE:
        (st / f"{s}.done").write_text("old\n")
    return st


def _init_dir(tmp_path) -> Path:
    d = tmp_path / "cs"
    for f in ("encoder.pt", "oneshot/fold0/encoder.pt", "oneshot/fold1/encoder.pt"):
        (d / f).parent.mkdir(parents=True, exist_ok=True)
        (d / f).write_bytes(f"init {f}".encode())
    return d


def _calls(tmp_path, *names):
    f = tmp_path / "fake_py.log"
    lines = f.read_text().splitlines() if f.exists() else []
    return [c for c in lines if any(f" {n} " in f" {c} " for n in names)]


def test_driver_default_is_unchanged(tmp_path, monkeypatch):
    """No ENC_INIT: --init auto for both steps, no --init-dir, no new dry-run line, no new stamp text."""
    st = _stamps(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "oneshot", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warm start" not in r.stdout and "ENC_INIT" not in r.stdout
    assert "would run pretrain_dedup" in r.stdout and "would run pretrain" in r.stdout
    r = _driver(tmp_path, monkeypatch, "research", "--force", "pretrain_dedup,pretrain,encoder,oneshot")
    assert r.returncode == 0, r.stdout + r.stderr
    enc, one = _calls(tmp_path, "train-encoder", "train-oneshot")
    assert " --init auto --exclude-sources wslp --bridge-p 12" in enc and " --init auto --exclude-sources" in one
    assert "--init-dir" not in one and "enc_init" not in (st / "encoder.done").read_text()
    assert _calls(tmp_path, "pretrain-dedup") and _calls(tmp_path, "pretrain")
    assert not _calls(tmp_path, "app.isl.training.enc_init")
    assert "skipped" not in (st / "pretrain.done").read_text()


def test_driver_enc_init_dry_run_and_run(tmp_path, monkeypatch):
    st = _stamps(tmp_path)
    cs = _init_dir(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "oneshot", "--dry-run",
                enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"encoder warm start: ENC_INIT={cs}" in r.stdout
    for f in ("encoder.pt", "oneshot/fold0/encoder.pt", "oneshot/fold1/encoder.pt"):  # with their SHA-256
        assert f"  init {cs / f} sha256:{_sha(cs / f)}" in r.stdout
    assert "would stamp pretrain_dedup skipped: ENC_INIT" in r.stdout
    assert "would stamp pretrain skipped: ENC_INIT" in r.stdout and "would run encoder" in r.stdout
    assert not _calls(tmp_path, "train-encoder") and len(_calls(tmp_path, "app.isl.training.enc_init")) == 1
    (tmp_path / "fake_py.log").unlink()
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "oneshot", enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    enc, one = _calls(tmp_path, "train-encoder", "train-oneshot")
    assert f" --init {cs}/encoder.pt --exclude-sources wslp" in enc and "--init auto" not in enc
    assert f" --init-dir {cs} --exclude-sources wslp" in one and "--init " not in one and "--folds 0 1" in one
    assert not _calls(tmp_path, "pretrain-dedup", "pretrain")  # skipped: nothing trained
    check = _calls(tmp_path, "app.isl.training.enc_init")  # the start-time check, before anything runs
    assert len(check) == 1 and "--profile research --hidden 192 --layers 2 --emb-dim 256" in check[0]
    assert str(cs / "oneshot/fold1/encoder.pt") in check[0]
    for s in ("pretrain_dedup", "pretrain"):
        assert f"skipped: ENC_INIT={cs}" in (st / f"{s}.done").read_text()
    for s in ("encoder", "oneshot"):
        assert (st / f"{s}.done").read_text().rstrip().endswith(f"enc_init={cs}")
    assert f"enc_init={cs}" in r.stdout  # the start line
    # a later run without ENC_INIT: the encoder stamps are flagged, and training the encoders without the skipped
    # pre-training is refused (they would silently start from nothing)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "classifier", "--dry-run")
    assert r.returncode == 0 and "warning: encoder stamp settings differ" in r.stdout
    r = _driver(tmp_path, monkeypatch, "research", "--force", "encoder", "--dry-run")
    assert r.returncode == 2 and "skipped by ENC_INIT" in r.stderr
    # another ENC_INIT than the stamps': flagged too
    other = _init_dir(tmp_path / "o")
    r = _driver(tmp_path, monkeypatch, "research", "--from", "classifier", "--dry-run", enc_init=other)
    assert r.returncode == 0 and "warning: oneshot stamp settings differ" in r.stdout


def test_driver_enc_init_refusals(tmp_path, monkeypatch):
    _stamps(tmp_path)
    cs = _init_dir(tmp_path)
    (cs / "oneshot/fold1/encoder.pt").unlink()
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--dry-run", enc_init=cs)
    assert r.returncode == 2 and "oneshot/fold1/encoder.pt is missing" in r.stderr
    # only the files of the steps that run are needed
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "encoder", "--dry-run",
                enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    # the dry run says what the warm start does in this run: the encoder only; the oneshot file is not listed
    assert f"encoder warm start: ENC_INIT={cs} (encoder --init; pretrain_dedup and pretrain seeded / run / kept" \
        in r.stdout and "oneshot is kept without ENC_INIT" in r.stdout
    assert f"  init {cs / 'encoder.pt'} sha256:" in r.stdout and "oneshot/fold1" not in r.stdout
    r = _driver(tmp_path, monkeypatch, "research", "--dry-run", enc_init=tmp_path / "nowhere")
    assert r.returncode == 2 and "is not a folder" in r.stderr
    r = _driver(tmp_path, monkeypatch, "research", "--dry-run", enc_init=tmp_path / "art" / "research")
    assert r.returncode == 2 and "own work folder" in r.stderr
    r = _driver(tmp_path, monkeypatch, "both", "--dry-run", enc_init=cs)
    assert r.returncode == 2 and "one profile per run" in r.stderr


def test_driver_enc_init_scratch_run_does_not_seed_the_pretraining(tmp_path, monkeypatch):
    """A scratch run --from encoder seeds the earlier steps; with ENC_INIT the pre-training is not seeded (no
    pretrained_encoder.pt is needed) but stamped skipped."""
    seed = tmp_path / "seed"
    (seed / "research").mkdir(parents=True)
    (seed / "research" / "vocabulary.jsonl").write_text("{}\n")
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "encoder", "--dry-run",
                seed_base=seed)
    assert r.returncode == 2 and "pretrained_encoder.pt is missing" in r.stderr  # as before
    cs = _init_dir(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "encoder", "--dry-run",
                seed_base=seed, enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would seed vocab" in r.stdout and "would seed pretrain\n" not in r.stdout
    assert "would stamp pretrain skipped: ENC_INIT" in r.stdout and "would run encoder" in r.stdout


# ------------------------------------------------------------------------------------------- checks (unit)

def _ck(path: Path, meta: dict, state_dict=None) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict or {}, "meta": meta}, path)
    return path


def _meta(profile="commercial-safe", excl=(), train_sources=None, pre_sources=None, **kw):
    from app.isl.training.common import ISL_FEATURES

    return {"kind": "isl-encoder", "encoder_version": "isl-enc-1", "profile": profile,
            "feature_config": ISL_FEATURES.to_dict(),
            "model_kwargs": {"hidden": 16, "layers": 1, "emb_dim": 16, "dropout": 0.3},
            "excluded_concepts": sorted(excl),
            "data": {"split": "s", "train_sources": train_sources or {"include": 5, "islrtc": 5}},
            "pretraining": {"labels_used": False, "sources": pre_sources or {"islrtc": 5, "isl_csltr": 1}}, **kw}


def test_check_init_refusals(tmp_path):
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.enc_init import check_init

    mk = {"hidden": 16, "layers": 1, "emb_dim": 16}
    ok = _ck(tmp_path / "cs" / "encoder.pt", _meta(excl={"isl.apple", "isl.book"}))
    kw = dict(fcfg=ISL_FEATURES, model_kwargs=mk, split="s")
    assert check_init(ok, profile="research", **kw) == []  # commercial-safe into research: allowed
    assert check_init(ok, profile="commercial-safe", must_exclude={"isl.apple"}, **kw) == []  # a superset
    bad = check_init(ok, profile="research", must_exclude={"isl.apple", "isl.cat"}, **kw)
    assert len(bad) == 1 and "superset" in bad[0] and "isl.cat" in bad[0]
    res = _ck(tmp_path / "res" / "encoder.pt", _meta(profile="research"))
    bad = check_init(res, profile="commercial-safe", **kw)  # licence direction: never research -> commercial-safe
    assert len(bad) == 1 and "licence direction" in bad[0]
    assert check_init(res, profile="research", **kw) == []
    assert "missing" in check_init(tmp_path / "none.pt", profile="research", **kw)[0]
    assert "model shape" in check_init(ok, profile="research", fcfg=ISL_FEATURES, model_kwargs={**mk, "hidden": 32})[0]
    fc = _ck(tmp_path / "fc" / "encoder.pt", _meta(feature_config={**ISL_FEATURES.to_dict(), "hand_size_floor": 0.5}))
    assert "feature config" in check_init(fc, profile="research", **kw)[0]
    assert "split" in check_init(ok, profile="research", fcfg=ISL_FEATURES, split="other")[0]
    noprof = _ck(tmp_path / "np" / "encoder.pt", _meta(profile=None))
    assert "licence direction" in check_init(noprof, profile="research", **kw)[0]
    # a self-supervised checkpoint saw no label: no concept check
    pre = _ck(tmp_path / "pre" / "pretrained_encoder.pt",
              {k: v for k, v in _meta(encoder_version="isl-enc-pre-1").items() if k != "excluded_concepts"})
    assert check_init(pre, profile="research", must_exclude={"isl.cat"}, **kw) == []


def test_v2_split_guard_passes_a_proven_no_isl500_init(tmp_path, capsys):
    """Signer split v2 only moves ISL500 signers: an explicit init whose checkpoint proves it never read ISL500 passes
    without the 'cannot be checked' warning; anything else keeps today's warning (docs 15.43, 15.46)."""
    from app.isl import cli

    data = SimpleNamespace(link={"signer_split": "v2"})
    work = tmp_path / "work"
    work.mkdir()

    def guard(init):
        args = cli.build_parser().parse_args(["train-encoder", "--init", str(init)])
        ok = cli._pretrain_split_guard("train-encoder", args, work, data)
        return ok, capsys.readouterr().err

    cs = _ck(tmp_path / "cs" / "encoder.pt", _meta())
    (tmp_path / "cs" / "pretrain_clips.txt").write_text("islrtc/a\n")  # an encoder.pt: never read as its list
    ok, err = guard(cs)
    assert ok and "warning" not in err and "never read an ISL500 clip" in err and "profile commercial-safe" in err
    for name, meta in (("res", _meta(profile="research")),  # research profile reads ISL500: not proven
                       ("forged", _meta(train_sources={"include": 1, "isl500": 3})),
                       ("forged_pre", _meta(pre_sources={"isl500": 2})),
                       ("noprof", _meta(profile=None)),
                       ("ws", _meta(warm_start={"path": "x", "profile": "research"}))):
        ok, err = guard(_ck(tmp_path / name / "encoder.pt", meta))
        assert ok and "warning" in err and "not checked" in err, name
    # v1: not checked, as before
    data.link = {}
    ok, err = guard(cs)
    assert ok and err == ""


def test_resume_with_another_init_is_refused(tmp_path):
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_encoder import train_encoder
    from app.training.trainer import TrainConfig
    from tests.isl.test_encoder_bridge import TINY

    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex
    from tests.isl.isl_world import make_world

    w = make_world(tmp_path / "w")
    data = load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "research", w["split"])
    kw = dict(emb_dim=16, p=4, k=2, smoke=True, threads=1, batches_per_epoch=1, use_extra=False)
    a = tmp_path / "a" / "encoder.pt"
    train_encoder(data, a, ISL_FEATURES, TrainConfig(**{**TINY, "epochs": 1}), **kw)
    b = tmp_path / "b" / "encoder.pt"
    b.parent.mkdir()
    shutil.copy2(a, b)
    import torch

    ck = torch.load(b, map_location="cpu", weights_only=False)
    ck["meta"]["note"] = "another init"
    torch.save(ck, b)
    out = tmp_path / "new.pt"
    cfg = TrainConfig(**{**TINY, "epochs": 2})
    train_encoder(data, out, ISL_FEATURES, cfg, init_from=a, **kw)  # its state is kept next to it
    with pytest.raises(StateMismatch):
        train_encoder(data, out, ISL_FEATURES, cfg, init_from=b, resume=True, **kw)
    with pytest.raises(StateMismatch):  # none instead of the init: refused too
        train_encoder(data, out, ISL_FEATURES, cfg, resume=True, **kw)
    assert train_encoder(data, out, ISL_FEATURES, cfg, init_from=a, resume=True, **kw)["resumed_from"]["finished"]


# --------------------------------------------------------------------------------------------- CLI (toy world)

@pytest.fixture()
def two_profiles(tmp_path, monkeypatch):
    """The toy world with an ISL500-like source (research only), a one-shot protocol with two folds, and a trained
    commercial-safe work folder (encoder + both fold encoders) to warm-start from."""
    from app.isl import cli
    from app.isl.training import multisource as ms
    from tests.isl.isl_world import make_world
    from tests.isl.test_encoder_bridge import _Adapter
    from tests.isl.test_eval_runners import _extra_world

    w = make_world(tmp_path)
    extra = _extra_world(tmp_path, users=8, takes=2)
    ms.register_adapter("isl500", lambda p, v: _Adapter("isl500", extra))
    base = tmp_path / "art"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    folds = [{"fold": 0, "held_out_concepts": ["isl.apple"]}, {"fold": 1, "held_out_concepts": ["isl.book"]}]
    (w["paths"].splits / "oneshot_concepts.json").write_text(json.dumps(
        {"concepts": [{"concept_id": f"isl.{c}", "usable": True} for c in CORE], "protocol": {"k": 2, "folds": folds}}))
    root = ["--root", str(w["paths"].root)]
    common = ["--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1", "--all-sources", "--smoke", *TRAIN]
    cs = ["--profile", "commercial-safe", *common]
    assert cli.main(root + ["train-encoder", *cs]) == 0
    assert cli.main(root + ["train-oneshot", *cs, "--folds", "0", "1"]) == 0
    yield {**w, "base": base, "root": root, "common": common, "cs": base / "commercial-safe"}
    ms.register_adapter("isl500", None)


def test_cli_warm_start_per_fold_lineage_and_refusals(two_profiles, capsys):
    import torch

    from app.isl import cli

    t = two_profiles
    root, cs, base = t["root"], t["cs"], t["base"]
    res = ["--profile", "research", *t["common"]]
    capsys.readouterr()
    # the main encoder from the commercial-safe encoder.pt
    assert cli.main(root + ["train-encoder", *res, "--init", str(cs / "encoder.pt")]) == 0
    meta = torch.load(base / "research" / "encoder.pt", map_location="cpu", weights_only=False)["meta"]
    ws = meta["warm_start"]
    assert ws["path"] == str((cs / "encoder.pt").resolve()) and ws["sha256"] == _sha(cs / "encoder.pt")
    assert ws["profile"] == "commercial-safe" and meta["profile"] == "research"
    assert meta["pretraining"]["init_from"] == str(cs / "encoder.pt")
    # one-shot: fold k from the commercial-safe fold k (never another fold's encoder)
    assert cli.main(root + ["train-oneshot", *res, "--folds", "0", "1", "--init-dir", str(cs)]) == 0
    for k in (0, 1):
        m = torch.load(base / "research" / "oneshot" / f"fold{k}" / "encoder.pt", map_location="cpu",
                       weights_only=False)["meta"]
        src = cs / "oneshot" / f"fold{k}" / "encoder.pt"
        assert m["warm_start"]["path"] == str(src.resolve()) and m["warm_start"]["sha256"] == _sha(src)
        assert m["warm_start"]["fold"] == k and m["warm_start"]["excluded_concepts"] == len(m["excluded_concepts"])
    # superset refusal: fold 0 from a folder whose fold 0 encoder saw "apple" (it is the fold 1 encoder)
    swapped = t["base"].parent / "swapped"
    for k, src in ((0, 1), (1, 0)):
        (swapped / "oneshot" / f"fold{k}").mkdir(parents=True)
        shutil.copy2(cs / "oneshot" / f"fold{src}" / "encoder.pt", swapped / "oneshot" / f"fold{k}" / "encoder.pt")
    capsys.readouterr()
    assert cli.main(root + ["train-oneshot", *res, "--folds", "0", "1", "--init-dir", str(swapped)]) == 2
    assert "superset" in capsys.readouterr().err
    # the main encoder (it excluded nothing) as every fold's --init: refused
    assert cli.main(root + ["train-oneshot", *res, "--folds", "0", "--init", str(cs / "encoder.pt")]) == 2
    assert "superset" in capsys.readouterr().err
    assert cli.main(root + ["train-oneshot", *res, "--init-dir", str(cs), "--init", "auto"]) == 2
    part = t["base"].parent / "part"  # fold 1 trains (held-out concepts) but has no file: refused before loading
    (part / "oneshot" / "fold0").mkdir(parents=True)
    shutil.copy2(cs / "oneshot" / "fold0" / "encoder.pt", part / "oneshot" / "fold0" / "encoder.pt")
    capsys.readouterr()
    assert cli.main(root + ["train-oneshot", *res, "--folds", "0", "1", "--init-dir", str(part)]) == 2
    err = capsys.readouterr().err
    assert "missing" in err and "fold1" in err and "fold0" not in err
    # a fold with no held-out concepts is not trained: its file is not needed (no fold2 file, protocol has no fold 2)
    assert cli.main(root + ["train-oneshot", *res, "--folds", "2", "--init-dir", str(cs)]) == 0
    # licence direction: a research encoder never starts a commercial-safe one
    assert cli.main(root + ["train-encoder", "--profile", "commercial-safe", *t["common"], "--ckpt",
                            str(base.parent / "x.pt"), "--init", str(base / "research" / "encoder.pt")]) == 2
    assert "licence direction" in capsys.readouterr().err
    # another model shape
    assert cli.main(root + ["train-encoder", *res, "--hidden", "24", "--init", str(cs / "encoder.pt")]) == 2
    assert "model shape" in capsys.readouterr().err
    # a resume with another init: refused by the state's signature (init SHA-256)
    other = base.parent / "other" / "encoder.pt"
    other.parent.mkdir()
    ck = torch.load(cs / "encoder.pt", map_location="cpu", weights_only=False)
    ck["meta"]["note"] = "a different file"
    torch.save(ck, other)
    from app.isl.training.loop import StateMismatch

    with pytest.raises(StateMismatch):
        cli.main(root + ["train-encoder", *res, "--resume", "--init", str(other)])


# ------------------------------------------------------------------------------------- evaluate / build-pack

def test_evaluate_and_pack_report_the_warm_start_lineage(phase2, tmp_path):
    """E2's note and the leakage guard describe the warm start's own pre-training list (never dropped), and the pack
    records the lineage; a commercial-safe pack never ships an encoder warm-started from research."""
    import torch

    from app.isl.evaluation.runners import evaluate_all, pretraining_guard
    from app.isl.training import multisource as ms
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.enc_init import init_lineage
    from app.isl.training.pipeline import build_pack

    w, data = phase2["world"], phase2["data"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work, ignore=shutil.ignore_patterns("eval"))

    def fix_index():  # the toy index records the encoder file's SHA-256; these edits change only its meta
        im = json.loads((work / "index" / "meta.json").read_text())
        im["encoder_sha256"] = _sha(work / "encoder.pt")
        (work / "index" / "meta.json").write_text(json.dumps(im))

    # an init folder: its encoders and its own pre-training list (5 ISLRTC clips, unlabelled)
    cs = tmp_path / "cs"
    cs.mkdir()
    seen = sorted(data.islrtc, key=lambda c: c.clip_id)[:5]
    lst = ms.write_pretrain_list(cs, seen)
    for f in ("encoder.pt", "oneshot/fold0/encoder.pt"):
        ck = torch.load(work / f, map_location="cpu", weights_only=False)
        ck["meta"]["profile"] = "commercial-safe"
        ck["meta"]["pretraining"] = {"init_from": str(cs / "pretrained_encoder.pt"), "clip_list": lst,
                                     "sources": {"islrtc": 5}, "labels_used": False}
        _ck(cs / f, ck["meta"], ck["state_dict"])
    # this folder has its own (other) pre-training, which the warm-started encoders did not use
    ms.write_pretrain_list(work, [])
    (work / "pretrained_encoder.pt").write_bytes(b"x")
    for f in ("encoder.pt", "oneshot/fold0/encoder.pt"):
        ck = torch.load(work / f, map_location="cpu", weights_only=False)
        ck["meta"]["warm_start"] = init_lineage(cs / f)
        torch.save(ck, work / f)
    fix_index()
    lin = init_lineage(cs / "encoder.pt")["self_supervised"]
    assert lin["list_verified"] and lin["list_file"] == str(cs / "pretrain_clips.txt") and lin["clips"] == 5
    # the guard of this folder's own list vs the warm start's
    assert pretraining_guard(work, data)["pretrain_clips"] == 0
    g = pretraining_guard(work, data, torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)["meta"])
    assert g["pretrain_clips"] == 5 and g["islrtc_clips_seen_unlabelled"] == 5 and g["excludes_eval_clips"] is True
    assert g["lineage"]["pretrain_list"] == str(cs / "pretrain_clips.txt")
    doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[], e4_ks=(1,), latency_n=1)
    e2 = doc["protocols"]["E2"]
    assert e2["run"], e2
    assert "fold 0 was warm-started from" in e2["label"] and str(cs / "pretrain_clips.txt") in e2["label"]
    assert f"unlabelled clips: 5 of {len(data.islrtc)} ISLRTC clips" in e2["label"]
    assert e2["pretraining_saw_gallery_unlabelled"] is True
    assert e2["pretraining_lineage"][0]["warm_start"] == str((cs / "oneshot/fold0/encoder.pt").resolve())
    lg = doc["leakage_guards"]
    assert lg["pretraining_check"]["lineage"]["warm_start_profile"] == "commercial-safe"
    assert lg["pretraining_excludes_eval_clips"] is True and lg["pretraining_check"]["pretrain_clips"] == 5
    md = (work / "eval" / "report.md").read_text()
    assert "pre-training lineage: the encoder was warm-started from" in md
    # a changed list cannot be checked: said, never silently dropped
    (cs / "pretrain_clips.txt").write_text("islrtc/other\n")
    doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[], e4_ks=(1,), latency_n=1)
    assert doc["leakage_guards"]["pretraining_excludes_eval_clips"] is None
    assert "changed since" in doc["leakage_guards"]["pretraining_check"]["reason"]
    assert "cannot be checked" in doc["protocols"]["E2"]["label"]
    assert doc["protocols"]["E2"]["pretraining_saw_gallery_unlabelled"] is None
    # the pack records the lineage; the warm-started encoder is a research component (no mixed_profiles)
    build_pack(work, "research", ISL_FEATURES, tmp_path / "pack", smoke=True, vocab_path=w["vocab"])
    pj = json.loads((tmp_path / "pack" / "pack.json").read_text())
    assert pj["components"]["encoder"]["warm_start"]["sha256"] == _sha(cs / "encoder.pt")
    assert pj["components"]["encoder"]["warm_start"]["profile"] == "commercial-safe"
    assert "mixed_profiles" not in pj
    # the reverse licence direction is refused
    ck = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)
    ck["meta"]["warm_start"]["profile"] = "research"
    ck["meta"]["profile"] = "commercial-safe"
    torch.save(ck, work / "encoder.pt")
    fix_index()
    with pytest.raises(ValueError, match="warm-started from"):
        build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "pack2", smoke=True, vocab_path=w["vocab"])


# ------------------------------------------------------------------------- review fixes (docs 15.46, review 1)

def _seed_folder(tmp_path) -> Path:
    """A real work folder to seed a scratch run from: its own pre-training and encoders (no warm start)."""
    seed = tmp_path / "seed" / "research"
    for f in ("vocabulary.jsonl", "pretrained_encoder.pt", "pretrain_clips.txt", "pretrain_dedup.json", "encoder.pt",
              "oneshot/fold0/encoder.pt", "oneshot/fold1/encoder.pt"):
        (seed / f).parent.mkdir(parents=True, exist_ok=True)
        (seed / f).write_text(f"seed {f}\n")
    return seed.parent


@pytest.mark.parametrize("start", ["oneshot", "classifier"])
def test_driver_scratch_run_seeds_the_pretraining_when_encoders_are_seeded(tmp_path, monkeypatch, start):
    """A scratch run --from oneshot / classifier with ENC_INIT seeds encoders that descend from the real folder's own
    pre-training: the pre-training is seeded as before (evaluate then checks the list they used), never stamped
    "skipped: ENC_INIT", and the dry run says what the warm start really does."""
    seed = _seed_folder(tmp_path)
    cs = _init_dir(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", start, "--dry-run", seed_base=seed, enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    for s in ("pretrain_dedup", "pretrain", "encoder"):
        assert f"would seed {s}\n" in r.stdout, s
    assert "skipped: ENC_INIT" not in r.stdout and "would stamp" not in r.stdout
    assert f"would copy {seed / 'research' / 'pretrained_encoder.pt'}" in r.stdout
    assert f"would copy {seed / 'research' / 'pretrain_clips.txt'}" in r.stdout
    if start == "classifier":
        assert "would seed oneshot" in r.stdout and "neither encoder nor oneshot runs" in r.stdout
        assert f"encoder warm start: ENC_INIT={cs} not used (" in r.stdout and "  init " not in r.stdout
        assert not _calls(tmp_path, "app.isl.training.enc_init")
        return
    assert f"encoder warm start: ENC_INIT={cs} (oneshot --init-dir; pretrain_dedup and pretrain seeded" in r.stdout
    assert f"encoder is seeded from {seed / 'research'}" in r.stdout
    assert f"  init {cs / 'oneshot/fold0/encoder.pt'} sha256:" in r.stdout and f"  init {cs / 'encoder.pt'}" \
        not in r.stdout
    # the real (fake-python) run: the seeded pre-training files are there, the oneshot step warm-starts
    r = _driver(tmp_path, monkeypatch, "research", "--from", start, "--until", "oneshot", seed_base=seed,
                enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    work = tmp_path / "art" / "research"
    st = work / "driver" / "phase2_research.stamps"
    for f in ("pretrained_encoder.pt", "pretrain_clips.txt", "pretrain_dedup.json", "encoder.pt"):
        assert (work / f).read_text() == f"seed {f}\n"
    for s in ("pretrain_dedup", "pretrain", "encoder"):
        assert "seeded from" in (st / f"{s}.done").read_text()
    (one,) = _calls(tmp_path, "train-oneshot")
    assert f"--init-dir {cs}" in one and not _calls(tmp_path, "pretrain", "train-encoder")


def test_driver_enc_init_from_encoder_scratch_and_until(tmp_path, monkeypatch):
    """--from encoder --until encoder in a scratch run: the folder holds only the warm-started encoder (no oneshot):
    the pre-training is skipped. --until encoder on a folder whose oneshot was trained without ENC_INIT: it runs."""
    seed = _seed_folder(tmp_path)
    cs = _init_dir(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "encoder", "--dry-run",
                seed_base=seed, enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would stamp pretrain skipped: ENC_INIT" in r.stdout and "would seed pretrain\n" not in r.stdout
    st = _stamps(tmp_path)  # a finished folder (oneshot stamped "old": its own pre-training)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "encoder", "--dry-run",
                enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would run pretrain_dedup" in r.stdout and "would run pretrain\n" in r.stdout
    assert "oneshot is kept without ENC_INIT (old)" in r.stdout
    # the same with a oneshot stamp of an earlier ENC_INIT run: skipped
    (st / "oneshot.done").write_text(f"t 1s enc_exclude=wslp x enc_init={cs}\n")
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "encoder", "--dry-run",
                enc_init=cs)
    assert r.returncode == 0 and "would stamp pretrain skipped: ENC_INIT" in r.stdout


def test_driver_skipped_pretrain_keeps_its_files(tmp_path, monkeypatch):
    """A pretrain skipped by ENC_INIT keeps this folder's pretrained_encoder.pt with its clip list and state (no
    orphaned checkpoint); without ENC_INIT --from pretrain removes them as before."""
    st = _stamps(tmp_path)
    work = tmp_path / "art" / "research"
    files = {f: work / f for f in ("pretrained_encoder.pt", "pretrain_clips.txt", "pretrained_encoder.state.pt")}
    for p in files.values():
        p.write_text("old\n")
    cs = _init_dir(tmp_path)
    r = _driver(tmp_path, monkeypatch, "research", "--from", "link", "--until", "oneshot", "--dry-run", enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    for name in ("pretrain_clips.txt", "pretrained_encoder.state.pt"):
        assert f"would keep {files[name]} (pretrain skipped: ENC_INIT)" in r.stdout
        assert f"would remove {files[name]}" not in r.stdout
    r = _driver(tmp_path, monkeypatch, "research", "--from", "link", "--until", "oneshot", enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all(p.read_text() == "old\n" for p in files.values())
    assert "skipped: ENC_INIT" in (st / "pretrain.done").read_text()
    r = _driver(tmp_path, monkeypatch, "research", "--from", "pretrain_dedup", "--until", "oneshot", "--dry-run")
    assert r.returncode == 0 and f"would remove {files['pretrain_clips.txt']}" in r.stdout


def test_driver_passes_the_oneshot_concepts_to_the_start_check(tmp_path, monkeypatch):
    _stamps(tmp_path)
    cs = _init_dir(tmp_path)
    oc = tmp_path / "isl" / "splits" / "oneshot_concepts.json"
    oc.parent.mkdir(parents=True)
    oc.write_text("{}")
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "oneshot", "--dry-run",
                enc_init=cs)
    assert r.returncode == 0, r.stdout + r.stderr
    (check,) = _calls(tmp_path, "app.isl.training.enc_init")
    assert f"--oneshot-concepts {oc}" in check
    (tmp_path / "fake_py.log").unlink()
    r = _driver(tmp_path, monkeypatch, "research", "--from", "encoder", "--until", "encoder", "--dry-run",
                enc_init=cs)
    (check,) = _calls(tmp_path, "app.isl.training.enc_init")
    assert "--oneshot-concepts" not in check  # oneshot does not run


def test_start_check_refuses_a_swapped_fold(tmp_path, capsys):
    """The driver's start-time check: a fold file must exclude its fold's raw held-out concepts; a fold without
    held-out concepts needs no file."""
    from app.isl.training.enc_init import main

    oc = tmp_path / "oneshot_concepts.json"
    oc.write_text(json.dumps({"protocol": {"folds": [{"fold": 0, "held_out_concepts": ["isl.apple"]},
                                                     {"fold": 1, "held_out_concepts": ["isl.book"]},
                                                     {"fold": 2, "held_out_concepts": []}]}}))
    d = tmp_path / "cs"
    f0 = _ck(d / "oneshot" / "fold0" / "encoder.pt", _meta(excl={"isl.apple"}))
    f1 = _ck(d / "oneshot" / "fold1" / "encoder.pt", _meta(excl={"isl.apple"}))  # fold 0's encoder: swapped
    mk = ["--profile", "research", "--hidden", "16", "--layers", "1", "--emb-dim", "16", "--oneshot-concepts", str(oc)]
    assert main(mk + [str(f0)]) == 0
    assert main(mk + [str(f0), str(f1)]) == 2
    assert "isl.book" in capsys.readouterr().err
    assert main(mk + [str(f0), str(d / "oneshot" / "fold2" / "encoder.pt")]) == 0  # not trained: not needed
    assert "not needed" in capsys.readouterr().out
    assert main(mk[:-2] + [str(f0), str(f1)]) == 0  # without the protocol: shape / profile only, as before


def test_clip_level_exclusion_uses_the_init_linker(tmp_path):
    """The superset check compares concept ids of two linkers; the clip-level check relabels this run's clips of a
    held-out concept with the init's linker and refuses one it did not exclude."""
    from app.isl.training.enc_init import clip_level_exclusion, relabel

    def clip(cid, src="include", gloss="x", q=None):
        return SimpleNamespace(clip_id=f"{src}/{cid}/{gloss}", concept_id=cid, source=src, raw_gloss=gloss,
                               quality=q or {})

    class Lk:
        variants = {}

        def link(self, gloss, source):
            return ({"pomme": "isl.fruit"}.get(gloss), "gloss")

    held = {"isl.apple"}
    data = SimpleNamespace(include_part=lambda part: [clip("isl.apple"), clip("isl.cat")], islrtc=[],
                           extra_part=lambda part: [clip("isl.apple", "wslp", "pomme", {"link": "alias"})])
    ok_meta = _meta(excl={"isl.apple", "isl.fruit"}, train_sources={"include": 1, "wslp": 1})
    bad_meta = _meta(excl={"isl.apple"}, train_sources={"include": 1, "wslp": 1})
    init = _ck(tmp_path / "cs" / "oneshot" / "fold0" / "encoder.pt", ok_meta)
    import app.isl.training.enc_init as ei

    orig = ei.init_linker
    ei.init_linker = lambda m, p, v: (Lk(), {"work": "cs", "vocab": "-"})
    try:
        assert relabel(Lk(), clip("isl.apple", "wslp", "pomme", {"link": "alias"})) == "isl.fruit"
        assert relabel(Lk(), clip("isl.apple", q={"variant_of": "isl.apple"})) == "isl.apple"
        r = clip_level_exclusion(data, ok_meta, init, held, None)
        assert r["checked"] == 2 and r["violations"] == []
        r = clip_level_exclusion(data, bad_meta, init, held, None)
        assert [v["init"] for v in r["violations"]] == ["isl.fruit"]
        # a source the init never trained on is not its training clip
        r = clip_level_exclusion(data, {**bad_meta, "data": {"train_sources": {"include": 1}}}, init, held, None)
        assert r["checked"] == 1 and r["violations"] == []
    finally:
        ei.init_linker = orig


def test_warm_start_from_a_fine_tuned_init_with_pretraining(two_profiles, monkeypatch):
    """The real shape: the commercial-safe encoder was itself fine-tuned from its pre-training, so its
    meta.pretraining has init_from / loaded_tensors / skipped of ITS load. The research encoder's meta.pretraining
    names the file it really loaded, keeps the inherited record apart, and its warm start's lineage finds the
    commercial-safe pretrain_clips.txt; the one-shot folds record the clip-level check."""
    import torch

    from app.isl import cli

    t = two_profiles
    root, base = t["root"], t["base"]
    cs2_base = base.parent / "cs2"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(cs2_base))
    cs = ["--profile", "commercial-safe", *t["common"]]
    pre = ["--profile", "commercial-safe", "--vocab", str(t["vocab"]), "--split", t["split"], "--threads", "1",
           "--smoke", "--hidden", "16", "--layers", "1", "--emb-dim", "16", "--epochs", "1", "--batches", "1",
           "--batch-size", "8", "--max-clips", "40"]
    assert cli.main(root + ["pretrain", *pre]) == 0
    assert cli.main(root + ["train-encoder", *cs, "--init", "auto"]) == 0
    csw = cs2_base / "commercial-safe"
    cs_meta = torch.load(csw / "encoder.pt", map_location="cpu", weights_only=False)["meta"]
    assert cs_meta["pretraining"]["init_from"] == str(csw / "pretrained_encoder.pt")
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    res = ["--profile", "research", *t["common"]]
    assert cli.main(root + ["train-encoder", *res, "--init", str(csw / "encoder.pt")]) == 0
    meta = torch.load(base / "research" / "encoder.pt", map_location="cpu", weights_only=False)["meta"]
    pt = meta["pretraining"]
    assert pt["init_from"] == str(csw / "encoder.pt")  # the file really loaded, not the init's own pretrain
    assert pt["inherited_load"]["init_from"] == str(csw / "pretrained_encoder.pt")
    assert pt["sources"] == cs_meta["pretraining"]["sources"]  # the inherited pre-training record is kept
    ss = meta["warm_start"]["self_supervised"]
    assert ss["list_verified"] and ss["list_file"] == str(csw / "pretrain_clips.txt")
    # the folds: per-fold warm start with the clip-level check recorded
    assert cli.main(root + ["train-oneshot", *res, "--folds", "0", "1", "--init-dir", str(t["cs"])]) == 0
    m = torch.load(base / "research" / "oneshot" / "fold0" / "encoder.pt", map_location="cpu",
                   weights_only=False)["meta"]
    assert m["warm_start"]["clip_level_check"]["violations"] == 0
    assert m["warm_start"]["clip_level_check"]["checked"] > 0


@pytest.mark.parametrize("which", ["encoder", "folds"])
def test_evaluate_guard_when_only_some_encoders_were_warm_started(phase2, tmp_path, which):
    """Only the main encoder warm-started: the leakage guard checks the warm start's list, E2's note this folder's own
    (the fold encoders used it). Only the folds: the guard checks this folder's list, E2 names the warm start's."""
    import torch

    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training import multisource as ms
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.enc_init import init_lineage

    w, data = phase2["world"], phase2["data"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work, ignore=shutil.ignore_patterns("eval"))
    isl = sorted(data.islrtc, key=lambda c: c.clip_id)
    cs = tmp_path / "cs"
    cs.mkdir()
    lst = ms.write_pretrain_list(cs, isl[:5])
    f = "encoder.pt" if which == "encoder" else "oneshot/fold0/encoder.pt"
    ck = torch.load(work / f, map_location="cpu", weights_only=False)
    ck["meta"]["profile"] = "commercial-safe"
    ck["meta"]["pretraining"] = {"init_from": str(cs / "pretrained_encoder.pt"), "clip_list": lst,
                                 "sources": {"islrtc": 5}, "labels_used": False}
    _ck(cs / f, ck["meta"], ck["state_dict"])
    ms.write_pretrain_list(work, isl[:3])  # this folder's own pre-training: 3 ISLRTC clips
    (work / "pretrained_encoder.pt").write_bytes(b"x")
    ck = torch.load(work / f, map_location="cpu", weights_only=False)
    ck["meta"]["warm_start"] = init_lineage(cs / f)
    torch.save(ck, work / f)
    im = json.loads((work / "index" / "meta.json").read_text())
    im["encoder_sha256"] = _sha(work / "encoder.pt")
    (work / "index" / "meta.json").write_text(json.dumps(im))
    doc = evaluate_all(work, data, ISL_FEATURES, profile="research", smoke=True, vocab_path=w["vocab"],
                       paths=w["paths"], out_dirs=[], e4_ks=(1,), latency_n=1)
    lg, e2 = doc["leakage_guards"], doc["protocols"]["E2"]
    md = (work / "eval" / "report.md").read_text()
    n = len(data.islrtc)
    if which == "encoder":
        assert lg["pretraining_check"]["pretrain_clips"] == 5 and lg["pretraining_check"]["lineage"]
        sv = lg["pretraining_check"]["supervised"]
        assert sv["sources"] and sv["reconstructed_clips"] > 0 and "reconstructed" in sv["how"]
        assert lg["warm_start_supervised_excludes_eval_clips"] == sv["excludes_eval_clips"]
        assert "warm start's supervised training excludes evaluation clips" in md
        assert f"(3 of {n} ISLRTC clips were in the pre-training list)" in e2["label"]
        assert "warm-started" not in e2["label"] and e2["pretraining_saw_gallery_unlabelled"] is True
    else:
        assert lg["pretraining_check"]["pretrain_clips"] == 3 and "lineage" not in lg["pretraining_check"]
        assert "warm_start_supervised_excludes_eval_clips" not in lg
        assert "fold 0 was warm-started from" in e2["label"] and f"5 of {n} ISLRTC clips" in e2["label"]
        assert e2["pretraining_lineage"][0]["fold"] == 0


def test_warm_start_supervised_check_says_when_it_cannot_check():
    from app.isl.evaluation.runners import warm_start_supervised_check

    c = lambda i, s: SimpleNamespace(clip_id=i, source=s)  # noqa: E731
    data = SimpleNamespace(include_part=lambda p: [c("include/a", "include")], islrtc=[c("islrtc/b", "islrtc")],
                           extra_part=lambda p: [c("wslp/c", "wslp")], include=[], extra=[], assign={},
                           extra_assign={}, multisource={"dedup": {"dropped_train_extra": 0,
                                                                   "dropped_train_islrtc": 0}})
    ws = {"train_sources": {"include": 1, "islrtc": 1}, "split": "s"}
    r = warm_start_supervised_check(ws, data)
    assert r["excludes_eval_clips"] is True and r["reconstructed_clips"] == 2
    data.multisource["dedup"]["dropped_train_islrtc"] = 2
    r = warm_start_supervised_check(ws, data)
    assert r["excludes_eval_clips"] is None and "dropped 2" in r["reason"]
    data.multisource = {}
    assert warm_start_supervised_check(ws, data)["excludes_eval_clips"] is None
    assert warm_start_supervised_check({}, data)["excludes_eval_clips"] is None


def test_unreachable_heldout_excuses_only_proven_variant_ids(tmp_path):
    """Only a variant id this run's linker made, for a source the init cannot read, of a concept the init excluded is
    excused (e.g. ``isl.mean~isl500``); never an id merely absent from the loaded clips (dedup drops, caps), one a
    profile-only source or the init's own linker carries, one of an unexcluded base, or a known vocabulary id."""
    from app.isl.training.enc_init import unreachable_heldout

    def clip(cid, src="include", gloss="x", q=None):
        return SimpleNamespace(clip_id=f"{src}/{cid}/{gloss}", concept_id=cid, source=src, raw_gloss=gloss,
                               quality=q or {})

    class Lk:
        variants = {}

        def link(self, gloss, source):
            return ({"pomme": "isl.pear~isl500"}.get(gloss), "gloss")

    variants = {"isl500": {"isl.mean": "isl.mean~isl500", "isl.cat": "isl.cat~isl500", "isl.pear": "isl.pear~isl500",
                           "isl.fig": "isl.fig~isl500", "isl.owl": "isl.owl~isl500", "isl.hat": "isl.hat"},
                "mendeley_emergency": {"isl.owl": "isl.owl~isl500"}}
    linker = SimpleNamespace(variants=variants, known={"isl.mean", "isl.cat", "isl.pear", "isl.fig", "isl.hat"})
    data = SimpleNamespace(include=[clip("isl.mean"), clip("isl.cat")],
                           islrtc=[clip("isl.dog", "islrtc", "pomme", {"link": "gloss"})],
                           extra=[clip("isl.mean~isl500", "isl500"), clip("isl.fig~isl500", "mendeley_emergency"),
                                  clip("isl.zebra", "isl500")],
                           linker=linker)
    # mendeley_emergency is a commercial-safe profile source, not among the recorded training sources
    meta = _meta(excl={"isl.mean", "isl.pear", "isl.fig", "isl.owl", "isl.hat"},
                 train_sources={"include": 2, "islrtc": 1})
    init = _ck(tmp_path / "cs" / "oneshot" / "fold1" / "encoder.pt", meta)
    import app.isl.training.enc_init as ei

    orig = ei.init_linker
    ei.init_linker = lambda m, p, v: (Lk(), {"work": "cs", "vocab": "-"})
    try:
        held = {"isl.mean", "isl.mean~isl500", "isl.cat~isl500", "isl.pear~isl500", "isl.fig~isl500",
                "isl.owl~isl500", "isl.zebra", "isl.gone~isl500"}
        r = unreachable_heldout(data, meta, init, held, None)
        # isl.cat~isl500: its base was not excluded; isl.pear~isl500: the init's linker labels an ISLRTC clip with it;
        # isl.fig~isl500: a profile-only source's clip carries it; isl.owl~isl500: also made for an init source;
        # isl.zebra / isl.gone~isl500: carried by no loaded clip, but not a proven variant (never excused)
        assert r["concepts"] == ["isl.mean~isl500"] and "isl500" not in r["init_sources"]
        assert "mendeley_emergency" in r["init_sources"] and r["checked"] == 4
        assert unreachable_heldout(data, {**meta, "profile": None}, init, held, None)["concepts"] == []
        assert unreachable_heldout(data, {**meta, "data": {}}, init, held, None)["concepts"] == []
        nolink = SimpleNamespace(**{**vars(data), "linker": None})
        assert unreachable_heldout(nolink, meta, init, held, None)["concepts"] == []
        # a research init reads ISL500: nothing is unreachable
        rmeta = _meta(profile="research", excl={"isl.mean"}, train_sources={"include": 2, "isl500": 3})
        assert unreachable_heldout(data, rmeta, init, held, None)["concepts"] == []
    finally:
        ei.init_linker = orig


def test_cli_oneshot_excuse_wiring(two_profiles, monkeypatch, capsys):
    """train-oneshot drops an excused id from the superset check only: the fold encoder and the clip-level check keep
    the full held-out set and the lineage records the excuse; without the excuse the same run is refused."""
    import app.isl.evaluation.runners as runners
    import app.isl.training.enc_init as ei
    import app.isl.training.train_encoder as te
    from app.isl import cli

    t = two_profiles
    res = ["--profile", "research", *t["common"]]
    orig_held = runners.heldout_set
    monkeypatch.setattr(runners, "heldout_set",
                        lambda lk, cs: orig_held(lk, cs) | ({"isl.book~isl500"} if "isl.book" in cs else set()))
    seen = {}
    orig_cl = ei.clip_level_exclusion
    monkeypatch.setattr(ei, "clip_level_exclusion",
                        lambda data, m, p, held, v: seen.setdefault("cl", set(held)) and orig_cl(data, m, p, held, v))

    def fake_train(data, ckpt, *a, exclude_concepts=(), warm_start=None, **kw):
        seen.setdefault("train", {})[str(ckpt)] = (set(exclude_concepts), warm_start)
        return {"finished": True}

    monkeypatch.setattr(te, "train_encoder", fake_train)
    capsys.readouterr()
    assert cli.main(t["root"] + ["train-oneshot", *res, "--folds", "1", "--init-dir", str(t["cs"])]) == 2
    assert "superset" in capsys.readouterr().err and "train" not in seen
    monkeypatch.setattr(ei, "unreachable_heldout", lambda data, m, p, held, v: {
        "concepts": ["isl.book~isl500"], "init_sources": ["include"], "checked": 1, "rule": "test"})
    assert cli.main(t["root"] + ["train-oneshot", *res, "--folds", "1", "--init-dir", str(t["cs"])]) == 0
    assert "isl.book~isl500" in capsys.readouterr().err
    (excl, ws), = seen["train"].values()
    assert "isl.book~isl500" in excl and "isl.book" in excl and "isl.book~isl500" in seen["cl"]
    assert ws["excused_concepts"]["concepts"] == ["isl.book~isl500"] and ws["excused_concepts"]["rule"] == "test"
