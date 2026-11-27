"""Encoder: dictionary-anchored bridge sampler and a validation that can see the dictionary (QA plan 1.5, critics 3,
11 and 14; docs/ISL_DESIGN.md 15.40).

* ``loop.pk_batches`` with a bridge plan: every batch has ``bridge_p`` concepts drawn as one ISLRTC row + one signer
  clip, the signer source by weight; "multi" concepts need two recording groups; without the options the draws are
  exactly the old ones;
* the ``dictionary`` validation protocol: queries from INCLUDE / ISL500 (per signer) / Mendeley validation, a gallery of
  ISLRTC rows only, held-out concepts in neither, per-source recall in the history, the selection key;
* ``cli encoder-probe`` measures any encoder on that protocol and writes nothing into the work folder;
* the driver: research defaults (bridge 12, dictionary validation, 800 steps for encoder and oneshot), commercial-safe
  unchanged, ``--until`` for a short probe run.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.sources import Clip
from app.isl.training import common
from app.isl.training.loop import BridgePlan, bridge_plan, expected_bridge_pairs, pk_batches
from app.isl.training.train_encoder import (DEFAULT_BRIDGE_WEIGHTS, DICT_SELECT_KEY, dictionary_select,
                                            dictionary_val_metrics, dictionary_val_sets, parse_bridge_weights,
                                            signer_key)

TINY = dict(hidden=16, layers=1, epochs=2, batch_size=8, patience=5, lr=3e-3)


def _old_pk_batches(labels, p, k, n_batches, seed, epoch, weights=None):
    """The sampler before 15.40, verbatim (the reference for the default draws)."""
    rng = random.Random(f"{seed}:{epoch}")
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    multi = sorted(y for y, v in by.items() if len(v) > 1)
    allc = sorted(by)
    p = min(p, len(allc))
    for _ in range(n_batches):
        n_multi = min(len(multi), p // 2)
        chosen = set(rng.sample(multi, n_multi)) if n_multi else set()
        rest = [y for y in allc if y not in chosen]
        chosen |= set(rng.sample(rest, min(len(rest), p - len(chosen))))
        batch = []
        for y in sorted(chosen):
            idx = by[y]
            if weights is not None and len(idx) > k:
                batch += rng.choices(idx, weights=[weights[i] for i in idx], k=k)
            else:
                batch += rng.sample(idx, k) if len(idx) >= k else [rng.choice(idx) for _ in range(k)]
        yield batch


def _toy_items(n_bridge=20, n_dict2=10, n_multi=10, n_single=10):
    """(labels, sources, groups): bridge concepts (1 ISLRTC row + INCLUDE x2 signers + ISL500 + Mendeley), concepts
    with two ISLRTC rows only (one recording group), ISL500-only concepts of two signers, single dictionary rows."""
    labels, sources, groups = [], [], []

    def add(y, src, grp):
        labels.append(y)
        sources.append(src)
        groups.append(grp)

    y = 0
    for _ in range(n_bridge):
        add(y, "islrtc", "dictionary")
        for s in range(2):
            add(y, "include", f"include:s{s}")
        add(y, "isl500", "isl500:u1")
        add(y, "isl500", "isl500:u2")
        add(y, "mendeley_emergency", "mendeley_emergency:m1")
        y += 1
    for _ in range(n_dict2):
        add(y, "islrtc", "dictionary")
        add(y, "islrtc", "dictionary")
        y += 1
    for _ in range(n_multi):
        add(y, "isl500", "isl500:u1")
        add(y, "isl500", "isl500:u2")
        y += 1
    for _ in range(n_single):
        add(y, "islrtc", "dictionary")
        y += 1
    return labels, sources, groups


def _plan(labels, sources, p=12, weights=None):
    w = dict(weights or DEFAULT_BRIDGE_WEIGHTS)
    return bridge_plan(labels, [s == "islrtc" for s in sources], [signer_key(s, w) for s in sources], w, p)


# ---------------------------------------------------------------------------------------------- sampler

def test_default_draws_are_exactly_the_old_ones():
    labels, sources, _ = _toy_items()
    w = [0.5 if s.startswith("mendeley") else 1.0 for s in sources]
    for seed, epoch in ((0, 1), (3, 7)):
        for weights in (None, w):
            assert (list(pk_batches(labels, 32, 2, 25, seed, epoch, weights))
                    == list(_old_pk_batches(labels, 32, 2, 25, seed, epoch, weights)))


def test_every_batch_has_bridge_p_cross_source_dictionary_pairs():
    """The plan's check: in a toy world every batch of P=32 has at least 12 pairs of one ISLRTC row and one signer
    clip of the same concept (the old sampler gives far fewer)."""
    labels, sources, groups = _toy_items()
    plan = _plan(labels, sources)
    assert plan.labels == list(range(20))  # only concepts with an ISLRTC row and a signer clip

    def pairs(batch):
        n = 0
        for a in range(0, len(batch), 2):
            i, j = batch[a], batch[a + 1]
            if labels[i] == labels[j] and {sources[i] == "islrtc", sources[j] == "islrtc"} == {True, False}:
                n += 1
        return n

    new = [pairs(b) for b in pk_batches(labels, 32, 2, 200, 0, 1, groups=groups, bridge=plan)]
    assert min(new) >= 12
    old = [pairs(b) for b in pk_batches(labels, 32, 2, 200, 0, 1)]
    assert np.mean(old) < 6
    # K = one ISLRTC row + K-1 signer clips for a bridge label (K = 3 here)
    for b in pk_batches(labels, 32, 3, 20, 0, 2, groups=groups, bridge=plan):
        chunks = [b[a:a + 3] for a in range(0, len(b), 3)]
        bridged = [c for c in chunks if sources[c[0]] == "islrtc" and all(sources[x] != "islrtc" for x in c[1:])
                   and len({labels[x] for x in c}) == 1]
        assert len(bridged) >= 12


def test_signer_source_shares_follow_the_weights():
    labels, sources, groups = _toy_items(n_bridge=20)
    plan = _plan(labels, sources, p=20)
    fam = Counter()
    for b in pk_batches(labels, 20, 2, 1500, 0, 1, groups=groups, bridge=plan):
        for a in range(0, len(b), 2):
            if sources[b[a]] == "islrtc" and labels[b[a]] < 20:
                fam[signer_key(sources[b[a + 1]], DEFAULT_BRIDGE_WEIGHTS)] += 1
    tot = sum(fam.values())
    for s, w in DEFAULT_BRIDGE_WEIGHTS.items():
        assert abs(fam[s] / tot - w) < 0.02, (s, fam)
    # renormalised over the sources a concept has: INCLUDE + Mendeley only -> 0.5 / 0.65 and 0.15 / 0.65
    lab2 = [0, 0, 0, 1, 1]
    src2 = ["islrtc", "include", "mendeley_agri", "islrtc", "include"]
    p2 = _plan(lab2, src2, p=2)
    assert p2.source_probs(0) == pytest.approx({"include": 0.5 / 0.65, "mendeley": 0.15 / 0.65})
    assert p2.source_probs(1) == {"include": 1.0}


def test_multi_labels_need_two_recording_groups():
    """Two ISLRTC rows of one word are one recording group (the dictionary): with ``groups`` such a concept is no
    "multi" concept, so the 2 real multi-signer concepts fill the multi half of every batch."""
    labels, groups = [], []
    for y in range(50):  # two dictionary rows each
        labels += [y, y]
        groups += ["dictionary", "dictionary"]
    for y in (50, 51):  # two signers each
        labels += [y, y]
        groups += ["include:s1", "include:s2"]
    for y in range(52, 100):
        labels.append(y)
        groups.append("dictionary")
    new = list(pk_batches(labels, 4, 2, 100, 0, 1, groups=groups))
    assert all({labels[i] for i in b} >= {50, 51} for b in new)
    old = list(pk_batches(labels, 4, 2, 100, 0, 1))
    assert sum({labels[i] for i in b} >= {50, 51} for b in old) < 10


def test_expected_pairs_match_the_draws():
    labels, sources, groups = _toy_items(n_bridge=20)
    plan = _plan(labels, sources, p=12)
    exp = expected_bridge_pairs(plan, 32, 2, len(set(labels)), 400, "include")
    seen = Counter()
    for b in pk_batches(labels, 32, 2, 400, 0, 1, groups=groups, bridge=plan):
        for a in range(0, len(b), 2):
            if sources[b[a]] == "islrtc" and sources[b[a + 1]] == "include":
                seen[labels[b[a]]] += 1
    assert exp["labels"] == 20
    assert abs(np.mean(list(seen.values())) - exp["median"]) < 0.1 * exp["median"]
    assert expected_bridge_pairs(BridgePlan(12, {}, {}, {}), 32, 2, 10, 100, "include") == {"labels": 0}


def test_bridge_weights_parsing_and_refusals():
    from app.isl.cli import build_parser

    assert parse_bridge_weights(None) is None and parse_bridge_weights("default") is None
    assert parse_bridge_weights("include=1,mendeley=0.2") == {"include": 1.0, "mendeley": 0.2}
    for bad, msg in (("islrtc=1", "anchor"), ("wslp=1", "anchor"), ("incldue=1", "unknown"),
                     ("include=-1", ">= 0"), ("include=0", "at least one")):
        with pytest.raises(ValueError, match=msg):
            parse_bridge_weights(bad)
    p = build_parser()
    a = p.parse_args(["train-encoder", "--bridge-p", "12", "--source-weights", "include=0.6,isl500=0.4",
                      "--val-protocol", "dictionary"])
    assert a.bridge_p == 12 and a.source_weights == {"include": 0.6, "isl500": 0.4} and a.val_protocol == "dictionary"
    d = p.parse_args(["train-oneshot"])
    assert d.bridge_p == 0 and d.source_weights is None and d.val_protocol == "mixed" and d.dict_val_per_signer == 300
    for cmd in ("train-encoder", "train-oneshot"):
        with pytest.raises(SystemExit) as e:
            p.parse_args([cmd, "--source-weights", "islrtc=1"])
        assert e.value.code == 2


# ------------------------------------------------------------------------------------------- validation

def _c(cid, src, concept, group=None):
    return Clip(cid, src, concept, concept, None, None, Path("/nonexistent"), group=group)


def _val_world():
    islrtc = [_c(f"islrtc/{w}", "islrtc", f"isl.{w}") for w in ("a", "b", "c", "held")]
    include = [_c("inc/a1", "include", "isl.a", "s1"), _c("inc/b1", "include", "isl.b", "s1"),
               _c("inc/held", "include", "isl.held", "s1"), _c("inc/nodict", "include", "isl.zzz", "s1"),
               _c("inc/train", "include", "isl.a", "s2")]
    extra = [_c(f"isl500/u{u}/{i}", "isl500", "isl.c" if i % 2 else "isl.a", f"isl500-u{u}")
             for u in (1, 2) for i in range(5)]
    extra += [_c("men/1", "mendeley_emergency", "isl.b", "m"), _c("wslp/a", "wslp", "isl.a"),
              _c("isl500/train", "isl500", "isl.a", "isl500-u9")]
    assign = {c.clip_id: "val" for c in include}
    assign["inc/train"] = "train"
    ex = {c.clip_id: "val" for c in extra}
    ex["isl500/train"] = "train"
    return common.IslData(include=include, islrtc=islrtc, assign=assign, split={"name": "toy"}, profile="research",
                          extra=extra, extra_assign=ex)


def test_dictionary_val_sets_gallery_queries_and_held_out_concepts():
    data = _val_world()
    q, g, info = dictionary_val_sets(data, excl={"isl.held"}, per_signer=3, seed=0)
    assert {c.source for c in g} == {"islrtc"}  # dictionary rows only, never a training clip
    assert "isl.held" not in {c.concept_id for c in q} | {c.concept_id for c in g}  # critic 11
    ids = {c.clip_id for c in q}
    assert "inc/nodict" not in ids and "inc/train" not in ids and "wslp/a" not in ids and "isl500/train" not in ids
    assert {"inc/a1", "inc/b1", "men/1"} <= ids
    per = Counter(c.group for c in q if c.source == "isl500")
    assert per == {"isl500-u1": 3, "isl500-u2": 3}  # stratified per signer
    assert info["queries"]["include"] == {"candidates": 3, "with_dictionary_row": 2}
    assert info["gallery"] == 3 and info["isl500_signers"] == 2
    # the same seed gives the same sample; per_signer 0 = every clip; an excluded source gives no query
    assert [c.clip_id for c in dictionary_val_sets(data, {"isl.held"}, per_signer=3, seed=0)[0]] == [c.clip_id for c in q]
    assert sum(c.source == "isl500" for c in dictionary_val_sets(data, per_signer=0)[0]) == 10
    assert not [c for c in dictionary_val_sets(data, xsrc=("isl500",))[0] if c.source == "isl500"]


def test_dictionary_metrics_per_source_and_selection_key():
    g_lab = ["isl.a", "isl.b", "isl.c"]
    g = np.eye(3, dtype=np.float32)
    qc = [_c("i1", "include", "isl.a"), _c("i2", "include", "isl.b"), _c("x1", "isl500", "isl.c"),
          _c("x2", "isl500", "isl.a"), _c("m1", "mendeley_agri", "isl.b")]
    q = np.array([[1, 0, 0], [1, 0.1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]], np.float32)
    m = dictionary_val_metrics(q, qc, g, g_lab)
    assert m["val_dict_r1_include"] == 0.5 and m["val_dict_r1_isl500"] == 0.5 and m["val_dict_r1_mendeley"] == 1.0
    assert m["val_dict_n_include"] == 2 and m["val_dict_r5_include"] == 1.0 and m["val_dict_r1"] == 3 / 5
    assert m[DICT_SELECT_KEY] == pytest.approx(0.5 * 0.5 + 0.5 * (0.5 + 1.0) / 2)
    assert dictionary_select({"val_dict_r1_include": 0.4, "val_dict_n_include": 3}) == 0.4
    assert dictionary_select({"val_dict_r1_isl500": 0.3, "val_dict_n_isl500": 3}) == 0.3
    assert dictionary_val_metrics(q[:0], [], g, g_lab)[DICT_SELECT_KEY] == 0.0


# ------------------------------------------------------------------------------------- training (toy world)

@pytest.fixture(scope="module")
def bridge_world(tmp_path_factory):
    """The toy world (INCLUDE + ISLRTC) plus ISL500 clips of 4 signers (2 train, 2 val) and Mendeley val clips, with
    concept ids set."""
    from app.dataset.synthetic import make_sequence
    from app.isl.training.common import load_isl_data
    from app.isl.training.multisource import GlossLinker
    from app.isl.vocab.build import VocabIndex
    from tests.isl.isl_world import CORE, make_world

    root = tmp_path_factory.mktemp("bridge")
    w = make_world(root)
    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(w["paths"], vocab, "research", w["split"])
    data.linker = GlossLinker.from_vocab(vocab)
    d = root / "extra"
    d.mkdir()
    extra, assign = [], {}
    for ci, word in enumerate(CORE):
        for u in range(4):
            for t in range(2):
                raw = make_sequence(ci, signer=300 + u, n_frames=40, fps=12.0, n_classes=6,
                                    rng=np.random.default_rng(100 * ci + 10 * u + t))
                p = d / f"{word}_{u}_{t}.npz"
                raw.save(p)
                c = Clip(f"isl500/u{u}/{word}_{t}", "isl500", word, f"isl.{word}", None, None, p,
                         group=f"isl500-u{u}", quality={"n_frames": 40})
                extra.append(c)
                assign[c.clip_id] = "train" if u < 2 else "val"
        raw = make_sequence(ci, signer=400, n_frames=40, fps=12.0, n_classes=6)
        p = d / f"men_{word}.npz"
        raw.save(p)
        c = Clip(f"mendeley/{word}", "mendeley_emergency", word, f"isl.{word}", None, None, p, group="men",
                 quality={"n_frames": 40})
        extra.append(c)
        assign[c.clip_id] = "val"
    data.extra, data.extra_assign = extra, assign
    return {"world": w, "data": data, "root": root}


def test_train_encoder_bridge_and_dictionary_validation(bridge_world, tmp_path):
    import torch

    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_encoder import train_encoder
    from app.training.trainer import TrainConfig

    data = bridge_world["data"]
    out = tmp_path / "encoder.pt"
    lines = []
    kw = dict(emb_dim=16, p=4, k=2, smoke=True, threads=1, batches_per_epoch=2, dict_val_per_signer=3)
    res = train_encoder(data, out, ISL_FEATURES, TrainConfig(**TINY), bridge_p=2, val_protocol="dictionary",
                        exclude_concepts={"isl.dog"}, progress=lambda f, m: lines.append(m), **kw)
    assert res["val_protocol"] == "dictionary" and res["select_key"] == DICT_SELECT_KEY
    meta = torch.load(out, map_location="cpu", weights_only=False)["meta"]
    for row in meta["history"]:
        for s in ("include", "isl500", "mendeley"):
            assert f"val_dict_r1_{s}" in row and f"val_dict_r5_{s}" in row
        assert "val_r1" in row and DICT_SELECT_KEY in row  # the old val_r1 is kept as a secondary
        assert row[DICT_SELECT_KEY] == pytest.approx(dictionary_select(row))
    best = next(h for h in meta["history"] if h["epoch"] == meta["best_epoch"])
    assert best[DICT_SELECT_KEY] == max(h[DICT_SELECT_KEY] for h in meta["history"])
    assert meta["select_key"] == DICT_SELECT_KEY and meta["val_dict"][DICT_SELECT_KEY] == best[DICT_SELECT_KEY]
    sets = meta["val_dictionary_sets"]
    assert sets["excluded_concepts"] == 1 and sets["gallery"] == len(data.islrtc) - 1  # held out of the gallery too
    assert sets["queries"]["isl500"]["candidates"] == 6  # 2 val signers x 3, dog held out
    sm = meta["sampler"]
    assert sm["kind"] == "bridge" and sm["bridge_p"] == 2 and sm["weights"] == DEFAULT_BRIDGE_WEIGHTS
    assert sm["bridge_labels"] == 3  # apple, book, cat (dog held out; egg, fish have no signer clip)
    assert sm["expected_pairs_per_label"]["include"]["labels"] == 3
    assert any("include->islrtc r1" in m for m in lines) and any("bridge sampler" in m for m in lines)
    # a changed bridge setting or validation protocol is never resumed on (rule of 15.19)
    for change in ({"bridge_p": 1, "val_protocol": "dictionary"}, {"bridge_p": 2, "val_protocol": "mixed"},
                   {"bridge_p": 2, "val_protocol": "dictionary", "bridge_weights": {"include": 1.0}}):
        with pytest.raises(StateMismatch):
            train_encoder(data, out, ISL_FEATURES, TrainConfig(**TINY), exclude_concepts={"isl.dog"}, resume=True,
                          **kw, **change)
    # the same settings resume (finished: the saved weights come back)
    again = train_encoder(data, out, ISL_FEATURES, TrainConfig(**TINY), bridge_p=2, val_protocol="dictionary",
                          exclude_concepts={"isl.dog"}, resume=True, **kw)
    assert again["resumed_from"]["finished"] is True


def test_default_options_add_no_new_resume_keys(bridge_world, monkeypatch, tmp_path):
    """Backward compatibility: without the new options, ``LoopConfig.extra`` has no bridge / validation key (a state
    written before 15.40 still resumes) and the sampler is the old one; with them, both keys are there."""
    from app.isl.training import train_encoder as te
    from app.isl.training.common import ISL_FEATURES
    from app.training.trainer import TrainConfig

    class Captured(Exception):
        pass

    seen = []

    def fake_loop(model, batches, loss_fn, cfg, state_path, **kw):
        seen.append((cfg.extra, kw["select_key"], [b[2] for b in batches(1)]))
        raise Captured

    monkeypatch.setattr(te, "run_loop", fake_loop)
    data = bridge_world["data"]
    kw = dict(emb_dim=16, p=4, k=2, smoke=True, threads=1, batches_per_epoch=3)
    with pytest.raises(Captured):
        te.train_encoder(data, tmp_path / "e.pt", ISL_FEATURES, TrainConfig(**TINY), **kw)
    with pytest.raises(Captured):
        te.train_encoder(data, tmp_path / "e.pt", ISL_FEATURES, TrainConfig(**TINY), bridge_p=2,
                         val_protocol="dictionary", **kw)
    (old, old_key, old_batches), (new, new_key, _) = seen
    assert "bridge" not in old and "val_protocol" not in old and "val_dict" not in old and old_key == "val_r1"
    assert new["bridge"]["p"] == 2 and new["val_protocol"] == "dictionary" and new_key == DICT_SELECT_KEY
    clips = te.encoder_training_clips(data, set(), True)
    concepts = sorted({c.concept_id for c in clips})
    labels = [concepts.index(c.concept_id) for c in clips]
    weights = [float({**te.SAMPLE_WEIGHTS}.get(c.source, 1.0)) for c in clips]
    assert old_batches == list(_old_pk_batches(labels, 4, 2, 3, TINY.get("seed", 42), 1, weights))
    with pytest.raises(ValueError, match="val_protocol"):
        te.train_encoder(data, tmp_path / "e.pt", ISL_FEATURES, TrainConfig(**TINY), val_protocol="other", **kw)


# ------------------------------------------------------------------------------------------ encoder-probe

class _Adapter:
    def __init__(self, name, clips):
        self.name, self.commercial, self.licence = name, False, "test"
        self._clips = clips

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        yield from self._clips


@pytest.fixture()
def cli_world(tmp_path, monkeypatch):
    from app.isl.training import multisource as ms
    from tests.isl.isl_world import make_world
    from tests.isl.test_eval_runners import _extra_world

    w = make_world(tmp_path)
    extra = _extra_world(tmp_path, users=8, takes=2)
    ms.register_adapter("isl500", lambda p, v: _Adapter("isl500", extra))
    base = tmp_path / "art"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    common_args = ["--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1"]
    yield {**w, "base": base, "common": common_args, "root": ["--root", str(w["paths"].root)]}
    ms.register_adapter("isl500", None)


def test_cli_encoder_probe_measures_any_encoder_read_only(cli_world, tmp_path, capsys):
    from app.isl import cli

    root, base = cli_world["root"], cli_world["base"]
    common_args = ["--profile", "research", *cli_world["common"]]
    enc = tmp_path / "probe" / "encoder.pt"
    assert cli.main(root + ["train-encoder", *common_args, "--all-sources", "--hidden", "16", "--layers", "1",
                            "--emb-dim", "16", "--epochs", "3", "--batches", "2", "--p", "4", "--batch-size", "8",
                            "--bridge-p", "2", "--val-protocol", "dictionary", "--dict-val-per-signer", "2",
                            "--ckpt", str(enc)]) == 0
    out = capsys.readouterr().out
    assert out.count("include->islrtc r1") == 3  # logged every epoch (critic 14: a short probe run)
    assert enc.exists() and not (base / "research" / "encoder.pt").exists()  # --ckpt
    work = base / "research"
    before = {p: p.stat().st_mtime_ns for p in work.rglob("*")}
    probe = tmp_path / "probe.json"
    assert cli.main(root + ["encoder-probe", *common_args, "--encoder", str(enc), "--per-signer", "2",
                            "--out", str(probe), "--bar", "1.01"]) == 0
    err = capsys.readouterr().err
    assert {p: p.stat().st_mtime_ns for p in work.rglob("*")} == before  # nothing written into the work folder
    doc = json.loads(probe.read_text())
    assert doc["protocol"] == "dictionary" and doc["split"] == cli_world["split"]
    assert set(doc["by_source"]) == {"include", "isl500"} and doc["by_source"]["include"]["n"] > 0
    assert doc["include_r1"] == doc["by_source"]["include"]["r1"]
    assert doc["bar"]["below"] is True and "below the bar" in err
    # a previous probe as the bar: no warning when it is met
    assert cli.main(root + ["encoder-probe", *common_args, "--encoder", str(enc), "--per-signer", "2",
                            "--bar", str(probe)]) == 0
    assert "WARNING" not in capsys.readouterr().err
    # the probe numbers match the trained encoder's own best epoch on the same protocol and settings
    import torch

    meta = torch.load(enc, map_location="cpu", weights_only=False)["meta"]
    assert doc["by_source"]["include"]["r1"] == pytest.approx(meta["val_dict"]["val_dict_r1_include"], abs=1e-4)
    assert cli.main(root + ["encoder-probe", *common_args, "--encoder", str(tmp_path / "none.pt")]) == 2
    assert cli.main(root + ["encoder-probe", *common_args, "--encoder", str(enc), "--bar", "high"]) == 2


def test_probe_leaves_out_the_concepts_a_fold_encoder_never_saw(bridge_world, tmp_path):
    """A one-shot fold encoder (``excluded_concepts`` in its meta): its held-out concepts are in neither the probe's
    queries nor its gallery (critic 11)."""
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.train_encoder import probe_encoder, train_encoder
    from app.training.trainer import TrainConfig

    data = bridge_world["data"]
    out = tmp_path / "fold.pt"
    train_encoder(data, out, ISL_FEATURES, TrainConfig(**{**TINY, "epochs": 1}), emb_dim=16, p=4, k=2, smoke=True,
                  threads=1, batches_per_epoch=1, exclude_concepts={"isl.cat", "isl.egg"})
    full = probe_encoder(data, out, per_signer=0, threads=1)
    assert full["excluded_concepts"] == 2 and full["gallery_used"] == len(data.islrtc) - 2
    n_cat = sum(1 for c in data.include_part("val") + data.extra_part("val") if c.concept_id == "isl.cat")
    q_all, _, _ = dictionary_val_sets(data, per_signer=0)
    assert full["queries_used"] == len(q_all) - n_cat


# ---------------------------------------------------------------------------------------------- driver

def _driver(tmp_path, profile, *args, env_extra=None):
    import os

    from tests.isl.test_phase2_guards import _run_driver

    keys = ("ENC_BRIDGE_P", "ENC_SOURCE_WEIGHTS", "ENC_VAL", "ENC_STEPS", "ONESHOT_STEPS")
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(env_extra or {})
        return _run_driver(tmp_path, *args, work_base=tmp_path / "art", isl_root=tmp_path / "isl", profile=profile)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _stamped(tmp_path, profile):
    from tests.isl.test_phase2_guards import STEPS_DONE

    stamps = tmp_path / "art" / profile / "driver" / f"phase2_{profile}.stamps"
    stamps.mkdir(parents=True, exist_ok=True)
    for s in STEPS_DONE:
        (stamps / f"{s}.done").write_text("old\n")
    return stamps


def _train_calls(tmp_path):
    f = tmp_path / "fake_py.log"
    return [c for c in f.read_text().splitlines() if " train-encoder " in f" {c} " or " train-oneshot " in f" {c} "]


def test_driver_research_defaults_and_commercial_safe_unchanged(tmp_path):
    _stamped(tmp_path, "research")
    r = _driver(tmp_path, "research", "--force", "encoder,oneshot")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _train_calls(tmp_path)
    assert len(calls) == 2
    for c in calls:
        assert "--bridge-p 12 --val-protocol dictionary" in c and "--batches 800" in c and "--source-weights" not in c
    stamp = (tmp_path / "art/research/driver/phase2_research.stamps/encoder.done").read_text()
    assert "bridge_p=12 source_weights=default val=dictionary" in stamp
    (tmp_path / "fake_py.log").unlink()
    _stamped(tmp_path, "commercial-safe")
    r = _driver(tmp_path, "commercial-safe", "--force", "encoder,oneshot")
    assert r.returncode == 0, r.stdout + r.stderr
    enc, one = _train_calls(tmp_path)
    for c in (enc, one):
        assert "--bridge-p" not in c and "--val-protocol" not in c and "--source-weights" not in c
    assert "--batches 400" in enc and "--batches 300" in one
    r = _driver(tmp_path, "research", "--from", "encoder", "--dry-run",
                env_extra={"ENC_BRIDGE_P": "0", "ENC_VAL": "mixed", "ENC_STEPS": "400", "ONESHOT_STEPS": "300"})
    assert "encoder: bridge_p=0 source_weights=default val=mixed" in r.stdout
    assert "--batches 400" in r.stdout and "--batches 300" in r.stdout
    r = _driver(tmp_path, "research", "--from", "encoder", "--dry-run",
                env_extra={"ENC_SOURCE_WEIGHTS": "include=0.7,isl500=0.3"})
    assert r.returncode == 0 and "source_weights=include=0.7,isl500=0.3" in r.stdout
    for bad in ({"ENC_VAL": "best"}, {"ENC_BRIDGE_P": "x"}, {"ENC_SOURCE_WEIGHTS": "include"},
                {"ENC_BRIDGE_P": "0", "ENC_SOURCE_WEIGHTS": "include=1"}):
        assert _driver(tmp_path, "research", "--dry-run", env_extra=bad).returncode == 2, bad


def test_driver_until_runs_a_short_encoder_probe_only(tmp_path):
    """``--from encoder --until encoder`` (critic 14): only the encoder runs and only its stamp and state are
    forgotten; the later steps keep theirs."""
    stamps = _stamped(tmp_path, "research")
    work = tmp_path / "art" / "research"
    (work / "encoder.pt.state").write_bytes(b"s")
    (work / "classifier.pt.state").write_bytes(b"s")
    r = _driver(tmp_path, "research", "--from", "encoder", "--until", "encoder", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "would run encoder" in r.stdout and "would run oneshot" not in r.stdout
    assert "would remove" in r.stdout and "classifier.pt.state" not in r.stdout
    assert "would stop before oneshot (--until encoder)" in r.stdout
    r = _driver(tmp_path, "research", "--from", "encoder", "--until", "encoder", env_extra={"ENC_STEPS": "40"})
    assert r.returncode == 0, r.stdout + r.stderr
    calls = _train_calls(tmp_path)
    assert len(calls) == 1 and " train-encoder " in f" {calls[0]} " and "--batches 40" in calls[0]
    assert (work / "classifier.pt.state").exists() and not (work / "encoder.pt.state").exists()
    assert (stamps / "oneshot.done").read_text() == "old\n" and "bridge_p=12" in (stamps / "encoder.done").read_text()
    assert "stop before oneshot (--until encoder)" in r.stdout
    bad = _driver(tmp_path, "research", "--from", "oneshot", "--until", "encoder", "--dry-run")
    assert bad.returncode == 2 and "after --until" in bad.stderr
    assert _driver(tmp_path, "research", "--until", "nostep", "--dry-run").returncode == 2
