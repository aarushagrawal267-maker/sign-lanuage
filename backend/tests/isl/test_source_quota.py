"""Classifier source quota and worst-source selection (QA plan 1.4, critics 2, 9 and 10; docs/ISL_DESIGN.md 15.38).

1. the two-stage sampler draws each quota group at its share (10k draws within +-2 %); without a quota the draws are
   exactly the old ones; the expected-share formula matches the draws;
2. parsing and resolving a quota: renormalised over the sources present, typos and uncovered sources refused;
3. the classifier selects on the worst major source (the minimum of INCLUDE and ISL500), fits the temperature on
   source-balanced validation, logs per-source history and records the quota; a changed quota or selection is a
   changed resume signature;
4. the driver: research defaults on (800 steps, patience 8), commercial-safe off; e1a never gets the quota.
"""

from __future__ import annotations

import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from app.isl.training.loop import balanced_batches, expected_draw_shares
from app.isl.training.source_quota import (DEFAULT_SOURCE_QUOTA, parse_source_quota, quota_group,
                                           resolve_source_quota)

REPO = Path(__file__).resolve().parents[3]


def _items():
    """An unbalanced toy training list like the research one: ISL500 swamps INCLUDE inside shared classes."""
    labels, sources = [], []
    for c in range(10):  # INCLUDE: 10 clips per class
        labels += [c] * 10
        sources += ["include"] * 10
    for c in range(60):  # ISL500: 50 clips per class, 10 classes shared with INCLUDE
        labels += [c] * 50
        sources += ["isl500"] * 50
    for c in range(70):  # dictionary: one ISLRTC row per class, WSLP twins of half
        labels.append(c)
        sources.append("islrtc")
        if c % 2:
            labels.append(c)
            sources.append("wslp")
    for c in range(3):
        labels += [c] * 4
        sources += ["mendeley_agri"] * 4
    weights = [4.0 if s == "islrtc" else (0.8 if s == "wslp" else 1.0) for s in sources]
    return labels, sources, weights


def _old_balanced(labels, weights, batch, n_batches, seed, epoch, class_power=0.5):
    """The sampler before 15.38, verbatim: the no-quota path must draw exactly this."""
    rng = random.Random(f"{seed}:{epoch}")
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    cls = sorted(by)
    cw = np.array([len(by[c]) ** class_power for c in cls], np.float64)
    cw /= cw.sum()
    cum = np.cumsum(cw)
    w = np.asarray(weights if weights is not None else np.ones(len(labels)), np.float64)
    for _ in range(n_batches):
        out = []
        for _ in range(batch):
            c = cls[min(len(cls) - 1, int(np.searchsorted(cum, rng.random())))]
            idx = by[c]
            ww = w[idx]
            out.append(idx[rng.choices(range(len(idx)), weights=ww)[0]] if len(idx) > 1 else idx[0])
        yield out


# --------------------------------------------------------------------------------------------- 1. sampler

def test_quota_draws_each_group_at_its_share():
    labels, sources, weights = _items()
    group_of, shares = resolve_source_quota(DEFAULT_SOURCE_QUOTA, set(sources))
    assert group_of == {"include": "include", "isl500": "isl500", "islrtc": "dictionary", "wslp": "dictionary",
                        "mendeley_agri": "mendeley"}
    assert shares == pytest.approx(DEFAULT_SOURCE_QUOTA)
    groups = [group_of[s] for s in sources]
    draws = [i for b in balanced_batches(labels, weights, 100, 100, seed=3, epoch=1, groups=groups,
                                         group_quota=shares) for i in b]
    assert len(draws) == 10_000
    got = Counter(groups[i] for i in draws)
    for g, q in shares.items():
        assert abs(got[g] / len(draws) - q) <= 0.02, (g, got[g], q)
    # deterministic per (seed, epoch), another epoch differs
    again = [i for b in balanced_batches(labels, weights, 100, 100, seed=3, epoch=1, groups=groups,
                                         group_quota=shares) for i in b]
    assert again == draws
    other = [i for b in balanced_batches(labels, weights, 100, 100, seed=3, epoch=2, groups=groups,
                                         group_quota=shares) for i in b]
    assert other != draws
    # the expected-share formula of the meta agrees with the draws (per source, inside the dictionary group too)
    exp = expected_draw_shares(labels, weights, sources, 0.5, groups, shares)
    emp = Counter(sources[i] for i in draws)
    for s, v in exp.items():
        assert abs(emp[s] / len(draws) - v) <= 0.02, (s, emp[s], v)
    assert sum(exp.values()) == pytest.approx(1.0)


def test_without_quota_the_draws_are_exactly_the_old_ones():
    labels, sources, weights = _items()
    for cp in (0.0, 0.5, 1.0):
        new = list(balanced_batches(labels, weights, 32, 40, seed=7, epoch=4, class_power=cp))
        assert new == list(_old_balanced(labels, weights, 32, 40, seed=7, epoch=4, class_power=cp))
    # and the old expected share (one stage) matches the old draws
    draws = [i for b in balanced_batches(labels, weights, 100, 100, seed=1, epoch=1) for i in b]
    exp = expected_draw_shares(labels, weights, sources, 0.5)
    emp = Counter(sources[i] for i in draws)
    assert all(abs(emp[s] / len(draws) - v) <= 0.02 for s, v in exp.items())
    assert exp["isl500"] > 0.6  # the problem the quota fixes: ISL500 takes most draws


def test_quota_inside_a_group_draws_classes_by_count_power():
    labels = [0] * 90 + [1] * 10 + [0] * 5
    groups = ["a"] * 100 + ["b"] * 5
    draws = [i for b in balanced_batches(labels, None, 100, 100, seed=0, epoch=1, class_power=1.0, groups=groups,
                                         group_quota={"a": 0.5, "b": 0.5}) for i in b]
    a = [labels[i] for i in draws if groups[i] == "a"]
    assert abs(a.count(0) / len(a) - 0.9) < 0.03  # power 1: natural frequency inside the group
    with pytest.raises(ValueError):
        next(balanced_batches(labels, None, 4, 1, seed=0, epoch=1, group_quota={"a": 1.0}))  # no groups given
    with pytest.raises(ValueError):
        next(balanced_batches(labels, None, 4, 1, seed=0, epoch=1, groups=groups, group_quota={"c": 1.0}))


# --------------------------------------------------------------------------------------------- 2. parsing

def test_parse_and_resolve_source_quota():
    assert parse_source_quota(None) is None
    for off in ("", "none", "None", "off"):
        assert parse_source_quota(off) is None
    assert parse_source_quota("default") == DEFAULT_SOURCE_QUOTA
    q = parse_source_quota("include=0.45, isl500=0.35,dictionary=0.12,mendeley=0.08")
    assert q == DEFAULT_SOURCE_QUOTA
    for bad in ("include", "include=x", "include=-1", "include=0", "incldue=1", "include=1,include=2", "include=nan"):
        with pytest.raises(ValueError):
            parse_source_quota(bad)
    # an exact source name wins over its family
    q = {"islrtc": 0.2, "dictionary": 0.1, "include": 0.7}
    assert quota_group("islrtc", q) == "islrtc" and quota_group("wslp", q) == "dictionary"
    assert quota_group("mendeley_emergency", q) is None
    # renormalised over the sources present (commercial-safe has no ISL500 and no WSLP)
    g, s = resolve_source_quota(DEFAULT_SOURCE_QUOTA, {"include", "islrtc", "mendeley_emergency", "mendeley_agri"})
    assert s == pytest.approx({"include": 0.45 / 0.65, "dictionary": 0.12 / 0.65, "mendeley": 0.08 / 0.65})
    with pytest.raises(ValueError, match="no share"):  # a present source without a group is never silently dropped
        resolve_source_quota({"include": 1.0}, {"include", "isl500"})
    with pytest.raises(ValueError, match="share 0"):
        resolve_source_quota({"include": 0.0, "isl500": 1.0}, {"include"})


def test_dictionary_family_matches_common_and_needs_no_torch():
    from app.isl.training.common import DICTIONARY_SOURCES
    from app.isl.training.source_quota import DICTIONARY_FAMILY, check_source_quota

    assert DICTIONARY_FAMILY == DICTIONARY_SOURCES
    assert check_source_quota("default", ("include", "isl500", "wslp")) is None
    assert "unknown" in check_source_quota("incldue=1", ("include",))
    # the driver's pre-flight runs with any python: the module imports no torch
    r = subprocess.run([sys.executable, "-c", "import sys, app.isl.training.source_quota, app.isl.profiles; "
                        "sys.exit(int('torch' in sys.modules))"], cwd=REPO / "backend", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_cli_refuses_a_bad_quota():
    import argparse

    from app.isl.cli import _source_quota_arg

    assert _source_quota_arg("default") == DEFAULT_SOURCE_QUOTA
    assert _source_quota_arg("none") is None
    with pytest.raises(argparse.ArgumentTypeError):
        _source_quota_arg("include=0.5,isl5000=0.5")


def test_source_balanced_temperature_weights():
    from app.isl.models.calibration import fit_temperature
    from app.isl.training.train_classifier import source_balanced_weights

    src = np.array(["isl500"] * 90 + ["include"] * 30 + ["mendeley_emergency"] * 5)
    w = source_balanced_weights(src)  # no quota: the research default shares weigh the quota groups
    tot = {s: w[src == s].sum() for s in ("isl500", "include", "mendeley_emergency")}
    norm = 0.45 + 0.35 + 0.08 * 5 / 20
    assert sum(tot.values()) == pytest.approx(1.0)
    assert tot["include"] == pytest.approx(0.45 / norm) and tot["isl500"] == pytest.approx(0.35 / norm)
    assert tot["mendeley_emergency"] == pytest.approx(0.08 * 5 / 20 / norm)
    # the review case (docs 15.39): ISL500 12,261, INCLUDE 856, Mendeley 46 validation clips, WSLP / CISLR too.
    # The families count once (dictionary, Mendeley), so a 46-clip niche source never weighs like INCLUDE and the
    # dictionary never gets two votes; with a quota its shares weigh the groups
    big = np.array(["isl500"] * 12261 + ["include"] * 856 + ["mendeley_emergency"] * 46 + ["wslp"] * 30
                   + ["cislr"] * 30 + ["mendeley_agri"] * 10)
    for q in (None, {"include": 0.6, "isl500": 0.2, "dictionary": 0.1, "mendeley": 0.1}):
        wb = source_balanced_weights(big, q)
        qq = q or DEFAULT_SOURCE_QUOTA
        t = {s: wb[big == s].sum() for s in set(big.tolist())}
        assert t["include"] / t["isl500"] == pytest.approx(qq["include"] / qq["isl500"])
        assert t["wslp"] + t["cislr"] == pytest.approx(t["include"] * qq["dictionary"] / qq["include"])
        assert t["mendeley_emergency"] + t["mendeley_agri"] == pytest.approx(
            t["include"] * qq["mendeley"] / qq["include"])
        assert t["mendeley_emergency"] < t["include"] / 3
    # every group present with share 0: plain weights, never a division by zero
    assert source_balanced_weights(np.array(["include"] * 4), {"include": 0.0, "isl500": 1.0}) == \
        pytest.approx(np.full(4, 0.25))
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(125, 6)) * 3
    y = rng.integers(0, 6, 125)
    assert fit_temperature(lg, y, weights=np.ones(125)) == fit_temperature(lg, y)  # uniform = the old fit
    # ISL500 rows are confidently right, INCLUDE rows confidently wrong: balancing raises T
    lg2 = np.zeros((120, 4))
    y2 = np.zeros(120, int)
    lg2[:90, 0] = 4.0
    lg2[90:, 1] = 4.0
    s2 = np.array(["isl500"] * 90 + ["include"] * 30)
    assert fit_temperature(lg2, y2, weights=source_balanced_weights(s2)) > fit_temperature(lg2, y2)


def test_validation_row_falls_back_to_macro_f1_without_a_major_source():
    from app.isl.training.train_classifier import per_source_metrics, validation_row

    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    src = np.array(["mendeley_emergency"] * 3 + ["mendeley_agri"] * 3)
    assert per_source_metrics(y, pred, src)["val_worst_major_f1"] is None
    row = validation_row(y, pred, src)
    assert row["val_worst_major_f1"] == row["val_macro_f1"]
    assert {"val_f1_mendeley_emergency", "val_f1_mendeley_agri", "val_acc_mendeley_agri"} <= set(row)
    src2 = np.array(["include"] * 3 + ["isl500"] * 3)
    row2 = validation_row(y, pred, src2)
    assert row2["val_worst_major_f1"] == min(row2["val_f1_include"], row2["val_f1_isl500"])


# --------------------------------------------------------------------------------------------- 3. classifier

@pytest.fixture(scope="module")
def isl500_world(tmp_path_factory):
    """The toy world plus ISL500-like clips (other signers of the same words) in train and val."""
    from app.dataset.synthetic import make_sequence
    from app.isl.data.sources import Clip
    from app.isl.training.common import load_isl_data
    from app.isl.training.multisource import GlossLinker
    from app.isl.vocab.build import VocabIndex
    from tests.isl.isl_world import CORE, concept, make_world

    root = tmp_path_factory.mktemp("isl500q")
    w = make_world(root)
    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(w["paths"], vocab, "research", w["split"])
    data.linker = GlossLinker.from_vocab(vocab)
    d = root / "isl500"
    d.mkdir()
    extra, assign = [], {}
    for ci, word in enumerate(CORE):
        for s in range(6):
            raw = make_sequence(ci, signer=20 + s, n_frames=40, fps=12.0, n_classes=6)
            p = d / f"{word}_{s}.npz"
            raw.save(p)
            cid = f"isl500/{word}/{s}"
            extra.append(Clip(cid, "isl500", word, concept(word), None, None, p, group=f"s{s}",
                              quality={"n_frames": 40}))
            assign[cid] = "train" if s < 4 else "val"
    data.extra = extra
    data.extra_assign = assign
    return data


def test_classifier_selects_on_the_worst_major_source_and_records_the_quota(isl500_world, tmp_path):
    import torch

    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_classifier import train_classifier
    from app.training.trainer import TrainConfig

    cfg = TrainConfig(hidden=16, layers=1, epochs=4, batch_size=8, patience=10, lr=3e-3)
    out = tmp_path / "cls.pt"

    def run(**kw):
        return train_classifier(isl500_world, out, ISL_FEATURES, cfg, smoke=True, threads=1, steps_per_epoch=3,
                                use_extra=True, **kw)

    import app.isl.training.train_classifier as tc

    seen = []
    real_fit = tc.fit_temperature

    def spy(logits, y, grid=None, weights=None):
        seen.append((logits.copy(), y.copy(), None if weights is None else weights.copy()))
        return real_fit(logits, y, grid, weights=weights)

    tc.fit_temperature = spy
    try:
        res = run(source_quota="default", select_on="worst_source")
    finally:
        tc.fit_temperature = real_fit
    assert res["select_key"] == "val_worst_major_f1"
    # the temperature really is the source-balanced fit of the best checkpoint's validation logits
    (lg, vy, tw), = seen
    assert tw is not None and tw.sum() == pytest.approx(1.0)
    assert res["temperature"] == real_fit(lg, vy, weights=tw)
    meta = torch.load(out, map_location="cpu", weights_only=False)["meta"]
    q = meta["source_quota"]
    assert q["requested"] == DEFAULT_SOURCE_QUOTA
    assert set(q["shares"]) == {"include", "isl500", "dictionary"}  # no Mendeley clip: renormalised
    assert q["shares"]["include"] == pytest.approx(0.45 / 0.92)
    assert meta["select_on"] == "worst_source" and meta["select_key"] == "val_worst_major_f1"
    assert meta["temperature_fit"] == "source_balanced"
    assert set(meta["val_by_source"]) == {"include", "isl500"}
    wbs = meta["temperature_weight_by_source"]
    assert set(wbs) == {"include", "isl500"} and sum(wbs.values()) == pytest.approx(1.0)
    nv = meta["data"]["val_sources"]  # weighted by the quota shares, a group under 20 clips by n / 20 of it
    assert wbs["include"] / wbs["isl500"] == pytest.approx(
        0.45 * min(1, nv["include"] / 20) / (0.35 * min(1, nv["isl500"] / 20)), rel=1e-4)
    shares = meta["data"]["expected_draw_share"]
    assert shares["include"] == pytest.approx(q["shares"]["include"], abs=1e-6)
    hist = meta["history"]
    assert len(hist) == 4
    for row in hist:  # per-source history, and the key is the minimum of the major sources
        assert row["val_worst_major_f1"] == min(row["val_f1_include"], row["val_f1_isl500"])
        assert {"val_acc_include", "val_acc_isl500", "val_macro_f1"} <= set(row)
    keys = [r["val_worst_major_f1"] for r in hist]
    assert res["best_epoch"] == 1 + int(np.argmax(keys))  # selection used that key (first maximum)
    assert meta["best_epoch"] == res["best_epoch"]
    # every new option is in the resume signature
    assert run(source_quota="default", select_on="worst_source", resume=True)["finished"]
    for kw in ({"source_quota": "include=0.6,isl500=0.2,dictionary=0.2", "select_on": "worst_source"},
               {"source_quota": None, "select_on": "worst_source"},
               {"source_quota": "default", "select_on": "macro_f1"},
               {"source_quota": "default", "select_on": "worst_source", "temperature_fit": "all"}):
        with pytest.raises(StateMismatch):
            run(resume=True, **kw)


def test_classifier_default_is_unchanged_but_logs_per_source(isl500_world, tmp_path):
    import torch

    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.train_classifier import train_classifier
    from app.training.trainer import TrainConfig

    cfg = TrainConfig(hidden=16, layers=1, epochs=2, batch_size=8, patience=10, lr=3e-3)
    res = train_classifier(isl500_world, tmp_path / "c.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                           steps_per_epoch=2, use_extra=True)
    meta = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=False)["meta"]
    assert res["select_key"] == "val_macro_f1" and meta["source_quota"] is None
    assert meta["temperature_fit"] == "all"
    assert meta["temperature_weight_by_source"] == pytest.approx(
        {s: n / sum(meta["data"]["val_sources"].values()) for s, n in meta["data"]["val_sources"].items()})
    assert res["best_epoch"] == 1 + int(np.argmax([r["val_macro_f1"] for r in meta["history"]]))
    assert "val_f1_isl500" in meta["history"][0]
    with pytest.raises(ValueError):
        train_classifier(isl500_world, tmp_path / "d.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                         steps_per_epoch=1, select_on="best")
    with pytest.raises(ValueError):
        train_classifier(isl500_world, tmp_path / "d.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                         steps_per_epoch=1, temperature_fit="weighted")
    # the temperature fit is its own option: source-balanced with macro-F1 selection
    res = train_classifier(isl500_world, tmp_path / "e.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                           steps_per_epoch=2, use_extra=True, temperature_fit="source_balanced")
    meta = torch.load(tmp_path / "e.pt", map_location="cpu", weights_only=False)["meta"]
    assert res["temperature_fit"] == meta["temperature_fit"] == "source_balanced"
    assert meta["select_key"] == "val_macro_f1"
    with pytest.raises(ValueError, match="no share"):  # ISL500 is training data but the quota forgot it
        train_classifier(isl500_world, tmp_path / "d.pt", ISL_FEATURES, cfg, smoke=True, threads=1,
                         steps_per_epoch=1, use_extra=True, source_quota="include=0.5,dictionary=0.5")


# --------------------------------------------------------------------------------------------- 4. driver

def _driver(tmp_path, profile, *args, env_extra=None, py=None):
    from tests.isl.test_phase2_guards import _run_driver

    import os

    old = {k: os.environ.get(k) for k in ("CLS_SOURCE_QUOTA", "CLS_SELECT", "CLS_STEPS", "CLS_PATIENCE")}
    try:
        for k in old:
            os.environ.pop(k, None)
        os.environ.update(env_extra or {})
        return _run_driver(tmp_path, *args, work_base=tmp_path / "art", isl_root=tmp_path / "isl", profile=profile,
                           py=py)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_driver_defaults_research_on_commercial_safe_off(tmp_path):
    r = _driver(tmp_path, "research", "--from", "classifier", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("classifier: source quota=include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08, "
            "selection=worst_source") in r.stdout
    assert "--batches 800 --patience 8" in r.stdout
    r = _driver(tmp_path, "commercial-safe", "--from", "classifier", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "classifier: source quota=none, selection=macro_f1" in r.stdout
    assert "--batches 300 --patience 10" in r.stdout
    r = _driver(tmp_path, "research", "--dry-run", env_extra={"CLS_SOURCE_QUOTA": "", "CLS_SELECT": "macro_f1",
                                                              "CLS_STEPS": "300", "CLS_PATIENCE": "10"})
    assert "classifier: source quota=none, selection=macro_f1" in r.stdout and "--batches 300 --patience 10" in r.stdout
    assert _driver(tmp_path, "research", "--dry-run", env_extra={"CLS_SELECT": "best"}).returncode == 2
    assert _driver(tmp_path, "research", "--dry-run", env_extra={"CLS_SOURCE_QUOTA": "include"}).returncode == 2


def test_driver_passes_the_quota_to_classifier_and_cv_not_e1a(tmp_path):
    from tests.isl.test_phase2_guards import STEPS_DONE

    work = tmp_path / "art" / "research"
    stamps = work / "driver" / "phase2_research.stamps"
    stamps.mkdir(parents=True)
    for s in STEPS_DONE:
        (stamps / f"{s}.done").write_text("old\n")
    r = _driver(tmp_path, "research", "--force", "classifier,e1a")
    assert r.returncode == 0, r.stdout + r.stderr
    calls = [c for c in (tmp_path / "fake_py.log").read_text().splitlines() if " train-classifier " in c]
    main = [c for c in calls if "e1a" not in c]
    e1a = [c for c in calls if "e1a" in c]
    assert main and e1a
    q = "--source-quota include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08 --select-on worst_source"
    assert all(q in c and "--batches 800 --patience 8" in c for c in main), main
    assert all("--source-quota" not in c and "--select-on" not in c for c in e1a), e1a
    assert all("--patience 10" in c.split("--patience 8", 1)[-1] for c in e1a)  # e1a keeps patience 10 (last wins)
    assert "quota=include=0.45" in (stamps / "classifier.done").read_text()
    assert "select=worst_source" in (stamps / "classifier.done").read_text()


def test_driver_checks_the_quota_before_anything_runs(tmp_path):
    """A typo or a source left out is refused by the pre-flight (exit 2), not hours later by the classifier step."""
    py = REPO / ".venv" / "bin" / "python"
    if not py.exists():
        pytest.skip("no project venv")
    ok = _driver(tmp_path, "research", "--from", "classifier", "--dry-run", py=py)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    for bad, why in (("incldue=1", "unknown source"), ("include=0.5,isl500=0.5", "no share"),
                     ("include=0,isl500=0,dictionary=0,mendeley=0", "at least one share")):
        r = _driver(tmp_path, "research", "--from", "classifier", "--dry-run", env_extra={"CLS_SOURCE_QUOTA": bad},
                    py=py)
        assert r.returncode == 2, (bad, r.stdout, r.stderr)
        assert why in r.stderr and "nothing was run" in r.stderr, r.stderr
        assert "would run classifier" not in r.stdout
    # commercial-safe has no ISL500: the research default is fine there (renormalised); default = that quota
    r = _driver(tmp_path, "commercial-safe", "--from", "classifier", "--dry-run",
                env_extra={"CLS_SOURCE_QUOTA": "default"}, py=py)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "classifier: source quota=include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08," in r.stdout
    # not checked when neither the classifier nor cv runs
    r = _driver(tmp_path, "research", "--from", "fingerspell", "--dry-run", env_extra={"CLS_SOURCE_QUOTA": "incldue=1"},
                py=py)
    assert r.returncode == 0, r.stdout + r.stderr


def test_driver_warns_on_a_classifier_stamp_of_other_settings(tmp_path):
    from tests.isl.test_phase2_guards import STEPS_DONE

    work = tmp_path / "art" / "research"
    stamps = work / "driver" / "phase2_research.stamps"
    stamps.mkdir(parents=True)
    for s in STEPS_DONE:
        (stamps / f"{s}.done").write_text("2026-11-20T00:00:00Z 10s dict_group=1\n")  # a stamp from before 15.38
    r = _driver(tmp_path, "research", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    warn = [ln for ln in r.stdout.splitlines() if ln.startswith("warning: ")]
    assert any(ln.startswith("warning: classifier stamp settings differ") and "quota=include=0.45" in ln
               and "--from classifier" in ln for ln in warn), r.stdout
    assert any(ln.startswith("warning: cv stamp settings differ") for ln in warn)
    assert not any(ln.startswith(("warning: e1a", "warning: encoder")) for ln in warn)
    cur = " dict_group=1 quota=include=0.45,isl500=0.35,dictionary=0.12,mendeley=0.08 select=worst_source"
    for s in ("classifier", "cv"):
        (stamps / f"{s}.done").write_text(f"2026-11-28T00:00:00Z 10s{cur}\n")
    r = _driver(tmp_path, "research", "--dry-run")
    assert r.returncode == 0 and "stamp settings differ" not in r.stdout, r.stdout
    # a real (non-dry) resume skips with the warning in the log and runs nothing
    for s in ("classifier",):
        (stamps / f"{s}.done").write_text("old\n")
    r = _driver(tmp_path, "research")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warning: classifier stamp settings differ" in r.stdout
    assert not [c for c in (tmp_path / "fake_py.log").read_text().splitlines() if " train-classifier " in c]
