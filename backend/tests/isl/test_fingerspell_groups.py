"""Letter set: content duplicates across labels, grouped hold-out for E5 and the temperature, balanced sampling
(docs/ISL_DESIGN.md 15.12). Synthetic manifests in a temporary data root, real adapters, no network."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from app.dataset.synthetic import make_sequence
from app.isl.data.sources import Clip
from app.isl.fingerspell.model import LETTERS, NOISE
from app.isl.paths import isl_paths
from app.isl.training import train_fingerspell as tf
from app.isl.training.train_fingerspell import (
    BalancedEpochs,
    LetterSet,
    _water_fill,
    calibration_holdout,
    collect_letter_examples,
    dedupe_by_content,
    grouped_units,
    letter_data,
    train_fingerspell,
    unit_of,
)

@pytest.fixture(autouse=True)
def _small_calibration_sets(monkeypatch):
    monkeypatch.setattr(tf, "MIN_CAL_EXAMPLES", 10)  # the synthetic sets are far below the real 200


AGES, SLEEVES = ("kids", "teenagers", "adults"), ("full", "half")
CLS = {"A": 0, "B": 3, "C": 6, "3": 1, "4": 4, "5": 7, "6": 8}


def _photo(root, name: str, sym: str, person: int, k: int) -> tuple[str, str]:
    """Save a one-frame landmark file; returns (manifest path, sha256 of a fake source photo)."""
    r = make_sequence(CLS[sym], signer=person, n_frames=1, fps=12.0, rng=np.random.default_rng(1000 * person + k))
    p = root / "landmarks" / "src" / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    r.save(p)
    return f"data/isl/landmarks/src/{name}.npz", hashlib.sha256(name.encode()).hexdigest()


def _row(rid, gloss, path, sha, group=None, **kw):
    r = {"id": rid, "gloss": gloss, "path": path, "sha256": sha, "n_frames": 1, "hands_frac": 1.0, "ok": True, **kw}
    if group:
        r["group"] = group
    return r


def _write(root, name, rows):
    m = root / "manifests" / f"{name}.jsonl"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture(scope="module")
def static_root(tmp_path_factory):
    """mendeley_static: 3 age groups x 2 sleeves x letters A/B/C x 4 photos, plus
    * a photo filed under '3' and its byte-identical copy under '4' (engine duplicate_of row: same landmark file),
    * a photo under 'A' with a duplicate_of copy under 'A' (same label: kept once),
    * two separate extractions with the same SHA-256 under '5' and '6' (different landmark files)."""
    root = tmp_path_factory.mktemp("isl")
    rows, person = [], 0
    for ai, age in enumerate(AGES):
        for sl in SLEEVES:
            person = ai  # one "person" per age group, in both sleeve groups
            for sym in "ABC":
                for k in range(4):
                    name = f"{age}_{sl}_{sym}_{k}"
                    path, sha = _photo(root, name, sym, person, k + (10 if sl == "half" else 0))
                    rows.append(_row(f"isl_static/{name}", sym, path, sha, f"static-{age}-{sl}",
                                     age_group=age, sleeves=sl))
    p3, s3 = _photo(root, "dup34", "3", 0, 99)
    rows.append(_row("isl_static/dup34_orig", "3", p3, s3, "static-teenagers-full"))
    rows.append(_row("isl_static/dup34_copy", "4", p3, s3, "static-adults-full", duplicate_of="isl_static/dup34_orig"))
    pa, sa = _photo(root, "dupAA", "A", 1, 98)
    rows.append(_row("isl_static/dupAA_orig", "A", pa, sa, "static-kids-half"))
    rows.append(_row("isl_static/dupAA_copy", "A", pa, sa, "static-kids-full", duplicate_of="isl_static/dupAA_orig"))
    p5, s5 = _photo(root, "sha5", "5", 2, 97)
    p6, _ = _photo(root, "sha6", "6", 2, 96)
    rows.append(_row("isl_static/sha5", "5", p5, s5, "static-adults-half"))
    rows.append(_row("isl_static/sha6", "6", p6, s5, "static-adults-half"))  # same source photo, other label
    _write(root, "mendeley_static", rows)
    return root


def test_unit_of_kinds():
    mk = lambda src, g: Clip("x", src, "a", None, None, None, "p.npz", group=g)  # noqa: E731
    assert unit_of(mk("mendeley_static", "static-adults-half")) == ("mendeley_static:adults", "age_group",
                                                                    "mendeley_static:adults-half")
    assert unit_of(mk("isl_fingerspelling", "signer_02"))[:2] == ("isl_fingerspelling:signer_02", "signer")
    assert unit_of(mk("hf_akrit_alphabet", None)) == ("hf_akrit_alphabet", "whole_source", "hf_akrit_alphabet")
    assert unit_of(mk("openhands_fs", "openhands_fs:test/a/4"))[:2] == ("openhands_fs", "whole_source")
    assert unit_of(mk("mendeley_static", None))[1] == "whole_source"  # no group: never held out


def test_dedupe_drops_cross_label_contents_and_same_label_copies(static_root):
    ls = letter_data(isl_paths(static_root), "commercial-safe")
    d = ls.dedup
    assert d["items"] == 72 + 6 and d["cross_label_contents"] == 2 and d["cross_label_items"] == 4
    assert d["same_label_copies"] == 1 and d["kept"] == 72 + 1
    assert sorted(map(sorted, (e["labels"] for e in d["cross_label_examples"]))) == [["3", "4"], ["5", "6"]]
    # neither label of a doubly-filed photo is learnt; the same-label copy counts once
    assert {LETTERS[c] for c in ls.y} == {"A", "B", "C"} and len(ls.y) == 73
    assert ls.summary()["groups"]["mendeley_static:kids"] == {"kind": "age_group", "examples": 25}


def test_dedupe_without_duplicate_rows_still_sees_sha_and_paths():
    c = lambda i, g, path, sha=None, src="s": Clip(i, src, g, None, None, None, path,  # noqa: E731
                                                  quality={"sha256": sha} if sha else {})
    clips = [c("a", "A", "p1"), c("b", "B", "p1"), c("c", "C", "p2", "h"), c("d", "C", "p3", "h", "t"),
             c("e", "D", "p4", "h2"), c("f", "D", "p4"), c("g", "E", "p5")]
    keep, st = dedupe_by_content(clips)
    assert keep == [False, False, True, False, True, False, True]
    assert st["cross_label_items"] == 2 and st["same_label_copies"] == 2
    # a slice of a file is not the same content as another slice of it
    s1 = Clip("x1", "s", "A", None, None, None, "v.npz", start=0, end=10, quality={"sha256": "v"})
    s2 = Clip("x2", "s", "B", None, None, None, "v.npz", start=10, end=20, quality={"sha256": "v"})
    assert dedupe_by_content([s1, s2])[0] == [True, True]


def test_commercial_grouped_e5_and_calibration_by_age_group(static_root, tmp_path):
    ls = letter_data(isl_paths(static_root), "commercial-safe")
    units, why = grouped_units(ls)
    assert units == ["mendeley_static:adults", "mendeley_static:kids", "mendeley_static:teenagers"] and why is None
    r = train_fingerspell(ls, tmp_path / "l.pt", profile="commercial-safe", epochs=3, max_per_class=10)
    e5, cal = r["e5"], r["calibration"]
    assert e5["measurable"] and e5["groups"] == units and e5["signers"] == []
    assert e5["n"] == 73 and set(e5["folds"]) == set(units)
    assert e5["intervals"]["letter_top1"] is None  # 3 groups: no interval, and the reason says why
    assert e5["interval_reasons"]["letter_top1"] == "too few signer groups (3)"
    assert e5["interval_groups"]["letter_top1"] == 3
    assert "letter_top1_mendeley_static" in e5["metrics"]
    assert cal["status"] == "grouped" and cal["held_out_units"] == units and cal["n"] == 73
    assert r["temperature"] == cal["temperature"] and "ece" in cal
    # every photo was used by the final fit although each epoch takes at most 10 per class
    assert r["training"]["epoch_examples"] == 30 and r["training"]["coverage"] == 1.0
    import torch

    meta = torch.load(tmp_path / "l.pt", weights_only=False)["meta"]
    assert meta["calibration"]["status"] == "grouped" and meta["data"]["dedup"]["cross_label_items"] == 4


def test_calibration_without_e5_holds_out_one_whole_age_group(static_root, tmp_path):
    ls = letter_data(isl_paths(static_root), "commercial-safe")
    assert calibration_holdout(ls, min_examples=20) == ["mendeley_static:adults"]  # smallest with >= 20 letters
    r = train_fingerspell(ls, tmp_path / "l.pt", profile="commercial-safe", epochs=2, evaluate=False)
    assert r["calibration"]["status"] == "grouped" and r["calibration"]["n_groups"] == 1
    assert r["e5"]["measurable"] is False


def _hf_rows(root, name, n_per, splits=("train",)):
    rows = []
    for sym in "ABC":
        for k in range(n_per):
            sp = splits[k % len(splits)]
            path, sha = _photo(root, f"{name}_{sym}_{k}", sym, 5 + len(name), k)
            rows.append(_row(f"{name}/{sp}/{sym}{k}", sym, path, sha, split=sp))
    _write(root, name, rows)


def test_no_group_ids_means_not_grouped(tmp_path):
    """HF alphabets only (research profile without Mendeley / ISL-Fingerspelling): akrit's own 'test' rows are
    never used as a hold-out, so the temperature stays 1.0 and the report says 'not grouped'."""
    _hf_rows(tmp_path, "hf_akrit_alphabet", 6, splits=("train", "test"))
    _hf_rows(tmp_path, "hf_hemg_alphabet", 6)
    ls = letter_data(isl_paths(tmp_path), "research")
    assert grouped_units(ls)[0] == []
    r = train_fingerspell(ls, tmp_path / "l.pt", profile="research", epochs=2)
    assert r["e5"]["measurable"] is False and "needs at least 2" in r["e5"]["reason"]
    assert "hf_akrit_alphabet" in r["e5"]["reason"]
    assert r["calibration"]["status"] == "not grouped" and r["temperature"] == 1.0


def test_research_mix_keeps_whole_sources_on_training_side(static_root, tmp_path):
    import shutil

    root = tmp_path / "root"
    shutil.copytree(static_root, root)
    _hf_rows(root, "hf_akrit_alphabet", 4, splits=("train", "test"))
    ls = letter_data(isl_paths(root), "research")
    units, _ = grouped_units(ls)
    assert "hf_akrit_alphabet" in ls.group and "hf_akrit_alphabet" not in units and len(units) == 3
    r = train_fingerspell(ls, tmp_path / "l.pt", profile="research", epochs=2, max_per_class=12)
    assert set(r["e5"]["groups"]) == set(units)
    assert all(f["train_examples"] > 12 for f in r["e5"]["folds"].values())  # akrit is in every fold's training


def test_interval_only_from_five_groups(tmp_path):
    """Letter top-1 gets an interval at >= 5 held-out groups of one source, not below and not pooled over sources."""
    def mk(n_signers, n_ages):
        ls = LetterSet()
        rng = np.random.default_rng(0)
        for s in range(n_signers):
            u = f"isl_fingerspelling:signer_{s}"
            ls.unit_kind[u] = "signer"
            for c in (0, 1, 2, NOISE):
                ls.add(rng.normal(c, 0.3, (6, 138)), c, u, "isl_fingerspelling")
        for a in range(n_ages):
            u = f"mendeley_static:age{a}"
            ls.unit_kind[u] = "age_group"
            for c in (0, 1, 2):
                ls.add(rng.normal(c, 0.3, (6, 138)), c, u, "mendeley_static", f"{u}-full")
        return ls

    r4 = train_fingerspell(mk(2, 2), tmp_path / "a.pt", profile="research", epochs=2)
    assert r4["e5"]["intervals"]["letter_top1"] is None
    assert r4["e5"]["interval_reasons"]["letter_top1"] == "too few signer groups (4)"
    r5 = train_fingerspell(mk(2, 3), tmp_path / "b.pt", profile="research", epochs=2)
    # 5 groups, but from two sources: no pooled interval (it would measure the source mix, not people)
    assert r5["e5"]["intervals"]["letter_top1"] is None and r5["e5"]["interval_groups"]["letter_top1"] == 5
    assert r5["e5"]["interval_reasons"]["letter_top1"].startswith("pooled over 2 sources")
    # per-source metrics carry their own (smaller) group counts and reasons
    assert r5["e5"]["interval_reasons"]["letter_top1_isl_fingerspelling"] == "too few signer groups (2)"
    assert r5["calibration"]["n_groups"] == 5
    r5s = train_fingerspell(mk(5, 0), tmp_path / "c.pt", profile="research", epochs=2)  # one source, 5 signers
    assert r5s["e5"]["intervals"]["letter_top1"] is not None and r5s["e5"]["interval_groups"]["letter_top1"] == 5
    assert r5s["e5"]["intervals"]["letter_top1_isl_fingerspelling"] is not None


def test_water_fill_and_balanced_epochs_cover_every_group():
    assert _water_fill([100, 3, 50], 60) == [29, 3, 28]
    assert _water_fill([9, 9, 9], 4) == [2, 1, 1] and _water_fill([9, 9, 9], 4, start=1) == [1, 2, 1]
    assert _water_fill([5, 5], 100) == [5, 5]
    y = np.array([0] * 90 + [1] * 10)
    sub = np.array(["big"] * 80 + ["small"] * 10 + ["s1"] * 10)
    plan = BalancedEpochs(y, sub, 20, np.random.default_rng(0))
    assert plan.quota == {0: 20, 1: 10} and plan.size == 30
    for _ in range(8):
        e = plan.epoch()
        assert len(e) == 30 and (y[e] == 0).sum() == 20
        assert (sub[e] == "small").sum() == 10 and (sub[e] == "big").sum() == 10  # even split over subgroups
    assert plan.seen.all()  # every example used within 8 epochs
    again = BalancedEpochs(y, sub, 20, np.random.default_rng(0))
    first = again.epoch()
    assert np.array_equal(first, BalancedEpochs(y, sub, 20, np.random.default_rng(0)).epoch())  # deterministic
    assert len(BalancedEpochs(y, sub, None, np.random.default_rng(0)).epoch()) == 100  # no cap: all of them


def test_e5_runner_and_report_lines(static_root, tmp_path):
    from app.isl.evaluation.runners import _e5_lines, e5_from_letters

    ls = letter_data(isl_paths(static_root), "commercial-safe")
    train_fingerspell(ls, tmp_path / "letters.pt", profile="commercial-safe", epochs=2)
    p = e5_from_letters(tmp_path / "letters.pt")
    assert p["run"] and p["n_groups"] == 3 and p["calibration"]["status"] == "grouped"
    assert "leave-one-group-out" in p["label"]
    text = "\n".join(_e5_lines(p))
    assert "letter_top1: " in text and "none: too few signer groups (3)" in text
    assert "calibration: **grouped**" in text and "mendeley_static:adults" in text


def test_too_few_held_out_examples_keep_temperature_one(static_root, tmp_path, monkeypatch):
    monkeypatch.setattr(tf, "MIN_CAL_EXAMPLES", 200)
    ls = letter_data(isl_paths(static_root), "commercial-safe")
    r = train_fingerspell(ls, tmp_path / "l.pt", profile="commercial-safe", epochs=2)
    assert r["e5"]["measurable"]  # E5 still reports (with its interval reason)
    assert r["calibration"]["status"] == "too few held-out examples" and r["temperature"] == 1.0
    assert r["calibration"]["n"] == 73 and "< 200" in r["calibration"]["reason"]
