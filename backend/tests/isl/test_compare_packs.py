"""Selection gates (docs/ISL_DESIGN.md 15.44): compare-packs selects on validation / out-of-fold numbers only and
confirms once on test with paired tests; evaluate --all --per-clip writes what it pairs on; a licence-compatible
component of another data profile is loaded only when the pack manifest records it."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from app.isl.evaluation.compare_packs import (DEFAULT_MARGINS, LEDGER, Side, compare, gate_status, isl500_val_check,
                                              lower_bound_check, mcnemar_exact, paired_difference,
                                              unpaired_difference)
from tests.isl.test_eval_runners import phase2  # noqa: F401  (module fixture: a tiny trained research world)

PRIMARY = "include-session-cv5-fold0"
CV1 = "include-session-cv5-fold1"
CV4 = "include-session-cv5-fold4"
# after every file a test writes (a component newer than its report makes the report stale)
CREATED = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600))


# --------------------------------------------------------------------------------------------- statistics

def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def _ids(n, prefix="include/c"):
    return [f"{prefix}{i:04d}" for i in range(n)]


def _hits(ids, rate, offset=0):
    """Deterministic: the first ``rate`` share of every 100 clips is a hit (nested for growing rates)."""
    return {c: int(((i + offset) % 100) < round(rate * 100)) for i, c in enumerate(ids)}


def test_paired_difference_verdicts_and_group_bootstrap():
    ids = _ids(1000)
    g6 = {c: f"g{i % 6}" for i, c in enumerate(ids)}
    same = paired_difference(_hits(ids, 0.7), _hits(ids, 0.7), g6, 0.02)
    assert same["status"] == "pass" and same["delta"] == 0 and same["p_mcnemar"] == 1.0
    worse = paired_difference(_hits(ids, 0.6), _hits(ids, 0.7), g6, 0.02)
    assert worse["status"] == "fail" and worse["significantly_worse"] and worse["cand_only"] == 0
    assert worse["group_interval"] is not None and worse["n_groups"] == 6
    better = paired_difference(_hits(ids, 0.75), _hits(ids, 0.7), g6, 0.02)
    assert better["status"] == "pass" and better["significantly_better"]
    # a small sample cannot show non-inferiority: inconclusive, not pass
    few = _ids(40)
    small = paired_difference(_hits(few, 0.7, 3), _hits(few, 0.7), {c: "g" for c in few}, 0.02)
    assert small["status"] == "inconclusive"
    # 4 signer groups: no group interval, the reason is given, the clip interval decides
    g4 = {c: f"g{i % 4}" for i, c in enumerate(ids)}
    r4 = paired_difference(_hits(ids, 0.7), _hits(ids, 0.7), g4, 0.02)
    assert r4["group_interval"] is None and "too few signer groups (4)" in r4["group_interval_reason"]
    # only common clip ids are paired
    part = paired_difference(_hits(ids[:500], 0.7), _hits(ids, 0.7), g6, 0.02)
    assert part["common"] == 500
    assert paired_difference({}, _hits(ids, 0.7), g6, 0.02)["status"] == "not_available"


def test_group_bootstrap_widens_the_interval_when_groups_disagree():
    ids = _ids(1200)
    g = {c: f"g{i % 6}" for i, c in enumerate(ids)}
    ref = _hits(ids, 0.7)
    # the candidate is much better on 3 groups and much worse on the other 3: mean difference near 0
    cand = {c: (1 if int(g[c][1:]) < 3 else 0) for c in ids}
    r = paired_difference(cand, ref, g, 0.02)
    lo_clip, hi_clip = r["clip_interval"]
    assert r["group_interval"][0] < lo_clip and r["interval"][0] == min(lo_clip, r["group_interval"][0])


def test_unpaired_fallback():
    r = unpaired_difference((0.70, 800), (0.70, 800), 0.02)
    assert r["paired"] is False and r["status"] == "inconclusive" and "per-clip" in r["note"]
    assert unpaired_difference((0.50, 800), (0.70, 800), 0.02)["status"] == "fail"
    assert unpaired_difference((None, 0), (0.7, 10), 0.02)["status"] == "not_available"


# --------------------------------------------------------------------------------------------- synthetic work folders

def _rec(hits: dict, groups=None, sources=None, parts=None) -> dict:
    ids = list(hits)
    out = {"clip_ids": ids, "hit1": "".join(str(hits[c]) for c in ids),
           "groups": [groups(c) if groups else f"g{i % 6}" for i, c in enumerate(ids)],
           "sources": [sources(c) if sources else c.split("/")[0] for c in ids]}
    if parts is not None:
        out["parts"] = [parts(c) for c in ids]
    return out


def _fold_ids(split: str) -> list[str]:
    return _ids(600, "include/v" if split == CV1 else f"include/f{split[-1]}_")


def _splits(root: Path, fold4_selects_on_test: bool = True) -> Path:
    """Split files of the synthetic world: the primary test is include/t*; fold 1 trains on it (disclosed, kept); fold
    4's validation part IS the primary test (as in the real cv5 splits), so its classifier was selected on it."""
    d = root / "splits"
    if d.is_dir():
        return d
    d.mkdir(parents=True)
    t = _ids(600, "include/t")

    def write(name, assign):
        (d / f"{name}.json").write_text(json.dumps({"name": name, "assign": assign}))

    write(PRIMARY, {**{c: "test" for c in t}, **{c: "val" for c in _fold_ids(CV1)}})
    write(CV1, {**{c: "test" for c in _fold_ids(CV1)}, **{c: "train" for c in t}})
    write(CV4, {**{c: "test" for c in _fold_ids(CV4)},
                **{c: ("val" if fold4_selects_on_test else "train") for c in t}})
    return d


def _work(root: Path, name: str, *, cv: float, e1b: float, lb: float, test_strict_prec: float = 0.95,
          e2_val: float = 0.1, e2_test: float = 0.1, seen: float = 0.4, isl500: float | None = None,
          per_clip: bool = True, folds: dict | None = None) -> Path:
    """``folds``: {split: top-1} of the CV folds besides the primary (default {CV1: cv})."""
    w = root / name
    (w / "eval").mkdir(parents=True)
    folds = {CV1: cv} if folds is None else folds
    test_ids = _ids(600, "include/t")
    e2_ids = _ids(300, "include/o")
    e2h = {**_hits(e2_ids[:150], e2_val), **_hits(e2_ids[150:], e2_test)}
    isl_ids = _ids(600, "isl500/user005/x") if isl500 is not None else []
    e1c = {**_hits(isl_ids, isl500 or 0)}
    e1b_h, seen_h = _hits(test_ids, e1b), _hits(test_ids, seen)
    cv_h = {f: _hits(_fold_ids(f), r) for f, r in folds.items()}
    pr = {"E1b": {"split": PRIMARY, "n": 600, "metrics": {"top1": e1b}, "clip_ids": test_ids,
                  "per_clip": {"hit1": "".join(str(e1b_h[c]) for c in test_ids)},
                  "cv": {"pooled_top1": (e1b + sum(folds.values())) / (1 + len(folds)),
                         "pooled_n": 600 * (1 + len(folds)),
                         "per_fold": [{"split": "E1b core words ...", "n": 600, "top1": e1b}]
                         + [{"split": f, "n": 600, "top1": r} for f, r in folds.items()]}},
          "E1c": {"n": len(isl_ids), "metrics": {"top1": isl500}, "clip_ids": isl_ids,
                  "per_clip": {"hit1": "".join(str(e1c[c]) for c in isl_ids)}} if isl_ids else {"run": False},
          "E2_seen": {"n": 600, "metrics": {"top1": seen}},
          "E2": {"n": 300, "metrics": {"top1": (e2_val + e2_test) / 2}},
          "E3": {"policies": {"strict": {"by_source": {"include": {"threshold_precision": test_strict_prec,
                                                                   "threshold_n_accepted": 200}}}},
                 "top1_by_source": {"isl500": {"n": len(isl_ids), "top1": isl500}} if isl_ids else {}}}
    report = {"created": CREATED, "protocols": pr}
    (w / "eval" / "report.json").write_text(json.dumps(report))
    if per_clip:
        pc = {"E1b": _rec(e1b_h), "E1b_cv": {f: _rec(h) for f, h in cv_h.items()}, "E2_seen": _rec(seen_h),
              "E2": _rec(e2h, parts=lambda c: "val" if int(c[-4:]) < 150 else "test"),
              "E3": _rec({**e1b_h, **e1c})}
        if isl_ids:
            pc["E1c"] = _rec(e1c, groups=lambda c: c.split("/")[1])
        (w / "eval" / "per_clip.json").write_text(json.dumps({"schema": 1, "report_created": CREATED,
                                                              "protocols": pc}))
    fusion = {"achieved": {"strict": {"precision_lower_bound": lb, "worst_source": "include", "rule": "confirm_only",
                                      "by_source": {"include": {"lower_bound": lb, "major": True, "n": 856}}}},
              "certification": {"calibration_clips": 856}}
    (w / "fusion.json").write_text(json.dumps(fusion))
    return w


def test_selection_uses_validation_only_and_confirms_once(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    valgood = _work(tmp_path, "valgood", cv=0.75, e1b=0.50, lb=0.86, test_strict_prec=0.80)
    testgood = _work(tmp_path, "testgood", cv=0.55, e1b=0.90, lb=0.80)
    out = tmp_path / "cmp"
    doc = compare(ref, {"valgood": valgood, "testgood": testgood}, splits_dir=_splits(tmp_path), out=out)
    c = doc["candidates"]
    # the candidate with the better TEST numbers is rejected on validation and its test numbers are never looked at
    assert doc["selected"] == "valgood"
    assert c["testgood"]["selection"]["gates"]["a"] == "fail" and c["testgood"]["selection"]["gates"]["c"] == "fail"
    assert "confirmation" not in c["testgood"] and "rejected on validation" in c["testgood"]["verdict"]
    sel = {x["check"]: x for x in c["valgood"]["selection"]["checks"]}
    assert all(x["stage"] == "validation" for x in sel.values())
    cvchk = sel["CV out-of-fold top-1 (folds other than the primary)"]
    assert cvchk["paired"] and cvchk["common"] == 600 and cvchk["folds"] == [CV1]  # the primary fold is never used
    assert sel["E2 one-shot top-1, validation-part queries"]["common"] == 150
    # one confirmation on test: E1b is much worse (paired, McNemar) and the strict certificate does not hold
    cf = c["valgood"]["confirmation"]
    assert cf["gates"]["a"] == "fail" and cf["gates"]["c"] == "fail" and cf["ship"] is False
    e1 = next(x for x in cf["checks"] if x["check"].startswith("E1b top-1"))
    assert e1["paired"] and e1["p_mcnemar"] < 1e-6 and e1["common"] == 600
    e2t = next(x for x in cf["checks"] if x["check"].startswith("E2 one-shot"))
    assert e2t["common"] == 150  # test-part queries only
    assert "NOT confirmed" in c["valgood"]["verdict"] and cf["test_looks_before"] == 0 and cf["warning"] is None
    led = (out / LEDGER).read_text().splitlines()
    assert len(led) == 1 and json.loads(led[0])["candidate"] == "valgood"
    assert (out / "compare.json").exists() and "Confirmation (test, once)" in (out / "compare.md").read_text()
    # a second look from the same folder is reported
    doc2 = compare(ref, {"valgood": valgood, "testgood": testgood}, splits_dir=_splits(tmp_path), out=out)
    cf2 = doc2["candidates"]["valgood"]["confirmation"]
    assert cf2["test_looks_before"] == 1 and "no longer a single confirmation" in cf2["warning"]
    assert len((out / LEDGER).read_text().splitlines()) == 2


def test_select_only_never_looks_at_test(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    cand = _work(tmp_path, "cand", cv=0.71, e1b=0.70, lb=0.85)
    out = tmp_path / "cmp"
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=out, select_only=True)
    assert doc["selected"] == "cand" and "confirmation" not in doc["candidates"]["cand"]
    assert "test not looked at" in doc["candidates"]["cand"]["verdict"] and not (out / LEDGER).exists()


def test_confirmed_candidate_ships_and_no_candidate_falls_back(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    good = _work(tmp_path, "good", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    doc = compare(ref, {"good": good}, splits_dir=_splits(tmp_path), out=tmp_path / "c1")
    cf = doc["candidates"]["good"]["confirmation"]
    assert cf["gates"] == {"a": "pass", "b": "pass", "c": "pass", "d": "not_applicable"} and cf["ship"] is True
    assert doc["candidates"]["good"]["stale"] == [] and "confirmed on test: ship" in doc["verdict"]
    bad = _work(tmp_path, "bad", cv=0.50, e1b=0.74, lb=0.86)
    doc = compare(ref, {"bad": bad}, splits_dir=_splits(tmp_path), out=tmp_path / "c2")
    assert doc["selected"] is None and "commercial-safe's encoder and classifier" in doc["verdict"]
    assert not (tmp_path / "c2" / LEDGER).exists()


def test_isl500_gate_against_the_baseline(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    base = _work(tmp_path, "base", cv=0.60, e1b=0.60, lb=0.80, isl500=0.88)
    worse = _work(tmp_path, "worse", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12, isl500=0.80)
    doc = compare(ref, {"worse": worse}, baseline=base, splits_dir=_splits(tmp_path), out=tmp_path / "c")
    cf = doc["candidates"]["worse"]["confirmation"]
    d = [x for x in cf["checks"] if x["gate"] == "d" and x["role"] == "gate"]
    assert {x["check"] for x in d} == {"ISL500 test top-1 (classifier alone)", "ISL500 test top-1 (fused)"}
    assert all(x["status"] == "fail" and x["paired"] for x in d) and cf["ship"] is False
    # ISL500 groups are its users (2 test signers): no group interval, said so
    assert d[0]["group_interval"] is None and d[0]["n_groups"] == 1
    # a candidate without ISL500 clips: gate (d) does not apply
    plain = _work(tmp_path, "plain", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    doc = compare(ref, {"plain": plain}, baseline=base, splits_dir=_splits(tmp_path), out=tmp_path / "c2")
    assert doc["candidates"]["plain"]["confirmation"]["gates"]["d"] == "not_applicable"


def test_unpaired_fallback_and_stale_inputs(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84, per_clip=False)
    cand = _work(tmp_path, "cand", cv=0.74, e1b=0.74, lb=0.86, per_clip=False)
    # E1b still pairs on report.json's clip ids and hit bits; E2_seen falls back to the two rates
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c1")
    ch = {x["check"]: x for x in doc["candidates"]["cand"]["confirmation"]["checks"]}
    assert ch["E1b top-1 on the common test clips"]["paired"] is True
    assert ch["E2_seen top-1 (INCLUDE test -> ISLRTC rows)"]["paired"] is False
    sel = {x["check"]: x for x in doc["candidates"]["cand"]["selection"]["checks"]}
    assert sel["CV out-of-fold top-1 (folds other than the primary)"]["paired"] is False
    # a per-clip file of another evaluate run is ignored; a report of another encoder.pt is stale and never ships
    good = _work(tmp_path, "good", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    pc = json.loads((good / "eval" / "per_clip.json").read_text())
    (good / "eval" / "per_clip.json").write_text(json.dumps({**pc, "report_created": "2026-01-01T00:00:00Z"}))
    s = Side("good", good)
    assert s.per_clip is None and any("another evaluate run" in n for n in s.notes)
    rep = json.loads((good / "eval" / "report.json").read_text())
    rep["leakage_guards"] = {"index": {"encoder_sha256": "0" * 64}}
    (good / "eval" / "report.json").write_text(json.dumps(rep))
    (good / "encoder.pt").write_bytes(b"not the encoder the report measured")
    (good / "eval" / "per_clip.json").write_text(json.dumps(pc))
    doc = compare(ref, {"good": good}, splits_dir=_splits(tmp_path), out=tmp_path / "c2")
    e = doc["candidates"]["good"]
    # a stale candidate is not selected: no confirmation, no ledger line (its test numbers are of another model)
    assert e["stale"] == ["report.json measured another encoder.pt"] and not e["selection"]["eligible"]
    assert doc["selected"] is None and "confirmation" not in e and "stale inputs" in e["verdict"]
    assert not (tmp_path / "c2" / LEDGER).exists() and any("STALE" in n for n in e["notes"])
    # a stale reference: selection runs, but stage 2 does not (no test look, no ledger line)
    doc = compare(good, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c3")
    assert doc["selected"] == "cand" and "confirmation" not in doc["candidates"]["cand"]
    assert "reference is stale" in doc["verdict"] and not (tmp_path / "c3" / LEDGER).exists()
    # a stale ISL500 baseline blocks stage 2 too
    doc = compare(ref, {"cand": cand}, baseline=good, splits_dir=_splits(tmp_path), out=tmp_path / "c4")
    assert "ISL500 baseline is stale" in doc["verdict"] and not (tmp_path / "c4" / LEDGER).exists()
    # a component newer than its report is stale too
    rep["created"] = "2020-01-01T00:00:00Z"
    (good / "eval" / "report.json").write_text(json.dumps(rep))
    assert "fusion.json is newer than report.json" in Side("good", good).stale


def test_gate_status_takes_the_worst_required_check():
    checks = [{"gate": "a", "role": "gate", "status": "pass"}, {"gate": "a", "role": "gate", "status": "inconclusive"},
              {"gate": "a", "role": "info", "status": "fail"}, {"gate": "d", "role": "gate", "status": "not_applicable"}]
    g = gate_status(checks)
    assert g["a"] == "inconclusive" and g["d"] == "not_applicable" and g["b"] == "not_available"
    assert DEFAULT_MARGINS["top1"] == 0.02


def test_cli_compare_packs(tmp_path, capsys):
    from app.isl.cli import main

    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    cand = _work(tmp_path, "cand", cv=0.74, e1b=0.74, lb=0.86)
    out = tmp_path / "cmp"
    assert main(["compare-packs", "--reference", str(ref), "--candidate", f"cand={cand}", "--out", str(out),
                 "--select-only", "--margin-top1", "0.03"]) == 0
    doc = json.loads((out / "compare.json").read_text())
    assert doc["selected"] == "cand" and doc["margins"]["top1"] == 0.03
    assert main(["compare-packs", "--reference", str(ref), "--candidate", "nope", "--out", str(out)]) == 2
    assert main(["compare-packs", "--reference", str(ref), "--out", str(out)]) == 2
    assert main(["compare-packs", "--reference", str(tmp_path / "missing"), "--candidate", f"c={cand}"]) == 2


# --------------------------------------------------------------------------------------------- review fixes (15.45)

def test_a_fold_selected_on_the_primary_test_never_serves_selection(tmp_path):
    folds = {CV1: 0.70, CV4: 0.70}
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84, folds=folds)
    # the candidate is much better on fold 4 only: the fold whose classifier was early-stopped on the E1b test
    cand = _work(tmp_path, "cand", cv=0.70, e1b=0.70, lb=0.85, folds={CV1: 0.70, CV4: 0.95})
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c", select_only=True)
    chk = next(x for x in doc["candidates"]["cand"]["selection"]["checks"] if x["check"].startswith("CV out-of-fold"))
    assert chk["folds"] == [CV1] and chk["common"] == 600 and chk["delta"] == 0
    sk = chk["folds_skipped"]
    assert [r["split"] for r in sk] == [CV4] and sk[0]["selected_on_primary_test_clips"] == 600
    assert "early-stopped on the E1b test" in sk[0]["reason"] and CV4 in chk["note"]
    # fold 1 trained on E1b test clips: kept (training data, not selection) and disclosed
    assert chk["folds_disclosed"][0]["split"] == CV1 and chk["folds_disclosed"][0]["trained_on_primary_test_clips"] == 600
    # one signer group in two folds is one cluster: 6 real groups, not 6 per fold
    assert chk["n_groups"] == 6
    # only behind the explicit option, recorded
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c2", select_only=True,
                  allow_test_selected_folds=True)
    chk = next(x for x in doc["candidates"]["cand"]["selection"]["checks"] if x["check"].startswith("CV out-of-fold"))
    assert chk["folds"] == [CV1, CV4] and chk["allow_test_selected_folds"] and chk["delta"] > 0
    assert doc["options"]["allow_test_selected_folds"] is True
    assert "USED although selected" in chk["folds_disclosed"][1]["note"]
    # without split files nothing can be verified: no fold serves selection (fail closed)
    doc = compare(ref, {"cand": cand}, out=tmp_path / "c3", select_only=True)
    chk = next(x for x in doc["candidates"]["cand"]["selection"]["checks"] if x["check"].startswith("CV out-of-fold"))
    assert chk["status"] == "not_available" and "cannot verify" in chk["reason"] and chk["folds"] == []


def _probe(work: Path, hits: dict, *, same=None, protocol="mixed"):
    (work / "eval" / "encoder_probe.json").write_text(json.dumps({
        "hits": {f"include|{k}": v for k, v in hits.items()}, "profile": "research", "split": PRIMARY,
        "per_signer": 300, "same_sets_as_training": same, "trained_val": {"val_protocol": protocol}}))


def test_probe_of_an_encoder_selected_on_its_queries_is_information(tmp_path):
    q = _ids(800, "include/q")
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    cand = _work(tmp_path, "cand", cv=0.71, e1b=0.70, lb=0.85, e2_val=0.02)
    _probe(ref, _hits(q, 0.5))
    _probe(cand, _hits(q, 0.6), same=True, protocol="dictionary")
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c", select_only=True)
    sel = doc["candidates"]["cand"]["selection"]
    pr = next(x for x in sel["checks"] if x["check"].startswith("encoder probe"))
    assert pr["role"] == "info" and pr["status"] == "pass" and "same_sets_as_training" in pr["note"]
    assert pr["val_protocol"] == {"cand": "dictionary", "ref": "mixed"}
    # gate (b) is carried by E2 one-shot on validation-part queries (here much worse): rejected
    assert sel["gates"]["b"] == "fail" and doc["selected"] is None
    # both encoders selected otherwise (the old mixed protocol on other sets): the probe gates
    _probe(cand, _hits(q, 0.6), same=False, protocol="mixed")
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c2", select_only=True)
    pr = next(x for x in doc["candidates"]["cand"]["selection"]["checks"] if x["check"].startswith("encoder probe"))
    assert pr["role"] == "gate" and pr["selected_on_probe_queries"] is None
    # a probe that does not say how its encoder was selected cannot gate
    (cand / "eval" / "encoder_probe.json").write_text(json.dumps({"hits": {f"include|{k}": 1 for k in q}}))
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c3", select_only=True)
    pr = next(x for x in doc["candidates"]["cand"]["selection"]["checks"] if x["check"].startswith("encoder probe"))
    assert pr["role"] == "info" and "does not record" in pr["note"]


def test_isl500_clips_without_a_baseline_cannot_ship(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    cand = _work(tmp_path, "cand", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12, isl500=0.9)
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c")
    e = doc["candidates"]["cand"]
    d = next(x for x in e["selection"]["checks"] if x["gate"] == "d")
    assert d["status"] == "not_available" and "--isl500-baseline" in d["reason"]
    # selected, but no test look: it could not ship anyway
    assert doc["selected"] == "cand" and "confirmation" not in e and "--isl500-baseline" in e["verdict"]
    assert not (tmp_path / "c" / LEDGER).exists()
    # explicitly without the gate: recorded, confirmed, gate (d) not applicable
    doc = compare(ref, {"cand": cand}, splits_dir=_splits(tmp_path), out=tmp_path / "c2", no_isl500_gate=True)
    cf = doc["candidates"]["cand"]["confirmation"]
    assert doc["options"]["no_isl500_gate"] is True and cf["gates"]["d"] == "not_applicable"
    assert all("--no-isl500-gate" in x["reason"] for x in cf["checks"] if x["gate"] == "d" and x["role"] == "gate")


def test_isl500_validation_needs_the_same_signer_split():
    ref = Side("a", Path("/nonexistent-a"))
    base = Side("b", Path("/nonexistent-b"))
    data = {"extra_sources": {"isl500": {"val": 900}}}
    ref._cls_meta = {"val_by_source": {"isl500": {"top1": 0.9, "n": 900}}, "data": data, "train_config": {"seed": 0}}
    base._cls_meta = {"val_by_source": {"isl500": {"top1": 0.9, "n": 900}}, "data": {**data, "signer_split": "v2"},
                      "train_config": {"seed": 0}}
    r = isl500_val_check(ref, base, DEFAULT_MARGINS)
    assert r["status"] == "not_available" and "signer_split v1 vs v2" in r["reason"]
    base._cls_meta["data"] = data
    r = isl500_val_check(ref, base, DEFAULT_MARGINS)
    assert r["status"] != "not_available" and r["same_validation_set"]["signer_split"] == "v1"
    base._cls_meta.pop("data")
    assert "cannot verify" in isl500_val_check(ref, base, DEFAULT_MARGINS)["reason"]


def test_lower_bound_gate_reads_how_each_side_was_calibrated(tmp_path):
    ref = Side("ref", _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84))
    cw = _work(tmp_path, "cand", cv=0.70, e1b=0.70, lb=0.86)
    fu = json.loads((cw / "fusion.json").read_text())
    fu["certification"]["calibration_sets"] = {"clips": 2566, "folds_used": [1, 2, 3], "fold_classifiers": {
        "1": {"selected_on_primary_test_clips": 0}, "4": {"selected_on_primary_test_clips": 856}}}
    (cw / "fusion.json").write_text(json.dumps(fu))
    r = lower_bound_check(Side("cand", cw), ref, DEFAULT_MARGINS)
    assert r["status"] == "pass" and "not like for like" in r["note"]
    assert r["calibration"]["cand"]["kind"] == "out-of-fold" and r["calibration"]["ref"]["kind"] == "validation"
    # a fold classifier selected on the E1b test among the folds used: the bound is not a validation number
    fu["certification"]["calibration_sets"]["folds_used"] = [1, 2, 3, 4]
    (cw / "fusion.json").write_text(json.dumps(fu))
    r = lower_bound_check(Side("cand", cw), ref, DEFAULT_MARGINS)
    assert r["status"] == "not_available" and "selected on the E1b test" in r["reason"]


def test_one_look_per_folder_and_the_fallback_after_a_failed_confirmation(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    first = _work(tmp_path, "first", cv=0.75, e1b=0.50, lb=0.86, e2_val=0.12)
    second = _work(tmp_path, "second", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    out = tmp_path / "cmp"
    doc = compare(ref, {"first": first, "second": second}, splits_dir=_splits(tmp_path), out=out)
    v = doc["candidates"]["first"]["verdict"]
    assert doc["selected"] == "first" and "NOT confirmed" in v
    assert "commercial-safe's encoder and classifier" in v and "selection on test" in v
    led = json.loads((out / LEDGER).read_text().splitlines()[0])
    assert [r["candidate"] for r in led["ranking"]] == ["first", "second"] and led["ship"] is False
    # confirming the next-ranked candidate from the same folder is refused: no test look, no ledger line
    doc = compare(ref, {"second": second}, splits_dir=_splits(tmp_path), out=out)
    assert "confirmation" not in doc["candidates"]["second"] and "--force-second-look" in doc["verdict"]
    assert len((out / LEDGER).read_text().splitlines()) == 1
    # forced: it runs, is recorded and flagged as selection on test
    doc = compare(ref, {"second": second}, splits_dir=_splits(tmp_path), out=out, force_second_look=True)
    cf = doc["candidates"]["second"]["confirmation"]
    assert "selection on test" in cf["warning"] and doc["options"]["force_second_look"] is True
    assert json.loads((out / LEDGER).read_text().splitlines()[1])["force_second_look"] is True


def test_plan_floors_are_information_and_ranking_counts_gates(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    good = _work(tmp_path, "good", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    doc = compare(ref, {"good": good}, splits_dir=_splits(tmp_path), out=tmp_path / "c")
    fl = {x["check"]: x for x in doc["candidates"]["good"]["confirmation"]["checks"] if "plan floor" in x["check"]}
    assert fl["E1b top-1 (common clips) >= plan floor 0.722"]["status"] == "pass"
    assert fl["E2_seen top-1 >= plan floor 0.386"]["status"] == "pass"
    assert all(x["role"] == "info" for x in fl.values())
    assert doc["ranking"][0]["key"][0] == 4  # gates passed or not applicable, not raw checks
    assert doc["plan_floors"]["E2"] == 0.06


def test_a_recorded_fusion_or_index_file_is_checked(tmp_path):
    from app.packs.isl import ProfileMismatch, _file_sha256, check_component_profiles

    fu, emb = tmp_path / "fusion.json", tmp_path / "embeddings.f16.npy"
    fu.write_text("{}")
    emb.write_bytes(b"rows")
    rec = {"fusion": {"profile": "commercial-safe", "sha256": _file_sha256(fu)},
           "index": {"profile": "commercial-safe", "sha256": _file_sha256(emb)}}
    prof = {"fusion": "commercial-safe", "index": "commercial-safe"}
    assert set(check_component_profiles(_manifest("research", rec), prof, {"fusion": fu, "index": emb})) == {
        "fusion", "index"}
    fu.write_text('{"swapped": true}')
    with pytest.raises(ProfileMismatch, match="another fusion file"):
        check_component_profiles(_manifest("research", rec), prof, {"fusion": fu, "index": emb})
    fu.write_text("{}")
    emb.write_bytes(b"other rows")
    with pytest.raises(ProfileMismatch, match="another index file"):
        check_component_profiles(_manifest("research", rec), prof, {"fusion": fu, "index": emb})


# --------------------------------------------------------------------------------------------- evaluate --per-clip

def test_evaluate_per_clip_is_opt_in_and_matches_the_report(phase2, tmp_path):  # noqa: F811
    from app.isl.evaluation.runners import PER_CLIP_FILE, evaluate_all
    from app.isl.training.common import ISL_FEATURES

    w, data = phase2["world"], phase2["data"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work)
    shutil.rmtree(work / "eval", ignore_errors=True)
    kw = dict(profile="research", smoke=True, vocab_path=w["vocab"], paths=w["paths"], out_dirs=[], e4_ks=(1, 2),
              latency_n=1)
    plain = evaluate_all(work, data, ISL_FEATURES, **kw)
    assert "per_clip_file" not in plain and not (work / "eval" / PER_CLIP_FILE).exists()
    assert "\"_per_clip\"" not in (work / "eval" / "report.json").read_text()
    doc = evaluate_all(work, data, ISL_FEATURES, per_clip=True, **kw)
    pc = json.loads((work / "eval" / PER_CLIP_FILE).read_text())
    assert doc["per_clip_file"] == PER_CLIP_FILE and pc["report_created"] == doc["created"]
    assert "\"_per_clip\"" not in (work / "eval" / "report.json").read_text()
    p = pc["protocols"]
    assert p["E1b"]["clip_ids"] == doc["protocols"]["E1b"]["clip_ids"]
    assert p["E1b"]["hit1"] == doc["protocols"]["E1b"]["per_clip"]["hit1"]
    for k in ("E1b", "E1c", "E3", "E2_seen", "E2"):
        r = p[k]
        assert len(r["clip_ids"]) == len(r["hit1"]) == len(r["groups"]) == len(r["sources"]) > 0, k
    assert p["E3"]["hit1"].count("1") / len(p["E3"]["hit1"]) == pytest.approx(doc["protocols"]["E3"]["metrics"]["top1"])
    assert set(p["E2"]["parts"]) <= {"val", "test"} and len(p["E2"]["parts"]) == doc["protocols"]["E2"]["n"]
    test_ids = {c.clip_id for c in data.include_part("test")}
    assert all((pt == "test") == (cid in test_ids) for cid, pt in zip(p["E2"]["clip_ids"], p["E2"]["parts"]))
    assert len(p["E2_seen"]["clip_ids"]) == doc["protocols"]["E2_seen"]["n"]
    # the side reader pairs on it
    s = Side("x", work)
    assert s.per_clip is not None and len(s.hits("E1b")[0]) == doc["protocols"]["E1b"]["n"]


# --------------------------------------------------------------------------------------------- profile allowance

def _manifest(profile: str, mixed: dict | None = None):
    from app.packs.schema import PackManifest

    return PackManifest.model_validate({"id": "isl-research" if profile == "research" else "isl", "kind": "isl-hybrid",
                                        "sign_language": "isl", "display_name": {"en": "x"}, "profile": profile,
                                        **({"mixed_profiles": mixed} if mixed is not None else {})})


def test_component_profile_rule():
    from app.isl.profiles import component_profile_allowed
    from app.packs.isl import ProfileMismatch, check_component_profiles

    assert component_profile_allowed("commercial-safe", "research")
    assert not component_profile_allowed("research", "commercial-safe")
    assert not component_profile_allowed("unknown", "research")
    # same profile: nothing to record
    assert check_component_profiles(_manifest("research"), {"pack": "research", "encoder": "research"}) == {}
    # licence-compatible but not recorded: refused, with the way to record it
    with pytest.raises(ProfileMismatch, match="allow-profile-mix"):
        check_component_profiles(_manifest("research"), {"pack": "research", "encoder": "commercial-safe"})
    rec = {"encoder": {"profile": "commercial-safe", "reason": "test"}}
    ok = check_component_profiles(_manifest("research", rec), {"pack": "research", "encoder": "commercial-safe"})
    assert ok == {"encoder": {"profile": "commercial-safe", "reason": "test"}}
    # the reverse is never allowed, even when a manifest claims it
    with pytest.raises(ProfileMismatch, match="never allowed"):
        check_component_profiles(_manifest("commercial-safe", {"encoder": {"profile": "research"}}),
                                 {"pack": "commercial-safe", "encoder": "research"})
    # a record of another profile, or of another file, does not count
    with pytest.raises(ProfileMismatch):
        check_component_profiles(_manifest("research", {"encoder": {"profile": "other"}}),
                                 {"pack": "research", "encoder": "commercial-safe"})


def test_component_profile_rule_checks_the_recorded_file(tmp_path):
    from app.packs.isl import ProfileMismatch, _file_sha256, check_component_profiles

    f = tmp_path / "encoder.pt"
    f.write_bytes(b"a")
    rec = {"encoder": {"profile": "commercial-safe", "sha256": _file_sha256(f)}}
    assert check_component_profiles(_manifest("research", rec), {"encoder": "commercial-safe"}, {"encoder": f})
    f.write_bytes(b"b")
    with pytest.raises(ProfileMismatch, match="another encoder file"):
        check_component_profiles(_manifest("research", rec), {"encoder": "commercial-safe"}, {"encoder": f})


def test_research_pack_with_a_commercial_safe_encoder(phase2, tmp_path):  # noqa: F811
    import torch

    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack
    from app.packs.isl import IslPack, ProfileMismatch
    from app.packs.schema import PackManifest

    w, data = phase2["world"], phase2["data"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work)
    for d in ("index", "index_final", "eval"):
        shutil.rmtree(work / d, ignore_errors=True)
    ck = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)
    ck["meta"]["profile"] = "commercial-safe"  # the fallback run's encoder (docs 15.31)
    torch.save(ck, work / "encoder.pt")
    build_index(EncoderPredictor(work / "encoder.pt"), data.islrtc, data.include_part("train"), work / "index",
                "research", extra=data.extra_part("train"), split=data.split["name"])

    def load(folder):
        man = PackManifest.model_validate(json.loads((folder / "pack.json").read_text()))
        return man, IslPack(man, folder)

    # default build: unchanged, no record, and the app refuses it with the way out
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "p1", smoke=True, vocab_path=w["vocab"])
    man, pack = load(Path(r["pack"]))
    assert "mixed_profiles" not in json.loads((Path(r["pack"]) / "pack.json").read_text())
    with pytest.raises(ProfileMismatch, match="allow-profile-mix"):
        pack._load()
    # --allow-profile-mix: recorded in the manifest (profile, file SHA-256, reason) and loaded
    r = build_pack(work, "research", ISL_FEATURES, tmp_path / "p2", smoke=True, vocab_path=w["vocab"],
                   allow_profile_mix=True)
    raw = json.loads((Path(r["pack"]) / "pack.json").read_text())
    mp = raw["mixed_profiles"]
    assert set(mp) == {"encoder"} and mp["encoder"]["profile"] == "commercial-safe"
    assert mp["encoder"]["sha256"] == raw["checksums"]["encoder.pt"] and "licence-compatible" in mp["encoder"]["reason"]
    man, pack = load(Path(r["pack"]))
    rec, info = pack._load()
    assert rec.encoder is not None and info["mixed_profiles"]["encoder"]["profile"] == "commercial-safe"
    # never the reverse: research components in a commercial-safe pack are refused at build time
    with pytest.raises(ValueError, match="another data profile"):
        build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "p3", smoke=True, vocab_path=w["vocab"],
                   allow_profile_mix=True)
