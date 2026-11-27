"""Selection gates (docs/ISL_DESIGN.md 15.44): compare-packs selects on validation / out-of-fold numbers only and
confirms once on test with paired tests; evaluate --all --per-clip writes what it pairs on; a licence-compatible
component of another data profile is loaded only when the pack manifest records it."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest

from app.isl.evaluation.compare_packs import (DEFAULT_MARGINS, LEDGER, Side, compare, gate_status, mcnemar_exact,
                                              paired_difference, unpaired_difference)
from tests.isl.test_eval_runners import phase2  # noqa: F401  (module fixture: a tiny trained research world)

PRIMARY = "include-session-cv5-fold0"
CV1 = "include-session-cv5-fold1"
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


def _work(root: Path, name: str, *, cv: float, e1b: float, lb: float, test_strict_prec: float = 0.95,
          e2_val: float = 0.1, e2_test: float = 0.1, seen: float = 0.4, isl500: float | None = None,
          per_clip: bool = True) -> Path:
    w = root / name
    (w / "eval").mkdir(parents=True)
    test_ids, cv_ids = _ids(600, "include/t"), _ids(600, "include/v")
    e2_ids = _ids(300, "include/o")
    e2h = {**_hits(e2_ids[:150], e2_val), **_hits(e2_ids[150:], e2_test)}
    isl_ids = _ids(600, "isl500/user005/x") if isl500 is not None else []
    e1c = {**_hits(isl_ids, isl500 or 0)}
    e1b_h, cv_h, seen_h = _hits(test_ids, e1b), _hits(cv_ids, cv), _hits(test_ids, seen)
    pr = {"E1b": {"split": PRIMARY, "n": 600, "metrics": {"top1": e1b}, "clip_ids": test_ids,
                  "per_clip": {"hit1": "".join(str(e1b_h[c]) for c in test_ids)},
                  "cv": {"pooled_top1": (e1b + cv) / 2, "pooled_n": 1200,
                         "per_fold": [{"split": "E1b core words ...", "n": 600, "top1": e1b},
                                      {"split": CV1, "n": 600, "top1": cv}]}},
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
        pc = {"E1b": _rec(e1b_h), "E1b_cv": {CV1: _rec(cv_h)}, "E2_seen": _rec(seen_h),
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
    doc = compare(ref, {"valgood": valgood, "testgood": testgood}, out=out)
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
    doc2 = compare(ref, {"valgood": valgood, "testgood": testgood}, out=out)
    cf2 = doc2["candidates"]["valgood"]["confirmation"]
    assert cf2["test_looks_before"] == 1 and "no longer a single confirmation" in cf2["warning"]
    assert len((out / LEDGER).read_text().splitlines()) == 2


def test_select_only_never_looks_at_test(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    cand = _work(tmp_path, "cand", cv=0.71, e1b=0.70, lb=0.85)
    out = tmp_path / "cmp"
    doc = compare(ref, {"cand": cand}, out=out, select_only=True)
    assert doc["selected"] == "cand" and "confirmation" not in doc["candidates"]["cand"]
    assert "test not looked at" in doc["candidates"]["cand"]["verdict"] and not (out / LEDGER).exists()


def test_confirmed_candidate_ships_and_no_candidate_falls_back(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    good = _work(tmp_path, "good", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    doc = compare(ref, {"good": good}, out=tmp_path / "c1")
    cf = doc["candidates"]["good"]["confirmation"]
    assert cf["gates"] == {"a": "pass", "b": "pass", "c": "pass", "d": "not_applicable"} and cf["ship"] is True
    assert doc["candidates"]["good"]["stale"] == [] and "confirmed on test: ship" in doc["verdict"]
    bad = _work(tmp_path, "bad", cv=0.50, e1b=0.74, lb=0.86)
    doc = compare(ref, {"bad": bad}, out=tmp_path / "c2")
    assert doc["selected"] is None and "commercial-safe's encoder and classifier" in doc["verdict"]
    assert not (tmp_path / "c2" / LEDGER).exists()


def test_isl500_gate_against_the_baseline(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84)
    base = _work(tmp_path, "base", cv=0.60, e1b=0.60, lb=0.80, isl500=0.88)
    worse = _work(tmp_path, "worse", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12, isl500=0.80)
    doc = compare(ref, {"worse": worse}, baseline=base, out=tmp_path / "c")
    cf = doc["candidates"]["worse"]["confirmation"]
    d = [x for x in cf["checks"] if x["gate"] == "d"]
    assert {x["check"] for x in d} == {"ISL500 test top-1 (classifier alone)", "ISL500 test top-1 (fused)"}
    assert all(x["status"] == "fail" and x["paired"] for x in d) and cf["ship"] is False
    # ISL500 groups are its users (2 test signers): no group interval, said so
    assert d[0]["group_interval"] is None and d[0]["n_groups"] == 1
    # a candidate without ISL500 clips: gate (d) does not apply
    plain = _work(tmp_path, "plain", cv=0.74, e1b=0.74, lb=0.86, seen=0.45, e2_val=0.12, e2_test=0.12)
    doc = compare(ref, {"plain": plain}, baseline=base, out=tmp_path / "c2")
    assert doc["candidates"]["plain"]["confirmation"]["gates"]["d"] == "not_applicable"


def test_unpaired_fallback_and_stale_inputs(tmp_path):
    ref = _work(tmp_path, "ref", cv=0.70, e1b=0.70, lb=0.84, per_clip=False)
    cand = _work(tmp_path, "cand", cv=0.74, e1b=0.74, lb=0.86, per_clip=False)
    # E1b still pairs on report.json's clip ids and hit bits; E2_seen falls back to the two rates
    doc = compare(ref, {"cand": cand}, out=tmp_path / "c1")
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
    doc = compare(ref, {"good": good}, out=tmp_path / "c2")
    e = doc["candidates"]["good"]
    assert e["stale"] == ["report.json measured another encoder.pt"] and e["confirmation"]["ship"] is False
    assert any("STALE" in n for n in e["notes"])
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
