"""Linker and label granularity (QA plan 1.2, critic 12, docs/ISL_DESIGN.md 15.34).

* the link step can strip WSLP's "(N)" numbering, so adult(1) links to isl.adult;
* twin concepts form alias groups (female / woman-female, ...); you / you~plural never;
* with ``FusionParams.aliases`` the fused decision sums a group before the accept rule (the app and fit-fusion);
* evaluation reports strict-label AND alias-label numbers; the metrics, gates and certification stay strict;
* everything is opt-in: the link step's ``--link-aliases`` (driver ``LINK_ALIASES``, research only).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.sources import Clip
from app.isl.models.fusion import FusionInputs, FusionParams, decide, fuse
from app.isl.training import multisource as ms
from app.isl.vocab import aliases as al

ROWS = [{"concept_id": c, "headword": h, "kind": "word", "synonyms": syn}
        for c, h, syn in (("isl.adult", "adult", []), ("isl.female", "female", []),
                          ("isl.woman-female", "woman", ["woman", "female"]), ("isl.big", "big", ["big", "large"]),
                          ("isl.large", "large", []), ("isl.you", "you", []), ("isl.you~plural", "you", []),
                          ("isl.cat", "cat", []))]


# ------------------------------------------------------------------------------------------------ linking

def test_link_strips_wslp_numbering_only_when_asked():
    old = ms.GlossLinker(ROWS)
    assert old.link("adult(1)", "wslp") == ("isl.adult-1", "new")  # before: a separate concept
    new = ms.GlossLinker(ROWS, strip_numbering=True)
    assert new.link("adult(1)", "wslp") == ("isl.adult", "exact")
    assert new.link("Adult (2)", "wslp") == ("isl.adult", "exact")
    assert new.counts["numbering_stripped"] == 2
    # a word the vocabulary does not know: all its numbered forms are one new concept, named without the number
    assert new.link("Zebra(1)", "wslp") == ("isl.zebra", "new") == new.link("zebra(2)", "wslp")
    assert new.extra_concepts()[0]["headword"] == "zebra"
    # only a trailing "(digits)": "(academic)" and a bare "(1)" are kept
    assert al.strip_sign_number("x (academic)") == "x (academic)" and al.strip_sign_number("(1)") == "(1)"
    assert al.strip_sign_number("Bank(12) ") == "Bank"


def test_link_clips_keep_the_numbered_variant_id():
    lk = ms.GlossLinker(ROWS, strip_numbering=True)
    c = Clip("wslp/a1", "wslp", "Adult(1)", None, None, None, Path("/x"))
    (out,) = lk.link_clips([c])
    assert out.concept_id == "isl.adult" and out.variant_id == "wslp:adult1"  # the form stays visible


def test_alias_table_resolution_and_the_never_alias_pair():
    assert al.resolve_aliases(al.TWIN_ALIASES, known=[r["concept_id"] for r in ROWS]) == {
        "isl.large": "isl.big", "isl.woman-female": "isl.female"}  # rat-mouse: not in this vocabulary
    assert al.resolve_aliases({"a": "b", "b": "c"}) == {"a": "c", "b": "c"}  # chains go to the final root
    with pytest.raises(ValueError, match="cycle"):
        al.resolve_aliases({"a": "b", "b": "a"})
    with pytest.raises(ValueError, match="different senses"):
        al.resolve_aliases({"isl.you~plural": "isl.you"})
    assert not any({"isl.you", "isl.you~plural"} <= set(m) for m in al.alias_groups(al.TWIN_ALIASES).values())
    assert al.alias_groups({"isl.large": "isl.big"}) == {"isl.big": ["isl.big", "isl.large"]}


def test_linker_alias_of_and_expand_cover_the_group():
    assert ms.GlossLinker(ROWS).alias_of == {}
    lk = ms.GlossLinker(ROWS, aliases=al.TWIN_ALIASES)
    assert lk.alias_of == {"isl.large": "isl.big", "isl.woman-female": "isl.female"}
    assert {"isl.big", "isl.large"} <= lk.expand(["isl.large"])  # a held-out concept takes its twin with it
    assert "isl.cat" not in lk.expand(["isl.large"])


def test_link_settings_round_trip(tmp_path):
    assert ms.read_link_settings(tmp_path) == {}  # a work folder linked before 15.34
    assert ms.linker_options({}) == {"strip_numbering": False, "aliases": None}
    s = ms.link_settings(True)
    lk = ms.GlossLinker(ROWS, **ms.linker_options(s))
    lk.link("adult(1)", "wslp")
    d = ms.write_link_settings(tmp_path, s, lk)
    back = ms.read_link_settings(tmp_path)
    assert back == d and back["strip_sign_numbers"] and back["aliases"] and back["numbering_stripped"] == 1
    assert back["alias_of"] == {"isl.large": "isl.big", "isl.woman-female": "isl.female"} and back["alias_sha256"]
    off = ms.write_link_settings(tmp_path, ms.link_settings(False), lk)
    assert off["alias_of"] == {} and not off["strip_sign_numbers"]


# ------------------------------------------------------------------------------------------------ fusion

def _cls_only(probs: dict, crops: int = 3) -> FusionInputs:
    ids = list(probs)
    lp = np.log(np.array([probs[k] for k in ids], np.float64))
    return FusionInputs(cls_logp=np.tile(lp, (crops, 1)), cls_ids=ids)


def _params(**kw) -> FusionParams:
    # classifier term only, temperature 1: the fused probabilities are the classifier's; ungated (in-code params)
    return FusionParams(alpha=1.0, beta=0.0, gamma=0.0, t_c=1.0, top_n=10, margin=0.1, min_agree=2,
                        accept={"strict": 0.8, "balanced": 0.7, "lenient": 0.55}, **kw)


def test_summing_over_the_alias_group_changes_the_accept_decision():
    """female 0.45 + woman-female 0.40: strictly neither reaches 0.7 and the margin is 0.05; summed, the group has
    0.85 and is accepted. The shown word is the group's best member (female)."""
    inp = _cls_only({"isl.female": 0.45, "isl.woman-female": 0.40, "isl.cat": 0.15})
    strict = fuse(inp, _params())
    assert strict.roots is None and strict.keys[0] == "isl.female"
    assert decide(strict.probs, strict.agree, 3, _params(), "balanced")[0] != "accept"
    p = _params(aliases={"isl.woman-female": "isl.female"})
    res = fuse(inp, p)
    assert res.keys == ["isl.female", "isl.cat"] and res.roots == ["isl.female", "isl.cat"]
    assert res.probs[0] == pytest.approx(0.85) and res.agree == 3
    assert decide(res.probs, res.agree, 3, p, "balanced") == ("accept", "ok")
    assert decide(res.probs, res.agree, 3, p, "strict") == ("accept", "ok")  # 0.85 >= 0.8


def test_summing_can_change_the_winner_and_the_group_member_shown():
    inp = _cls_only({"isl.cat": 0.45, "isl.big": 0.30, "isl.large": 0.25})
    p = _params(aliases={"isl.large": "isl.big"})
    res = fuse(inp, p)
    assert res.keys[0] == "isl.big" and res.probs[0] == pytest.approx(0.55) and res.roots[0] == "isl.big"
    assert fuse(inp, _params()).keys[0] == "isl.cat"
    # the alias member wins inside its group: it is the word shown, the root is recorded
    res2 = fuse(_cls_only({"isl.large": 0.5, "isl.big": 0.3, "isl.cat": 0.2}), p)
    assert res2.keys[0] == "isl.large" and res2.roots[0] == "isl.big"


def test_crop_votes_are_counted_on_the_summed_groups():
    """Crop-level agreement uses the per-crop summed groups: a crop that ranks big then large first counts for the
    group even when 'cat' is its best single key."""
    rows = np.log(np.array([[0.40, 0.35, 0.25], [0.40, 0.35, 0.25], [0.2, 0.5, 0.3]]))
    inp = FusionInputs(cls_logp=rows, cls_ids=["isl.cat", "isl.big", "isl.large"])
    strict = fuse(inp, _params())
    p = _params(aliases={"isl.large": "isl.big"})
    res = fuse(inp, p)
    assert res.keys[0] == "isl.big" and res.agree == 3 and res.per_crop_top1 == ["isl.big"] * 3
    assert strict.per_crop_top1.count("isl.cat") == 2


def test_user_signs_and_components_survive_the_merge():
    inp = _cls_only({"isl.big": 0.5, "isl.large": 0.3, "isl.cat": 0.2})
    inp.user = np.full((3, 1), 0.4, np.float32)
    inp.user_ids = ["mysign:u1"]
    res = fuse(inp, _params(aliases={"isl.large": "isl.big"}))
    assert "mysign:u1" in res.keys and len(res.keys) == len(set(res.keys)) == 3
    assert res.components["classifier"] >= {"isl.big"}
    assert math.isclose(float(res.probs.sum()), 1.0, rel_tol=1e-6)


def test_fusion_params_store_aliases_and_old_files_have_none(tmp_path):
    p = _params(aliases={"isl.large": "isl.big", "isl.big": "isl.big"})
    assert p.aliases == {"isl.large": "isl.big"}  # self maps dropped
    f = tmp_path / "fusion.json"
    p.save(f)
    assert FusionParams.load(f).aliases == {"isl.large": "isl.big"}
    d = json.loads(f.read_text())
    d.pop("aliases")
    f.write_text(json.dumps(d))
    assert FusionParams.load(f).aliases == {}  # a fusion.json written before 15.34: strict keys, as before
    with pytest.raises(ValueError):
        FusionParams(aliases={"isl.you~plural": "isl.you"})


def test_the_app_decision_uses_the_summed_group():
    """``user_sign_decision`` (the app's segment mode and E4) on a fused result of parameters with aliases."""
    from app.isl.recognizer import user_sign_decision

    inp = _cls_only({"isl.female": 0.45, "isl.woman-female": 0.40, "isl.cat": 0.15})
    p = _params(aliases={"isl.woman-female": "isl.female"})
    d, why, keys, _ = user_sign_decision(fuse(inp, p), inp, 3, p, "balanced", lambda k: 1.0)
    assert (d, why, keys[0]) == ("accept", "ok", "isl.female")
    d0, _, _, _ = user_sign_decision(fuse(inp, _params()), inp, 3, _params(), "balanced", lambda k: 1.0)
    assert d0 != "accept"


# ------------------------------------------------------------------------------------------------ evaluation

def test_e1_alias_rates_count_a_twin_as_correct_and_keep_the_strict_number():
    from app.isl.evaluation.runners import e1_alias_rates

    ids = ["isl.big", "isl.large", "isl.cat"]
    P = np.array([[0.6, 0.3, 0.1],   # truth large, predicted big: strict wrong, alias right
                  [0.1, 0.1, 0.8],   # truth large, predicted cat: wrong both ways
                  [0.1, 0.2, 0.7]])  # truth cat: right both ways
    y = np.array([1, 1, 2])
    hit = (P.argmax(1) == y).astype(float)
    top5 = np.ones(3)
    r = e1_alias_rates(P, y, ids, {"isl.large": "isl.big"}, hit, top5)
    assert r["top1_strict"] == pytest.approx(1 / 3) and r["top1"] == pytest.approx(2 / 3)
    assert r["top1_only_alias"] == 1 and r["top1_only_strict"] == 0 and r["groups_among_classes"] == 1
    assert "STRICT" in r["rule"]


def test_e3_alias_rates_report_strict_and_alias_precision_side_by_side():
    from app.isl.evaluation.runners import e3_alias_rates

    p = _params(aliases={"isl.woman-female": "isl.female"})
    inp = _cls_only({"isl.female": 0.45, "isl.woman-female": 0.40, "isl.cat": 0.15})
    results = [fuse(inp, p), fuse(inp, p)]
    truths = ["isl.woman-female", "isl.female"]
    r = e3_alias_rates(results, truths, p, p.aliases)
    b = r["policies"]["balanced"]
    assert b["n_accepted"] == 2
    assert b["accepted_precision_strict"] == pytest.approx(0.5) and b["accepted_precision"] == pytest.approx(1.0)
    assert r["top1_strict"] == pytest.approx(0.5) and r["top1"] == pytest.approx(1.0) and r["fusion_sums_groups"]


def test_alias_summary_prefers_the_link_step_and_flags_a_stale_fusion():
    from types import SimpleNamespace

    from app.isl.evaluation.runners import alias_summary

    assert alias_summary(SimpleNamespace(alias_of={}), FusionParams())["used"] is False
    a = alias_summary(SimpleNamespace(alias_of={"isl.large": "isl.big"}), FusionParams())
    assert a["used"] and a["source"] == "link step" and not a["fusion_sums_groups"] and not a["mismatch"]
    b = alias_summary(SimpleNamespace(alias_of={"isl.large": "isl.big"}),
                      FusionParams(aliases={"isl.woman-female": "isl.female"}))
    assert b["mismatch"] and b["fusion_sums_groups"]
    c = alias_summary(SimpleNamespace(alias_of={}), FusionParams(aliases={"isl.large": "isl.big"}))
    assert c["source"] == "fusion.json"


def test_markdown_prints_alias_rows_as_not_a_gate():
    from app.isl.evaluation.runners import _alias_lines

    doc = {"aliases": {"used": True, "groups": {"isl.big": ["isl.big", "isl.large"]}, "source": "link step",
                       "fusion_sums_groups": True, "mismatch": False},
           "protocols": {"E1b": {"alias_labels": {"n": 3, "top1_strict": 0.5, "top1": 0.6, "top5": 0.9}}}}
    lines = "\n".join(_alias_lines(doc))
    assert "NOT a gate" in lines and "top-1 strict 0.500 / alias 0.600" in lines
    assert _alias_lines({"aliases": {"used": False}, "protocols": {}}) == []


def test_fit_fusion_with_aliases_fits_thresholds_on_summed_groups_with_strict_labels(monkeypatch):
    """The weights are the strict fit's; the parameters carry the alias map, and the thresholds / certification are
    fitted on the summed groups (one extra fuse pass): ``fit_thresholds`` sees results with roots, one column per
    group, strict truths, and a closed set that holds every member of a group with a class (docs 15.35)."""
    import app.isl.training.fit_fusion as ff
    from tests.isl.test_fusion_efficiency import _ToyRec, _toy_world

    seen = []
    real = ff.fit_thresholds

    def spy(results, truths, params, *a, **kw):
        seen.append((list(results), list(truths), params, kw.get("closed_set")))
        return real(results, truths, params, *a, **kw)

    monkeypatch.setattr(ff, "fit_thresholds", spy)
    clips, truth_of = _toy_world()
    base, _ = ff.fit_fusion(_ToyRec(truth_of), clips, "research")
    res0, truths0, _, _ = seen[-1]
    assert all(r.roots is None for r in res0)  # strict fit: no groups
    assert any({"isl.c10", "isl.c11"} <= set(r.keys) for r in res0)  # both twins have their own column
    alias = {"isl.c11": "isl.c10", "isl.c21": "isl.c20", "isl.d0": "isl.c10"}  # d0: retrieval only
    p, rep = ff.fit_fusion(_ToyRec(truth_of), clips, "research", aliases=alias)
    res, truths, fp, closed = seen[-1]
    assert fp.aliases == alias and truths == truths0  # strict labels: the clips' own concept ids
    assert {"isl.c11", "isl.c21"} & set(truths)  # the toy has clips of the aliases themselves
    assert all(r.roots is not None and len(r.roots) == len(r.keys) for r in res)
    for r in res:  # the members of a group were merged: at most one column per group, named by a member
        assert not ({"isl.c10", "isl.c11"} <= set(r.keys) or {"isl.c10", "isl.d0"} <= set(r.keys))
        assert all(al.alias_root(k, alias) == rt for k, rt in zip(r.keys, r.roots))
    assert "isl.d0" in closed and "isl.d1" not in closed  # the group of a class is a classifier word
    assert any(a.keys[0] != b.keys[0] or not np.isclose(a.probs[0], b.probs[0]) for a, b in zip(res, res0))
    alias = {"isl.c11": "isl.c10", "isl.c21": "isl.c20"}
    p, rep = ff.fit_fusion(_ToyRec(truth_of), clips, "research", aliases=alias)
    assert base.aliases == {} and p.aliases == alias
    assert all(getattr(p, k) == getattr(base, k) for k in ("alpha", "beta", "gamma", "tau_r", "t_c"))
    assert rep["aliases"]["n_aliases"] == 2 and rep["aliases"]["labels"].startswith("strict")
    assert p.achieved["balanced"]["n"] == len(clips)
    assert FusionParams.from_json(p.to_json()).aliases == alias


# ------------------------------------------------------------------------------------------------ CLI + driver

class _Wslp:
    name, commercial, licence = "wslp", False, "test"

    def __init__(self, clips):
        self._clips = clips

    def available(self):
        return True

    def clips(self, *, only_ok=True):
        yield from self._clips


def test_cli_link_sources_writes_the_settings_every_later_step_reads(tmp_path, monkeypatch):
    from app.isl import cli
    from tests.isl.isl_world import make_world

    w = make_world(tmp_path)
    with Path(w["vocab"]).open("a") as fh:  # two twins of the alias table
        for c, h in (("isl.big", "big"), ("isl.large", "large")):
            fh.write(json.dumps({"concept_id": c, "headword": h, "display": {"en": h}, "kind": "word",
                                 "category": "things", "variants": [], "in_classifier": False,
                                 "indexable": True}) + "\n")
    lm = next((w["paths"].landmarks / "islrtc").glob("*.npz"))
    clip = Clip("wslp/apple-1", "wslp", "Apple(1)", None, None, None, lm)
    ms.register_adapter("wslp", lambda p, v: _Wslp([clip]))
    base = tmp_path / "art"
    monkeypatch.setenv("SLR_ISL_WORKDIR", str(base))
    work = base / "research"
    common = ["--profile", "research", "--vocab", str(w["vocab"]), "--split", w["split"], "--threads", "1"]
    root = ["--root", str(w["paths"].root)]
    try:
        assert cli.main(root + ["link-sources", *common]) == 0
        off = json.loads((work / "link_settings.json").read_text())
        assert off["strip_sign_numbers"] is False and off["alias_of"] == {}
        assert "isl.apple-1" in (work / "extra_concepts.jsonl").read_text()  # the old linking
        assert cli.main(root + ["link-sources", *common, "--link-aliases"]) == 0
        on = json.loads((work / "link_settings.json").read_text())
        assert on["strip_sign_numbers"] and on["numbering_stripped"] == 1
        assert on["alias_of"] == {"isl.large": "isl.big"}
        assert "isl.apple-1" not in (work / "extra_concepts.jsonl").read_text()
        assert json.loads((work / "multisource.json").read_text())["links"]["numbering_stripped"] == 1
        # a later step (here: the context of build-index) links the same way without any flag
        args = cli.build_parser().parse_args(root + ["build-index", *common, "--all-sources"])
        _, data, _, _ = cli._model_ctx(w["paths"], args)
        assert data.alias_of == {"isl.large": "isl.big"}
        assert [c.concept_id for c in data.extra] == ["isl.apple"]
    finally:
        ms.register_adapter("wslp", None)


def test_driver_passes_link_aliases_for_research_only(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import _calls, _run_driver

    monkeypatch.delenv("LINK_ALIASES", raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"

    def link_calls(profile):
        f = tmp_path / "fake_py.log"
        if f.exists():
            f.unlink()
        r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile=profile)
        assert r.returncode == 0, r.stdout + r.stderr
        return [c for c in _calls(tmp_path) if "link-sources" in c], r

    calls, r = link_calls("research")
    assert calls and all("--link-aliases" in c for c in calls)
    assert "link_aliases=1" in r.stdout and "link_aliases=1" in next(base.rglob("link.done")).read_text()
    calls, _ = link_calls("commercial-safe")
    assert calls and not any("--link-aliases" in c for c in calls)
    # only the link step gets the flag
    f = tmp_path / "fake_py.log"
    f.unlink(missing_ok=True)
    assert _run_driver(tmp_path, "--only", "fusion", work_base=base, isl_root=isl, profile="research").returncode == 0
    assert not any("--link-aliases" in c for c in _calls(tmp_path))
    monkeypatch.setenv("LINK_ALIASES", "0")
    calls, _ = link_calls("research")
    assert calls and not any("--link-aliases" in c for c in calls)
    monkeypatch.setenv("LINK_ALIASES", "yes")
    bad = _run_driver(tmp_path, "--only", "link", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert bad.returncode == 2 and "LINK_ALIASES" in bad.stderr
    monkeypatch.setenv("LINK_ALIASES", "1")
    dry = _run_driver(tmp_path, "--only", "link", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert dry.returncode == 0 and "aliases and WSLP number stripping=1" in dry.stdout


# ------------------------------------------------------------------------------------------------ evaluate --all
from tests.isl.test_eval_runners import phase2  # noqa: E402,F401  (the toy phase-2 world: module fixture)


def test_evaluate_all_reports_alias_labels_next_to_strict_ones(phase2, tmp_path):
    """With the link step's aliases, E1b / E1c / E3 get ``alias_labels`` next to the strict metrics; the metrics
    (and so every gate and headline) stay strict; report.md says the alias rows are not a gate. A fusion.json fitted
    with the same map sums the groups in E3; one fitted with another map is flagged."""
    import dataclasses
    import shutil

    from app.isl.evaluation.runners import evaluate_all
    from app.isl.training.common import ISL_FEATURES

    w = phase2["world"]
    work = tmp_path / "work"
    shutil.copytree(phase2["work"], work)
    alias = {"isl.book": "isl.apple"}  # a toy group of two classifier classes
    data = dataclasses.replace(phase2["data"], link={"aliases": True, "alias_of": alias})
    kw = dict(profile="research", smoke=True, vocab_path=w["vocab"], paths=w["paths"], out_dirs=[], e4_ks=(1,),
              latency_n=1)
    doc = evaluate_all(work, data, ISL_FEATURES, **kw)
    a = doc["aliases"]
    assert a["used"] and a["source"] == "link step" and not a["fusion_sums_groups"] and not a["mismatch"]
    for k in ("E1b", "E3"):
        p = doc["protocols"][k]
        r = p["alias_labels"]
        assert r["top1"] >= r["top1_strict"] - 1e-12
        assert p["metrics"]["top1"] == pytest.approx(r["top1_strict"])  # the metrics are the strict numbers
    assert "NOT a gate" in (work / "eval" / "report.md").read_text()
    # strict-only report without aliases: no alias rows at all
    plain = evaluate_all(work, phase2["data"], ISL_FEATURES, **kw)
    assert plain["aliases"]["used"] is False and "alias_labels" not in plain["protocols"]["E1b"]
    # fusion.json fitted with the same map: E3 sums the groups; with another map: flagged
    FusionParams.load(work / "fusion.json").with_(aliases=alias).save(work / "fusion.json")
    doc2 = evaluate_all(work, data, ISL_FEATURES, **kw)
    assert doc2["aliases"]["fusion_sums_groups"] and doc2["protocols"]["E3"]["alias_labels"]["fusion_sums_groups"]
    FusionParams.load(work / "fusion.json").with_(aliases={"isl.cat": "isl.dog"}).save(work / "fusion.json")
    doc3 = evaluate_all(work, data, ISL_FEATURES, **kw)
    assert doc3["aliases"]["mismatch"] and "another alias map" in (work / "eval" / "report.md").read_text()


# ------------------------------------------------------------------------------------------------ review fixes (15.35)

def test_the_race_pair_is_not_aliased_and_every_pair_has_evidence():
    """INCLUDE's race (ethnicity) matches none of ISLRTC's three race rows: no race pair in the table (docs 15.35)."""
    assert not any("race" in a or "race" in r for a, r in al.TWIN_ALIASES.items())
    src = Path(al.__file__).read_text()
    table = src[src.index("TWIN_ALIASES: dict"):src.index("NEVER_ALIAS:")]
    for a in al.TWIN_ALIASES:  # a comment with the evidence right above each pair
        line = next(i for i, ln in enumerate(table.splitlines()) if f'"{a}"' in ln)
        assert table.splitlines()[line - 1].strip().startswith("#")


def test_numbering_is_stripped_for_wslp_only_and_counted_by_source():
    lk = ms.GlossLinker(ROWS, strip_numbering=True)
    assert lk.link("adult(1)", "isl500") == ("isl.adult-1", "new")  # another source keeps its number
    assert lk.link("adult(1)", "wslp") == ("isl.adult", "exact")
    assert lk.counts["numbering_stripped"] == 1 and lk.counts["numbering_stripped:wslp"] == 1
    assert "numbering_stripped:isl500" not in lk.counts


def test_later_steps_use_the_alias_map_the_link_step_froze(tmp_path):
    """A settings file whose ``alias_of`` differs from the table in the code: every later step (the linker, its
    ``expand`` for the held-out guard) uses the file's map, not ``TWIN_ALIASES`` (docs 15.35)."""
    frozen = {"isl.cat": "isl.adult"}  # not in TWIN_ALIASES
    d = {**ms.link_settings(True), "alias_of": frozen}
    (tmp_path / ms.LINK_SETTINGS).write_text(json.dumps(d))
    opts = ms.linker_options(ms.read_link_settings(tmp_path))
    assert opts["aliases"] == frozen and opts["strip_numbering"]
    lk = ms.GlossLinker(ROWS, **opts)
    assert lk.alias_of == frozen and "isl.large" not in lk.alias_of
    assert lk.expand(["isl.cat"]) >= {"isl.cat", "isl.adult"}
    assert "isl.adult" not in ms.GlossLinker(ROWS, aliases=al.TWIN_ALIASES).expand(["isl.cat"])
    # a file of a link step with aliases on but no known pair: no groups (never the code's table)
    assert ms.linker_options({**ms.link_settings(True), "alias_of": {}})["aliases"] == {}
    # only the link step itself (settings without alias_of yet) reads the code's table
    assert ms.linker_options(ms.link_settings(True))["aliases"] == al.TWIN_ALIASES


def test_a_group_with_a_class_is_a_classifier_word_whichever_member_leads():
    """``isl.large`` is retrieval only, ``isl.big`` a class: with the group summed, the open-vocabulary gate treats
    the group as closed whether "large" or "big" is shown (docs 15.35). Without aliases: as before."""
    from types import SimpleNamespace

    from app.isl.recognizer import IslRecognizer

    alias = {"isl.large": "isl.big"}
    assert al.closed_with_aliases({"isl.big", "isl.cat"}, alias) == {"isl.big", "isl.cat", "isl.large"}
    assert al.closed_with_aliases({"isl.cat"}, alias) == {"isl.cat"}  # no class in the group: stays open
    assert al.closed_with_aliases(None, alias) is None and al.closed_with_aliases({"x"}, {}) == {"x"}
    clf = SimpleNamespace(class_names=["isl.big", "isl.cat"])
    rec = SimpleNamespace(classifier=clf, fusion=FusionParams(), _class_set=None)
    assert IslRecognizer.is_open_vocabulary(rec, "isl.large")
    rec.fusion = FusionParams(aliases=alias)  # a new fusion: the cache is rebuilt
    assert not IslRecognizer.is_open_vocabulary(rec, "isl.large")
    assert IslRecognizer.is_open_vocabulary(rec, "isl.zebra")


def test_alias_reports_per_group_counts_top5_strict_and_a_stale_fusion():
    from types import SimpleNamespace

    from app.isl.evaluation.runners import _alias_lines, alias_summary, e1_alias_rates, e3_alias_rates

    ids = ["isl.big", "isl.large", "isl.cat"]
    P = np.array([[0.6, 0.3, 0.1], [0.1, 0.1, 0.8], [0.1, 0.2, 0.7]])
    y = np.array([1, 1, 2])
    r = e1_alias_rates(P, y, ids, {"isl.large": "isl.big"}, (P.argmax(1) == y).astype(float), np.ones(3))
    assert r["by_group"] == {"isl.big": {"n": 2, "top1_only_alias": 1, "top1_only_strict": 0}}
    p = _params(aliases={"isl.woman-female": "isl.female"})
    inp = _cls_only({"isl.female": 0.45, "isl.woman-female": 0.40, "isl.cat": 0.15})
    e3 = e3_alias_rates([fuse(inp, p)], ["isl.woman-female"], p, p.aliases)
    assert e3["top5_strict"] == 0.0 and e3["top5"] == 1.0  # the summed group shows only its best member
    data = SimpleNamespace(alias_of={"isl.large": "isl.big"}, link={"alias_version": al.ALIAS_VERSION})
    a = alias_summary(data, FusionParams())
    assert a["stale"] and not a["mismatch"] and a["link_alias_version"] == al.ALIAS_VERSION
    assert not alias_summary(data, FusionParams(aliases={"isl.large": "isl.big"}))["stale"]
    doc = {"aliases": a, "protocols": {"E1b": {"alias_labels": r}}}
    lines = "\n".join(_alias_lines(doc))
    assert "fusion.json has none" in lines and "isl.big 1" in lines and "top-5 strict 1.000 / alias" in lines


def test_driver_start_line_shows_the_folder_state_when_link_is_not_run(tmp_path, monkeypatch):
    from tests.isl.test_phase2_guards import _run_driver

    monkeypatch.delenv("LINK_ALIASES", raising=False)
    isl, base = tmp_path / "isl", tmp_path / "art"
    r = _run_driver(tmp_path, "--only", "fusion", work_base=base, isl_root=isl, profile="research")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "link_aliases=1 (link step not run; folder: none)" in out and "warning: LINK_ALIASES=1" in out
    work = next(p for p in base.iterdir() if p.is_dir())
    (work / "link_settings.json").write_text(json.dumps({**ms.link_settings(True), "alias_of": {}}, indent=1))
    r = _run_driver(tmp_path, "--only", "fusion", work_base=base, isl_root=isl, profile="research")
    out = r.stdout + r.stderr
    assert "link_aliases=1 (link step not run; folder: on)" in out and "warning: LINK_ALIASES" not in out
    dry = _run_driver(tmp_path, "--only", "fusion", "--dry-run", work_base=base, isl_root=isl, profile="research")
    assert "linker this run: link_aliases=1 (link step not run; folder: on)" in dry.stdout
    r = _run_driver(tmp_path, "--only", "link", work_base=base, isl_root=isl, profile="research")
    assert "link_aliases=1 " in r.stdout and "link step not run" not in r.stdout
