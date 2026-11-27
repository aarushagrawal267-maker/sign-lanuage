"""Per-policy thresholds (docs/ISL_DESIGN.md 15.17).

Regression for the commercial-safe pack of 2026-11-26: no policy reached its validation target, so ``fit_thresholds``
gave strict, balanced and lenient the same best-precision threshold (0.96 / 0.96 / 0.96) and the app's policy switch
did nothing; sliding mode used one classifier threshold (0.60) for every policy; and the "did you mean" mass was
fitted on the balanced pool only.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.features.spec import FeatureConfig
from app.isl.models.calibration import (FLOORS, TARGETS, _fit_suggest_mass, choose_policy_thresholds,
                                        fit_thresholds, wilson_lower)
from app.isl.models.fusion import POLICIES, FusionInputs, FusionParams, FusionResult, decide

# the commercial-safe fusion.json as written before 15.17 (``achieved`` left out)
OLD_PACK = {"alpha": 0.5, "beta": 1.0, "gamma": 1.0, "tau_r": 0.1, "t_c": 0.7211247851537042,
            "s_min": 0.6814198911190033, "accept": {"strict": 0.96, "balanced": 0.96, "lenient": 0.96}, "margin": 0.1,
            "min_agree": 2, "suggest_min_top5_mass": 0.0, "user_bonus": 1.0, "user_margin": 0.05,
            "user_theta_floor": 0.0, "noncore_logp": -6.907755278982137, "user_cls_mode": "neutral", "top_n": 50,
            "version": "fusion-1", "profile": "commercial-safe", "fitted": True,
            "enrol": {"c_min": 0.5, "min_takes": 3, "max_takes": 5}}
PHI = (5 ** 0.5 - 1) / 2


def _ceiling_set(n: int = 2000, ceiling: float = 0.9):
    """Fused results whose precision rises with p1 but never exceeds ``ceiling``: no target is reachable
    (deterministic: a low-discrepancy sequence decides which queries are right)."""
    results, truths = [], []
    for i in range(n):
        p1 = 0.2 + 0.79 * i / n
        ok = (i * PHI) % 1.0 < min(ceiling, p1)
        truth = f"w{i % 7}"
        top = truth if ok else "other"
        rest = [k for k in (truth, "x1", "x2", "x3", "x4", "x5") if k != top][:5]
        probs = np.array([p1] + [(1 - p1) / 5] * 5)
        results.append(FusionResult([top, *rest], probs, [top] * 3, 3, 0.9, 0.9, None, {}))
        truths.append(truth)
    return results, truths


# ------------------------------------------------------------------------------- fused thresholds

def test_unreachable_targets_are_confirm_only_and_say_so():
    """Precision capped at 0.9 (best bound 0.859): strict and balanced miss their floors (0.95 / 0.90) and are
    confirm-only; lenient (floor 0.85) is certified at the best bound (docs 15.25; before: graded fallback)."""
    results, truths = _ceiling_set()
    p, rep = fit_thresholds(results, truths, FusionParams())
    a = p.accept
    assert a["strict"] >= a["balanced"] >= a["lenient"], a
    best = rep["accept_fit"]["best_lower_bound"]
    assert best < TARGETS["lenient"]
    pols = rep["accept_fit"]["policies"]
    for pol in POLICIES:
        assert rep[f"{pol}_target_reached"] is False
        certified = pols[pol]["precision_lower_bound"] >= FLOORS[pol]
        assert pols[pol]["rule"] == ("certified" if certified else "confirm_only")
        assert p.achieved[pol]["rule"] == pols[pol]["rule"] and p.achieved[pol]["floor"] == FLOORS[pol]
        assert p.achieved[pol]["precision_lower_bound"] == pols[pol]["precision_lower_bound"]
        assert p.achieved[pol]["threshold"] == a[pol]
        assert p.policy_rule(pol) == pols[pol]["rule"]  # fitted parameters are gated
    assert [pols[pol]["rule"] for pol in POLICIES] == ["confirm_only", "confirm_only", "certified"]
    assert p.achieved["lenient"]["certified_precision"] == pytest.approx(best, abs=1e-4)
    assert p.achieved["strict"]["certified_precision"] is None
    assert "confirm-only" in rep["accept_fit"]["note"] and "strict" in rep["accept_fit"]["note"]


def test_flat_precision_is_confirm_only_everywhere():
    # precision does not depend on p1 at all (0.8): the best bound is below every floor
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(1000, 100, len(ts)).astype(int)
    n_ok = (0.8 * n_acc).astype(int)
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    assert acc["strict"] == acc["balanced"] == acc["lenient"] == 0.2
    assert all(rep["policies"][p]["rule"] == "confirm_only" for p in POLICIES)


def test_partly_reachable_targets_certify_what_reaches_its_floor():
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(2000, 600, len(ts)).astype(int)
    prec = np.linspace(0.80, 0.955, len(ts))  # lenient (0.90) reachable, balanced (0.95) and strict not
    n_ok = np.round(prec * n_acc).astype(int)
    lbs = np.array([wilson_lower(int(k), int(m)) for k, m in zip(n_ok, n_acc)])
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    pols = rep["policies"]
    assert pols["lenient"]["rule"] == "certified" and pols["lenient"]["target_reached"]
    assert acc["lenient"] == float(ts[np.flatnonzero(lbs >= 0.90)[0]])  # the smallest threshold at the target
    # balanced: target 0.95 out of reach, best bound 0.935 >= its floor 0.90 -> certified at the best bound
    assert pols["balanced"]["rule"] == "certified" and not pols["balanced"]["target_reached"]
    assert pols["balanced"]["certified_precision"] == pytest.approx(lbs.max(), abs=1e-4)
    assert pols["strict"]["rule"] == "confirm_only" and "floor" in pols["strict"]["reason"]
    assert acc["strict"] >= acc["balanced"] > acc["lenient"]


def test_reachable_targets_unchanged_from_the_old_rule():
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(5000, 3000, len(ts)).astype(int)
    n_ok = np.round(np.linspace(0.85, 0.999, len(ts)) * n_acc).astype(int)
    lbs = np.array([wilson_lower(int(k), int(m)) for k, m in zip(n_ok, n_acc)])
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    for pol, tg in TARGETS.items():
        assert rep["policies"][pol]["rule"] == "certified" and rep["policies"][pol]["target_reached"]
        assert acc[pol] == float(ts[np.flatnonzero(lbs >= tg)[0]])
    assert "note" not in rep


def test_too_little_data_keeps_defaults_and_certifies_nothing():
    p0 = FusionParams()
    acc, rep = choose_policy_thresholds(np.array([0.5, 0.6]), np.array([2, 1]), np.array([2, 1]), dict(TARGETS))
    assert acc is None and all(v["rule"] == "confirm_only" for v in rep["policies"].values())
    res, tr = _ceiling_set(3)
    p, rep2 = fit_thresholds(res, tr, p0)
    assert p.accept == p0.accept and rep2["strict_target_reached"] is False
    assert all(p.confirm_only(pol) for pol in POLICIES)
    p_empty, _ = fit_thresholds([], [], p0)
    assert p_empty.gated and all(p_empty.confirm_only(pol) for pol in POLICIES)


# ------------------------------------------------------------------------------- "did you mean"

def _res(p1, mass, hit, truth="t"):
    """One fused result with top-1 ``p1``, top-5 mass ``mass`` and the truth inside the top 5 or not."""
    tail = [(mass - p1) / 4] * 4 + [0.01] * int(round((1 - mass) / 0.01))
    probs = np.array([p1, *tail])
    keys = [truth if hit else "o0", "o1", "o2", "o3", "o4"] + [f"z{i}" for i in range(len(probs) - 5)]
    if not hit:
        keys[-1] = truth
    return FusionResult(keys, probs, [keys[0]] * 3, 3, -1.0, -1.0, None, {})


def test_suggest_mass_is_fitted_on_each_policys_own_pool():
    p = FusionParams(accept={"strict": 0.9, "balanced": 0.7, "lenient": 0.5}, s_min=-1.0)
    res = ([_res(0.8, 0.95, True)] * 60 + [_res(0.6, 0.9, True)] * 20 + [_res(0.3, 0.35, False)] * 10
           + [_res(0.3, 0.7, True)] * 30)
    truths = ["t"] * len(res)
    m = {pol: _fit_suggest_mass(res, truths, p, pol, 0.25) for pol in POLICIES}
    assert m["strict"][0] == 0.0 and m["balanced"][0] == 0.0  # their pools already have >= 80 % top-5 hits
    assert m["lenient"][0] == pytest.approx(0.4) and m["lenient"][1]["top5_hit"] == 1.0
    assert m["lenient"][1]["pool"] == 40  # lenient's pool: only the hard cases (0.75 top-5 hits at mass 0)
    pp = p.with_(suggest_min_top5_mass=0.0, suggest_min_top5_mass_by_policy={k: v[0] for k, v in m.items()})
    low = np.array(_res(0.3, 0.35, False).probs)
    assert decide(low, 3, 3, pp, "balanced")[0] == "suggest" and decide(low, 3, 3, pp, "lenient")[0] == "abstain"


def test_fit_thresholds_stores_per_policy_masses_and_keeps_the_single_value_for_old_readers():
    results, truths = _ceiling_set()
    p, rep = fit_thresholds(results, truths, FusionParams())
    assert set(p.suggest_min_top5_mass_by_policy) == set(POLICIES)
    assert p.suggest_min_top5_mass == p.suggest_min_top5_mass_by_policy["balanced"]
    assert set(rep["suggest_fit"]) == set(POLICIES) and "suggest_target_reached" in rep


# ------------------------------------------------------------------------------- old packs

def test_old_pack_loads_and_behaves_as_before(tmp_path):
    import json

    (tmp_path / "fusion.json").write_text(json.dumps(OLD_PACK))
    p = FusionParams.load(tmp_path / "fusion.json")
    assert p.accept == {"strict": 0.96, "balanced": 0.96, "lenient": 0.96}
    assert p.suggest_min_top5_mass_by_policy == {} and all(p.suggest_mass(pol) == 0.0 for pol in POLICIES)
    assert p.classifier_accept == {} and p.classifier_fit == {}
    probs = np.array([0.5, 0.2, 0.1, 0.05, 0.05, 0.1])
    for pol in POLICIES:
        assert decide(probs, 3, 3, p, pol, best_cos=0.9) == ("suggest", "low_confidence")


def test_new_fields_roundtrip_and_bad_values_are_dropped(tmp_path):
    p = FusionParams(suggest_min_top5_mass_by_policy={"strict": 0.0, "lenient": 0.65, "yolo": 1.0, "balanced": "x"},
                     classifier_accept={"strict": 0.96, "lenient": 0.46},
                     classifier_fit={"fingerprint": "abc", "temperature": 0.72}, certification={"version": "cert-1"})
    assert p.suggest_min_top5_mass_by_policy == {"strict": 0.0, "lenient": 0.65}
    assert p.suggest_mass("balanced") == p.suggest_min_top5_mass  # missing policy: the single value
    assert p.classifier_accept == {"strict": 0.96, "balanced": 0.96, "lenient": 0.46}  # filled and ordered
    p.save(tmp_path / "f.json")
    assert FusionParams.load(tmp_path / "f.json") == p


# ------------------------------------------------------------------------------- classifier alone (sliding mode)

class _Clf:
    def __init__(self, probs, meta, names=("a", "b", "c")):
        self.probs = np.asarray(probs, float)
        self.meta = {"split": "s", "class_names": list(names), "train_seconds": 12.5, **meta}
        self.class_names = list(names)
        self.feature_config = FeatureConfig(canonical_hand=True)

    def predict(self, feats):
        return self.probs.copy(), 1.0


def _cls_inputs(n: int = 1500, names=("a", "b", "c")):
    """Validation evidence for the classifier alone: confidence rises with i, precision capped at 0.9."""
    inputs, truths = [], []
    for i in range(n):
        p1 = 0.36 + 0.63 * i / n
        ok = (i * PHI) % 1.0 < min(0.9, p1)
        truth = names[i % 3]
        top = truth if ok else names[(i + 1) % 3]
        row = np.full(3, (1 - p1) / 2)
        row[names.index(top)] = p1
        inputs.append(FusionInputs(cls_logp=np.log(np.tile(row, (3, 1))), cls_ids=list(names)))
        truths.append(truth)
    inputs.append(FusionInputs(cls_logp=np.log(np.tile([0.98, 0.01, 0.01], (3, 1))), cls_ids=list(names)))
    truths.append("not_a_class")  # outside the classifier: can only be a wrong accept
    return inputs, truths


def test_fit_fusion_fits_per_policy_classifier_thresholds_for_this_classifier():
    from app.isl.recognizer import classifier_fingerprint
    from app.isl.training.fit_fusion import fit_classifier_alone

    clf = _Clf([[0.5, 0.3, 0.2]], {"temperature": 1.0, "recommended_threshold": 0.6})
    inputs, truths = _cls_inputs()
    p, rep = fit_classifier_alone(SimpleNamespace(classifier=clf), inputs, truths, FusionParams())
    a = p.classifier_accept
    assert a["strict"] >= a["balanced"] >= a["lenient"]
    assert p.classifier_fit["fingerprint"] == classifier_fingerprint(clf) and p.classifier_fit["temperature"] == 1.0
    assert rep["n"] == len(inputs) and rep["n_in_classes"] == len(inputs) - 1
    assert all(rep["policies"][pol]["target_reached"] is False for pol in POLICIES) and "confirm-only" in rep["note"]
    # the segment proxy is never certified for the app's sliding mode (docs 15.25), even where it reaches a floor
    assert rep["rule"] == "segment proxy" and rep["best_lower_bound"] >= FLOORS["lenient"]
    assert all(rep["policies"][pol]["rule"] == "confirm_only" for pol in POLICIES)
    # no classifier: nothing stored, the engine keeps the checkpoint threshold
    p2, rep2 = fit_classifier_alone(SimpleNamespace(classifier=None), inputs, truths, p)
    assert p2.classifier_accept == {} and p2.classifier_fit == {} and "no classifier" in rep2["note"]


def _rec(clf, fusion):
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    return IslRecognizer("isl", clf.feature_config, Vocabulary([]), classifier=clf, fusion=fusion)


def _fitted_pack(clf, acc=None):
    from app.isl.recognizer import classifier_fingerprint

    return FusionParams.from_json(OLD_PACK).with_(
        classifier_accept=acc or {"strict": 0.96, "balanced": 0.74, "lenient": 0.46},
        classifier_fit={"fingerprint": classifier_fingerprint(clf), "temperature": clf.meta.get("temperature", 1.0)})


def test_sliding_mode_uses_the_policy_threshold_and_a_user_threshold_still_wins():
    from app.inference.engine import EngineConfig, RecognitionEngine

    clf = _Clf([[0.5, 0.3, 0.2]], {"temperature": 0.72, "recommended_threshold": 0.6})
    rec = _rec(clf, _fitted_pack(clf))
    got = {pol: RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding", policy=pol)).threshold
           for pol in POLICIES}
    assert got == {"strict": 0.96, "balanced": 0.74, "lenient": 0.46}
    assert RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding", policy="strict",
                                                                       threshold=0.5)).threshold == 0.5
    # segment mode keeps the fused thresholds
    assert RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="segment", policy="lenient")).threshold == 0.96
    assert rec.classifier_threshold == 0.6  # the checkpoint's own value is still exposed


def test_sliding_mode_falls_back_to_the_checkpoint_threshold():
    from app.inference.engine import EngineConfig, RecognitionEngine
    from app.isl.recognizer import classifier_policy_thresholds

    clf = _Clf([[0.5, 0.3, 0.2]], {"temperature": 0.72, "recommended_threshold": 0.6})
    other = _Clf([[0.5, 0.3, 0.2]], {"temperature": 0.72, "recommended_threshold": 0.6, "train_seconds": 99.0})
    retuned = _Clf([[0.5, 0.3, 0.2]], {"temperature": 0.72, "recommended_threshold": 0.6})
    cases = {
        "old pack": FusionParams.from_json(OLD_PACK),
        "another classifier (E1a / cv fold swap)": _fitted_pack(other),
        "temperature differs": _fitted_pack(retuned).with_(classifier_fit={
            **_fitted_pack(retuned).classifier_fit, "temperature": 1.3}),
    }
    for why, fusion in cases.items():
        rec = _rec(clf, fusion)
        for pol in POLICIES:
            eng = RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding", policy=pol))
            assert eng.threshold == 0.6, why
        cal = classifier_policy_thresholds(clf, fusion)
        assert cal["per_policy"] is False and cal["reason"], why
    # a recogniser without the per-policy method (older fakes) still works
    fake = SimpleNamespace(classifier_threshold=0.55, fusion=FusionParams(), recommended_threshold=0.7, classifier=clf)
    assert RecognitionEngine(None, recognizer=fake, config=EngineConfig(mode="sliding")).threshold == 0.55


# ------------------------------------------------------------------------------- E1 / E3 runners and report

def test_e1_reports_the_policy_it_uses_and_every_policy(monkeypatch):
    from app.isl.evaluation import runners

    monkeypatch.setattr(runners, "_features", lambda clip, fcfg: np.zeros((3, 4, 2), np.float32))
    clf = _Clf([[0.8, 0.1, 0.1]] * 3, {"temperature": 1.0, "recommended_threshold": 0.6})
    clips = [SimpleNamespace(clip_id=f"c{i}", concept_id=t, group=f"g{i % 2}", category=None)
             for i, t in enumerate(["a", "a", "a", "b"])]
    rec = SimpleNamespace(classifier=clf, fusion=_fitted_pack(clf, {"strict": 0.9, "balanced": 0.75,
                                                                    "lenient": 0.5}),
                          feature_config=clf.feature_config)
    out = runners.e1_closed_set(rec, clips, "E1b", "s", "primary")
    d = out["decision"]
    assert d["policy"] == "balanced" and d["threshold"] == 0.75 and d["per_policy"] is True
    assert d["checkpoint_threshold"] == 0.6 and "fusion.json" in d["source"]
    assert out["metrics"]["coverage"] == 1.0 and out["policies"]["strict"]["coverage"] == 0.0
    assert out["policies"]["strict"]["not_sure_rate"] == 1.0  # E1's not sure = 1 - accepted, as in 15.16
    assert out["metrics"]["wrong_accept_rate"] == pytest.approx(0.25)
    assert "balanced policy" in out["decision_note"] and "0.75" in out["decision_note"]
    # an old pack: every policy at the checkpoint threshold, said so
    rec_old = SimpleNamespace(classifier=clf, fusion=FusionParams.from_json(OLD_PACK), feature_config=clf.feature_config)
    old = runners.e1_closed_set(rec_old, clips, "E1b", "s", "primary")
    assert old["decision"]["per_policy"] is False and old["decision"]["threshold"] == 0.6
    assert {v["threshold"] for v in old["policies"].values()} == {0.6}
    assert "same for every policy" in old["decision_note"]


def test_e3_and_report_show_each_policy(monkeypatch):
    from app.isl.evaluation import runners
    from app.isl.training import fit_fusion as ff

    names = ["a", "b", "c"]
    inputs, truths = [], []
    for i, p1 in enumerate(np.linspace(0.4, 0.99, 30)):
        row = np.full(3, (1 - p1) / 2)
        row[i % 3] = p1
        inputs.append(FusionInputs(cls_logp=np.log(np.tile(row, (3, 1))), cls_ids=names))
        truths.append(names[i % 3] if i % 5 else names[(i + 1) % 3])
    used = [SimpleNamespace(clip_id=f"c{i}", group=f"g{i % 3}", source="include") for i in range(len(inputs))]
    monkeypatch.setattr(ff, "collect_evidence", lambda rec, clips: (inputs, truths, used))
    fusion = FusionParams(gamma=0.0, accept={"strict": 0.95, "balanced": 0.8, "lenient": 0.5},
                          suggest_min_top5_mass_by_policy={"strict": 0.0, "balanced": 0.0, "lenient": 0.5},
                          suggest_min_top5_mass=0.0, fitted=True,
                          achieved={pol: {"target": TARGETS[pol], "target_reached": False, "rule": "fallback",
                                          "precision_lower_bound": 0.8} for pol in POLICIES})
    out = runners.e3_fused(SimpleNamespace(fusion=fusion), used, "E3")
    pols = out["policies"]
    assert [pols[p]["threshold"] for p in POLICIES] == [0.95, 0.8, 0.5]
    assert pols["strict"]["coverage"] < pols["balanced"]["coverage"] < pols["lenient"]["coverage"]
    assert pols["lenient"]["suggest_min_top5_mass"] == 0.5 and pols["strict"]["val_target_reached"] is False
    assert out["metrics_policy"] == "balanced" and out["metrics"]["coverage"] == pols["balanced"]["coverage"]
    doc = {"profile": "research", "label": "x", "created": "now", "smoke": False,
           "protocols": {"E3": out, "E1b": {"run": True, "n": 4, "label": "E1b", "metrics": {"top1": 0.75},
                                            "decision": {"source": "fusion.json classifier_accept"},
                                            "policies": {p: {"threshold": t, "coverage": c} for p, t, c in
                                                         (("strict", 0.96, 0.3), ("balanced", 0.74, 0.6),
                                                          ("lenient", 0.46, 0.8))}}},
           "leakage_guards": {"pretraining_check": {"reason": "none"}}}
    md = runners.markdown_report(doc)
    assert "## Decision policies" in md
    rows = [ln for ln in md.splitlines() if ln.startswith(("| E3 | ", "| E1b | "))
            and ln.split("|")[2].strip() in POLICIES]
    assert len(rows) == 6
    assert any("| E3 | lenient | 0.500 |" in r and "NOT reached" in r for r in rows)
    assert any("| E1b | strict | 0.960 | 0.300 |" in r for r in rows)
    header = next(ln for ln in md.splitlines() if ln.startswith("| Protocol | policy"))
    assert all(r.count("|") == header.count("|") for r in rows)


# ------------------------------------------------------------------------------- per-source certification (15.25)

def _two_source_set(prec_a=0.99, prec_b=0.90, n_a=3000, n_b=300, groups_a=6, groups_b=4):
    """Fused results of two sources whose precision does not depend on p1: A right ``prec_a`` of the time, B
    ``prec_b``. Errors are spread evenly over each source's signer groups (homogeneous: DEFF about 1)."""
    results, truths, sources, groups = [], [], [], []
    for src, n, prec, ng in (("A", n_a, prec_a, groups_a), ("B", n_b, prec_b, groups_b)):
        for i in range(n):
            p1 = 0.5 + 0.49 * ((i * 0.618034) % 1.0)
            ok = ((i // ng) * PHI) % 1.0 < prec
            truth = f"w{i % 7}"
            top = truth if ok else "other"
            rest = [k for k in (truth, "x1", "x2", "x3", "x4", "x5") if k != top][:5]
            results.append(FusionResult([top, *rest], np.array([p1] + [(1 - p1) / 5] * 5), [top] * 3, 3, 0.9, 0.9,
                                        None, {}))
            truths.append(truth)
            sources.append(src)
            groups.append(f"{src}{i % ng}")
    return results, truths, sources, groups


def test_pooled_rule_passes_but_the_worst_source_fails():
    res, tr, src, grp = _two_source_set()
    pooled, prep = fit_thresholds(res, tr, FusionParams())  # no sources: the pre-15.25 pooled clip-level bound
    assert pooled.policy_rule("strict") == "certified"  # 3,240 of 3,300 right: pooled bound about 0.977
    p, rep = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp)
    af = rep["accept_fit"]
    assert af["major_sources"] == ["A", "B"]
    for pol in ("strict", "balanced"):
        assert p.policy_rule(pol) == "confirm_only" and p.achieved[pol]["worst_source"] == "B"
        assert p.achieved[pol]["precision_lower_bound"] < FLOORS[pol] <= p.achieved[pol]["pooled_lower_bound"]
    by = p.achieved["strict"]["by_source"]
    assert set(by) == {"A", "B"} and by["A"]["lower_bound"] > 0.98 > by["B"]["lower_bound"]
    for row in by.values():  # what fusion.json reports per source
        assert {"precision", "lower_bound", "coverage", "n_groups", "deff"} <= set(row)
    assert by["B"]["n_groups"] == 4 and by["B"]["deff"] == pytest.approx(1.0)
    # B alone (0.90, n=300) has a bound of about 0.86: lenient (floor 0.85) is certified on the worst source
    assert p.policy_rule("lenient") == "certified" and p.achieved["lenient"]["certified_precision"] >= 0.85
    # the stored parameters gate decide(): a confident query is a "did you mean" under strict, accepted under lenient
    probs = np.array([0.999, 0.0005, 0.0005])
    assert decide(probs, 3, 3, p, "strict", 0.95) == ("suggest", "confirm_only")
    assert decide(probs, 3, 3, p, "lenient", 0.95) == ("accept", "ok")
    assert p.certification["version"] and p.certification["policies"]["strict"] == "confirm_only"


def test_a_source_too_small_to_be_major_cannot_hide_the_others_but_is_not_certified_on():
    res, tr, src, grp = _two_source_set(n_b=20)  # B: 20 queries, not a major source (needs >= 30 accepts)
    p, rep = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp)
    assert rep["accept_fit"]["major_sources"] == ["A"] and p.achieved["strict"]["by_source"]["B"]["major"] is False
    assert p.policy_rule("strict") == "certified"
    res1, tr1, src1, grp1 = _two_source_set(n_b=0, groups_a=1)  # one source, one signer group: nothing to certify
    q, _ = fit_thresholds(res1, tr1, FusionParams(), sources=src1, groups=grp1)
    assert all(q.policy_rule(pol) == "confirm_only" for pol in POLICIES)
    assert "no major source" in q.achieved["strict"]["reason"]


def test_design_effect_lowers_the_bound_only_for_heterogeneous_groups():
    from app.isl.models.calibration import DEFF_FALLBACK, pearson_dispersion, scan_by_source, wilson_lower_deff

    def matrix(errors_per_group, n_per_group=100):
        acc, ok, grp = [], [], []
        for g, e in enumerate(errors_per_group):
            for j in range(n_per_group):
                acc.append(True)
                ok.append(j >= e)
                grp.append(f"g{g}")
        return np.array([acc]), np.array([ok]), grp

    homo = matrix([10] * 6)
    hetero = matrix([0, 0, 0, 20, 20, 20])  # the same 60 errors in 600, all from three signers
    bh = scan_by_source(*homo[:2], sources=["s"] * 600, groups=homo[2])["s"]
    bx = scan_by_source(*hetero[:2], sources=["s"] * 600, groups=hetero[2])["s"]
    assert bh["deff"][0] == 1.0 and bh["lb"][0] == pytest.approx(wilson_lower(540, 600))
    assert bx["deff"][0] > 2.0 and bx["lb"][0] < bh["lb"][0] - 0.01
    assert bx["lb"][0] == pytest.approx(wilson_lower_deff(540, 600, bx["deff"][0]))
    assert pearson_dispersion(np.array([100, 100]), np.array([0, 20])) is None  # < 3 groups: not estimable
    # fewer than 3 groups: the prior (e.g. from cross-validation groups) or the 5.0 fallback
    two = matrix([10, 10])
    b2 = scan_by_source(*two[:2], sources=["s"] * 200, groups=two[2])["s"]
    assert b2["deff"][0] == DEFF_FALLBACK and b2["phi_from"][0] == "fallback"
    b3 = scan_by_source(*two[:2], sources=["s"] * 200, groups=two[2], phi_prior={"s": 2.0})["s"]
    assert b3["deff"][0] == 2.0 and b3["phi_from"][0] == "prior"


def test_a_weak_scan_is_confirm_only_everywhere_and_decide_never_accepts():
    res, tr = _ceiling_set(ceiling=0.6)  # best bound about 0.58
    src = ["include" if i % 2 else "isl500" for i in range(len(res))]
    grp = [f"{s}-{i % 5}" for i, s in enumerate(src)]
    p, rep = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp)
    assert rep["accept_fit"]["best_lower_bound"] < 0.62
    assert all(p.policy_rule(pol) == "confirm_only" for pol in POLICIES)
    rng = np.random.default_rng(0)
    for _ in range(300):
        probs = np.sort(rng.dirichlet(np.full(6, rng.uniform(0.02, 2.0))))[::-1]
        for pol in POLICIES:
            assert decide(probs, 3, 3, p, pol, best_cos=0.99)[0] != "accept"
    for pol in POLICIES:
        assert p.achieved[pol]["auto_accept"] is False
        from app.isl.models.calibration import summarize
        assert summarize(res, tr, p, pol)["n_accepted"] == 0  # what the app does
        assert p.achieved[pol]["n_accepted"] > 0  # what the threshold alone would accept (the confirm tap)


def test_the_fine_grid_above_099_lets_strict_differ_from_balanced():
    from app.isl.models.calibration import THRESHOLD_GRID

    assert {0.991, 0.995, 0.999} <= set(np.round(THRESHOLD_GRID, 3)) and THRESHOLD_GRID.max() == 0.999
    results, truths = [], []

    def add(n, lo, hi, prec):
        for i in range(n):
            p1 = lo + (hi - lo) * i / n
            ok = (i * PHI) % 1.0 < prec
            top = "w" if ok else "o"
            results.append(FusionResult([top, "x1", "x2"], np.array([p1, (1 - p1) / 2, (1 - p1) / 2]), [top] * 3, 3,
                                        0.9, 0.9, None, {}))
            truths.append("w")

    add(2000, 0.995, 0.9999, 1.0)  # very confident: always right
    add(2000, 0.99, 0.995, 0.93)
    add(4000, 0.5, 0.99, 0.70)
    p, rep = fit_thresholds(results, truths, FusionParams())
    a = p.accept
    assert a["strict"] > 0.99 and a["strict"] > a["balanced"], a
    assert p.policy_rule("strict") == "certified" and p.achieved["strict"]["target_reached"]
    assert p.policy_rule("balanced") == "certified"


def test_s_min_is_the_minimum_over_major_sources():
    from app.isl.models.calibration import _fit_s_min

    rng = np.random.default_rng(1)
    cos = np.r_[rng.uniform(0.8, 1.0, 200), rng.uniform(0.5, 0.9, 100), np.full(25, 0.1)]
    src = ["isl500"] * 200 + ["include"] * 100 + ["tiny"] * 25  # tiny: < 30 queries, not a major source
    grp = [f"g{i % 4}" for i in range(len(cos))]
    s_min, rep = _fit_s_min(cos, src, grp)
    q_inc = float(np.quantile(cos[200:300], 0.05))
    assert s_min == pytest.approx(q_inc) and s_min < float(np.quantile(cos[:200], 0.05))
    assert s_min > float(np.quantile(cos, 0.05))  # the pooled quantile is pulled down by the tiny source
    assert rep["by_source"]["include"]["pass_rate"] >= 0.95 and "major" not in rep["by_source"]["tiny"]
    pooled, prep = _fit_s_min(cos)  # no sources: the pooled quantile, as before
    assert pooled == pytest.approx(float(np.quantile(cos, 0.05))) and "pooled" in prep["rule"]
    # fit_thresholds stores the per-source guard
    res = [FusionResult(["w"], np.array([0.9]), ["w"] * 3, 3, float(c), float(c), None, {}) for c in cos]
    p, frep = fit_thresholds(res, ["w"] * len(res), FusionParams(), sources=src, groups=grp)
    assert p.s_min == pytest.approx(q_inc) and frep["s_min_fit"]["s_min"] == pytest.approx(q_inc, abs=1e-4)


def test_suggest_mass_is_chosen_on_the_worst_source():
    p = FusionParams(accept={"strict": 0.9, "balanced": 0.7, "lenient": 0.5}, s_min=-1.0)
    res = ([_res(0.3, 0.5, True)] * 30 + [_res(0.3, 0.95, True)] * 30  # A: always in the top 5
           + [_res(0.3, 0.5, False)] * 20 + [_res(0.3, 0.9, True)] * 20)  # B: only the confident ones
    src = ["A"] * 60 + ["B"] * 40
    truths = ["t"] * len(res)
    m_pooled, r_pooled = _fit_suggest_mass(res, truths, p, "balanced", 0.25)
    assert m_pooled == 0.0 and r_pooled["top5_hit"] == pytest.approx(0.8)  # pooled: 80 % at mass 0
    m, rep = _fit_suggest_mass(res, truths, p, "balanced", 0.25, sources=src, major=["A", "B"])
    assert 0.45 < m <= 0.9 and rep["target_reached"] and rep["worst_top5_hit"] >= 0.8 and rep["worst_source"]
    assert rep["by_source"]["B"]["top5_hit"] == 1.0 and rep["by_source"]["B"]["shown"] == 20


def test_old_fusion_json_is_uncertified_confirm_only(tmp_path):
    import json

    (tmp_path / "fusion.json").write_text(json.dumps(OLD_PACK))
    p = FusionParams.load(tmp_path / "fusion.json")
    assert p.gated and p.certification == {}
    probs = np.array([0.99, 0.005, 0.005])  # above the old 0.96: was accepted before 15.25
    for pol in POLICIES:
        assert p.policy_rule(pol) == "confirm_only"
        assert decide(probs, 3, 3, p, pol, best_cos=0.9) == ("suggest", "confirm_only")
        info = p.certification_info(pol)
        assert info["rule"] == "confirm_only" and info["certified_precision"] is None and info["confirm_tap"]
        assert "no certification record" in info["reason"]
    # a pre-15.25 achieved block (rule "fallback", no by_source) is not a certification either
    old = dict(OLD_PACK, achieved={pol: {"rule": "target", "target_reached": True} for pol in POLICIES},
               certification={"version": "cert-1"})
    q = FusionParams.from_json(old)
    assert all(q.confirm_only(pol) for pol in POLICIES)
    # in-code parameters (tests, toy tools) are not gated
    assert FusionParams().policy_rule("strict") == "ungated"
    assert decide(probs, 3, 3, FusionParams(), "strict") == ("accept", "ok")


def test_open_vocabulary_words_stay_confirm_only_unless_certified():
    # B: about 7 errors in 1,500, too few to estimate its dispersion, so DEFF is the 5.0 fallback (docs 15.26)
    res, tr, src, grp = _two_source_set(prec_b=0.995, n_b=1500)
    closed = {f"w{i}" for i in range(7)} | {"other"}  # every top-1 here is a classifier class
    p, rep = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp, closed_set=closed)
    assert p.policy_rule("strict") == "certified"
    ov = rep["open_vocabulary"]["policies"]["strict"]
    assert ov["rule"] == "confirm_only" and ov["n_accepted"] == 0  # no open-vocabulary accept to certify on
    probs = np.array([0.999, 0.0005, 0.0005])
    assert decide(probs, 3, 3, p, "strict", 0.95) == ("accept", "ok")
    assert decide(probs, 3, 3, p, "strict", 0.95, open_vocabulary=True) == ("suggest", "open_vocabulary")
    # retrieval only (no classifier): every word is open vocabulary, certified only at the strict floor
    q, qrep = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp, closed_set=set())
    for pol in POLICIES:
        o = qrep["open_vocabulary"]["policies"][pol]
        assert o["rule"] == ("certified" if q.policy_rule(pol) == "certified" and o["worst_lower_bound"] >= 0.95
                             else "confirm_only")
    assert q.achieved["strict"]["open_vocabulary_rule"] == "certified"
    # no class list given: open-vocabulary words are never auto-accepted
    r, _ = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp)
    assert decide(probs, 3, 3, r, "strict", 0.95, open_vocabulary=True) == ("suggest", "open_vocabulary")


def test_recognizer_exposes_the_certified_worst_source_precision():
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    res, tr, src, grp = _two_source_set()
    p, _ = fit_thresholds(res, tr, FusionParams(), sources=src, groups=grp)
    clf = _Clf([[0.98, 0.01, 0.01]] * 3, {"temperature": 1.0, "recommended_threshold": 0.6})
    rec = IslRecognizer("isl", clf.feature_config, Vocabulary([]), classifier=clf, fusion=p)
    info = rec.certification_info("lenient")
    assert info["rule"] == "certified" and info["certified_precision"] == p.achieved["lenient"]["certified_precision"]
    assert info["worst_source"] == "B" and info["floor"] == 0.85 and info["target"] == 0.90 and info["confirm_tap"]
    assert rec.certification_info("strict")["rule"] == "confirm_only"
    assert rec.is_open_vocabulary("isl.not_a_class") and not rec.is_open_vocabulary("a")
    assert not rec.is_open_vocabulary("mysign:abc")
    # no fusion.json: the recogniser's own defaults are gated and uncertified
    bare = IslRecognizer("isl", clf.feature_config, Vocabulary([]), classifier=clf)
    assert bare.fusion.gated and all(bare.fusion.confirm_only(pol) for pol in POLICIES)
    # the sliding rule: the old pack's per-policy classifier thresholds carry no certification
    assert _rec(clf, _fitted_pack(clf)).sliding_rule("lenient") == "confirm_only"
    assert _rec(clf, FusionParams(fitted=True)).sliding_rule("lenient") == "ungated"


def test_e3_per_source_rows_and_the_headline(monkeypatch):
    from app.isl.evaluation import runners
    from app.isl.training import fit_fusion as ff

    names = ["a", "b", "c"]
    inputs, truths, used = [], [], []
    for i, p1 in enumerate(np.linspace(0.5, 0.99, 240)):  # 120 per source: big enough for the headline (15.26)
        row = np.full(3, (1 - p1) / 2)
        row[i % 3] = p1
        inputs.append(FusionInputs(cls_logp=np.log(np.tile(row, (3, 1))), cls_ids=names))
        src = "include" if i % 2 else "isl500"
        wrong = (i % 7 == 0) if src == "include" else (i % 3 == 0)  # isl500 is the worse source here
        truths.append(names[(i + 1) % 3] if wrong else names[i % 3])
        used.append(SimpleNamespace(clip_id=f"c{i}", group=f"{src}{i % 3}", source=src))
    monkeypatch.setattr(ff, "collect_evidence", lambda rec, clips: (inputs, truths, used))
    ach = {pol: {"rule": "certified" if pol == "lenient" else "confirm_only", "by_source": {},
                 "certified_precision": 0.9 if pol == "lenient" else None, "target": TARGETS[pol],
                 "floor": FLOORS[pol]} for pol in POLICIES}
    fusion = FusionParams(gamma=0.0, accept={"strict": 0.9, "balanced": 0.8, "lenient": 0.5}, fitted=True,
                          achieved=ach, certification={"version": "cert-1"})
    rec = SimpleNamespace(fusion=fusion, classifier=SimpleNamespace(class_names=names))
    out = runners.e3_fused(rec, used, "E3")
    pols = out["policies"]
    for pol in ("strict", "balanced"):  # confirm-only: nothing auto-accepted, the threshold rule still measured
        assert pols[pol]["rule"] == "confirm_only" and pols[pol]["coverage"] == 0.0
        assert pols[pol]["threshold_rule"]["coverage"] > 0
    assert pols["lenient"]["rule"] == "certified" and pols["lenient"]["coverage"] > 0
    by = pols["lenient"]["by_source"]
    assert set(by) == {"include", "isl500"} and by["isl500"]["n_groups"] == 3
    assert pols["lenient"]["worst_source"] == "isl500"
    assert pols["lenient"]["val_certified_precision"] == 0.9 and pols["strict"]["certification"]["confirm_tap"]
    md = runners.markdown_report({"profile": "research", "label": "x", "created": "now", "smoke": False,
                                  "protocols": {"E3": out}, "leakage_guards": {"pretraining_check": {}}})
    head = [ln for ln in md.splitlines() if ln.startswith("| E3 fused")]
    assert any("INCLUDE: include" in ln for ln in head) and any("worst source: isl500" in ln for ln in head)
    assert md.index("## Headline") < md.index("| Protocol | n |")  # the headline comes first


def test_oov_false_accepts_count_every_accept_as_wrong():
    from app.isl.evaluation.runners import e2_pool, oov_false_accepts

    res = [FusionResult(["x", "y"], np.array([0.97, 0.01]), ["x"] * 3, 3, 0.9, 0.9, None, {})] * 3 + \
          [FusionResult(["x", "y"], np.array([0.4, 0.3]), ["x"] * 3, 3, 0.2, 0.2, None, {})]
    gated = FusionParams(accept={"strict": 0.95, "balanced": 0.9, "lenient": 0.5}, s_min=0.5,
                         certification={"version": "cert-1"})
    o = oov_false_accepts(res, gated)
    assert o["strict"]["threshold_false_accept_rate"] == 0.75 and o["strict"]["false_accept_rate"] == 0.0
    assert o["strict"]["out_of_vocabulary_rate"] == 0.25
    pooled = e2_pool([{"run": True, "oov_false_accept": o}, {"run": True, "oov_false_accept": o}])
    assert pooled["oov_false_accept"]["n"] == 8 and pooled["oov_false_accept"]["strict"]["threshold_false_accept_rate"] == 0.75


# ------------------------------------------------------------------- review of 15.25: major sources, DEFF (15.26)

def _renamed(results, sources, groups, name, best_cos=None):
    from dataclasses import replace

    res = [replace(r, best_cos=best_cos, best_global_cos=best_cos) if best_cos is not None else r for r in results]
    return res, [name] * len(sources), [f"{name}-{g}" for g in groups]


def test_a_39_clip_source_is_reported_but_never_major():
    """Mendeley on research: 39 clips, 3 signers, poor precision. wilson(39, 39) = 0.91 < 0.95, so as a major source
    it made strict uncertifiable by construction and set every threshold and s_min (review of 15.25)."""
    from app.isl.models.calibration import MAJOR_MIN_CLIPS, is_major_size

    assert wilson_lower(39, 39) < FLOORS["strict"] <= wilson_lower(MAJOR_MIN_CLIPS, MAJOR_MIN_CLIPS)
    assert not is_major_size(39, 3, 35) and is_major_size(100, 2, 30) and not is_major_size(500, 1, 400)
    res, tr, src, grp = _two_source_set(prec_b=0.995, n_b=1500)  # A = include-like, B = isl500-like: both strong
    m_res, m_tr, m_src, m_grp = _two_source_set(prec_a=0.6, n_a=39, groups_a=3, n_b=0)
    m_res, m_src, m_grp = _renamed(m_res, m_src, m_grp, "mendeley", best_cos=0.3)  # far below the others' cosines
    p, rep = fit_thresholds(res + m_res, tr + m_tr, FusionParams(), sources=src + m_src, groups=grp + m_grp)
    af = rep["accept_fit"]
    assert af["major_sources"] == ["A", "B"] and af["sources"]["mendeley"]["major"] is False
    assert "too small to certify" in af["sources"]["mendeley"]["note"]
    row = p.achieved["strict"]["by_source"]["mendeley"]
    assert row["major"] is False and row["n"] == 39 and row["n_groups"] == 3 and "too small" in row["note"]
    assert p.policy_rule("strict") == "certified" and p.achieved["strict"]["worst_source"] in ("A", "B")
    # it does not set the out-of-vocabulary guard either: s_min stays at the major sources' 5 % quantile
    assert p.s_min == pytest.approx(0.9) and rep["s_min_fit"]["by_source"]["mendeley"]["pass_rate"] == 0.0
    # nor is it judged for the "did you mean" mass
    for pol in POLICIES:
        assert "mendeley" not in (rep["suggest_fit"][pol].get("worst_source") or "")
    # an explicit list (e.g. from a profile) overrides the size rule
    q, qrep = fit_thresholds(res + m_res, tr + m_tr, FusionParams(), sources=src + m_src, groups=grp + m_grp,
                             major_sources=["A", "mendeley"])
    assert qrep["accept_fit"]["major_sources"] == ["A", "mendeley"] and q.policy_rule("strict") == "confirm_only"
    assert q.achieved["strict"]["worst_source"] == "mendeley" and "listed" in q.certification["major_rule"]


def test_deff_is_carried_up_to_thresholds_where_the_errors_run_out():
    """Errors clustered in three of six signers at a low threshold and none at a high one: before, phi was
    re-estimated per row and fell to 1 at the high row, which certified on clip-level counts (review of 15.25)."""
    from app.isl.models.calibration import PHI_MIN_ERRORS, pearson_dispersion, scan_by_source, wilson_lower_deff

    def matrix(errors_per_group, n_per_group=100):
        acc, ok, grp = [[], [], []], [[], [], []], []
        for g, e in enumerate(errors_per_group):
            for j in range(n_per_group):
                wrong = j < e
                acc[0].append(True)  # low threshold: everything accepted
                acc[1].append(not wrong or j < 1)  # mid: one error per erring group left (3 < PHI_MIN_ERRORS)
                acc[2].append(not wrong)  # high: error-free
                for row in ok:
                    row.append(not wrong)
                grp.append(f"g{g}")
        a = np.array(acc)
        return a, a & np.array(ok), grp

    acc, ok, grp = matrix([0, 0, 0, 20, 20, 20])
    b = scan_by_source(acc, ok, sources=["s"] * 600, groups=grp)["s"]
    phi_low = pearson_dispersion(np.full(6, 100), np.array([0, 0, 0, 20, 20, 20]))
    assert phi_low > 2 and (b["n_acc"] - b["n_ok"]).tolist() == [60, 3, 0] and 3 < PHI_MIN_ERRORS
    assert b["phi_rows"] == 1 and b["phi"] == pytest.approx(phi_low, abs=1e-4)
    assert all(d >= phi_low - 1e-9 for d in b["deff"])  # carried up: never back to 1 at the high rows
    assert b["lb"][2] == pytest.approx(wilson_lower_deff(540, 540, phi_low))
    assert b["lb"][2] < wilson_lower(540, 540) - 0.01
    # homogeneous errors at the low threshold: the same scan keeps DEFF 1 everywhere (nothing to penalise)
    acc, ok, grp = matrix([10] * 6)
    h = scan_by_source(acc, ok, sources=["s"] * 600, groups=grp)["s"]
    assert list(h["deff"]) == [1.0, 1.0, 1.0] and h["lb"][2] == pytest.approx(wilson_lower(540, 540))
    # never an error at all: nothing to estimate, so the prior or the 5.0 fallback, never 1
    acc, ok, grp = matrix([0] * 6)
    z = scan_by_source(acc, ok, sources=["s"] * 600, groups=grp)["s"]
    assert z["phi_from"][0] == "fallback" and z["deff"][2] == 5.0


def test_a_user_sign_is_never_auto_accepted_under_a_confirm_only_policy(monkeypatch):
    """A direct "My signs" match was accepted after decide() even when the policy was confirm-only (review of
    15.25). Now it stays the top "did you mean" candidate; in-code parameters (ungated) keep the old rule."""
    from app.isl import recognizer as R
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    vocab = Vocabulary([{"concept_id": "a", "headword": "a", "display": {"en": "a"}}])
    ukey = "mysign:x"
    fused = FusionResult([ukey, "a"], np.array([0.6, 0.4]), [ukey] * 3, 3, 0.95, 0.5, (ukey, 0.95), {})
    monkeypatch.setattr(R, "fuse", lambda inp, p: fused)

    def make(fusion):
        rec = IslRecognizer("isl", FeatureConfig(canonical_hand=True), vocab, fusion=fusion)
        rec._features = lambda crops: np.zeros((3, 4, 2), np.float32)
        rec.evidence = lambda feats, use=True: (FusionInputs(user=np.full((3, 1), 0.95), user_ids=[ukey]), {"z": None})
        rec._user_theta = lambda key: 0.8
        rec._candidates = lambda keys, probs, comps, z, top_k: []
        return rec

    ungated = make(FusionParams()).recognize([None] * 3, policy="balanced")
    assert (ungated.decision, ungated.reason) == ("accept", "my_signs")
    for fusion in (FusionParams(certification={}),  # a pack without a certification record (old fusion.json)
                   FusionParams.from_json(OLD_PACK)):
        rec = make(fusion)
        for pol in POLICIES:
            r = rec.recognize([None] * 3, policy=pol)
            assert (r.decision, r.reason) == ("suggest", "confirm_only")
            assert r.detail["certification"]["rule"] == "confirm_only"
