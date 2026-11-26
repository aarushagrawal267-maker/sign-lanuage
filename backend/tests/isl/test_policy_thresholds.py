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
from app.isl.models.calibration import (TARGETS, _fit_suggest_mass, choose_policy_thresholds, fit_thresholds,
                                        wilson_lower)
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

def test_unreachable_targets_give_distinct_ordered_thresholds_and_say_so():
    results, truths = _ceiling_set()
    p, rep = fit_thresholds(results, truths, FusionParams())
    a = p.accept
    assert a["strict"] > a["balanced"] > a["lenient"], a  # before 15.17: all three equal
    best = rep["accept_fit"]["best_lower_bound"]
    assert best < TARGETS["lenient"]
    pols = rep["accept_fit"]["policies"]
    for pol in POLICIES:
        assert rep[f"{pol}_target_reached"] is False and pols[pol]["rule"] == "fallback"
        assert pols[pol]["precision_lower_bound"] >= pols[pol]["effective_target"] - 1e-9
        # the report and the stored params both carry the bound each policy actually achieves
        assert p.achieved[pol]["target_reached"] is False and p.achieved[pol]["rule"] == "fallback"
        assert p.achieved[pol]["precision_lower_bound"] == pols[pol]["precision_lower_bound"]
        assert p.achieved[pol]["threshold"] == a[pol]
    # the targets' spacing below the best bound: 0 / 0.03 / 0.08
    assert pols["strict"]["effective_target"] == pytest.approx(best, abs=1e-4)
    assert pols["balanced"]["effective_target"] == pytest.approx(best - 0.03, abs=1e-4)
    assert pols["lenient"]["effective_target"] == pytest.approx(best - 0.08, abs=1e-4)
    assert pols["strict"]["threshold"] == rep["accept_fit"]["best_threshold"]  # strictest: the best bound itself
    assert "not reachable" in rep["accept_fit"]["note"] and "lenient" in rep["accept_fit"]["note"]
    # more lenient = more accepted, lower precision
    n = [p.achieved[pol]["n_accepted"] for pol in POLICIES]
    assert n[0] < n[1] < n[2]


def test_flat_precision_cannot_be_split_and_ties_are_not_invented():
    # precision does not depend on p1 at all: every threshold has the same precision, so the best bound is at the
    # lowest threshold (most data) and nothing is gained by being stricter. One threshold for all is honest here.
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(1000, 100, len(ts)).astype(int)
    n_ok = (0.8 * n_acc).astype(int)
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    assert acc["strict"] == acc["balanced"] == acc["lenient"] == 0.2
    assert all(rep["policies"][p]["rule"] == "fallback" for p in POLICIES)


def test_partly_reachable_targets_keep_the_reached_rule_and_spread_the_others():
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(2000, 600, len(ts)).astype(int)
    prec = np.linspace(0.80, 0.955, len(ts))  # lenient (0.90) reachable, balanced (0.95) and strict not
    n_ok = np.round(prec * n_acc).astype(int)
    lbs = np.array([wilson_lower(int(k), int(m)) for k, m in zip(n_ok, n_acc)])
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    pols = rep["policies"]
    assert pols["lenient"]["rule"] == "target" and pols["lenient"]["target_reached"]
    assert acc["lenient"] == float(ts[np.flatnonzero(lbs >= 0.90)[0]])  # the pre-15.17 rule for a reached target
    assert pols["balanced"]["rule"] == pols["strict"]["rule"] == "fallback"
    assert 0.90 < pols["balanced"]["effective_target"] < pols["strict"]["effective_target"] == pytest.approx(lbs.max(), abs=1e-4)
    assert acc["strict"] > acc["balanced"] > acc["lenient"]


def test_reachable_targets_unchanged_from_the_old_rule():
    ts = np.round(np.arange(0.2, 0.991, 0.01), 2)
    n_acc = np.linspace(5000, 3000, len(ts)).astype(int)
    n_ok = np.round(np.linspace(0.85, 0.999, len(ts)) * n_acc).astype(int)
    lbs = np.array([wilson_lower(int(k), int(m)) for k, m in zip(n_ok, n_acc)])
    acc, rep = choose_policy_thresholds(ts, n_acc, n_ok, dict(TARGETS))
    for pol, tg in TARGETS.items():
        assert rep["policies"][pol]["rule"] == "target"
        assert acc[pol] == float(ts[np.flatnonzero(lbs >= tg)[0]])
    assert "note" not in rep


def test_too_little_data_keeps_defaults():
    p0 = FusionParams()
    acc, rep = choose_policy_thresholds(np.array([0.5, 0.6]), np.array([2, 1]), np.array([2, 1]), dict(TARGETS))
    assert acc is None and all(v["rule"] == "default" for v in rep["policies"].values())
    res, tr = _ceiling_set(3)
    p, rep2 = fit_thresholds(res, tr, p0)
    assert p.accept == p0.accept and rep2["strict_target_reached"] is False


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
                     classifier_fit={"fingerprint": "abc", "temperature": 0.72})
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
    assert a["strict"] > a["balanced"] > a["lenient"]
    assert p.classifier_fit["fingerprint"] == classifier_fingerprint(clf) and p.classifier_fit["temperature"] == 1.0
    assert rep["n"] == len(inputs) and rep["n_in_classes"] == len(inputs) - 1
    assert all(rep["policies"][pol]["target_reached"] is False for pol in POLICIES) and "not reachable" in rep["note"]
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
