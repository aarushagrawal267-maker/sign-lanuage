"""Sliding-mode thresholds are fitted on what sliding mode actually does (docs/ISL_DESIGN.md 15.18).

Regression for 93d2d5c (docs 15.17): the per-policy classifier-alone thresholds were fitted on a three-crop segment
proxy (crop-mean probability of the whole clip, >= 2 of 3 crops agree), but the app's sliding mode classifies
single-crop windows, smooths them (EMA), needs a streak of 3 windows and commits possibly several times per sign.
Replaying the commercial-safe validation clips through the real engine, the proxy's thresholds 0.96 / 0.74 / 0.46 had
sliding-mode lower bounds 0.819 / 0.763 / 0.658, not the 0.865 / 0.835 / 0.785 stored in fusion.json, and on test the
lenient 0.46 gave a wrong commit on 23.5 % of clips (0.60 before: 17.6 %).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.features.spec import FeatureConfig
from app.isl.models.calibration import (THRESHOLD_GRID, fit_sliding_thresholds, sliding_commit_counts,
                                        sliding_commits, wilson_lower)
from app.isl.models.fusion import POLICIES, FusionInputs, FusionParams

NAMES = ("a", "b", "c")


class _ScriptedClf:
    """Classifier stub: call k returns ``script[k % len]`` (one row per crop), so a replay and a live engine run see
    the same window probabilities in the same order."""

    def __init__(self, script, temperature=1.0, threshold=0.6):
        self.script = [np.asarray(p, float) for p in script]
        self.k = 0
        self.class_names = list(NAMES)
        self.meta = {"split": "s", "class_names": list(NAMES), "temperature": temperature,
                     "recommended_threshold": threshold}
        self.feature_config = FeatureConfig(canonical_hand=True)

    def predict(self, feats):
        p = self.script[self.k % len(self.script)]
        self.k += 1
        return np.repeat(p[None], len(feats), 0), 1.0


def _script(seed=0, n=40):
    """Dominant class switching every few windows, confidence varying: commits, repeats, ambiguous windows."""
    rng = np.random.default_rng(seed)
    out, cls = [], 0
    for i in range(n):
        if i % int(rng.integers(3, 6)) == 0:
            cls = int(rng.integers(0, 3))
        top = float(rng.uniform(0.4, 0.99))
        rest = rng.dirichlet([1, 1]) * (1 - top)
        p = np.insert(rest, cls, top)
        out.append(p)
    return out


def _rec(clf):
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    vocab = Vocabulary([{"concept_id": n, "headword": n, "display": {"en": n}} for n in NAMES])
    return IslRecognizer("isl", clf.feature_config, vocab, classifier=clf, fusion=FusionParams(fitted=True))


def _raw(n_frames=90, gap=(40, 55), seed=1):
    """A synthetic clip with hands, and the hands gone for frames ``gap`` (>= end_absent_ms: a sign boundary)."""
    from app.dataset.synthetic import make_sequence

    raw = make_sequence(1, seed, n_frames=n_frames)
    if gap:
        raw.left[gap[0]:gap[1]] = np.nan
        raw.right[gap[0]:gap[1]] = np.nan
    return raw


def _live_commits(clf, raw, threshold):
    from app.inference.engine import EngineConfig, RecognitionEngine

    clf.k = 0
    eng = RecognitionEngine(None, recognizer=_rec(clf), config=EngineConfig(mode="sliding", threshold=threshold))
    eng.set_size(raw.width, raw.height)
    out = []
    for i in range(len(raw.t_ms)):
        msg = eng.push(raw.frame_to_json(i))
        if "commit" in msg:
            out.append(NAMES.index(msg["commit"]["concept_id"]))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_replayed_commits_equal_the_live_engine_for_every_threshold(seed):
    from app.inference.engine import EngineConfig
    from app.isl.training.fit_fusion import sliding_windows

    clf = _ScriptedClf(_script(seed))
    raw = _raw(seed=seed + 1)
    clf.k = 0
    windows = sliding_windows(_rec(clf), raw)
    assert len(windows) >= 10 and any(w[3] for w in windows)  # several windows and one sign boundary
    c = EngineConfig()
    total = 0
    for thr in (0.3, 0.45, 0.6, 0.75, 0.9):
        live = _live_commits(clf, raw, thr)
        assert sliding_commits(windows, thr, c.min_margin, c.agree_n, c.cooldown_ms) == live, thr
        total += len(live)
    assert total >= 3  # the scripts do commit, so the comparison is not vacuous


def test_commit_rule_streak_margin_repeat_boundary_and_cooldown():
    p = lambda a: np.array([a, (1 - a) / 2, (1 - a) / 2])  # noqa: E731
    w = [(0, p(0.9), 1, False), (200, p(0.9), 2, False), (400, p(0.9), 3, False),  # streak reached at 400
         (600, p(0.9), 4, False),  # same class again: suppressed (repeat)
         (1400, p(0.9), 5, True),  # after a sign boundary the same word may be committed again
         (1600, np.array([0.1, 0.85, 0.05]), 3, False)]  # another word, but within the 700 ms cooldown
    assert sliding_commits(w, 0.8, 0.1, 3, 700.0) == [0, 0]
    assert sliding_commits(w, 0.95, 0.1, 3, 700.0) == []
    amb = [(t, np.array([0.5, 0.45, 0.05]), 5, False) for t in (0, 1000)]  # margin 0.05 < 0.10
    assert sliding_commits(amb, 0.3, 0.1, 3, 700.0) == []
    # the engine's class key decides repeats: two classes with the same key are one word to the engine
    keyed = [(0, p(0.9), 3, False, 7), (1000, np.array([0.05, 0.9, 0.05]), 3, False, 7)]
    assert sliding_commits(keyed, 0.8, 0.1, 3, 700.0) == [0]


def test_counts_are_per_clip_and_a_clip_with_any_wrong_commit_is_wrong():
    good = [(0, np.array([0.9, 0.05, 0.05]), 3, False)]
    mixed = good + [(1000, np.array([0.05, 0.9, 0.05]), 3, False)]  # commits a, then b
    a, o, k = sliding_commit_counts([good, mixed, good, []], [0, 0, -1, 1], [0.5], 0.1, 3, 700.0)
    assert (a[0], o[0], k[0]) == (3, 1, 4)  # out-of-classifier word (-1): its commit can only be wrong


def test_fit_sliding_thresholds_bound_is_the_bound_of_sliding_commits():
    rng = np.random.default_rng(3)
    windows, truths = [], []
    for i in range(300):
        y = i % 3
        conf = rng.uniform(0.3, 1.0)
        right = rng.uniform() < conf  # higher confidence, more often right
        top = y if right else (y + 1) % 3
        ema = np.full(3, (1 - conf) / 2)
        ema[top] = conf
        windows.append([(0.0, ema, 3, False)])
        truths.append(y)
    acc, rep = fit_sliding_thresholds(windows, truths, 0.1, 3, 700.0)
    assert acc["strict"] >= acc["balanced"] >= acc["lenient"]
    n_acc, n_ok, _ = sliding_commit_counts(windows, truths, THRESHOLD_GRID, 0.1, 3, 700.0)
    for pol in POLICIES:
        i = int(np.flatnonzero(np.isclose(THRESHOLD_GRID, acc[pol]))[0])
        b = rep["policies"][pol]
        assert b["n_accepted"] == n_acc[i] and b["precision_lower_bound"] == pytest.approx(
            wilson_lower(int(n_ok[i]), int(n_acc[i])), abs=1e-4)
    assert "clip" in rep["unit"]


def _inputs(clf, n):
    """Segment-proxy evidence that is always confidently right: the proxy alone would accept everything."""
    lp = np.log(np.array([[0.98, 0.01, 0.01]] * 3))
    return [FusionInputs(cls_logp=lp, cls_ids=list(NAMES)) for _ in range(n)]


def test_fit_classifier_alone_uses_the_sliding_replay_not_the_segment_proxy():
    from app.isl.recognizer import classifier_fingerprint, classifier_policy_thresholds
    from app.isl.training.fit_fusion import fit_classifier_alone

    # sliding mode sees confident windows of the WRONG word half of the time (transitions, partial signs), which the
    # three-crop proxy of the whole clip never shows
    script = [[0.95, 0.03, 0.02]] * 4 + [[0.03, 0.95, 0.02]] * 4
    clf = _ScriptedClf(script)
    rec = _rec(clf)
    def load():  # every clip starts at the top of the script: "a" windows, then "b" windows
        clf.k = 0
        return _raw(gap=None)

    clips = [SimpleNamespace(clip_id=f"c{i}", concept_id="a", load=load) for i in range(12)]
    p, rep = fit_classifier_alone(rec, _inputs(clf, 12), ["a"] * 12, FusionParams(), clips=clips)
    assert rep["rule"] == "sliding replay" and rep["n"] == 12 and rep["replay_failed"] == 0
    # low thresholds commit "a" and then the transition word "b" in every clip; only thresholds above the transition's
    # EMA commit "a" alone. The replay sees that; the proxy (always confidently right) cannot.
    assert p.classifier_fit["rule"] == "sliding replay" and "engine" in p.classifier_fit
    assert p.classifier_fit["fingerprint"] == classifier_fingerprint(clf)
    for pol in POLICIES:
        assert p.classifier_accept[pol] >= 0.85 and p.classifier_fit["policies"][pol]["precision"] == 1.0
    assert classifier_policy_thresholds(clf, p)["per_policy"] is True
    q, prep = fit_classifier_alone(rec, _inputs(clf, 12), ["a"] * 12, FusionParams())
    assert prep["rule"] == "segment proxy" and q.classifier_fit["rule"] == "segment proxy" and "proxy_note" in q.classifier_fit
    assert q.classifier_fit["policies"]["lenient"]["precision"] == 1.0 and q.classifier_accept["lenient"] < 0.5
    # ...and at the proxy's lenient threshold sliding mode commits a wrong word in every clip
    from app.inference.engine import EngineConfig
    from app.isl.training.fit_fusion import replay_sliding

    c = EngineConfig()
    windows, truths, _ = replay_sliding(rec, clips)
    n_acc, n_ok, _ = sliding_commit_counts(windows, truths, [q.classifier_accept["lenient"]], c.min_margin, c.agree_n,
                                           c.cooldown_ms)
    assert n_acc[0] == 12 and n_ok[0] == 0


def test_unreplayable_clips_are_counted_and_fall_back_to_the_proxy():
    from app.isl.training.fit_fusion import fit_classifier_alone

    clf = _ScriptedClf([[0.95, 0.03, 0.02]])

    def bad():
        raise OSError("missing file")

    clips = [SimpleNamespace(clip_id="x", concept_id="a", load=bad)]
    p, rep = fit_classifier_alone(_rec(clf), _inputs(clf, 6), ["a"] * 6, FusionParams(), clips=clips)
    assert rep["rule"] == "segment proxy" and rep["replay_failed"] == 1


def test_e1_reports_the_sliding_replay_for_the_main_classifier_only():
    from app.isl.evaluation import runners

    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    rec = _rec(clf)
    clips = [SimpleNamespace(clip_id=f"c{i}", concept_id=c, group=f"g{i % 2}", category=None,
                             load=lambda: _raw(gap=None)) for i, c in enumerate(["a", "a", "b"])]
    orig = runners._features
    runners._features = lambda clip, fcfg: np.zeros((3, 4, 2), np.float32)
    try:
        out = runners.e1_closed_set(rec, clips, "E1b", "s", "primary")
        cv = runners.e1_closed_set(rec, clips, "fold", "s", "cv")
    finally:
        runners._features = orig
    sr = out["policies"]["balanced"]["sliding_replay"]
    assert sr["n"] == 3 and sr["n_accepted"] == 3 and sr["accepted_precision"] == pytest.approx(2 / 3)
    assert sr["wrong_accept_rate"] == pytest.approx(1 / 3) and out["sliding_replay_note"]
    assert "sliding_replay" not in cv["policies"]["balanced"]
    md = runners._policy_lines({"protocols": {"E1b": out}})
    assert any(ln.startswith("| E1b sliding replay | balanced |") for ln in md)
    header = next(ln for ln in md if ln.startswith("| Protocol"))
    assert all(ln.count("|") == header.count("|") for ln in md if ln.startswith("| E1b"))


# ------------------------------------------------------------------------------- certification (docs 15.25)

def _one_window_clips(n, prec, y=0, start=0):
    """One-window clips whose top class is right ``prec`` of the time (deterministic), confidence 0.97."""
    windows, truths = [], []
    for i in range(n):
        ok = ((start + i) * 0.6180339887) % 1.0 < prec
        ema = np.array([0.015, 0.015, 0.015])
        ema[y if ok else (y + 1) % 3] = 0.97
        windows.append([(0.0, ema, 3, False)])
        truths.append(y)
    return windows, truths


def test_sliding_thresholds_are_certified_on_the_worst_source():
    wa, ta = _one_window_clips(3000, 0.995)
    wb, tb = _one_window_clips(300, 0.90, start=7)
    windows, truths = wa + wb, ta + tb
    src = ["include"] * 3000 + ["isl500"] * 300
    grp = [f"i{i % 6}" for i in range(3000)] + [f"s{i % 4}" for i in range(300)]
    acc0, rep0 = fit_sliding_thresholds(windows, truths, 0.1, 3, 700.0)  # pooled clip-level bound
    assert rep0["policies"]["strict"]["rule"] == "certified"
    acc, rep = fit_sliding_thresholds(windows, truths, 0.1, 3, 700.0, sources=src, groups=grp)
    pols = rep["policies"]
    assert rep["major_sources"] == ["include", "isl500"]
    assert pols["strict"]["rule"] == pols["balanced"]["rule"] == "confirm_only"
    assert pols["strict"]["worst_source"] == "isl500" and pols["strict"]["by_source"]["isl500"]["n_groups"] == 4
    assert pols["lenient"]["rule"] == "certified"  # isl500 alone: about 0.86 >= 0.85


def _certified_sliding_pack(clf, rule="certified"):
    from app.isl.recognizer import classifier_fingerprint

    pol = {"rule": rule, "by_source": {"include": {"lower_bound": 0.97}}, "certified_precision": 0.97,
           "worst_source": "include", "floor": 0.90, "target": 0.95}
    return FusionParams(fitted=True, certification={"version": "cert-1"},
                        classifier_accept={"strict": 0.9, "balanced": 0.9, "lenient": 0.9},
                        classifier_fit={"fingerprint": classifier_fingerprint(clf), "temperature": 1.0,
                                        "rule": "sliding replay", "policies": {p: dict(pol) for p in POLICIES}})


def _commits(rec, clf, policy="balanced"):
    from app.inference.engine import EngineConfig, RecognitionEngine

    clf.k = 0
    raw = _raw(gap=None)
    eng = RecognitionEngine(None, recognizer=rec, config=EngineConfig(mode="sliding", policy=policy))
    eng.set_size(raw.width, raw.height)
    msgs = [eng.push(raw.frame_to_json(i)) for i in range(len(raw.t_ms))]
    return eng, [m for m in msgs if "commit" in m], [m["provisional"]["reason"] for m in msgs if "provisional" in m]


def test_sliding_mode_never_auto_commits_with_a_confirm_only_policy():
    from app.isl.recognizer import IslRecognizer
    from app.packs.vocabulary import Vocabulary

    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    vocab = Vocabulary([{"concept_id": n, "headword": n, "display": {"en": n}} for n in NAMES])

    def rec(fusion):
        return IslRecognizer("isl", clf.feature_config, vocab, classifier=clf, fusion=fusion)

    certified = rec(_certified_sliding_pack(clf))
    assert certified.sliding_rule("balanced") == "certified"
    _, commits, _ = _commits(certified, clf)
    assert commits and commits[0]["commit"]["concept_id"] == "a"
    for fusion in (_certified_sliding_pack(clf, rule="confirm_only"),  # validation did not certify it
                   FusionParams.from_json({"accept": {"balanced": 0.9}, "fitted": True}),  # a pre-15.25 fusion.json
                   _certified_sliding_pack(clf).with_(classifier_fit={  # fitted on the segment proxy
                       **_certified_sliding_pack(clf).classifier_fit, "rule": "segment proxy"})):
        r = rec(fusion)
        assert r.sliding_rule("balanced") == "confirm_only"
        eng, commits, reasons = _commits(r, clf)
        assert commits == [] and "confirm_only" in reasons  # provisional words only
        assert eng.describe()["certification"]["rule"] == "confirm_only"
    # in-code parameters are not gated (tools, tests): unchanged behaviour
    _, commits, _ = _commits(rec(FusionParams(fitted=True)), clf)
    assert commits


def test_fit_classifier_alone_certifies_the_replay_per_source():
    from app.isl.training.fit_fusion import fit_classifier_alone

    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    rec = _rec(clf)

    def load():
        clf.k = 0
        return _raw(gap=None)

    clips = [SimpleNamespace(clip_id=f"c{i}", concept_id="a", load=load, source="include", group=f"g{i % 4}")
             for i in range(40)]
    p, rep = fit_classifier_alone(rec, _inputs(clf, 40), ["a"] * 40, FusionParams(), clips=clips)
    assert rep["rule"] == "sliding replay" and rep["major_sources"] == ["include"]
    b = p.classifier_fit["policies"]["lenient"]
    assert b["by_source"]["include"]["n_groups"] == 4 and b["by_source"]["include"]["n_accepted"] == 40
    # 40 of 40 right: bound 0.91 >= lenient / balanced floors, below strict's 0.95
    assert b["rule"] == "certified" and p.classifier_fit["policies"]["strict"]["rule"] == "confirm_only"


def test_e1_reports_the_rule_and_per_source_sliding_rows():
    from app.isl.evaluation import runners

    clf = _ScriptedClf([[0.95, 0.03, 0.02]])
    rec = _rec(clf)
    clips = [SimpleNamespace(clip_id=f"c{i}", concept_id=c, group=f"g{i % 2}", category=None, source="include",
                             load=lambda: _raw(gap=None)) for i, c in enumerate(["a", "a", "b"])]
    orig = runners._features
    runners._features = lambda clip, fcfg: np.zeros((3, 4, 2), np.float32)
    try:
        out = runners.e1_closed_set(rec, clips, "E1b", "s", "primary")
    finally:
        runners._features = orig
    b = out["policies"]["balanced"]
    assert b["rule"] == "ungated"  # in-code FusionParams
    row = b["sliding_replay"]["by_source"]["include"]
    assert row["n"] == 3 and row["threshold_precision"] == pytest.approx(2 / 3) and "top1" not in row
    md = runners.markdown_report({"profile": "x", "label": "x", "created": "now", "smoke": False,
                                  "protocols": {"E1b": out}, "leakage_guards": {"pretraining_check": {}}})
    assert "## Headline: INCLUDE and the worst source" in md
