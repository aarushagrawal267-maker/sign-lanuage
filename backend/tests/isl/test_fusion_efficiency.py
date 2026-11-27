"""Fit-fusion cost (docs/ISL_DESIGN.md 15.23, QA item 0.3 / RC11): compact retrieval evidence, a shared prior, one
evidence pass for fit-fusion and the "My signs" choice, the weight grid on a seeded source-stratified subsample and a
compact sliding replay. Research fit-fusion took 8,144 s and about 9.7 GB; none of this may change what fuse() or
fit-fusion decides.

``_fuse_ref`` below is fuse() exactly as it was at 574b447 (before 15.23), kept here as the reference."""

from __future__ import annotations

import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.isl.models.fusion import (POLICIES, ConceptPrior, FusionInputs, FusionParams, FusionResult, _log_softmax,
                                   compact_inputs, decide, fuse)
from app.packs.vocabulary import Vocabulary


# ------------------------------------------------------------------ reference: fuse() at 574b447, unchanged
def _fuse_ref(inp: FusionInputs, p: FusionParams) -> FusionResult:
    crops = inp.crops
    if crops == 0:
        return FusionResult([], np.zeros(0), [], 0, -1.0, -1.0, None, {})
    has_c = inp.cls_logp is not None and len(inp.cls_ids) > 0
    has_r = inp.ret is not None and len(inp.ret_ids) > 0
    has_u = inp.user is not None and len(inp.user_ids) > 0
    comps: dict[str, set[str]] = {"classifier": set(), "retrieval": set(), "my_signs": set()}
    cand: dict[str, None] = {}
    if has_c:
        clp = _log_softmax(inp.cls_logp / max(p.t_c, 1e-3))
        mean_c = clp.mean(0)
        order = np.argsort(-mean_c)
        for i in order[: p.top_n]:
            cand[inp.cls_ids[i]] = None
        comps["classifier"] = {inp.cls_ids[i] for i in order[:10]}
        c_pos = {c: i for i, c in enumerate(inp.cls_ids)}
    if has_r:
        mean_r = inp.ret.mean(0)
        order = np.argsort(-mean_r)
        for i in order[: p.top_n]:
            cand[inp.ret_ids[i]] = None
        comps["retrieval"] = {inp.ret_ids[i] for i in order[:10]}
        r_pos = {c: i for i, c in enumerate(inp.ret_ids)}
        floor_idx = order[min(p.top_n, len(order)) - 1]
    if has_u:
        mean_u = inp.user.mean(0)
        for k in inp.user_ids:
            cand[k] = None
        comps["my_signs"] = {inp.user_ids[i] for i in np.argsort(-mean_u)[:10]}
    keys = list(cand)
    K = len(keys)
    z = np.zeros((crops, K), np.float64)
    if has_c:
        col = np.array([c_pos.get(k, -1) for k in keys])
        fill = np.full((crops, K), p.noncore_logp)
        if has_u and p.user_cls_mode == "neutral":
            is_user = np.array([k in set(inp.user_ids) for k in keys])
            fill[:, is_user] = clp.max(1, keepdims=True)
        cl = np.where(col[None, :] >= 0, clp[:, np.maximum(col, 0)], fill)
        z += p.alpha * cl
    if has_r or has_u:
        logits_all = []
        if has_r:
            logits_all.append(inp.ret / p.tau_r)
        if has_u:
            logits_all.append(inp.user / p.tau_r + p.user_bonus)
        L = np.concatenate(logits_all, axis=1)
        logZ = np.log(np.exp(L - L.max(1, keepdims=True)).sum(1)) + L.max(1)
        u_pos = {k: i for i, k in enumerate(inp.user_ids)} if has_u else {}
        cols = []
        for k in keys:
            if k in u_pos:
                cols.append(inp.user[:, u_pos[k]] / p.tau_r + p.user_bonus)
            elif has_r and k in r_pos:
                cols.append(inp.ret[:, r_pos[k]] / p.tau_r)
            elif has_r:  # concept without a reference in the index: neutral floor (the 50th best score)
                cols.append(inp.ret[:, floor_idx] / p.tau_r)
            else:
                cols.append(np.full(crops, -1.0 / p.tau_r))
        z += p.beta * (np.stack(cols, axis=1) - logZ[:, None])
    if p.gamma and inp.prior:
        z += p.gamma * np.array([inp.prior.get(k, 0.0) for k in keys])[None, :]
    pf = np.exp(_log_softmax(z, axis=1))
    mean = pf.mean(0)
    order = np.argsort(-mean)
    keys = [keys[i] for i in order]
    pf, mean = pf[:, order], mean[order]
    per_crop = [keys[int(i)] for i in pf.argmax(1)]
    agree = sum(k == keys[0] for k in per_crop)
    best_g = float(inp.ret.mean(0).max()) if has_r else -1.0
    best_u = None
    if has_u:
        j = int(np.argmax(mean_u))
        best_u = (inp.user_ids[j], float(mean_u[j]))
    best = max(best_g, best_u[1] if best_u else -1.0)
    return FusionResult(keys, mean, per_crop, agree, best, best_g, best_u, comps)


# ------------------------------------------------------------------ a random but realistic evidence world
C, G = 300, 1500  # classifier classes, index concepts (research: 4,354 and 12,440)
CLS = [f"isl.c{i}" for i in range(C)]
# the index holds 250 of the classes, in another order, plus dictionary-only concepts
IDX = [f"isl.c{i}" for i in range(C - 1, 49, -1)] + [f"isl.d{i}" for i in range(G - (C - 50))]
assert len(IDX) == G


def _vocab() -> Vocabulary:
    rows = [{"concept_id": c, "headword": c, "kind": "word", "in_classifier": True} for c in CLS]
    rows += [{"concept_id": c, "headword": c, "kind": "word" if int(c[5:]) % 3 else "phrase", "in_classifier": False}
             for c in IDX if c.startswith("isl.d")]
    return Vocabulary(rows)


VOCAB = _vocab()


def _evidence(rng: np.random.Generator, truth: int, users: int = 0) -> FusionInputs:
    """3 crops: classifier log-probabilities and index cosines with the true concept boosted, two tied index columns
    (near-duplicate references), optionally user-sign cosines."""
    lg = rng.normal(0, 1.5, C) + rng.normal(0, 0.5, (3, C))
    if truth < C:
        lg[:, truth] += rng.uniform(1, 8)
    cls_logp = _log_softmax(lg).astype(np.float32)
    ret = (rng.uniform(-0.1, 0.45, G) + rng.normal(0, 0.03, (3, G))).astype(np.float32)
    name = CLS[truth] if truth < C else IDX[truth - C]
    if name in IDX:
        ret[:, IDX.index(name)] += np.float32(rng.uniform(0.05, 0.5))
    ret[:, G - 1] = ret[:, G - 2]  # an exact tie
    inp = FusionInputs(cls_logp=cls_logp, cls_ids=CLS, ret=ret, ret_ids=IDX)
    if users:
        inp.user = rng.uniform(0.0, 0.8, (3, users)).astype(np.float32)
        inp.user_ids = [f"mysign:u{j}" for j in range(users)]
    return inp


def _old_prior(inp: FusionInputs, top_n: int = 50) -> dict:
    """The per-segment prior dict the recogniser built before 15.23."""
    keys = set(inp.cls_ids) | {inp.ret_ids[i] for i in np.argsort(-inp.ret.mean(0))[:top_n]}
    return {k: VOCAB.prior(k) for k in keys}


TAUS = (0.02, 0.05, 0.1)
PARAMS = [FusionParams(alpha=a, beta=b, gamma=g, tau_r=t, t_c=tc)
          for a, b, g, t, tc in [(1.0, 1.0, 0.5, 0.05, 1.0), (0.5, 1.5, 1.0, 0.02, 1.7), (1.5, 0.5, 0.0, 0.1, 0.6),
                                 (1.0, 1.5, 0.5, 0.02, 2.4), (0.5, 0.5, 1.0, 0.1, 1.0)]]


def _same(a: FusionResult, b: FusionResult, atol: float):
    assert a.keys == b.keys
    np.testing.assert_allclose(a.probs, b.probs, rtol=0, atol=atol)
    assert a.per_crop_top1 == b.per_crop_top1 and a.agree == b.agree
    assert a.best_cos == b.best_cos and a.best_global_cos == b.best_global_cos and a.best_user == b.best_user
    assert a.components == b.components


T_CS = sorted({p.t_c for p in PARAMS})


@pytest.mark.parametrize("users", [0, 4])
@pytest.mark.parametrize("compact_cls", [False, True])
def test_compact_inputs_and_shared_prior_give_the_old_fuse_result(users, compact_cls):
    """On fixed inputs, fuse(compact, shared prior) equals the pre-15.23 fuse(full matrix, per-segment prior dict)
    within 1e-6 - bit for bit without user signs - and every policy decides the same. With ``compact_cls`` the
    classifier is compacted too (docs 15.24): its kept log-probabilities are bit for bit the old ones."""
    rng = np.random.default_rng(7 + users)
    shared = ConceptPrior(VOCAB)
    worst = 0.0
    for n in range(40):
        truth = int(rng.integers(0, C + 200))
        full = _evidence(rng, truth, users)
        old_in = dataclasses.replace(full, prior=_old_prior(full))
        new_in = compact_inputs(dataclasses.replace(full, prior=shared), TAUS, 50, t_cs=T_CS if compact_cls else None)
        assert new_in.compact and new_in.ret.shape[1] < 0.2 * G and new_in.ret_size == G
        assert new_in.compact_cls == compact_cls
        if compact_cls:
            assert new_in.cls_size == C and new_in.cls_top == 100 and new_in.cls_logp.shape[1] < 0.6 * C
            assert new_in.cls_lse.shape == (len(T_CS), 2, 3) and new_in.cls_lse.dtype == np.float32
        for p in PARAMS + [PARAMS[0].with_(user_cls_mode="neutral")]:
            a, b = _fuse_ref(old_in, p), fuse(new_in, p)
            _same(a, b, 1e-6)
            worst = max(worst, float(np.abs(a.probs - b.probs).max()))
            if not users:
                assert np.array_equal(a.probs, b.probs)
            for pol in POLICIES:
                fp = p.with_(s_min=0.3)
                assert (decide(a.probs, a.agree, 3, fp, pol, a.best_cos)
                        == decide(b.probs, b.agree, 3, fp, pol, b.best_cos))
    assert worst < 1e-6


def test_full_inputs_with_the_shared_prior_are_bit_identical_to_the_old_fuse():
    rng = np.random.default_rng(3)
    shared = ConceptPrior(VOCAB)
    for _ in range(10):
        full = _evidence(rng, int(rng.integers(0, C)))
        for p in PARAMS:
            a = _fuse_ref(dataclasses.replace(full, prior=_old_prior(full)), p)
            b = fuse(dataclasses.replace(full, prior=shared), p)
            _same(a, b, 0.0)


def test_compact_inputs_refuse_what_they_cannot_answer():
    rng = np.random.default_rng(1)
    inp = compact_inputs(_evidence(rng, 3), TAUS, 50)
    with pytest.raises(ValueError, match="tau_r"):
        fuse(inp, FusionParams(tau_r=0.07))
    with pytest.raises(ValueError, match="top_n"):
        fuse(inp, FusionParams(top_n=80))
    assert compact_inputs(inp, TAUS, 50) is inp  # already compact
    # compact classifier (docs 15.24): an unstored t_c, a top_n beyond the kept classes, full retrieval
    full = _evidence(rng, 4)
    ci = compact_inputs(full, TAUS, 50, t_cs=T_CS)
    with pytest.raises(ValueError, match="t_c"):
        fuse(ci, FusionParams(t_c=1.3))
    with pytest.raises(ValueError, match="top_n"):
        fuse(ci, FusionParams(top_n=120, tau_r=0.05))
    with pytest.raises(ValueError, match="compact retrieval"):
        fuse(dataclasses.replace(ci, ret=full.ret, ret_ids=IDX, ret_lse=None), FusionParams())
    assert compact_inputs(ci, TAUS, 50, t_cs=T_CS) is ci
    cls_only = compact_inputs(FusionInputs(cls_logp=full.cls_logp, cls_ids=CLS), TAUS, 50, t_cs=T_CS)
    assert cls_only.compact_cls and not cls_only.compact
    _same(_fuse_ref(FusionInputs(cls_logp=full.cls_logp, cls_ids=CLS), PARAMS[1]), fuse(cls_only, PARAMS[1]), 0.0)
    only_cls = FusionInputs(cls_logp=inp.cls_logp, cls_ids=CLS)
    assert compact_inputs(only_cls, TAUS) is only_cls  # nothing to compact
    # the retrieval-only E2 form (no classifier) compacts too
    r = _evidence(rng, 5)
    r2 = FusionInputs(ret=r.ret, ret_ids=IDX)
    _same(_fuse_ref(r2, PARAMS[1]), fuse(compact_inputs(r2, TAUS, 50), PARAMS[1]), 0.0)


def test_concept_prior_is_shared_read_only_and_matches_the_vocabulary():
    pr = ConceptPrior(VOCAB)
    assert pr.get("isl.c3") == 1.0 and pr.get("isl.d1") == 0.5 and pr.get("isl.d0") == 0.0
    assert pr.get("mysign:abc", 0.0) == 0.0 and pr.get("isl.unknown", 0.0) == 0.0 and "mysign:x" not in pr
    assert len(pr) == len(VOCAB) and bool(pr)
    with pytest.raises(TypeError):
        pr["isl.c3"] = 0.0  # noqa


def test_recogniser_evidence_shares_one_prior():
    from app.features.spec import FeatureConfig
    from app.isl.recognizer import IslRecognizer

    class Clf:
        class_names = CLS[:5]
        meta = {"class_names": CLS[:5]}
        feature_config = FeatureConfig(canonical_hand=True)

        def predict(self, feats):
            return np.full((len(feats), 5), 0.2, np.float32), 1.0

    rec = IslRecognizer("isl", FeatureConfig(canonical_hand=True), VOCAB, classifier=Clf())
    f = np.zeros((3, 10, 4), np.float32)
    a, _ = rec.evidence(f, use_my_signs=False)
    b, _ = rec.evidence(f, use_my_signs=False)
    assert a.prior is b.prior is rec.prior and isinstance(a.prior, ConceptPrior)
    assert {k: a.prior.get(k, 0.0) for k in CLS[:5]} == {k: VOCAB.prior(k) for k in CLS[:5]}


def test_temperature_terms_clip_by_clip_give_fit_temperatures_choice():
    """fit-fusion fits t_c from 49 numbers per clip (docs 15.24): the per-row terms are bit for bit those of the
    whole matrix, and the chosen temperature is fit_temperature's."""
    from app.isl.models.calibration import (TEMPERATURE_GRID, fit_temperature, fit_temperature_terms,
                                            temperature_nll_terms)

    rng = np.random.default_rng(5)
    for scale in (0.4, 1.0, 3.0):
        y = rng.integers(0, C, 257)
        lg = rng.normal(0, 1, (257, C)) * scale
        lg[np.arange(257), y] += 2.0
        lg = _log_softmax(lg).astype(np.float32)
        whole = temperature_nll_terms(lg, y)
        rows = np.stack([temperature_nll_terms(lg[i:i + 1], y[i:i + 1])[:, 0] for i in range(257)], axis=1)
        assert np.array_equal(whole, rows)
        assert fit_temperature_terms(rows, TEMPERATURE_GRID) == fit_temperature(lg, y)
    assert fit_temperature_terms(np.zeros((49, 0))) == 1.0


def test_classifier_summaries_give_the_same_segment_proxy_thresholds():
    from app.isl.models.calibration import classifier_summary, fit_classifier_thresholds
    from app.isl.recognizer import calibrated_probs

    rng = np.random.default_rng(9)
    per_crop, truths = [], []
    for i in range(400):
        t = int(rng.integers(0, C + 30))
        lg = rng.normal(0, 1.5, C) + rng.normal(0, 0.6, (3, C))
        if t < C:
            lg[:, t] += rng.uniform(0, 9)
        per_crop.append(calibrated_probs(np.exp(_log_softmax(lg).astype(np.float32)), 1.4))
        truths.append(CLS[t] if t < C else f"isl.x{t}")
    summ = [classifier_summary(p) for p in per_crop]
    for margin in (0.0, 0.1):
        assert (fit_classifier_thresholds(per_crop, CLS, truths, margin, min_agree=2)
                == fit_classifier_thresholds(summ, CLS, truths, margin, min_agree=2))


# ------------------------------------------------------------------ weight-grid subsample
def test_weight_subsample_is_seeded_and_source_stratified():
    from app.isl.training.fit_fusion import weight_subsample

    src = ["isl500"] * 9000 + ["include"] * 900 + ["mendeley"] * 7
    assert weight_subsample(src[:3000], 3000) is None  # small validation sets: every clip, as before
    a, b = weight_subsample(src, 3000, seed=0), weight_subsample(src, 3000, seed=0)
    assert a == b and a != weight_subsample(src, 3000, seed=1) and a == sorted(set(a))
    cnt = {s: sum(src[i] == s for i in a) for s in set(src)}
    assert sum(cnt.values()) == 3000
    assert cnt["isl500"] == pytest.approx(3000 * 9000 / 9907, abs=1)
    assert cnt["include"] == pytest.approx(3000 * 900 / 9907, abs=1) and cnt["mendeley"] >= 1


def test_weight_subsample_with_many_tiny_sources_stays_at_n_and_balances_signers():
    """Every source keeps at least one clip, but the minimums no longer push the total above n: the excess comes
    back from the largest sources. Within a source the signer groups keep their share (docs 15.24)."""
    from app.isl.training.fit_fusion import weight_subsample

    src = ["isl500"] * 4000 + ["include"] * 300 + [f"tiny{i}" for i in range(60)]
    grp = [f"s{i % 4}" for i in range(4000)] + [f"i{i % 30}" for i in range(300)] + [f"t{i}" for i in range(60)]
    for n in (100, 500):
        sub = weight_subsample(src, n, seed=2, groups=grp)
        assert len(sub) == n == len(set(sub))
        assert all(any(src[i] == f"tiny{k}" for i in sub) for k in range(60))  # every source kept
        sig = [grp[i] for i in sub if src[i] == "isl500"]
        assert max(sig.count(f"s{k}") for k in range(4)) - min(sig.count(f"s{k}") for k in range(4)) <= 1
    # more sources than n: one clip each, nothing more (the documented exception)
    assert len(weight_subsample(src, 50, seed=2, groups=grp)) == 62
    # without groups the source counts are the same as with them
    a, b = weight_subsample(src, 500, seed=2), weight_subsample(src, 500, seed=2, groups=grp)
    assert sorted(src[i] for i in a) == sorted(src[i] for i in b)


# ------------------------------------------------------------------ fit-fusion on a toy world
class _ToyRec:
    """A recogniser stub over real clips: evidence is drawn per clip (seeded by its features) around its word, with
    the large classifier / index of the world above. No index object, so no leak guard is involved."""

    def __init__(self, truth_of):
        from app.isl.training.common import ISL_FEATURES

        self.feature_config = ISL_FEATURES
        self.vocab = VOCAB
        self.index = None
        self.encoder = None
        self.classifier = None
        self.truth_of = truth_of
        self.prior = ConceptPrior(VOCAB)
        self.fusion = FusionParams()
        self.calls = 0

    def evidence(self, feats, use_my_signs=False):
        self.calls += 1
        h = hashlib.sha256(np.ascontiguousarray(feats).tobytes()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        inp = _evidence(rng, self.truth_of[h])
        inp.prior = self.prior
        return inp, {"z": None}


def _toy_world():
    from app.dataset.synthetic import make_sequence
    from app.isl.training.common import ISL_FEATURES, engine_crops
    from app.features.build import build_features

    clips, truth_of = [], {}
    for g in range(4):
        for w in range(8):
            for t in range(5):
                raw = make_sequence(w % 6, signer=g, n_frames=24, fps=12.0, n_classes=6,
                                    rng=np.random.default_rng(1000 * g + 10 * w + t))
                feats = np.stack([build_features(x, ISL_FEATURES) for x in engine_crops(raw)])
                h = hashlib.sha256(np.ascontiguousarray(feats).tobytes()).digest()
                truth = 10 * w + g if w < 6 else C + 7 * w  # six classifier words, two dictionary-only ones
                truth_of[h] = truth
                cid = CLS[truth] if truth < C else IDX[truth - C]
                clips.append(SimpleNamespace(clip_id=f"g{g}/w{w}/t{t}", concept_id=cid, group=f"grp{g}",
                                             source="include" if g < 1 else "isl500", load=lambda r=raw: r))
    return clips, truth_of


@pytest.fixture(scope="module")
def toy():
    return _toy_world()


def test_fit_fusion_gives_the_same_params_as_the_full_uncompacted_fit(toy, monkeypatch):
    """Below WEIGHT_FIT_CLIPS nothing is subsampled: the compact evidence, the shared E4 cache and the shared prior
    give exactly the parameters of the pre-15.23 path (full matrix, second evidence pass for "My signs")."""
    import app.isl.training.fit_fusion as ff

    clips, truth_of = toy
    new_rec = _ToyRec(truth_of)
    new, rep = ff.fit_fusion(new_rec, clips, "research")
    assert new_rec.calls == len(clips)  # one evidence pass: choose_user_mode reused it
    assert rep["my_signs"]["k"] is not None and set(rep["my_signs"]["modes"]) == {"penalty", "neutral"}
    assert new.weight_fit["subsample"] is None and new.weight_fit["n_full"] == len(clips)

    old_rec = _ToyRec(truth_of)
    monkeypatch.setattr(ff, "compact_inputs", lambda inp, *a, **k: inp)
    real_choose = ff.choose_user_mode
    monkeypatch.setattr(ff, "choose_user_mode", lambda rec, vc, p, ks=(3, 1), cache=None: real_choose(rec, vc, p, ks))
    monkeypatch.setattr(ff, "fuse", _fuse_ref)
    import app.isl.models.calibration as cal
    import app.isl.evaluation.runners as runners

    monkeypatch.setattr(cal, "fuse", _fuse_ref)
    monkeypatch.setattr(runners, "fuse", _fuse_ref)
    old, orep = ff.fit_fusion(old_rec, clips, "research")
    assert old_rec.calls > len(clips)  # the old path computes the "My signs" evidence a second time
    assert new == old
    assert rep["my_signs"] == orep["my_signs"] and rep["weights"] == orep["weights"]


def test_fit_fusion_grid_on_a_subsample_is_recorded_and_thresholds_use_every_clip(toy):
    import app.isl.training.fit_fusion as ff

    clips, truth_of = toy
    p, rep = ff.fit_fusion(_ToyRec(truth_of), clips, "research", weight_clips=60, weight_seed=5)
    wf = p.weight_fit
    assert wf["subsample"] == 60 and wf["seed"] == 5 and wf["n_full"] == len(clips) == rep["val_queries"]
    assert wf["subsample_by_source"] == {"include": 15, "isl500": 45} and wf["grid_points"] == 81
    assert wf["nll_full"] > 0 and wf["nll_subsample"] > 0 and rep["weights"]["nll"] == pytest.approx(wf["nll_full"], abs=1e-6)
    # the best 5 subsample points were re-scored on every clip and the best full-set NLL was kept (docs 15.24)
    rs = wf["rescored"]
    assert len(rs) == ff.WEIGHT_FIT_RESCORE == 5 and [r["nll_subsample"] for r in rs] == sorted(r["nll_subsample"] for r in rs)
    best = min(rs, key=lambda r: r["nll_full"])
    assert wf["nll_full"] == best["nll_full"] and all(getattr(p, k) == best[k] for k in ("alpha", "beta", "gamma", "tau_r"))
    assert wf["subsample_best_kept"] == (best is rs[0])
    assert rep["thresholds"]["per_policy"]["balanced"]["n_accepted"] <= len(clips)
    assert p.achieved["balanced"]["n"] == len(clips)  # thresholds and achieved numbers: every validation clip
    q = FusionParams.from_json(p.to_json())
    assert q == p and q.weight_fit == wf  # stored in fusion.json
    # same seed, same subsample, same parameters
    p2, _ = ff.fit_fusion(_ToyRec(truth_of), clips, "research", weight_clips=60, weight_seed=5)
    assert p2 == p


def test_old_fusion_json_without_new_fields_still_loads(tmp_path):
    old = {"alpha": 0.5, "beta": 1.0, "gamma": 1.0, "accept": {"strict": 0.96, "balanced": 0.96, "lenient": 0.96},
           "classifier_accept": None, "fitted": True}
    f = tmp_path / "fusion.json"
    import json

    f.write_text(json.dumps(old))
    p = FusionParams.load(f)
    assert p.weight_fit == {} and p.accept == {"strict": 0.96, "balanced": 0.96, "lenient": 0.96}


# ------------------------------------------------------------------ compact sliding replay
def test_prepared_windows_give_identical_sliding_fits():
    from app.isl.models.calibration import (THRESHOLD_GRID, PreparedWindows, fit_sliding_thresholds,
                                            prepare_windows, sliding_commit_counts, sliding_commits)

    rng = np.random.default_rng(11)
    windows, truths = [], []
    for i in range(120):
        w, t = [], 0.0
        for j in range(int(rng.integers(3, 12))):
            t += 200.0
            ema = rng.dirichlet(np.full(40, 0.3))
            w.append((t, ema, int(rng.integers(1, 6)), bool(rng.uniform() < 0.1), int(ema.argmax()) % 37))
        windows.append(w)
        truths.append(int(rng.integers(-1, 40)))
    prep = [prepare_windows(w) for w in windows]
    assert all(isinstance(p, PreparedWindows) for p in prep) and prepare_windows(prep[0]) is prep[0]
    for a, b in zip(sliding_commit_counts(windows, truths, THRESHOLD_GRID, 0.1, 3, 700.0),
                    sliding_commit_counts(prep, truths, THRESHOLD_GRID, 0.1, 3, 700.0)):
        assert np.array_equal(a, b)
    assert fit_sliding_thresholds(windows, truths, 0.1, 3, 700.0) == fit_sliding_thresholds(prep, truths, 0.1, 3, 700.0)
    assert sliding_commits(windows[0], 0.2, 0.05, 2, 300.0) == sliding_commits(prep[0], 0.2, 0.05, 2, 300.0)


def test_fit_fusion_with_compact_classifier_gives_the_old_temperature_and_segment_proxy(toy, monkeypatch):
    """With a classifier and no replayable clip (segment-proxy fallback), the compact classifier evidence (docs
    15.24) gives exactly the temperature, weights, thresholds and classifier-alone thresholds of the full rows."""
    import app.isl.evaluation.runners as runners
    import app.isl.models.calibration as cal
    import app.isl.training.fit_fusion as ff

    clips, truth_of = toy

    class Clf:
        class_names = CLS
        meta = {"temperature": 1.3, "recommended_threshold": 0.6, "class_names": CLS}

    def rec():
        r = _ToyRec(truth_of)
        r.classifier = Clf()
        return r

    monkeypatch.setattr(ff, "replay_sliding", lambda rec, clips, cfg=None, **k: ([], [], len(clips)))
    seen = []
    real_fct = ff.fit_classifier_thresholds
    monkeypatch.setattr(ff, "fit_classifier_thresholds", lambda pc, *a, **k: (seen.append(type(pc[0])), real_fct(pc, *a, **k))[1])
    new, rep = ff.fit_fusion(rec(), clips, "research")
    assert rep["classifier_alone"]["rule"] == "segment proxy" and new.classifier_accept
    assert seen[-1] is cal.ClassifierSummary  # the compact path read summaries, not full rows
    assert "temperature" in rep

    monkeypatch.setattr(ff, "compact_inputs", lambda inp, *a, **k: inp)
    for mod in (ff, cal, runners):
        monkeypatch.setattr(mod, "fuse", _fuse_ref)
    old, orep = ff.fit_fusion(rec(), clips, "research")
    assert seen[-1] is np.ndarray
    assert new == old and rep["temperature"] == orep["temperature"]
    assert rep["classifier_alone"] == orep["classifier_alone"] and rep["my_signs"] == orep["my_signs"]


# ------------------------------------------------------------------ batched sliding replay (docs 15.24)
class _RowClf:
    """A classifier whose output for a window does not depend on the batch it is in (row by row), so the batched
    replay must give exactly the reference windows. ``jitter`` adds a batch-dependent rounding-size change."""

    def __init__(self, k: int = 6, jitter: float = 0.0):
        from app.features.spec import FeatureConfig

        self.feature_config = FeatureConfig(canonical_hand=True)
        self.class_names = CLS[:k]
        self.meta = {"class_names": CLS[:k], "temperature": 1.2, "recommended_threshold": 0.5}
        self.k, self.jitter, self.calls = k, jitter, 0

    def predict(self, feats):
        self.calls += 1
        out = []
        for x in np.asarray(feats, np.float64):
            v = np.nan_to_num(x).reshape(len(x), -1)
            lg = np.array([v[:, j::self.k].mean() for j in range(self.k)]) * 60.0
            e = np.exp(lg - lg.max())
            out.append(e / e.sum())
        P = np.array(out, np.float32)
        if self.jitter and len(P) > 1:
            P = P + np.float32(self.jitter) * (np.arange(P.shape[1]) % 2)
        return P, 0.0


def _sliding_clips():
    from app.dataset.synthetic import make_sequence

    out = []
    for i in range(10):
        raw = make_sequence(i % 6, signer=i % 3, n_frames=int(40 + 9 * i), fps=25.0, n_classes=6,
                            rng=np.random.default_rng(50 + i))
        if i % 3 == 1:  # hands gone for a while: a sign boundary inside the clip
            a = len(raw.t_ms) // 3
            raw.left[a:a + 15] = np.nan
            raw.right[a:a + 15] = np.nan
        out.append(raw)
    return out


def _isl_rec(clf):
    from app.features.spec import FeatureConfig
    from app.isl.recognizer import IslRecognizer

    return IslRecognizer("isl", FeatureConfig(canonical_hand=True), VOCAB, classifier=clf)


def _same_windows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0] and x[2:] == y[2:] and np.array_equal(x[1], y[1])


def test_batched_sliding_replay_gives_the_engines_windows(monkeypatch):
    """One classifier call per clip, the engine's own EMA / streak / boundary code: the same windows as the
    window-by-window reference replay, and a clip near a decision edge falls back to the reference."""
    import app.isl.training.fit_fusion as ff

    clips = _sliding_clips()
    n_ref = n_bat = 0
    batched_used = 0
    for raw in clips:
        ref_clf, bat_clf = _RowClf(), _RowClf()
        ref = ff.sliding_windows(_isl_rec(ref_clf), raw)
        bat, replayed = ff.batched_sliding_windows(_isl_rec(bat_clf), raw)
        _same_windows(ref, bat)
        n_ref, n_bat = n_ref + ref_clf.calls, n_bat + bat_clf.calls
        batched_used += not replayed
    assert n_ref > 40 and batched_used >= 8 and n_bat <= len(clips) + (len(clips) - batched_used) * 100
    assert any(w[3] for raw in clips for w in ff.sliding_windows(_isl_rec(_RowClf()), raw))  # a boundary is tested
    monkeypatch.setattr(ff, "REPLAY_BAND", 2.0)  # every window "near an edge": the reference replays every clip
    for raw in clips[:3]:
        bat, replayed = ff.batched_sliding_windows(_isl_rec(_RowClf()), raw)
        assert replayed
        _same_windows(ff.sliding_windows(_isl_rec(_RowClf()), raw), bat)


def test_batched_replay_with_rounding_differences_commits_the_same():
    """A batch-dependent change of rounding size (as a real network's batched call has) moves EMA values slightly
    but never a commit: the sliding fit's counts and thresholds equal the reference replay's."""
    from types import SimpleNamespace

    import app.isl.training.fit_fusion as ff
    from app.inference.engine import EngineConfig
    from app.isl.models.calibration import THRESHOLD_GRID, fit_sliding_thresholds, sliding_commit_counts

    clips = [SimpleNamespace(load=lambda r=r: r, concept_id=CLS[i % 6]) for i, r in enumerate(_sliding_clips())]
    e = EngineConfig()
    ref = ff.replay_sliding(_isl_rec(_RowClf()), clips, e, batched=False)
    cnt: dict = {}
    bat = ff.replay_sliding(_isl_rec(_RowClf(jitter=3e-7)), clips, e, batched=True, counts=cnt)
    assert cnt["batched"] >= 8 and cnt["batched"] + cnt["reference"] == len(clips)
    assert ref[1] == bat[1] and ref[2] == bat[2]
    for a, b in zip(sliding_commit_counts(ref[0], ref[1], THRESHOLD_GRID, e.min_margin, e.agree_n, e.cooldown_ms),
                    sliding_commit_counts(bat[0], bat[1], THRESHOLD_GRID, e.min_margin, e.agree_n, e.cooldown_ms)):
        assert np.array_equal(a, b)
    assert (fit_sliding_thresholds(ref[0], ref[1], e.min_margin, e.agree_n, e.cooldown_ms)
            == fit_sliding_thresholds(bat[0], bat[1], e.min_margin, e.agree_n, e.cooldown_ms))
