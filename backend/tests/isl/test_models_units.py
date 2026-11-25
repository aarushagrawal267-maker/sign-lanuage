"""Unit tests of the ISL model pieces: features flag, augmentation, losses, encoder, index, fusion, calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from app.dataset.synthetic import make_sequence
from app.features.build import build_features, dominant_hand, prepare_body_frame, to_body_frame
from app.features.spec import FeatureConfig
from app.isl.augment import augment_strong
from app.isl.models.calibration import fit_temperature, fit_thresholds, summarize
from app.isl.models.encoder import FeatureStatsEmbedder, SignEncoder
from app.isl.models.fusion import FusionInputs, FusionParams, decide, fuse
from app.isl.models.index import EmbeddingIndex, IndexVersionMismatch
from app.isl.models.losses import SubCenterArcFace, SupConLoss


# ----------------------------------------------------------------------------------- features

def test_canonical_hand_is_opt_in_and_keeps_asl_config_identical():
    asl = FeatureConfig()
    assert "canonical_hand" not in asl.to_dict()  # ASL checkpoints / spec unchanged
    assert FeatureConfig.from_dict(asl.to_dict()) == asl
    isl = FeatureConfig(canonical_hand=True)
    assert isl.to_dict()["canonical_hand"] is True and isl.dim == asl.dim


def _swap_hands(raw):
    from app.landmarks.raw import RawSequence

    pose = raw.pose.copy()
    pose[..., 0] = 1.0 - pose[..., 0]
    perm = list(range(33))
    for a, b in [(11, 12), (13, 14), (15, 16), (23, 24)]:
        perm[a], perm[b] = b, a
    pose = pose[:, perm]
    left, right = raw.right.copy(), raw.left.copy()
    left[..., 0] = 1.0 - left[..., 0]
    right[..., 0] = 1.0 - right[..., 0]
    face = raw.face.copy()
    face[..., 0] = 1.0 - face[..., 0]
    return RawSequence(pose, left, right, face, raw.t_ms, raw.width, raw.height)


def test_canonical_hand_mirrors_left_dominant_signer():
    raw = make_sequence(3, 1)  # synthetic signs move the right hand most
    bf = to_body_frame(raw)
    dom = dominant_hand(bf)
    assert dom in ("left", "right")
    mirrored = _swap_hands(raw)
    assert dominant_hand(to_body_frame(mirrored)) != dom
    cfg = FeatureConfig(canonical_hand=True)
    a, b = build_features(raw, cfg), build_features(mirrored, cfg)
    # after canonicalisation both versions are "the same hand" (hand blocks agree closely)
    lay = {x["name"]: x for x in cfg.layout()}
    for blk in ("right_shape", "left_shape"):
        s = slice(lay[blk]["start"], lay[blk]["start"] + lay[blk]["size"])
        assert np.abs(a[:, s] - b[:, s]).mean() < 0.05
    # without the flag the mirrored clip looks different
    plain = FeatureConfig()
    assert np.abs(build_features(raw, plain) - build_features(mirrored, plain)).mean() > 0.01


def test_prepare_body_frame_without_flag_equals_to_body_frame():
    raw = make_sequence(1, 2)
    a, b = prepare_body_frame(raw, FeatureConfig()), to_body_frame(raw)
    assert np.allclose(a.left, b.left) and np.allclose(a.pose, b.pose)


def test_augment_strong_shapes_and_variety():
    bf = to_body_frame(make_sequence(2, 0))
    rng = np.random.default_rng(0)
    from app.features.build import featurize

    cfg = FeatureConfig(canonical_hand=True)
    x1 = featurize(augment_strong(bf, rng), cfg)
    x2 = featurize(augment_strong(bf, rng), cfg)
    assert x1.shape == (cfg.seq_len, cfg.dim) and np.isfinite(x1).all()
    assert not np.allclose(x1, x2)


# ----------------------------------------------------------------------------------- losses / encoder

def test_supcon_prefers_clustered_embeddings():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    good = torch.nn.functional.normalize(torch.eye(3)[labels] + 0.01 * torch.randn(6, 3), dim=-1)
    bad = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    loss = SupConLoss(0.1)
    assert loss(good, labels) < loss(bad, labels)
    assert float(loss(good, torch.arange(6))) == 0.0  # no positives -> 0


def test_subcenter_arcface_trains():
    torch.manual_seed(0)
    arc = SubCenterArcFace(8, 4, k=2)
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    y = torch.randint(0, 4, (16,))
    l0 = arc(z, y)
    opt = torch.optim.SGD(arc.parameters(), lr=0.5)
    for _ in range(30):
        opt.zero_grad()
        loss = arc(z, y)
        loss.backward()
        opt.step()
    assert math.isfinite(float(l0.detach())) and float(loss.detach()) < float(l0.detach())
    assert arc.cosine(z).shape == (16, 4)


def test_sign_encoder_outputs_unit_vectors():
    enc = SignEncoder(20, hidden=8, layers=1, emb_dim=6)
    z = enc(torch.randn(3, 10, 20), torch.ones(3, 10, dtype=torch.bool))
    assert z.shape == (3, 6) and torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_feature_stats_embedder():
    cfg = FeatureConfig(seq_len=16)
    e = FeatureStatsEmbedder(cfg)
    z, _ = e.embed(np.stack([build_features(make_sequence(c, 0), cfg) for c in (0, 0, 5)]))
    assert z.shape == (3, 2 * cfg.dim) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)


# ----------------------------------------------------------------------------------- index

def test_index_roundtrip_and_concept_scores(tmp_path):
    emb = np.eye(4, dtype=np.float32)
    rows = [{"variant_id": "v0", "concept_id": "isl.a", "source": "islrtc", "kind": "first_sign"},
            {"variant_id": "v1", "concept_id": "isl.a", "source": "islrtc", "kind": "alt_end", "weight": 0.9},
            {"variant_id": "v2", "concept_id": "isl.b", "source": "islrtc", "kind": "first_sign"},
            {"variant_id": "v3", "concept_id": "isl.c", "source": "include", "kind": "centroid"}]
    idx = EmbeddingIndex.build(emb, rows, "enc-x", {"seq_len": 48}, "commercial-safe")
    idx.save(tmp_path / "index")
    idx2 = EmbeddingIndex.load(tmp_path / "index", expect_encoder_version="enc-x")
    assert len(idx2) == 4 and idx2.meta.sha256 and idx2.meta.profile == "commercial-safe"
    q = np.array([[0, 1, 0, 0]], np.float32)
    assert idx2.search(q, 2)[0][0] == (1, pytest.approx(0.9, abs=1e-3))
    cs = idx2.concept_scores(q, 3)[0]
    assert list(cs)[0] == "isl.a" and cs["isl.a"] == pytest.approx(0.9, abs=1e-3)
    assert idx2.best_row(q, "isl.a")["variant_id"] == "v1"
    with pytest.raises(IndexVersionMismatch):
        EmbeddingIndex.load(tmp_path / "index", expect_encoder_version="enc-y")


# ----------------------------------------------------------------------------------- fusion

def _inp(cls=None, ret=None, user=None, crops=3):
    inp = FusionInputs()
    if cls is not None:
        ids, logp = zip(*cls.items())
        inp.cls_ids = list(ids)
        inp.cls_logp = np.log(np.tile(np.array(logp, np.float64), (crops, 1)))
    if ret is not None:
        ids, s = zip(*ret.items())
        inp.ret_ids = list(ids)
        inp.ret = np.tile(np.array(s, np.float64), (crops, 1))
    if user is not None:
        ids, s = zip(*user.items())
        inp.user_ids = list(ids)
        inp.user = np.tile(np.array(s, np.float64), (crops, 1))
    return inp


def test_fuse_combines_classifier_and_retrieval():
    p = FusionParams(gamma=0.0)
    r = fuse(_inp(cls={"isl.a": 0.6, "isl.b": 0.4}, ret={"isl.a": 0.5, "isl.b": 0.8, "isl.x": 0.79}), p)
    assert set(r.keys[:3]) == {"isl.a", "isl.b", "isl.x"} and abs(r.probs.sum() - 1) < 1e-6
    assert r.agree == 3 and r.best_cos == pytest.approx(0.8)
    only_c = fuse(_inp(cls={"isl.a": 0.9, "isl.b": 0.1}), p)
    assert only_c.keys[0] == "isl.a" and only_c.best_cos == -1.0
    only_r = fuse(_inp(ret={"isl.a": 0.2, "isl.b": 0.9}), p)
    assert only_r.keys[0] == "isl.b"


def test_fuse_user_bonus_and_best_user():
    p = FusionParams(gamma=0.0, user_bonus=2.0)
    r = fuse(_inp(ret={"isl.a": 0.8, "isl.b": 0.1}, user={"mysign:1": 0.78}), p)
    assert r.keys[0] == "mysign:1" and r.best_user == ("mysign:1", pytest.approx(0.78))


def test_decide_accept_suggest_abstain_oov():
    p = FusionParams(accept={"strict": 0.9, "balanced": 0.7, "lenient": 0.5}, margin=0.1, suggest_min_top5_mass=0.3,
                     s_min=0.4)
    assert decide(np.array([0.8, 0.1, 0.1]), 3, 3, p, "balanced") == ("accept", "ok")
    assert decide(np.array([0.8, 0.1, 0.1]), 3, 3, p, "strict") == ("suggest", "low_confidence")
    assert decide(np.array([0.55, 0.50, 0.15]), 3, 3, p, "lenient") == ("suggest", "ambiguous")
    assert decide(np.array([0.8, 0.1]), 1, 3, p, "balanced") == ("suggest", "unstable")
    flat = np.full(100, 0.01)
    assert decide(flat, 3, 3, p, "balanced") == ("abstain", "low_confidence")
    assert decide(np.array([0.9, 0.1]), 3, 3, p, "balanced", best_cos=0.2) == ("abstain", "out_of_vocabulary")
    assert decide(np.array([]), 0, 3, p) == ("abstain", "no_candidates")
    with pytest.raises(ValueError):
        decide(np.array([1.0]), 1, 1, p, "yolo")


def test_fusion_params_json_roundtrip(tmp_path):
    p = FusionParams(alpha=0.5, accept={"strict": 0.99, "balanced": 0.9, "lenient": 0.6}, fitted=True)
    p.save(tmp_path / "f.json")
    assert FusionParams.load(tmp_path / "f.json") == p


# ----------------------------------------------------------------------------------- calibration

def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    n, c = 4000, 5
    true_logits = rng.normal(0, 1.5, (n, c))
    y = np.array([rng.choice(c, p=np.exp(z) / np.exp(z).sum()) for z in true_logits])
    T = fit_temperature(true_logits * 2.0, y)  # over-confident by 2x
    assert 1.6 < T < 2.5


def test_wilson_lower_bound_is_conservative():
    from app.isl.models.calibration import wilson_lower

    assert wilson_lower(5, 5) < 0.6 and wilson_lower(950, 1000) > 0.93 and wilson_lower(0, 0) == 0.0


def _calib_set(n: int, hi: float, seed: int = 1):
    rng = np.random.default_rng(seed)
    inputs, truths = [], []
    for i in range(n):
        true = f"isl.c{i % 10}"
        conf = hi - (hi - 0.2) * rng.random() ** 3  # skewed towards confident, mostly correct
        top = true if rng.random() < conf else f"isl.c{(i + 1) % 10}"
        inputs.append(_inp(cls={f"isl.c{k}": (conf if f"isl.c{k}" == top else (1 - conf) / 9) for k in range(10)}))
        truths.append(true)
    return inputs, truths


def test_fit_thresholds_targets_precision_and_reports_achieved():
    p = FusionParams(gamma=0.0)
    inputs, truths = _calib_set(1500, 0.999)
    results = [fuse(x, p) for x in inputs]
    fitted, rep = fit_thresholds(results, truths, p)
    assert fitted.fitted and fitted.accept["strict"] >= fitted.accept["balanced"] >= fitted.accept["lenient"]
    assert rep["balanced_target_reached"]
    bal = summarize(results, truths, fitted, "balanced")
    assert bal["accepted_precision"] >= 0.95 and "balanced" in fitted.achieved and rep["n"] == 1500
    # little data: a "perfect" handful of accepted clips must not count as reaching the target
    small_in, small_t = _calib_set(60, 0.9, seed=3)
    _, rep2 = fit_thresholds([fuse(x, p) for x in small_in], small_t, p)
    assert rep2["strict_target_reached"] is False
