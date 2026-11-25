"""Self-supervised pre-training and the resumable training loop (synthetic data, no network)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from app.dataset.synthetic import make_sequence
from app.isl.data.sources import Clip
from app.isl.training.common import ISL_FEATURES
from app.isl.training.loop import LoopConfig, StateMismatch, balanced_batches, pk_batches, run_loop
from app.isl.training.pretrain import PretrainConfig, nt_xent, pretrain_encoder, span_mask

TINY = PretrainConfig(hidden=16, layers=1, emb_dim=16, batch=6, epochs=2, steps_per_epoch=3, val_batches=1)


@pytest.fixture(scope="module")
def unlabelled(tmp_path_factory) -> list[Clip]:
    d = tmp_path_factory.mktemp("pretrain")
    out = []
    for i in range(14):
        n = 120 if i % 2 else 30  # long sentence-like sequences and short word clips
        r = make_sequence(i % 5, signer=i, n_frames=n, fps=12.0)
        p = d / f"s{i}.npz"
        r.save(p)
        out.append(Clip(f"glosl/{i}" if i % 2 else f"include/x/{i}", "glosl" if i % 2 else "include", "", None, None,
                        None, p, group=f"g{i}"))
    return out


def test_nt_xent_and_span_mask():
    z = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    loss_same, acc = nt_xent(z, z.clone(), 0.1)
    loss_rand, _ = nt_xent(z, torch.nn.functional.normalize(torch.randn(4, 8), dim=-1), 0.1)
    assert acc == 1.0 and float(loss_same) < float(loss_rand)
    valid = np.ones((3, 40), bool)
    valid[2] = False
    m = span_mask(valid, 0.25, 3, np.random.default_rng(0))
    assert 0.1 < m[0].mean() < 0.45 and not m[2].any()


def test_pretrain_saves_encoder_resumes_and_warm_starts(unlabelled, tmp_path):
    from app.isl.models.encoder import EncoderPredictor, SignEncoder, load_encoder_weights

    out = tmp_path / "pre.pt"
    r = pretrain_encoder(unlabelled, out, ISL_FEATURES, TINY, profile="research", threads=1, smoke=True, fresh=True)
    assert r["finished"] and out.exists() and r["sources"] == {"glosl": 7, "include": 7}
    enc = EncoderPredictor(out)
    assert enc.version == "isl-enc-pre-1" and enc.meta["pretraining"]["labels_used"] is False
    z, _ = enc.embed(np.zeros((2, ISL_FEATURES.seq_len, ISL_FEATURES.dim), np.float32))
    assert z.shape == (2, 16) and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-4)
    # a finished run resumes as finished (no more training)
    r2 = pretrain_encoder(unlabelled, out, ISL_FEATURES, TINY, profile="research", threads=1, smoke=True)
    assert r2["resumed_from"]["finished"] is True
    # other settings refuse to continue the saved state
    with pytest.raises(StateMismatch):
        pretrain_encoder(unlabelled, out, ISL_FEATURES, PretrainConfig(**{**TINY.__dict__, "lr": 5e-4}),
                         profile="research", threads=1, smoke=True)
    # warm start: same trunk shape loads, another embedding size skips only the head
    e = SignEncoder(ISL_FEATURES.dim, hidden=16, layers=1, emb_dim=16)
    rep = load_encoder_weights(e, out)
    assert rep["loaded"] > 0 and rep["skipped"] == []
    e2 = SignEncoder(ISL_FEATURES.dim, hidden=16, layers=1, emb_dim=32)
    assert load_encoder_weights(e2, out)["skipped"] == ["head.2.bias", "head.2.weight"]


def test_finetune_records_pretraining(tmp_path, unlabelled):
    from app.isl.training.common import load_isl_data
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex
    from app.training.trainer import TrainConfig
    from tests.isl.isl_world import make_world

    pre = tmp_path / "pre.pt"
    pretrain_encoder(unlabelled, pre, ISL_FEATURES, TINY, profile="commercial-safe", threads=1, smoke=True, fresh=True)
    w = make_world(tmp_path / "w")
    data = load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "commercial-safe", w["split"])
    r = train_encoder(data, tmp_path / "enc.pt", ISL_FEATURES,
                      TrainConfig(hidden=16, layers=1, epochs=1, batch_size=8, patience=1, lr=3e-3),
                      emb_dim=16, p=4, k=2, threads=1, batches_per_epoch=2, init_from=pre, smoke=True)
    assert r["pretrained"] and r["finished"]
    meta = torch.load(tmp_path / "enc.pt", map_location="cpu", weights_only=False)["meta"]
    assert meta["pretraining"]["init_from"] == str(pre) and meta["pretraining"]["labels_used"] is False


class _Lin(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(3, 1)


def _toy(n_steps):
    torch.manual_seed(0)
    X = torch.randn(64, 3)
    Y = X @ torch.tensor([[1.0], [-2.0], [0.5]])
    seen = []

    def batches(epoch):
        g = np.random.default_rng([0, epoch])
        for s in range(n_steps):
            yield (epoch, s, g.choice(64, 8, replace=False))

    def loss_fn(m, item):
        seen.append(item[:2])
        idx = torch.from_numpy(item[2])
        return ((m.w(X[idx]) - Y[idx]) ** 2).mean()

    return batches, loss_fn, seen


def test_loop_resumes_mid_epoch_after_a_time_budget_stop(tmp_path):
    batches, loss_fn, seen = _toy(5)
    cfg = LoopConfig(epochs=3, steps_per_epoch=5, lr=0.05, max_minutes=1e-9, save_every_s=1e9)
    m = _Lin()
    r1 = run_loop(m, batches, loss_fn, cfg, tmp_path / "s.pt")
    assert not r1.finished and seen == [(1, 0)]  # one step, then the budget stops it (state saved)
    m2 = _Lin()
    r2 = run_loop(m2, batches, loss_fn, LoopConfig(**{**cfg.__dict__, "max_minutes": 0.0}), tmp_path / "s.pt")
    assert r2.finished and r2.resumed_from == {"epoch": 1, "step": 1, "finished": False}
    assert seen[1] == (1, 1) and len(seen) == 15  # no step repeated, none skipped
    assert [h["epoch"] for h in r2.history] == [1, 2, 3]
    # a fresh uninterrupted run reaches the same weights (deterministic batches, saved optimiser state)
    b3, l3, _ = _toy(5)
    m3 = _Lin()
    run_loop(m3, b3, l3, LoopConfig(**{**cfg.__dict__, "max_minutes": 0.0}), tmp_path / "t.pt")
    assert torch.allclose(m2.w.weight, m3.w.weight, atol=1e-5)


def test_samplers_are_deterministic_and_balanced():
    labels = [0] * 90 + [1] * 10
    a = list(balanced_batches(labels, None, 20, 30, seed=1, epoch=2, class_power=0.0))
    b = list(balanced_batches(labels, None, 20, 30, seed=1, epoch=2, class_power=0.0))
    assert a == b
    share = np.mean([labels[i] for bt in a for i in bt])
    assert 0.35 < share < 0.65  # uniform over classes although class 1 has 10 % of the items
    w = [1.0] * 50 + [0.0] * 50  # second half has weight 0 -> never drawn when a class has more than k items
    pk = list(pk_batches([0] * 100, p=1, k=4, n_batches=20, seed=0, epoch=1, weights=w))
    assert all(i < 50 for bt in pk for i in bt)
