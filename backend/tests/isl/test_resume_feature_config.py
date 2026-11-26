"""The feature config is part of the encoder / classifier resume signature, and a warm start records (and warns
about) the feature config its init checkpoint was trained with (docs/ISL_DESIGN.md 15.15)."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace

import pytest
import torch

from app.isl.models.encoder import SignEncoder
from app.isl.training.common import ISL_FEATURES, load_isl_data
from app.isl.training.loop import LoopConfig, StateMismatch
from app.isl.training.train_classifier import train_classifier
from app.isl.training.train_encoder import feature_config_diff, train_encoder
from app.isl.vocab.build import VocabIndex
from app.training.trainer import TrainConfig
from tests.isl.isl_world import make_world

UNFLOORED = replace(ISL_FEATURES, hand_size_floor=0.0)  # the feature config before 26 Nov 2026
CFG = TrainConfig(hidden=16, layers=1, epochs=1, batch_size=8, patience=1, lr=3e-3)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("resume_fc"))
    return load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "commercial-safe", w["split"])


def _enc(data, out, fcfg, **kw):
    return train_encoder(data, out, fcfg, CFG, emb_dim=16, p=4, k=2, threads=1, batches_per_epoch=2, smoke=True, **kw)


def _cls(data, out, fcfg, **kw):
    return train_classifier(data, out, fcfg, CFG, threads=1, steps_per_epoch=2, smoke=True, **kw)


def _init_ckpt(path, fcfg):
    enc = SignEncoder(fcfg.dim, hidden=16, layers=1, emb_dim=16)
    torch.save({"state_dict": enc.state_dict(),
                "meta": {"kind": "isl-encoder", "feature_config": fcfg.to_dict(),
                         "pretraining": {"objective": "test", "labels_used": False}}}, path)
    return path


def test_encoder_resume_accepted_on_same_features_refused_on_other(data, tmp_path):
    out = tmp_path / "enc.pt"
    _enc(data, out, ISL_FEATURES)
    assert _enc(data, out, ISL_FEATURES, resume=True)["resumed_from"]["finished"] is True
    with pytest.raises(StateMismatch):
        _enc(data, out, UNFLOORED, resume=True)


def test_classifier_resume_accepted_on_same_features_refused_on_other(data, tmp_path):
    out = tmp_path / "cls.pt"
    _cls(data, out, ISL_FEATURES)
    before = (tmp_path / "cls.pt.state").stat().st_mtime_ns
    assert _cls(data, out, ISL_FEATURES, resume=True)["finished"] is True
    assert (tmp_path / "cls.pt.state").stat().st_mtime_ns == before  # finished state returned, not retrained
    with pytest.raises(StateMismatch):
        _cls(data, out, UNFLOORED, resume=True)


def test_state_from_code_without_the_features_key_is_refused(data, tmp_path):
    """A .state written before this change has no "features" in its signature: it is refused (StateMismatch),
    never silently continued; the driver's --force (or deleting it) starts over."""
    out = tmp_path / "cls.pt"
    _cls(data, out, ISL_FEATURES)
    sp = tmp_path / "cls.pt.state"
    st = torch.load(sp, map_location="cpu", weights_only=False)
    old = dict(st["config"])
    old["extra"] = {k: v for k, v in old["extra"].items() if k != "features"}
    assert st["config"]["extra"]["features"] == ISL_FEATURES.to_dict()
    st["signature"], st["config"] = LoopConfig(**old).signature(), old
    torch.save(st, sp)
    with pytest.raises(StateMismatch):
        _cls(data, out, ISL_FEATURES, resume=True)


def test_feature_config_diff_fills_defaults():
    assert feature_config_diff(ISL_FEATURES.to_dict(), ISL_FEATURES) == {}
    assert feature_config_diff(None, ISL_FEATURES) == {}
    old = UNFLOORED.to_dict()
    assert "hand_size_floor" not in old  # how an unfloored checkpoint stores its config
    assert feature_config_diff(old, ISL_FEATURES) == {"hand_size_floor": [0.0, 0.1]}
    assert asdict(ISL_FEATURES).keys() >= feature_config_diff(old, ISL_FEATURES).keys()


def test_warm_start_records_init_feature_config_and_warns_on_mismatch(data, tmp_path, caplog):
    log_name = "app.isl.training.train_encoder"
    init = _init_ckpt(tmp_path / "pre_unfloored.pt", UNFLOORED)
    with caplog.at_level(logging.WARNING, logger=log_name):
        r = _enc(data, tmp_path / "enc.pt", ISL_FEATURES, init_from=init)
    assert r["pretrained"]
    warns = [x for x in caplog.records if x.name == log_name and x.levelno == logging.WARNING]
    assert len(warns) == 1 and "hand_size_floor" in warns[0].getMessage()
    pt = torch.load(tmp_path / "enc.pt", map_location="cpu", weights_only=False)["meta"]["pretraining"]
    assert pt["init_feature_config"] == UNFLOORED.to_dict()
    assert pt["init_feature_config_diff"] == {"hand_size_floor": [0.0, 0.1]}
    assert pt["objective"] == "test" and pt["init_from"] == str(init)

    caplog.clear()
    same = _init_ckpt(tmp_path / "pre_floored.pt", ISL_FEATURES)
    with caplog.at_level(logging.WARNING, logger=log_name):
        _enc(data, tmp_path / "enc2.pt", ISL_FEATURES, init_from=same)
    assert not [x for x in caplog.records if x.name == log_name]
    pt = torch.load(tmp_path / "enc2.pt", map_location="cpu", weights_only=False)["meta"]["pretraining"]
    assert pt["init_feature_config"] == ISL_FEATURES.to_dict() and pt["init_feature_config_diff"] == {}
