"""End to end on a synthetic ISL world: train every component with the real trainers, build a pack, load it
through the registry and use it over REST and the WebSocket (language switch, choose / reject, unknown language,
ASL unchanged), plus My signs endpoints and the DB migration."""

from __future__ import annotations

import io
import json
import os
import sqlite3

import numpy as np
import pytest

from app.dataset.synthetic import frames_for_stream, idle_frames, make_sequence
from app.isl.models.fusion import FusionParams
from app.isl.training.common import ISL_FEATURES, load_isl_data
from app.training.trainer import TrainConfig
from tests.conftest import fresh_data_dir
from tests.isl.isl_world import CORE, make_world

TINY = dict(hidden=16, layers=1, epochs=3, batch_size=8, patience=3, lr=3e-3)


@pytest.fixture(scope="module")
def isl_env(tmp_path_factory):
    saved = {k: os.environ.get(k) for k in ("SLR_DATA_DIR", "SLR_ARTIFACTS_DIR")}
    fresh_data_dir("isl-api")
    root = tmp_path_factory.mktemp("islworld")
    w = make_world(root)
    from app.isl.training.build_index import build_index
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import build_pack, evaluate, load_recognizer
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.isl.vocab.build import VocabIndex
    from app.isl.models.encoder import EncoderPredictor

    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(w["paths"], vocab, "commercial-safe", w["split"])
    assert data.summary()["include"] == {"train": 16, "val": 4, "test": 4} and len(data.islrtc) == 6
    work = root / "work"
    work.mkdir()
    rc = train_classifier(data, work / "classifier.pt", ISL_FEATURES, TrainConfig(**TINY), smoke=True, threads=1)
    re_ = train_encoder(data, work / "encoder.pt", ISL_FEATURES, TrainConfig(**TINY), emb_dim=16, p=4, k=2, smoke=True,
                        threads=1, batches_per_epoch=3)
    ri = build_index(EncoderPredictor(work / "encoder.pt"), data.islrtc, data.include_part("train"), work / "index",
                     "commercial-safe")
    rec = load_recognizer(work, ISL_FEATURES, w["vocab"], profile="commercial-safe")
    params, rf = fit_fusion(rec, data.include_part("val"), "commercial-safe")
    params.save(work / "fusion.json")
    doc = evaluate(work, data, ISL_FEATURES, smoke=True, vocab_path=w["vocab"])
    packs = root / "packs"
    rp = build_pack(work, "commercial-safe", ISL_FEATURES, packs / "isl", smoke=True, vocab_path=w["vocab"])
    # deterministic decisions for the API tests: never accept, always "did you mean"
    FusionParams.load(packs / "isl" / "fusion.json").with_(
        accept={"strict": 1.01, "balanced": 1.01, "lenient": 1.01}, suggest_min_top5_mass=0.0, s_min=-1.0
    ).save(packs / "isl" / "fusion.json")
    os.environ["SLR_PACKS_DIRS"] = str(packs)
    from app.core.config import get_settings
    from app.packs.registry import reset_registries

    get_settings.cache_clear()
    reset_registries()
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    yield {"client": client, "world": w, "reports": {"classifier": rc, "encoder": re_, "index": ri, "fusion": rf,
                                                      "results": doc, "pack": rp}, "packs": packs}
    client.close()
    os.environ.pop("SLR_PACKS_DIRS", None)
    for k, v in saved.items():  # give the session-wide (ASL) fixtures their data dir back
        if v is not None:
            os.environ[k] = v
    from app.db.session import reset_engine

    get_settings.cache_clear()
    reset_engine()
    reset_registries()


def _clip_frames(cls: int, signer: int = 5):
    return frames_for_stream(make_sequence(cls, signer=signer, n_frames=45, fps=25.0))


def test_training_reports_and_results_file(isl_env):
    r = isl_env["reports"]
    assert r["classifier"]["classes"] == len(CORE) and r["encoder"]["concepts"] == 6
    assert r["index"]["first_sign"] == 6 and r["index"]["centroids"] == len(CORE)
    assert r["fusion"]["val_queries"] == 4
    res = r["results"]
    assert res["smoke"] is True and "NOT the accuracy" in res["label"]
    assert {"E1", "E2", "E3", "E6"} <= set(res["protocols"]) and res["protocols"]["E1"]["n"] == 4
    assert r["pack"]["id"] == "isl" and "classifier.pt" in r["pack"]["files"]


def test_registry_lists_both_languages(isl_env):
    c = isl_env["client"]
    j = c.get("/api/languages").json()
    ids = {x["id"]: x for x in j["languages"]}
    assert j["default"] == "asl-wlasl100" and {"asl-wlasl100", "isl"} <= set(ids)
    isl = ids["isl"]
    assert isl["status"] == "ready" and isl["profile"] == "commercial-safe" and "retrieval" in isl["capabilities"]
    assert isl["display_name"]["hi"] == "भारतीय सांकेतिक भाषा"
    one = c.get("/api/languages/isl").json()
    assert one["model"]["components"]["classifier"]["classes"] == len(CORE) and one["results"]["smoke"] is True
    assert c.get("/api/languages/isl/results").json()["protocols"]["E3"]["n"] == 4
    assert c.get("/api/languages/xyz").status_code == 404


def test_vocabulary_search_and_entry(isl_env):
    c = isl_env["client"]
    v = c.get("/api/languages/isl/vocabulary", params={"q": "do"}).json()
    assert v["total"] == 1 and v["items"][0]["concept_id"] == "isl.dog" and v["items"][0]["text"]["hi"] is None
    assert c.get("/api/languages/isl/vocabulary", params={"in_classifier": False}).json()["total"] == 2
    e = c.get("/api/languages/isl/vocabulary/isl.dog").json()
    assert e["class_index"] == 3 and e["variants"][0]["source"] == "islrtc"
    assert c.get("/api/languages/isl/vocabulary/isl.nothing").status_code == 404


def test_lexicon_patch_gives_hindi(isl_env):
    c = isl_env["client"]
    r = c.patch("/api/languages/isl/lexicon/isl.dog", json={"hi": "कुत्ता", "review": "reviewed", "reviewer": "teacher-1"})
    assert r.status_code == 200 and r.json()["hi"] == "कुत्ता"
    assert c.get("/api/languages/isl/vocabulary", params={"q": "dog"}).json()["items"][0]["text"]["hi"] == "कुत्ता"
    assert c.patch("/api/languages/isl/lexicon/isl.dog", json={"hi": "x", "review": "bogus"}).status_code == 422


def test_analyze_landmarks_with_language(isl_env):
    c = isl_env["client"]
    body = {"width": 640, "height": 480, "frames": _clip_frames(1)}
    r = c.post("/api/languages/isl/analyze/landmarks", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["decision"] == "suggest" and 1 <= len(j["candidates"]) <= 5 and j["top"] == j["candidates"]
    cand = j["candidates"][0]
    assert cand["concept_id"].startswith("isl.") and cand["text"]["en"] and cand["source"] in (
        "classifier", "retrieval", "classifier+retrieval")
    assert abs(sum(x["prob"] for x in j["candidates"]) - 1) < 0.2
    legacy = c.post("/api/analyze/landmarks", json={**body, "language": "isl"})
    assert legacy.status_code == 200 and legacy.json()["decision"] == "suggest"
    assert c.post("/api/languages/isl/analyze/landmarks", json={**body, "policy": "x"}).status_code == 422
    assert c.post("/api/languages/isl/analyze/landmarks", json={**body, "spell": True}).status_code == 409


def _stream(ws, frames):
    out = []
    for f in frames:
        ws.send_json({"type": "frame", **f})
        out.append(ws.receive_json())
    return out


def test_websocket_language_query_choose_and_reject(isl_env):
    c = isl_env["client"]
    with c.websocket_connect("/api/ws/recognize?language=isl") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["language"]["id"] == "isl"
        assert {x["id"] for x in hello["languages"]} >= {"asl-wlasl100", "isl"}
        assert hello["engine"]["config"]["policy"] == "balanced" and hello["model"]["profile"] == "commercial-safe"
        frames = idle_frames(10, 0) + frames_for_stream(make_sequence(2, 5, n_frames=45), 400) + idle_frames(20, 2400)
        states = _stream(ws, frames)
        results = [s["segment_result"] for s in states if "segment_result" in s and "segment_id" in s["segment_result"]]
        assert results, [s.get("status") for s in states]
        seg = results[-1]
        assert seg["decision"] == "suggest" and not any("commit" in s for s in states)
        assert seg["candidates"] == seg["top"] and seg["candidates"][0]["text"]["en"]
        pick = seg["candidates"][1]["concept_id"]
        ws.send_json({"type": "choose", "segment_id": seg["segment_id"], "concept_id": pick})
        ack = ws.receive_json()
        assert ack["type"] == "choice_ack" and ack["commit"]["concept_id"] == pick and ack["commit"]["chosen_by_user"]
        ws.send_json({"type": "choose", "segment_id": seg["segment_id"], "concept_id": "isl.not-offered"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "reject", "segment_id": seg["segment_id"]})
        rej = ws.receive_json()
        assert rej["type"] == "reject_ack" and rej["rejected"] == pick
        ws.send_json({"type": "config", "engine": {"mode": "spell"}})
        assert "spelling" in ws.receive_json()["message"]  # no letter model and no alphabet calibration yet
        ws.send_json({"type": "config", "engine": {"policy": "nope"}})
        assert ws.receive_json()["type"] == "error"
    ev = c.get("/api/recognition/events", params={"language": "isl"}).json()
    assert ev[0]["source"] == "feedback" and ev[1]["detail"]["chosen_by_user"] is True
    assert ev[1]["detail"]["concept_id"] == pick
    assert all(e["language"] == "isl" for e in ev)


def test_websocket_switch_language_and_unknown(isl_env):
    c = isl_env["client"]
    with c.websocket_connect("/api/ws/recognize") as ws:
        hello = ws.receive_json()
        assert hello["language"]["id"] == "asl-wlasl100"
        assert "policy" not in hello["engine"]["config"]  # ASL engine config unchanged
        ws.send_json({"type": "config", "language": "isl", "width": 1280, "height": 720})
        h2 = ws.receive_json()
        assert h2["type"] == "hello" and h2["language"]["id"] == "isl"
        ws.send_json({"type": "config", "language": "xyz"})
        err = ws.receive_json()
        assert err["type"] == "error" and "unknown language 'xyz'" in err["message"] and "isl" in err["message"]
        ws.send_json({"type": "ping", "t": 3})
        assert ws.receive_json() == {"type": "pong", "t": 3}
        ws.send_json({"type": "config", "language": "asl-wlasl100"})
        assert ws.receive_json()["language"]["id"] == "asl-wlasl100"
    with c.websocket_connect("/api/ws/recognize?language=nope") as ws:
        assert ws.receive_json()["language"]["id"] == "asl-wlasl100"
        assert ws.receive_json()["type"] == "error"


def test_my_signs_endpoints_and_recognition(isl_env):
    c = isl_env["client"]
    takes = [{"width": 640, "height": 480, "frames": frames_for_stream(make_sequence(7, s, n_frames=40, fps=25.0))}
             for s in range(3)]
    dry = c.post("/api/languages/isl/my-signs", json={"text_en": "Poha", "takes": takes, "dry_run": True}).json()
    assert dry["sign"] is None and dry["take_quality"][0]["ok"] and dry["consistency"] is not None
    r = c.post("/api/languages/isl/my-signs", json={"text_en": "Poha", "text_hi": "पोहा", "region": "Indore, MP",
                                                    "takes": takes})
    assert r.status_code == 200 and r.json()["accepted"], r.json()
    sid = r.json()["sign"]["sign_id"]
    lst = c.get("/api/languages/isl/my-signs").json()["signs"]
    assert [s["text_en"] for s in lst] == ["Poha"]
    # the user's sign is now searchable: a new take of it is recognised as "My signs"
    j = c.post("/api/languages/isl/analyze/landmarks",
               json={"width": 640, "height": 480, "frames": frames_for_stream(make_sequence(7, 1, n_frames=40, fps=25.0))}).json()
    ours = [x for x in j["candidates"] if x["source"] == "my_signs"]
    assert ours and ours[0]["text"] == {"en": "Poha", "hi": "पोहा"} and ours[0]["region"] == "Indore, MP"
    # export -> delete -> import
    exp = c.get("/api/languages/isl/my-signs/export", params={"title": "Indore school", "region": "Indore"})
    assert exp.status_code == 200 and exp.headers["content-type"] == "application/zip"
    assert c.patch(f"/api/languages/isl/my-signs/{sid}", json={"enabled": False}).json()["enabled"] is False
    assert c.delete(f"/api/languages/isl/my-signs/{sid}").json() == {"deleted": True}
    assert c.delete(f"/api/languages/isl/my-signs/{sid}").status_code == 404
    imp = c.post("/api/languages/isl/my-signs/import", files={"file": ("x.signpack", io.BytesIO(exp.content))},
                 data={"mode": "merge"})
    assert imp.status_code == 200 and imp.json()["imported"] == 1
    bad = c.post("/api/languages/isl/my-signs/import", files={"file": ("x.signpack", io.BytesIO(b"nope"))})
    assert bad.status_code == 422
    assert c.get("/api/languages/asl-wlasl100/my-signs").status_code == 400


def test_alphabet_calibration_enables_spelling(isl_env):
    c = isl_env["client"]
    assert c.get("/api/languages/isl/my-signs/alphabet").json()["ready"] is False
    for i, letter in enumerate("ABCDE"):
        frames = frames_for_stream(make_sequence(i, 0, n_frames=30, fps=25.0))
        r = c.post("/api/languages/isl/my-signs/alphabet", json={"letter": letter,
                                                                 "take": {"width": 640, "height": 480, "frames": frames}})
        assert r.status_code == 200, r.text
    st = c.get("/api/languages/isl/my-signs/alphabet").json()
    assert st["ready"] and set(st["letters"]) == set("ABCDE")
    assert "spelling" in c.get("/api/languages/isl").json()["pack"]["capabilities"]
    body = {"width": 640, "height": 480, "frames": frames_for_stream(make_sequence(1, 0, n_frames=30, fps=25.0)),
            "spell": True}
    j = c.post("/api/languages/isl/analyze/landmarks", json=body).json()
    assert j["decision"] in ("suggest", "abstain") and j["reason"] in ("spelling", "no_letters")
    with c.websocket_connect("/api/ws/recognize?language=isl") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "engine": {"mode": "spell"}})
        assert ws.receive_json()["type"] == "config_ack"
    assert c.delete("/api/languages/isl/my-signs/alphabet").json() == {"deleted": True}


def test_asl_routes_unchanged_without_language(isl_env):
    c = isl_env["client"]
    with c.websocket_connect("/api/ws/recognize") as ws:
        hello = ws.receive_json()
        assert set(hello) == {"type", "model", "engine", "message", "language", "languages"}
        assert set(hello["engine"]["config"]) == {"mode", "threshold", "min_margin", "min_frames", "min_hand_ratio",
                                                  "start_frames", "pre_roll_ms", "end_absent_ms", "max_segment_ms",
                                                  "interval_ms", "window_ms", "ema_alpha", "agree_n", "cooldown_ms", "top_k"}


def test_missing_models_pack_and_profile_mismatch(tmp_path, isl_env):
    from app.packs.registry import PackRegistry

    src = isl_env["packs"] / "isl"
    bare = tmp_path / "r1" / "isl"
    bare.mkdir(parents=True)
    (bare / "pack.json").write_text((src / "pack.json").read_text())
    (bare / "vocabulary.jsonl").write_text((src / "vocabulary.jsonl").read_text())
    reg = PackRegistry([tmp_path / "r1"])
    info = reg.get("isl").info
    assert info.status == "missing_models" and "My signs" in info.message
    rec, _, problem = reg.get("isl").recognizer()
    assert rec is not None and not rec.has_global_model and problem  # My signs still work
    st = rec.my_signs
    takes = [make_sequence(7, s, n_frames=40, fps=25.0) for s in range(3)]
    assert st.add("Poha", takes, embedder=rec.user_embedder).accepted
    from app.isl.training.common import engine_crops

    same = rec.recognize(engine_crops(make_sequence(7, 4, n_frames=40, fps=25.0)))
    other = rec.recognize(engine_crops(make_sequence(2, 4, n_frames=40, fps=25.0)))
    assert same.candidates[0].gloss == "Poha" and same.decision in ("accept", "suggest")
    assert other.decision != "accept", "a different sign must never be silently accepted as the only user sign"
    st.delete_all()
    import shutil

    mixed = tmp_path / "r2" / "isl"
    shutil.copytree(src, mixed)
    man = json.loads((mixed / "pack.json").read_text())
    man["profile"] = "research"
    (mixed / "pack.json").write_text(json.dumps(man))
    rec, _, problem = PackRegistry([tmp_path / "r2"]).get("isl").recognizer()
    assert rec is None and "data profile" in problem
    broken = tmp_path / "r3" / "bad"
    broken.mkdir(parents=True)
    (broken / "pack.json").write_text("{not json")
    assert PackRegistry([tmp_path / "r3"]).get("bad").info.status == "error"


def test_db_migration_adds_columns(tmp_path):
    from sqlalchemy import create_engine

    from app.db.migrate import ensure_columns

    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE recognition_events (id INTEGER PRIMARY KEY, created_at TEXT, source TEXT, gloss TEXT, "
                "class_index INTEGER, score REAL, experiment_id INTEGER)")
    con.execute("INSERT INTO recognition_events (source, gloss, class_index, score) VALUES ('live', 'book', 1, 0.9)")
    con.commit()
    con.close()
    eng = create_engine(f"sqlite:///{p}")
    assert ensure_columns(eng) == ["recognition_events.language", "recognition_events.detail"]
    assert ensure_columns(eng) == []
    con = sqlite3.connect(p)
    assert con.execute("SELECT language FROM recognition_events").fetchone() == ("asl-wlasl100",)
    con.close()


def test_latency_e6_reported(isl_env):
    e6 = isl_env["reports"]["results"]["protocols"]["E6"]
    assert e6["n"] > 0 and e6["metrics"]["p50_ms"] > 0 and np.isfinite(e6["metrics"]["p95_ms"])
