"""Pipeline tests on SYNTHETIC landmark data (verifies wiring, not ASL accuracy)."""

import numpy as np

from app.dataset.synthetic import frames_for_stream, idle_frames, make_sequence
from app.inference.engine import EngineConfig, RecognitionEngine
from app.inference.registry import get_predictor


def test_training_produces_real_metrics(synthetic_env):
    c = synthetic_env["client"]
    e = c.get(f"/api/experiments/{synthetic_env['experiment_id']}").json()
    assert e["status"] == "completed"
    assert len(e["history"]) >= 1 and all("val_loss" in h for h in e["history"])
    vm = e["val_metrics"]
    assert vm["n"] > 0 and 0 <= vm["top1"] <= 1
    assert len(vm["per_class"]) == 6 and len(vm["confusion_matrix"]) == 6
    assert vm["top1"] > 0.6, "synthetic classes are separable; the model should learn them"
    assert e["test_metrics"] == {}, "test set must not be evaluated during training"
    assert e["is_synthetic"] is True


def test_test_evaluation_is_explicit(synthetic_env):
    c = synthetic_env["client"]
    r = c.post(f"/api/experiments/{synthetic_env['experiment_id']}/evaluate-test")
    assert r.status_code == 200, r.text
    tm = c.get(f"/api/experiments/{synthetic_env['experiment_id']}").json()["test_metrics"]
    assert tm["partition"] == "test" and tm["n"] > 0 and "latency" in tm


def _stream(engine, seqs, gap=20):
    t, commits, statuses = 0.0, [], []

    def push_all(frames):
        for f in frames:
            st = engine.push(f)
            statuses.append(st["status"])
            if "commit" in st:
                commits.append(st["commit"]["class_index"])

    for s in seqs:
        push_all(idle_frames(gap, t))
        t += gap * 40
        push_all(frames_for_stream(s, t))
        t += len(s) * 40
    push_all(idle_frames(gap, t))
    return commits, statuses


def test_segment_engine_emits_one_word_per_sign(synthetic_env):
    p, _ = get_predictor()
    eng = RecognitionEngine(p, 640, 480, EngineConfig(mode="segment", threshold=0.3))
    rng = np.random.default_rng(7)
    labels = [0, 3, 3, 5]
    commits, statuses = _stream(eng, [make_sequence(c, 7, rng=rng) for c in labels])
    assert len(commits) <= len(labels), "never more than one word per sign"
    assert sum(a == b for a, b in zip(commits, labels)) >= 3
    assert {"waiting_for_hands", "collecting"} <= set(statuses)


def test_high_threshold_rejects(synthetic_env):
    p, _ = get_predictor()
    eng = RecognitionEngine(p, 640, 480, EngineConfig(mode="segment", threshold=0.999, min_margin=0.99))
    commits, statuses = _stream(eng, [make_sequence(1, 7)])
    assert commits == [] and "uncertain" in statuses


def test_sliding_engine_suppresses_repeats(synthetic_env):
    p, _ = get_predictor()
    eng = RecognitionEngine(p, 640, 480, EngineConfig(mode="sliding", threshold=0.3, window_ms=1200, agree_n=2))
    long = make_sequence(2, 7, n_frames=150)  # one long continuous sign
    commits, _ = _stream(eng, [long])
    assert len(commits) <= 1


def test_no_model_reports_unavailable():
    eng = RecognitionEngine(None)
    st = eng.push(frames_for_stream(make_sequence(0, 0))[0])
    assert st["status"] == "unavailable"


def test_websocket_protocol(synthetic_env):
    c = synthetic_env["client"]
    with c.websocket_connect("/api/ws/recognize") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["model"]["experiment_id"] == synthetic_env["experiment_id"]
        ws.send_json({"type": "config", "width": 640, "height": 480, "engine": {"threshold": 0.3}})
        assert ws.receive_json()["type"] == "config_ack"
        ws.send_json({"type": "frame", "t": 0, "pose": [[0, 0, 0]], "left": None, "right": None, "face": None})
        assert ws.receive_json()["type"] == "error"
        got_commit = False
        frames = idle_frames(10, 0) + frames_for_stream(make_sequence(4, 7), 400) + idle_frames(20, 4000)
        for f in frames:
            ws.send_json({"type": "frame", **f})
            st = ws.receive_json()
            assert st["type"] == "state" and "status" in st and "server_ms" in st
            got_commit |= "commit" in st
        ws.send_json({"type": "ping", "t": 1})
        assert ws.receive_json()["type"] == "pong"
    ev = c.get("/api/recognition/events").json()
    assert got_commit == (len([e for e in ev if e["source"] == "live"]) > 0)


def test_api_overview_vocabulary_and_spec(synthetic_env):
    c = synthetic_env["client"]
    ov = c.get("/api/overview").json()
    assert ov["vocabulary_size"] == 6 and ov["active_model"]["is_synthetic"] is True
    voc = c.get("/api/vocabulary").json()
    assert len(voc["glosses"]) == 6 and voc["glosses"][0]["val"] is not None
    spec = c.get("/api/features/spec").json()
    assert spec["from_active_model"] and spec["feature_config"]["seq_len"] == 32
    assert len(spec["raw"]["face_indices"]) == spec["raw"]["face"]["count"]
    splits = c.get("/api/splits").json()
    names = {s["name"]: s for s in splits}
    assert names["signer-holdout"]["leakage_report"]["signer_independent"] is True
    assert c.get("/api/dataset/summary").status_code == 200
    assert c.get("/api/dataset/inventory").status_code == 200


def test_analyze_landmarks(synthetic_env):
    c = synthetic_env["client"]
    s = make_sequence(5, 6)
    r = c.post("/api/analyze/landmarks", json={"width": 640, "height": 480, "frames": frames_for_stream(s)})
    assert r.status_code == 200 and len(r.json()["top"]) == 5


def test_lstm_and_tcn_train(synthetic_env):
    from tests.conftest import wait_job

    c = synthetic_env["client"]
    for arch in ("lstm", "tcn"):
        r = c.post("/api/experiments", json={"architecture": arch, "split_name": "signer-holdout", "hidden": 32,
                                             "layers": 1, "epochs": 2, "batch_size": 16, "features": {"seq_len": 24}})
        assert r.status_code == 200, r.text
        wait_job(c, r.json()["job_id"], timeout=120)
        e = c.get(f"/api/experiments/{r.json()['experiment_id']}").json()
        assert e["status"] == "completed" and e["split"]["signer_independent"] is True


def test_csv_exports(synthetic_env):
    c = synthetic_env["client"]
    r = c.get("/api/experiments/export.csv")
    assert r.status_code == 200 and r.text.startswith("id,name,status")
    r = c.get(f"/api/experiments/{synthetic_env['experiment_id']}/per-class.csv")
    assert r.status_code == 200 and "partition" in r.text.splitlines()[0]
