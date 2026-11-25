"""Inference robustness: a missing or corrupt model file and malformed live messages must give
clear messages, never a server error or a dropped connection."""

import os

import pytest

from app.models.bundled import DEFAULT_BUNDLED

pytestmark = pytest.mark.skipif(not (DEFAULT_BUNDLED / "checkpoint.pt").is_file(), reason="bundled model not present")


@pytest.fixture()
def client(request):
    from fastapi.testclient import TestClient

    from app.main import app
    from tests.conftest import fresh_data_dir

    os.environ["SLR_IMPORT_BUNDLED_MODEL"] = "true"
    fresh_data_dir(f"robustness-{request.node.name}")
    with TestClient(app) as c:  # startup registers and activates the bundled model
        yield c
    os.environ["SLR_IMPORT_BUNDLED_MODEL"] = "false"
    fresh_data_dir("after-robustness")


def _checkpoint(c) -> str:
    info = c.get("/api/models/active").json()
    assert info["experiment_id"] is not None
    return info["checkpoint"]


def _assert_unavailable(c, needle: str) -> None:
    info = c.get("/api/models/active").json()
    assert info["experiment_id"] is None and needle in info["message"]
    with c.websocket_connect("/api/ws/recognize") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["model"] is None and needle in hello["message"]
        ws.send_json({"type": "frame", "t": 0, "pose": None, "left": None, "right": None, "face": None})
        assert ws.receive_json()["status"] == "unavailable"
    r = c.post("/api/analyze/landmarks", json={"width": 640, "height": 480, "frames": []})
    assert r.status_code == 409 and needle in r.json()["detail"]
    assert c.get("/api/features/spec").status_code == 200
    assert c.get("/api/vocabulary").status_code == 200


def test_missing_checkpoint_gives_clear_message(client):
    os.remove(_checkpoint(client))
    _assert_unavailable(client, "has no checkpoint file")


def test_corrupt_checkpoint_gives_clear_message(client):
    path = _checkpoint(client)
    with open(path, "wb") as f:
        f.write(b"not a torch checkpoint")
    from app.inference import registry

    registry._cache["id"] = registry._cache["predictor"] = None  # as after a restart
    _assert_unavailable(client, "could not be loaded")


def test_malformed_live_messages_keep_the_connection(client):
    with client.websocket_connect("/api/ws/recognize") as ws:
        assert ws.receive_json()["model"]["num_classes"] == 100
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json([1, 2, 3])
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(b"\x00\x01")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "config", "width": 0, "height": 480})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "t": {"bad": 1}, "pose": None, "left": None, "right": None, "face": None})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "t": 0, "pose": "abc", "left": None, "right": None, "face": None})
        assert ws.receive_json()["type"] == "error"
        # json.loads accepts Infinity/NaN: int(Infinity) raised OverflowError and a NaN timestamp was accepted
        ws.send_text('{"type": "config", "width": Infinity, "height": 480}')
        assert ws.receive_json()["type"] == "error"
        ws.send_text('{"type": "config", "engine": {"top_k": Infinity}}')
        assert ws.receive_json()["type"] == "error"
        ws.send_text('{"type": "frame", "t": NaN, "pose": null, "left": null, "right": null, "face": null}')
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "ping", "t": 7})
        assert ws.receive_json() == {"type": "pong", "t": 7}


def test_landmark_clip_needs_a_frame_size(client):
    r = client.post("/api/analyze/landmarks", json={"width": 0, "height": 480, "frames": []})
    assert r.status_code == 422
    r = client.post("/api/analyze/landmarks", json={"width": 640, "height": 480, "frames": [{"t": 0, "pose": {"a": 1}}]})
    assert r.status_code == 422
