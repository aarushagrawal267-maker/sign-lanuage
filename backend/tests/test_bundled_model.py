"""The pretrained WLASL100 model shipped in models/ is registered on first start, so a fresh
clone can run live recognition without downloading data or training."""

import os

import numpy as np
import pytest

from app.models.bundled import DEFAULT_BUNDLED

pytestmark = pytest.mark.skipif(not (DEFAULT_BUNDLED / "checkpoint.pt").is_file(), reason="bundled model not present")


@pytest.fixture()
def bundled_env():
    from tests.conftest import fresh_data_dir

    os.environ["SLR_IMPORT_BUNDLED_MODEL"] = "true"
    fresh_data_dir("bundled")
    yield
    os.environ["SLR_IMPORT_BUNDLED_MODEL"] = "false"
    fresh_data_dir("after-bundled")


def test_bundled_model_registered_on_startup(bundled_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:  # the context manager runs the startup hook
        exps = c.get("/api/experiments").json()
        assert len(exps) == 1
        e = exps[0]
        assert e["status"] == "completed" and e["is_active"] and not e["is_synthetic"]
        assert e["val"]["top1"] == pytest.approx(0.620, abs=1e-3)
        assert e["test"]["top1"] == pytest.approx(0.443, abs=1e-3)

        detail = c.get(f"/api/experiments/{e['id']}").json()
        assert detail["split"]["signer_independent"] is True
        # The original split is not shipped, so a new test evaluation is refused cleanly.
        assert c.post(f"/api/experiments/{e['id']}/evaluate-test").status_code == 400

    # Restarting does not import it twice.
    with TestClient(app) as c:
        assert len(c.get("/api/experiments").json()) == 1


def test_bundled_model_predicts_100_glosses(bundled_env):
    from app.inference.registry import get_predictor
    from app.models.bundled import ensure_bundled_model

    ensure_bundled_model()
    p, info = get_predictor()
    assert p is not None and info["num_classes"] == 100
    assert "book" in p.class_names
    feats = np.zeros((info["seq_len"], p.feature_config.dim), dtype=np.float32)
    probs, _ = p.predict(feats)
    assert probs.shape == (100,) and abs(float(probs.sum()) - 1) < 1e-4


def test_deleted_bundled_model_is_not_reimported(bundled_env):
    from fastapi.testclient import TestClient

    from app.main import app
    from app.models.bundled import ensure_bundled_model, import_bundled_model

    with TestClient(app) as c:
        eid = c.get("/api/experiments").json()[0]["id"]
        from app.db.models import Experiment
        from app.db.session import session_scope

        with session_scope() as db:
            db.get(Experiment, eid).is_active = False
        assert c.delete(f"/api/experiments/{eid}").status_code == 200
    ensure_bundled_model()
    with TestClient(app) as c:
        assert c.get("/api/experiments").json() == []
    # ...but it can be brought back explicitly (CLI: python -m app.cli import-model).
    res = import_bundled_model()
    assert res["imported"] and res["activated"]


def test_vocabulary_falls_back_to_model_classes(bundled_env):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        v = c.get("/api/vocabulary").json()
        assert v["model"] is not None and len(v["glosses"]) == 100
        assert v["glosses"][0]["gloss"] == "book"
