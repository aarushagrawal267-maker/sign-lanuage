"""Test fixtures. Every test session runs against an isolated temporary data dir,
so tests never touch real data in ./data or ./artifacts."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="slr-test-"))
os.environ["SLR_DATA_DIR"] = str(_TMP / "data")
os.environ["SLR_ARTIFACTS_DIR"] = str(_TMP / "artifacts")
os.environ["SLR_DEVICE"] = "cpu"
REPO_MODEL = ROOT.parent / "assets" / "mediapipe" / "holistic_landmarker.task"
if REPO_MODEL.exists():
    (_TMP / "assets" / "mediapipe").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_MODEL, _TMP / "assets" / "mediapipe" / REPO_MODEL.name)
os.environ["SLR_ASSETS_DIR"] = str(_TMP / "assets")


def fresh_data_dir(name: str) -> None:
    """Point the app at a new empty data dir (fresh SQLite DB)."""
    from app.core.config import get_settings
    from app.db.session import reset_engine

    os.environ["SLR_DATA_DIR"] = str(_TMP / name / "data")
    os.environ["SLR_ARTIFACTS_DIR"] = str(_TMP / name / "artifacts")
    get_settings.cache_clear()
    reset_engine()
    get_settings()


@pytest.fixture(scope="session")
def synthetic_env():
    """A synthetic dataset + official split + one trained & activated GRU model."""
    fresh_data_dir("synthetic")
    from app.dataset.synthetic import populate_synthetic_db
    from app.db.session import session_scope
    from app.splits.splits import ensure_default_splits

    populate_synthetic_db(n_classes=6, n_signers=8, per_signer=3, seed=1)
    with session_scope() as db:
        ensure_default_splits(db)
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    r = client.post("/api/experiments", json={
        "name": "test-gru", "split_name": "official", "architecture": "gru", "hidden": 64, "layers": 1,
        "epochs": 25, "patience": 25, "batch_size": 16, "lr": 3e-3, "features": {"seq_len": 32},
    })
    assert r.status_code == 200, r.text
    exp_id = r.json()["experiment_id"]
    wait_job(client, r.json()["job_id"], timeout=300)
    assert client.post(f"/api/experiments/{exp_id}/activate").status_code == 200
    yield {"client": client, "experiment_id": exp_id}
    client.close()


def wait_job(client, job_id: int, timeout: float = 120) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = {j["id"]: j for j in client.get("/api/jobs?limit=50").json()}
        j = jobs.get(job_id)
        if j and j["status"] in ("completed", "failed", "cancelled"):
            assert j["status"] == "completed", j
            return j
        time.sleep(0.3)
    raise TimeoutError(f"job {job_id} did not finish")


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TMP, ignore_errors=True)
