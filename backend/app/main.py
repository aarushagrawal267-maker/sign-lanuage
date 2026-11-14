"""FastAPI application entry point.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000   (from the backend/ folder)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import dataset, experiments, recognition, system
from app.core.config import get_settings
from app.core.jobs import recover_stale_jobs
from app.db.session import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DESCRIPTION = """
Educational and research implementation of **isolated, word-level** ASL sign
recognition on the WLASL100 vocabulary (MediaPipe Holistic landmarks +
recurrent sequence models).

* Output is an ASL **gloss label** per isolated sign - not sentence translation.
* WLASL is licensed under the Computational Use of Data Agreement (C-UDA):
  academic / computational use only, **no commercial use**. Videos are not redistributed.
* Live recognition receives landmark coordinates only; no webcam images reach this server.
"""


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_settings()
    get_engine()
    recover_stale_jobs()
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="Sign Intelligence Studio API", version="1.0.0", description=DESCRIPTION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.frontend_origin, "http://127.0.0.1:5173", "http://localhost:4173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (system.router, dataset.router, experiments.router, recognition.router):
        app.include_router(r)
    s.assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=s.assets_dir), name="assets")
    return app


app = create_app()
