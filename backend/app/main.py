"""FastAPI application entry point.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000   (from the backend/ folder)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    if s.public_demo:
        allowed = {"/api/analyze/video", "/api/analyze/landmarks"}

        @app.middleware("http")
        async def read_only_guard(request: Request, call_next):
            if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path not in allowed:
                return JSONResponse({"detail": "Disabled in the public demo. Run the project locally to download data or train models."},
                                    status_code=403)
            return await call_next(request)

    for r in (system.router, dataset.router, experiments.router, recognition.router):
        app.include_router(r)
    s.assets_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/assets/mediapipe", StaticFiles(directory=s.assets_dir / "mediapipe", check_dir=False), name="mediapipe-model")

    dist = s.frontend_dist
    if (dist / "index.html").exists():
        # Single-origin deployment: serve the built frontend (SPA) from the API server.
        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
