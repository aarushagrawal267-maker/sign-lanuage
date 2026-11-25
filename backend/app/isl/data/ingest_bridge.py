"""Import ``scripts/ingest_isl.py`` (not a package) so gated-dataset ingestion reuses exactly the same
extraction code (``work_item``, ``extract_full``) as the INCLUDE/ISLRTC jobs."""

from __future__ import annotations

import functools
import importlib.util

from app.core.config import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "ingest_isl.py"


@functools.lru_cache(maxsize=1)
def ingest_module():
    spec = importlib.util.spec_from_file_location("ingest_isl_script", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
