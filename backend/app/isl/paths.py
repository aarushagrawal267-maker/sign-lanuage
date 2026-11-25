"""Where ISL data lives on disk.

Everything derives from one root (default ``<SLR_DATA_DIR>/isl``, i.e. ``data/isl`` in the repo), which can
be overridden with ``SLR_ISL_DIR``. Manifests written by ``scripts/ingest_isl.py`` store clip paths relative
to the repository root (``data/isl/landmarks/...``); :meth:`IslPaths.resolve` maps them onto the current root
so a moved data folder (or a test fixture) still works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.core.config import REPO_ROOT

PACK_DIR = REPO_ROOT / "models" / "packs" / "isl"


@dataclass(frozen=True)
class IslPaths:
    root: Path

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def landmarks(self) -> Path:
        return self.root / "landmarks"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def official_splits(self) -> Path:
        return self.splits / "official"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    @property
    def nltk_data(self) -> Path:
        return self.root / "nltk_data"

    def manifest(self, name: str) -> Path:
        return self.manifests / f"{name}.jsonl"

    @property
    def trim_sidecar(self) -> Path:
        return self.derived / "islrtc_trim.jsonl"

    @property
    def include_groups(self) -> Path:
        return self.splits / "include_groups.json"

    @property
    def islrtc_index(self) -> Path:
        return self.manifests / "islrtc_index.json"

    def resolve(self, p: str | Path) -> Path:
        """Map a manifest ``path`` onto this root.

        ``data/isl/landmarks/x.npz`` (repo-relative, as the ingestion scripts write it) becomes
        ``<root>/landmarks/x.npz``; absolute paths are kept; other relative paths are taken relative to the root.
        """
        p = Path(p)
        if p.is_absolute():
            return p
        parts = p.parts
        if len(parts) >= 2 and parts[0] == "data" and parts[1] == "isl":
            return self.root.joinpath(*parts[2:])
        return self.root / p


def isl_paths(root: str | Path | None = None) -> IslPaths:
    if root is not None:
        return IslPaths(Path(root))
    env = os.environ.get("SLR_ISL_DIR")
    if env:
        return IslPaths(Path(env))
    from app.core.config import get_settings

    return IslPaths(get_settings().data_dir / "isl")
