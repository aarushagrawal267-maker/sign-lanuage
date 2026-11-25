"""Language-pack registry (docs/ISL_DESIGN.md sections 3.1 ADR-ISL-1 and 4.2).

A *language pack* is a folder with ``pack.json`` plus its artifacts. Roots are searched in order and a later
root overrides an earlier pack with the same id::

    models/packs/            tracked, shipped with the app (small files only)
    <artifacts>/packs/       local builds (git-ignored), e.g. the output of ``python -m app.isl.cli build-pack``
    $SLR_PACKS_DIRS          extra roots (os.pathsep-separated)

ASL is wrapped as the ``experiment-registry`` pack ``asl-wlasl100``: its recogniser is the active experiment
of the Model Laboratory, so the ASL code paths are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.core.config import REPO_ROOT, get_settings
from app.packs.schema import PackManifest

log = logging.getLogger(__name__)

STATUSES = ("ready", "missing_models", "error")


@dataclass(frozen=True)
class PackInfo:
    id: str
    kind: str
    sign_language: str
    standard: str
    display_name: dict
    output_languages: list
    capabilities: frozenset
    profile: str
    version: str | None
    licences: list
    status: str
    message: str | None
    folder: str = ""
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"id": self.id, "kind": self.kind, "sign_language": self.sign_language, "standard": self.standard,
                "display_name": dict(self.display_name), "output_languages": list(self.output_languages),
                "capabilities": sorted(self.capabilities), "profile": self.profile, "version": self.version,
                "licences": list(self.licences), "status": self.status, "message": self.message, **self.extra}


class LanguagePack(Protocol):
    manifest: PackManifest
    folder: Path

    @property
    def info(self) -> PackInfo: ...

    def recognizer(self) -> tuple[object | None, dict | None, str | None]:
        """(recognizer, model_info, problem): mirrors ``inference.registry.load_active()``."""

    def vocabulary(self): ...


class UnknownLanguage(KeyError):
    def __init__(self, pack_id: str, available: list[str]):
        super().__init__(pack_id)
        self.pack_id, self.available = pack_id, available

    def __str__(self) -> str:
        return f"unknown language {self.pack_id!r}; available: {', '.join(self.available)}"


class BrokenPack:
    """A pack folder whose ``pack.json`` cannot be read: listed with status ``error``."""

    def __init__(self, folder: Path, pack_id: str, message: str):
        self.folder, self.pack_id, self.message = folder, pack_id, message
        self.manifest = None

    @property
    def info(self) -> PackInfo:
        return PackInfo(self.pack_id, "unknown", "?", "", {"en": self.pack_id}, [], frozenset(), "", None, [], "error",
                        self.message, str(self.folder))

    def recognizer(self):
        return None, None, self.message

    def vocabulary(self):
        from app.packs.vocabulary import Vocabulary

        return Vocabulary([])


def build_pack(folder: Path):
    p = folder / "pack.json"
    try:
        man = PackManifest.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as e:
        return BrokenPack(folder, folder.name, f"invalid pack.json in {folder}: {str(e)[:300]}")
    if man.kind == "experiment-registry":
        from app.packs.asl import AslExperimentPack

        return AslExperimentPack(man, folder)
    if man.kind == "isl-hybrid":
        from app.packs.isl import IslPack

        return IslPack(man, folder)
    return BrokenPack(folder, man.id, f"unknown pack kind {man.kind!r}")


class PackRegistry:
    def __init__(self, roots: list[Path], default_id: str = "asl-wlasl100"):
        self.roots = [Path(r) for r in roots]
        self._default = default_id
        self._lock = threading.Lock()
        self._packs: dict[str, object] = {}
        self.reload()

    def reload(self) -> None:
        packs: dict[str, object] = {}
        for root in self.roots:
            if not root.is_dir():
                continue
            for folder in sorted(root.iterdir()):
                if (folder / "pack.json").is_file():
                    pk = build_pack(folder)
                    pid = pk.manifest.id if getattr(pk, "manifest", None) is not None else pk.pack_id
                    packs[pid] = pk  # later roots override
        with self._lock:
            self._packs = packs

    def ids(self) -> list[str]:
        return list(self._packs)

    def list(self) -> list[PackInfo]:
        return [p.info for p in self._packs.values()]

    def default_id(self) -> str:
        return self._default if self._default in self._packs else (next(iter(self._packs), self._default))

    def get(self, pack_id: str | None = None):
        pid = pack_id or self.default_id()
        try:
            return self._packs[pid]
        except KeyError:
            raise UnknownLanguage(pid, self.ids()) from None


def pack_roots() -> list[Path]:
    s = get_settings()
    roots = [REPO_ROOT / "models" / "packs", s.artifacts_dir / "packs"]
    roots += [Path(p) for p in (s.packs_dirs or "").split(os.pathsep) if p.strip()]
    return roots


_registries: dict[tuple, PackRegistry] = {}
_reg_lock = threading.Lock()


def get_registry() -> PackRegistry:
    """One registry per (roots, default) - tests that switch data directories get a fresh one."""
    roots = pack_roots()
    key = (tuple(str(r) for r in roots), get_settings().default_language)
    with _reg_lock:
        reg = _registries.get(key)
        if reg is None:
            reg = _registries[key] = PackRegistry(roots, key[1])
        return reg


def reset_registries() -> None:
    with _reg_lock:
        _registries.clear()
