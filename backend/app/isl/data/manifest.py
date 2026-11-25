"""Tolerant JSON-lines manifest reading.

Manifests are appended to by long-running ingestion jobs while they are read, so a reader must:
ignore a partial (not yet newline-terminated or invalid) last line, keep the **last** record per ``id``
(a retried failure is followed by a success), and never lock or modify the file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield every complete, valid JSON object line. A last line without a trailing newline is skipped
    (it may still be being written)."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break  # partial last line
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def read_manifest(path: Path, only_ok: bool = True) -> list[dict]:
    """Last record per ``id`` in file order of first appearance; with ``only_ok`` keep ``ok: true`` records."""
    last: dict[str, dict] = {}
    for rec in iter_jsonl(path):
        rid = rec.get("id")
        if not rid:
            continue
        last[rid] = rec  # dict keeps first-insertion order, value = latest record
    out = list(last.values())
    if only_ok:
        out = [r for r in out if r.get("ok")]
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
