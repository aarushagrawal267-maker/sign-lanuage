"""Official WLASL annotation handling.

WLASL100 is defined by the authors as the first 100 glosses of
``WLASL_v0.3.json``; the authors' ``nslt_100.json`` lists the 2,038 video ids of
that subset with their split and class index. We fetch both files unmodified,
record their checksums, and cross-check them before ingesting anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import VOCAB_SIZE, WLASL_ANNOTATION_URL, WLASL_NSLT100_URL, get_settings
from app.db.models import Gloss, MetaKV, Video

ANNOTATION_FILE = "WLASL_v0.3.json"
NSLT100_FILE = "nslt_100.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_annotations(force: bool = False, timeout: int = 60) -> dict:
    """Download the two official annotation files into ``data/annotations``."""
    s = get_settings()
    out = {}
    for name, url in ((ANNOTATION_FILE, WLASL_ANNOTATION_URL), (NSLT100_FILE, WLASL_NSLT100_URL)):
        dest = s.annotations_dir / name
        if force or not dest.exists():
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            json.loads(r.content)  # must parse before we accept it
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
        out[name] = {"url": url, "path": str(dest), "sha256": sha256_file(dest)}
    return out


@dataclass
class Wlasl100:
    glosses: list[str]
    instances: list[dict]  # each has 'gloss' and 'class_index' added
    checks: dict = field(default_factory=dict)


class AnnotationError(ValueError):
    pass


def parse_wlasl100(annotation_path: Path, nslt_path: Path | None = None) -> Wlasl100:
    """Select WLASL100 from the full annotation file and verify it against nslt_100."""
    full = json.loads(Path(annotation_path).read_text())
    if len(full) < VOCAB_SIZE:
        raise AnnotationError(f"annotation file has only {len(full)} glosses")
    top = full[:VOCAB_SIZE]
    glosses = [g["gloss"] for g in top]
    if len(set(glosses)) != VOCAB_SIZE:
        raise AnnotationError("duplicate gloss names in WLASL100 selection")

    instances: list[dict] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for ci, g in enumerate(top):
        for inst in g["instances"]:
            vid = str(inst["video_id"])
            if vid in seen:
                duplicates.append(vid)
                continue
            seen.add(vid)
            instances.append({**inst, "video_id": vid, "gloss": g["gloss"], "class_index": ci})

    checks: dict = {"n_glosses": len(glosses), "n_instances": len(instances), "duplicates_skipped": duplicates}
    if nslt_path is not None and Path(nslt_path).exists():
        nslt = json.loads(Path(nslt_path).read_text())
        by_id = {i["video_id"]: i for i in instances}
        missing_in_json = sorted(set(nslt) - set(by_id))
        missing_in_nslt = sorted(set(by_id) - set(nslt))
        split_mismatch = [k for k, v in nslt.items() if k in by_id and by_id[k]["split"] != v["subset"]]
        label_mismatch = [k for k, v in nslt.items() if k in by_id and by_id[k]["class_index"] != v["action"][0]]
        checks.update(
            nslt_entries=len(nslt),
            only_in_nslt=missing_in_json,
            only_in_annotation=missing_in_nslt,
            split_mismatch=split_mismatch,
            label_mismatch=label_mismatch,
        )
        checks["consistent"] = not (missing_in_json or missing_in_nslt or split_mismatch or label_mismatch)
    return Wlasl100(glosses=glosses, instances=instances, checks=checks)


def ingest(db: Session, data: Wlasl100, provenance: dict | None = None) -> dict:
    """Insert or update glosses and videos. Existing acquisition state is preserved."""
    for ci, g in enumerate(data.glosses):
        row = db.get(Gloss, ci)
        if row is None:
            db.add(Gloss(class_index=ci, gloss=g))
        elif row.gloss != g:
            raise AnnotationError(f"class {ci} changed from {row.gloss!r} to {g!r}; refusing to relabel")
    db.flush()

    existing = {v.video_id: v for v in db.scalars(select(Video))}
    added = updated = 0
    for inst in data.instances:
        fields = dict(
            class_index=inst["class_index"],
            gloss=inst["gloss"],
            signer_id=int(inst["signer_id"]),
            split=inst["split"],
            source=inst.get("source", ""),
            url=inst["url"],
            annotated_fps=int(inst.get("fps", 25)),
            frame_start=int(inst.get("frame_start", 1)),
            frame_end=int(inst.get("frame_end", -1)),
            bbox=inst.get("bbox"),
            variation_id=int(inst.get("variation_id", 0)),
            instance_id=int(inst.get("instance_id", 0)),
        )
        row = existing.get(inst["video_id"])
        if row is None:
            db.add(Video(video_id=inst["video_id"], **fields))
            added += 1
        else:
            for k, v in fields.items():
                setattr(row, k, v)
            updated += 1

    meta = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "checks": {k: (v if not isinstance(v, list) else len(v)) for k, v in data.checks.items()},
        "files": provenance or {},
    }
    kv = db.get(MetaKV, "annotations")
    if kv is None:
        db.add(MetaKV(key="annotations", value=meta))
    else:
        kv.value = meta
    return {"added": added, "updated": updated, **meta}


def dataset_version(db: Session) -> str | None:
    kv = db.get(MetaKV, "annotations")
    if not kv:
        return None
    files = kv.value.get("files", {})
    sha = files.get(ANNOTATION_FILE, {}).get("sha256")
    return f"WLASL_v0.3/WLASL100@{sha[:12]}" if sha else "WLASL100 (unverified annotation file)"


def load_and_ingest(db: Session, fetch: bool = True, force: bool = False) -> dict:
    s = get_settings()
    prov = fetch_annotations(force=force) if fetch else {}
    if not prov:
        for name in (ANNOTATION_FILE, NSLT100_FILE):
            p = s.annotations_dir / name
            if p.exists():
                prov[name] = {"path": str(p), "sha256": sha256_file(p)}
    ann = s.annotations_dir / ANNOTATION_FILE
    if not ann.exists():
        raise AnnotationError(f"{ann} not found; fetch the official annotations first")
    data = parse_wlasl100(ann, s.annotations_dir / NSLT100_FILE)
    if data.checks.get("consistent") is False:
        raise AnnotationError(f"WLASL100 selection disagrees with nslt_100.json: {data.checks}")
    return ingest(db, data, prov)
