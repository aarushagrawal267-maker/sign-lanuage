"""Sign packs: export and import of "My signs" as a ``.signpack`` zip (docs/ISL_DESIGN.md section 7.4).

Layout::

    manifest.json   {"format": "sign-pack", "version": 1, "pack", "title", "region", "author", "licence",
                     "created", "n_signs", "encoder_version", "app_version", "include_face"}
    signs.jsonl     MySign rows (no local paths)
    takes/<sign_id>/<n>.npz
    embeddings.npy  optional; ignored on import (takes are re-embedded with the local encoder)

Import is defensive: strict member names (no path traversal), size and count limits, ``allow_pickle=False``,
every array checked against the ``RawSequence`` shapes with finite-or-missing values, text fields cleaned.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

import numpy as np

from app.features.spec import N_FACE, N_HAND, N_POSE
from app.isl.mysigns.store import SIGN_ID_RE, TEXT_LIMITS, MySign, MySignsStore, clean_text, now_iso
from app.landmarks.raw import RawSequence

FORMAT = "sign-pack"
VERSION = 1
MAX_BYTES = 50 * 1024 * 1024
MAX_SIGNS = 2000
MAX_TAKES = 5
MAX_FRAMES = 300
MAX_UNCOMPRESSED = 400 * 1024 * 1024
MEMBER_RE = re.compile(r"^(manifest\.json|signs\.jsonl|embeddings\.npy|takes/[0-9a-f-]{36}/[0-9]\.npz)$")
CONSENT_TEXT = ("This file contains body and hand movement points of the people who recorded these signs (no video). "
                "Share it only with their permission.")
LICENCES = ("CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "CC-BY-NC-4.0", "all-rights-reserved")


class PackError(ValueError):
    pass


def _raw_bytes(raw: RawSequence, include_face: bool) -> bytes:
    face = raw.face if include_face else np.full_like(raw.face, np.nan)
    buf = io.BytesIO()
    np.savez_compressed(buf, pose=raw.pose.astype(np.float32), left=raw.left.astype(np.float32),
                        right=raw.right.astype(np.float32), face=face.astype(np.float32),
                        t_ms=raw.t_ms.astype(np.float64), width=raw.width, height=raw.height,
                        extractor_version=raw.extractor_version)
    return buf.getvalue()


def export_pack(store: MySignsStore, ids: list[str] | None = None, *, title: str = "My signs", region: str | None = None,
                author: str | None = None, licence: str = "CC-BY-4.0", include_face: bool = False,
                encoder_version: str | None = None, app_version: str = "") -> bytes:
    if licence not in LICENCES:
        raise PackError(f"licence must be one of {LICENCES}")
    signs = [s for s in store.list() if ids is None or s.sign_id in set(ids)]
    if not signs:
        raise PackError("no signs selected")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        manifest = {"format": FORMAT, "version": VERSION, "pack": store.pack_id,
                    "title": clean_text(title, 120) or "My signs", "region": clean_text(region, 80),
                    "author": clean_text(author, 80), "licence": licence, "created": now_iso(),
                    "n_signs": len(signs), "encoder_version": encoder_version, "app_version": app_version,
                    "include_face": include_face, "consent": CONSENT_TEXT}
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
        lines = []
        for s in signs:
            row = s.to_json()
            row["licence"] = licence
            lines.append(json.dumps(row, ensure_ascii=False))
            for i, raw in enumerate(store.takes(s.sign_id)):
                z.writestr(f"takes/{s.sign_id}/{i}.npz", _raw_bytes(raw, include_face))
        z.writestr("signs.jsonl", "\n".join(lines) + "\n")
    return out.getvalue()


def _check_raw(data: bytes, name: str) -> RawSequence:
    try:
        npz = np.load(io.BytesIO(data), allow_pickle=False)
        arrs = {k: npz[k] for k in ("pose", "left", "right", "face", "t_ms", "width", "height")}
        ev = str(npz["extractor_version"]) if "extractor_version" in npz.files else "holistic-v1"
    except Exception as e:  # noqa: BLE001 - any malformed npz is rejected with a clear message
        raise PackError(f"{name}: not a valid landmark file ({type(e).__name__})") from None
    T = arrs["pose"].shape[0] if arrs["pose"].ndim == 3 else -1
    shapes = {"pose": (T, N_POSE, 4), "left": (T, N_HAND, 3), "right": (T, N_HAND, 3), "face": (T, N_FACE, 3), "t_ms": (T,)}
    if not (1 <= T <= MAX_FRAMES):
        raise PackError(f"{name}: {T} frames (1 to {MAX_FRAMES} allowed)")
    for k, shp in shapes.items():
        a = arrs[k]
        if a.shape != shp or a.dtype.kind != "f":
            raise PackError(f"{name}: '{k}' has shape {a.shape}, expected {shp}")
        if np.isinf(a).any() or (k == "t_ms" and np.isnan(a).any()):
            raise PackError(f"{name}: '{k}' contains invalid numbers")
    w, h = int(arrs["width"]), int(arrs["height"])
    if not (0 < w <= 10000 and 0 < h <= 10000):
        raise PackError(f"{name}: invalid frame size {w}x{h}")
    return RawSequence(arrs["pose"].astype(np.float32), arrs["left"].astype(np.float32),
                       arrs["right"].astype(np.float32), arrs["face"].astype(np.float32),
                       arrs["t_ms"].astype(np.float64), w, h, clean_text(ev, 32) or "holistic-v1")


def read_pack(data: bytes) -> tuple[dict, list[tuple[MySign, list[RawSequence]]]]:
    """Validate a pack and return (manifest, [(sign, takes)]). Raises ``PackError``."""
    if len(data) > MAX_BYTES:
        raise PackError("sign pack larger than 50 MB")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise PackError("not a sign pack (zip) file") from None
    infos = z.infolist()
    names = [i.filename for i in infos]
    bad = [n for n in names if not MEMBER_RE.match(n)]
    if bad:
        raise PackError(f"unexpected file in sign pack: {bad[0]!r}")
    if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
        raise PackError("sign pack expands to more than 400 MB")
    if "manifest.json" not in names or "signs.jsonl" not in names:
        raise PackError("sign pack is missing manifest.json or signs.jsonl")
    try:
        manifest = json.loads(z.read("manifest.json"))
    except ValueError:
        raise PackError("manifest.json is not valid JSON") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
        raise PackError("not a sign pack of a supported version")
    out = []
    seen = set()
    for ln in z.read("signs.jsonl").decode("utf-8", "replace").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            raise PackError("signs.jsonl contains invalid JSON") from None
        if not isinstance(row, dict):
            raise PackError("signs.jsonl rows must be objects")
        sid = str(row.get("sign_id", ""))
        if not SIGN_ID_RE.match(sid) or sid in seen:
            raise PackError(f"invalid or repeated sign id {sid[:40]!r}")
        seen.add(sid)
        n = int(row.get("n_takes") or 0)
        if not (1 <= n <= MAX_TAKES):
            raise PackError(f"sign {sid}: {n} takes (1 to {MAX_TAKES} allowed)")
        takes = []
        for i in range(n):
            mn = f"takes/{sid}/{i}.npz"
            if mn not in names:
                raise PackError(f"missing {mn}")
            takes.append(_check_raw(z.read(mn), mn))
        cid = row.get("concept_id")
        sign = MySign(sign_id=sid, pack=str(manifest.get("pack") or ""),
                      concept_id=clean_text(cid, 120) if isinstance(cid, str) else None,
                      text_en=clean_text(row.get("text_en"), TEXT_LIMITS["text_en"]) or "",
                      text_hi=clean_text(row.get("text_hi"), TEXT_LIMITS["text_hi"]),
                      region=clean_text(row.get("region") or manifest.get("region"), TEXT_LIMITS["region"]),
                      note=clean_text(row.get("note"), TEXT_LIMITS["note"]),
                      author=clean_text(row.get("author") or manifest.get("author"), TEXT_LIMITS["author"]),
                      created=clean_text(row.get("created"), 32) or now_iso(),
                      theta=float(row["theta"]) if isinstance(row.get("theta"), (int, float)) and np.isfinite(row["theta"]) else None,
                      licence=clean_text(manifest.get("licence"), 40))
        if not sign.text_en:
            raise PackError(f"sign {sid}: empty English text")
        out.append((sign, takes))
        if len(out) > MAX_SIGNS:
            raise PackError(f"more than {MAX_SIGNS} signs")
    if "embeddings.npy" in names:  # validated, never trusted (re-embedded locally)
        try:
            np.load(io.BytesIO(z.read("embeddings.npy")), allow_pickle=False)
        except Exception:  # noqa: BLE001
            raise PackError("embeddings.npy is not a valid array file") from None
    return manifest, out


def import_pack(store: MySignsStore, data: bytes, mode: str = "merge", encoder_version: str | None = None) -> dict:
    if mode not in ("merge", "replace"):
        raise PackError("mode must be 'merge' or 'replace'")
    manifest, signs = read_pack(data)
    if manifest.get("pack") and manifest["pack"] != store.pack_id:
        raise PackError(f"this sign pack is for '{manifest['pack']}', not '{store.pack_id}'")
    if mode == "replace":
        store.delete_all()
    existing = {s.sign_id for s in store.list()}
    title = clean_text(manifest.get("title"), 120) or "sign pack"
    imported, skipped = 0, 0
    for sign, takes in signs:
        if sign.sign_id in existing:
            skipped += 1
            continue
        sign.pack = store.pack_id
        sign.origin = f"imported:{title}"
        sign.enabled = True
        store.save_sign(sign, takes)
        imported += 1
    return {"imported": imported, "skipped": skipped, "title": title, "licence": manifest.get("licence"),
            "reembedded": manifest.get("encoder_version") != encoder_version}
