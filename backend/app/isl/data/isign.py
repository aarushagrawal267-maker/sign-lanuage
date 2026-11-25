"""iSign poses ingestion (Exploration-Lab/iSign, gated, CC BY-NC-SA 4.0, research profile only).

The dataset card says the four ``iSign-poses_v1.1_part_a{a..d}`` files (3 x 48.3 GB + 25.3 GB) are one
**zip** split with ``split``: ``cat part_aa part_ab part_ac part_ad > iSign-poses_v1.1.zip``. Members are
``.pose`` files written with the ``pose-format`` library (MediaPipe Holistic layout).

Two ways to read it without ever having the 170 GB on disk:

* ``zip`` (default): :class:`~app.isl.data.remote.RangeFile` presents the parts as one seekable file; the zip
  central directory (at the end of part_ad) is read with a few range requests and each wanted member is
  fetched on its own. Resumable per member through ``data/isl/manifests/isign.jsonl``; only the selected
  sentence ids are transferred.
* ``stream``: the parts are piped one after another (:class:`~app.isl.data.remote.ConcatStream`) and the
  archive is walked sequentially (zip local headers, or ``tarfile`` ``r|*`` for tar / tar.gz). For an
  uncompressed archive each member's offset is recorded in ``isign_offsets.jsonl`` so a restart resumes
  there with a Range request; a compressed stream must be re-read from the start (``indexed_gzip`` seek
  points would remove that, see docs/DATASETS_ISL.md).

Pose-format -> ``RawSequence`` mapping (``pose_to_raw``):

=====================  ==========================  ============================================
pose-format component   points                      RawSequence field
=====================  ==========================  ============================================
POSE_LANDMARKS          33, XYZC (pixels)           ``pose`` (x/W, y/H, z/W, confidence as visibility)
LEFT_HAND_LANDMARKS     21, XYZC                    ``left`` (x/W, y/H, z/W); confidence 0 -> NaN frame
RIGHT_HAND_LANDMARKS    21, XYZC                    ``right``
FACE_LANDMARKS          468 (or 478), XYZC          ``face`` = points ``FACE_IDX`` (x/W, y/H, z/W)
POSE_WORLD_LANDMARKS    33 (optional)               ignored
=====================  ==========================  ============================================

``extractor_version = "isign-pose-format"``. Open question (checked once data is available): the older
MediaPipe Holistic used to create iSign may label hands from the camera's point of view; the converter
can swap them (``swap_hands=True``) and records the choice.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import struct
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import PurePosixPath

import numpy as np

from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.isl.data.manifest import append_jsonl, iter_jsonl, read_manifest
from app.isl.paths import IslPaths
from app.landmarks.raw import RawSequence

log = logging.getLogger(__name__)

ISIGN_REPO = "Exploration-Lab/iSign"
POSE_PARTS = tuple(f"iSign-poses_v1.1_part_{s}" for s in ("aa", "ab", "ac", "ad"))
CSV_FILES = ("iSign_v1.1.csv", "word-presence-dataset_v1.1.csv", "word-description-dataset_v1.1.csv")
EXTRACTOR_VERSION = "isign-pose-format"

_U16 = struct.Struct("<H")
_F32 = struct.Struct("<f")
_U32 = struct.Struct("<I")


# --------------------------------------------------------------------------------------------- pose-format

class PoseFormatError(ValueError):
    pass


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = memoryview(buf)
        self.off = 0

    def unpack(self, st: struct.Struct):
        if self.off + st.size > len(self.buf):
            raise PoseFormatError("truncated pose file")
        v = st.unpack_from(self.buf, self.off)
        self.off += st.size
        return v if len(v) > 1 else v[0]

    def string(self) -> str:
        n = self.unpack(_U16)
        s = bytes(self.buf[self.off: self.off + n]).decode("utf-8")
        self.off += n
        return s

    def array(self, count: int, shape: tuple) -> np.ndarray:
        nbytes = 4 * count
        if self.off + nbytes > len(self.buf):
            raise PoseFormatError("truncated pose body")
        a = np.frombuffer(self.buf[self.off: self.off + nbytes], dtype="<f4").reshape(shape)
        self.off += nbytes
        return a

    def left(self) -> int:
        return len(self.buf) - self.off


def read_pose(buf: bytes) -> dict:
    """Minimal reader for pose-format files (versions 0.1 and 0.2, numpy only, no dependency).

    Returns {version, width, height, depth, components: [{name, format, points}], fps, data (F, P, N, D),
    confidence (F, P, N)}."""
    r = _Reader(buf)
    version = round(float(r.unpack(_F32)), 3)
    if version not in (0.1, 0.2):
        raise PoseFormatError(f"unsupported pose-format version {version}")
    width, height, depth = r.unpack(struct.Struct("<HHH"))
    n_comp = r.unpack(_U16)
    comps = []
    for _ in range(n_comp):
        name = r.string()
        fmt = r.string()
        n_pts, n_limbs, n_cols = r.unpack(struct.Struct("<HHH"))
        pts = [r.string() for _ in range(n_pts)]
        r.off += 4 * n_limbs + 6 * n_cols
        comps.append({"name": name, "format": fmt, "points": pts})
    dims = max(1, len(comps[0]["format"]) - 1) if comps else 3
    n_points = sum(len(c["points"]) for c in comps)
    if version == 0.1:
        fps, _frames = r.unpack(struct.Struct("<HH"))
        people = r.unpack(_U16)
    else:
        fps = r.unpack(_F32)
        _frames = r.unpack(_U32)
        people = r.unpack(_U16)
    per_frame = people * n_points * (dims + 1) * 4
    frames = r.left() // per_frame if per_frame else 0
    data = r.array(frames * people * n_points * dims, (frames, people, n_points, dims))
    conf = r.array(frames * people * n_points, (frames, people, n_points))
    return {"version": version, "width": int(width), "height": int(height), "depth": int(depth),
            "components": comps, "fps": float(fps), "data": data, "confidence": conf}


def write_pose(path_or_buf, components: list[tuple[str, int]], data: np.ndarray, confidence: np.ndarray,
               width: int, height: int, fps: float = 25.0, version: float = 0.2) -> bytes:
    """Write a minimal pose-format (v0.1 / v0.2) file: used by tests and for round trips. ``components`` =
    [(name, n_points)] with format XYZC; ``data`` (F, P, N, 3) in pixels, ``confidence`` (F, P, N)."""
    b = io.BytesIO()

    def s(x: str):
        e = x.encode("utf-8")
        b.write(_U16.pack(len(e)) + e)

    b.write(_F32.pack(version))
    b.write(struct.pack("<HHH", width, height, 0))
    b.write(_U16.pack(len(components)))
    for name, n in components:
        s(name)
        s("XYZC")
        b.write(struct.pack("<HHH", n, 0, 0))
        for i in range(n):
            s(str(i))
    F, P = data.shape[:2]
    if version == 0.1:
        b.write(struct.pack("<HH", int(fps), min(F, 65535)))
    else:
        b.write(_F32.pack(fps) + _U32.pack(F))
    b.write(_U16.pack(P))
    b.write(np.asarray(data, "<f4").tobytes())
    b.write(np.asarray(confidence, "<f4").tobytes())
    out = b.getvalue()
    if path_or_buf is not None and not isinstance(path_or_buf, io.BytesIO):
        with open(path_or_buf, "wb") as fh:
            fh.write(out)
    return out


def _component(p: dict, *names: str) -> tuple[int, int] | None:
    off = 0
    for c in p["components"]:
        if c["name"].upper() in names:
            return off, len(c["points"])
        off += len(c["points"])
    return None


def pose_to_raw(buf: bytes, person: int = 0, swap_hands: bool = False) -> RawSequence:
    """Convert a pose-format Holistic file to the app's ``RawSequence`` layout (see module docstring)."""
    p = read_pose(buf)
    W, H = max(1, p["width"]), max(1, p["height"])
    data, conf = p["data"], p["confidence"]
    F = data.shape[0]
    if F == 0:
        raise PoseFormatError("pose file has no frames")
    person = min(person, data.shape[1] - 1)
    d = data[:, person].astype(np.float64)
    c = conf[:, person].astype(np.float64)
    dims = d.shape[-1]

    def norm(block: np.ndarray) -> np.ndarray:
        out = np.full(block.shape[:2] + (3,), np.nan)
        out[..., 0] = block[..., 0] / W
        out[..., 1] = block[..., 1] / H
        out[..., 2] = block[..., 2] / W if dims >= 3 else 0.0
        return out

    def comp(names, n_expected):
        loc = _component(p, *names)
        if loc is None:
            return None, None
        off, n = loc
        if n < n_expected:
            raise PoseFormatError(f"{names[0]} has {n} points, expected >= {n_expected}")
        return norm(d[:, off: off + n]), c[:, off: off + n]

    pose_xyz, pose_c = comp(("POSE_LANDMARKS",), N_POSE)
    if pose_xyz is None:
        raise PoseFormatError("no POSE_LANDMARKS component")
    pose = np.full((F, N_POSE, 4), np.nan, np.float32)
    present = pose_c[:, :N_POSE].max(axis=1) > 0
    pose[present, :, :3] = pose_xyz[present, :N_POSE]
    pose[present, :, 3] = pose_c[present, :N_POSE]
    hands = []
    for names in (("LEFT_HAND_LANDMARKS",), ("RIGHT_HAND_LANDMARKS",)):
        xyz, cc = comp(names, N_HAND)
        h = np.full((F, N_HAND, 3), np.nan, np.float32)
        if xyz is not None:
            ok = cc[:, :N_HAND].max(axis=1) > 0
            h[ok] = xyz[ok, :N_HAND]
        hands.append(h)
    if swap_hands:
        hands = hands[::-1]
    face = np.full((F, N_FACE, 3), np.nan, np.float32)
    fxyz, fc = comp(("FACE_LANDMARKS",), max(FACE_IDX) + 1)
    if fxyz is not None:
        ok = fc.max(axis=1) > 0
        face[ok] = fxyz[ok][:, FACE_IDX]
    fps = p["fps"] if p["fps"] > 0 else 25.0
    t_ms = np.arange(F, dtype=np.float64) * 1000.0 / fps
    return RawSequence(pose, hands[0], hands[1], face, t_ms, W, H,
                       EXTRACTOR_VERSION + ("+swap" if swap_hands else ""))


# --------------------------------------------------------------------------------------------- subset selection

def _norm_cols(row: dict) -> dict:
    return {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k is not None}


def read_sentences(paths: IslPaths) -> dict[str, str]:
    """uid -> English text from ``iSign_v1.1.csv`` (column names matched loosely)."""
    f = paths.raw / "isign" / "iSign_v1.1.csv"
    if not f.exists():
        return {}
    out = {}
    with f.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            r = _norm_cols(row)
            uid = next((r[k] for k in ("uid", "id", "video_id", "video", "file", "filename") if r.get(k)), None)
            text = next((r[k] for k in ("text", "sentence", "english", "translation", "label") if r.get(k)), None)
            if uid and text:
                out[PurePosixPath(str(uid)).stem] = str(text)
    return out


_TOKEN = re.compile(r"[a-z0-9']+")


def select_subset(sentences: dict[str, str], vocab_words: Iterable[str], n: int = 10_000) -> list[str]:
    """Sentence ids containing at least one vocabulary word (single-word headwords), most vocabulary words
    first, ties broken by uid (deterministic). ``word-presence-dataset_v1.1.csv`` can refine this later."""
    vw = {w.lower() for w in vocab_words if w and " " not in w}
    scored = []
    for uid, text in sentences.items():
        toks = set(_TOKEN.findall(text.lower()))
        k = len(toks & vw)
        if k:
            scored.append((-k, uid))
    scored.sort()
    return [u for _, u in scored[:n]]


# --------------------------------------------------------------------------------------------- ingestion

def _save(paths: IslPaths, uid: str, buf: bytes, text: str | None, member: str, swap_hands: bool) -> dict:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", uid)[:150] + ".npz"
    out = paths.landmarks / "isign" / name
    meta = {"id": f"isign/{uid}", "source": "isign", "uid": uid, "text": text, "member": member,
            "path": f"data/isl/landmarks/isign/{name}"}
    try:
        raw = pose_to_raw(buf, swap_hands=swap_hands)
        raw.save(out)
        pr = raw.presence()
        meta.update({"ok": True, "n_frames": len(raw), "fps": round(1000.0 / float(np.median(np.diff(raw.t_ms))), 3)
                     if len(raw) > 1 else None, "hands_frac": float((pr["left"] | pr["right"]).mean()),
                     "extractor_version": raw.extractor_version, "width": raw.width, "height": raw.height})
    except Exception as e:  # noqa: BLE001
        meta.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
    return meta


def ingest_isign_zip(paths: IslPaths, archive, wanted: set[str] | None = None, sentences: dict[str, str] | None = None,
                     limit: int = 0, swap_hands: bool = False, progress: Callable[[str], None] | None = None) -> dict:
    """Random-access mode over the (split) zip. ``archive`` is a seekable file (``RangeFile`` / ``BytesIO``)."""
    man = paths.manifest("isign")
    done = {r["id"] for r in read_manifest(man)}
    (paths.landmarks / "isign").mkdir(parents=True, exist_ok=True)
    sentences = sentences or {}
    zf = zipfile.ZipFile(archive)
    infos = sorted((i for i in zf.infolist() if i.filename.endswith(".pose")), key=lambda i: i.header_offset)
    stats = {"members": len(infos), "wanted": 0, "skipped_done": 0, "processed": 0, "ok": 0, "failed": 0}
    for info in infos:
        uid = PurePosixPath(info.filename).stem
        if wanted is not None and uid not in wanted:
            continue
        stats["wanted"] += 1
        if f"isign/{uid}" in done:
            stats["skipped_done"] += 1
            continue
        if limit and stats["processed"] >= limit:
            break
        meta = _save(paths, uid, zf.read(info), sentences.get(uid), info.filename, swap_hands)
        append_jsonl(man, meta)
        stats["processed"] += 1
        stats["ok" if meta.get("ok") else "failed"] += 1
        if progress:
            progress(f"{meta['id']} ok={meta.get('ok')}")
    return stats


def iter_zip_stream(stream) -> Iterator[tuple[str, int, bytes | None]]:
    """Sequential zip reader over a non-seekable stream (local file headers). Yields (name, header offset,
    data or None when compressed with an unsupported method / unknown size)."""
    import zlib

    pos = 0
    sig = struct.Struct("<IHHHHHIIIHH")
    while True:
        head = stream.read(30)
        if len(head) < 30:
            return
        (magic, _ver, flag, method, _t, _d, _crc, csize, usize, nlen, xlen) = sig.unpack(head)
        if magic != 0x04034B50:
            return  # central directory reached
        name = stream.read(nlen).decode("utf-8", "replace")
        extra = stream.read(xlen)
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:  # zip64 sizes in the extra field
            i = 0
            while i + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, i)
                if hid == 0x0001:
                    vals = struct.unpack_from("<" + "Q" * (hsz // 8), extra, i + 4)
                    if usize == 0xFFFFFFFF and vals:
                        usize, vals = vals[0], vals[1:]
                    if csize == 0xFFFFFFFF and vals:
                        csize = vals[0]
                    break
                i += 4 + hsz
        if flag & 0x08 and csize == 0:
            raise OSError("zip uses data descriptors without sizes; use the random-access (zip) mode")
        raw = stream.read(csize)
        data = raw if method == 0 else (zlib.decompress(raw, -15) if method == 8 else None)
        yield name, pos, data
        pos += 30 + nlen + xlen + csize


def ingest_isign_stream(paths: IslPaths, stream, fmt: str, wanted: set[str] | None = None,
                        sentences: dict[str, str] | None = None, limit: int = 0, swap_hands: bool = False,
                        start_offset: int = 0, progress: Callable[[str], None] | None = None) -> dict:
    """Sequential mode over a piped stream (``ConcatStream``): zip local headers or tar(.gz) via ``tarfile``."""
    man = paths.manifest("isign")
    offs = paths.manifests / "isign_offsets.jsonl"
    done = {r["id"] for r in read_manifest(man)}
    (paths.landmarks / "isign").mkdir(parents=True, exist_ok=True)
    sentences = sentences or {}
    stats = {"seen": 0, "wanted": 0, "skipped_done": 0, "processed": 0, "ok": 0, "failed": 0}
    if fmt == "zip":
        it = ((n, start_offset + o, d) for n, o, d in iter_zip_stream(stream))
    else:
        tf = tarfile.open(fileobj=stream, mode="r|*")

        def tar_iter():
            for m in tf:
                if not m.isfile():
                    continue
                f = tf.extractfile(m)
                yield m.name, start_offset + m.offset, (f.read() if f is not None else None)

        it = tar_iter()
    for name, off, data in it:
        stats["seen"] += 1
        if not name.endswith(".pose"):
            continue
        uid = PurePosixPath(name).stem
        if fmt in ("tar", "zip"):
            append_jsonl(offs, {"member": name, "offset": off})
        if wanted is not None and uid not in wanted:
            continue
        stats["wanted"] += 1
        if f"isign/{uid}" in done:
            stats["skipped_done"] += 1
            continue
        if limit and stats["processed"] >= limit:
            break
        if data is None:
            meta = {"id": f"isign/{uid}", "source": "isign", "ok": False, "error": "unsupported compression"}
        else:
            meta = _save(paths, uid, data, sentences.get(uid), name, swap_hands)
        append_jsonl(man, meta)
        stats["processed"] += 1
        stats["ok" if meta.get("ok") else "failed"] += 1
        if progress:
            progress(f"{meta['id']} ok={meta.get('ok')}")
    return stats


def last_offset(paths: IslPaths) -> int:
    """Resume point for an uncompressed stream: offset of the last recorded member header."""
    last = 0
    for r in iter_jsonl(paths.manifests / "isign_offsets.jsonl"):
        last = max(last, int(r.get("offset", 0)))
    return last
