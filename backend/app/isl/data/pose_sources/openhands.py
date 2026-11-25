"""OpenHands v1 raw ISL pose corpus (Zenodo 5558897, AI4Bharat): about 1,129 h of unlabelled MediaPipe Holistic
poses from five Indian deaf-community YouTube channels. Zenodo says CC BY 4.0, but the poses derive from
third-party YouTube videos, so it stays in the **research profile, pre-training only** until the owner decides.

Archive layout (inspected 25 Nov 2026): every zip holds one or two **huge deflate-compressed HDF5 files**
(``Sign-Library/sign_library_1_min_splits.h5`` 1.46 GB, ``sign_library_sub_based.h5`` 6.1 GB, ``NIOS/
corrected_pose_for_splitted.h5`` 30.2 GB, ...). Inside: groups ``keypoints`` and ``visibility`` with one
dataset per video chunk (``<youtube id><title>-<n>``), ``keypoints/<name>`` (T, 75, 3) float64 and
``visibility/<name>`` (T, 75) float64, stored **contiguous, unfiltered, in file order** (keypoints of a chunk,
then its visibility, then the next chunk). The 75 points are MediaPipe pose 0-32 (x, y normalised, raw z),
left hand 33-53, right hand 54-74 (the left hand's wrist sits on pose point 15, the signer's left wrist, as in
our extractor); missing = all zeros; hand visibility is 0/1. There is no face and no fps / frame size: OpenHands
assumes 25 fps (its ``WindowedDatasetHDF5`` sampler), and the aspect ratio is estimated per chunk from hand-bone
rigidity (:func:`~app.isl.data.pose_sources.common.hand_aspect`) and snapped to a common ratio.

Streaming without the archive on disk (:class:`SeekableInflate`): the zip member is inflated from HTTP range
requests (:class:`PrefetchReader`, parallel read-ahead). While inflating, a ``zlib`` decompressor snapshot
(``decompressobj.copy()``) is kept every ``every`` output bytes, so a backward seek restarts from the nearest
snapshot instead of the member start. ``h5py`` opens that file object to list the datasets and their object
header addresses (pass 1, metadata only, but the B-tree nodes are spread over the whole file, so it streams
through it once; 4 KB pages that look like HDF5 metadata are kept in memory while inflating, so h5py's scattered
reads rarely need a restart, see :func:`looks_like_metadata`); pass 2 visits the headers in address order, parses each small version-1 object header
itself (:func:`parse_v1_header`: shape, dtype, contiguous data address) and reads the data sequentially.
Network cost: about twice the compressed size, no temp files.

Restore points (25 Nov 2026): the snapshots are now ``zran`` access points on deflate block boundaries (bit
offset + last 32 KB of output, :mod:`~app.isl.data.pose_sources.zran`), and every 64 MB of output one is
appended to ``raw/openhands_raw/<member>.points``. A restarted job loads them and seeks straight to the point
before its next chunk (measured on ISH News: a chunk at 4.7 GB read with 27 MB of downloads instead of
re-streaming the member from byte 0).

One forward pass per member (26 Nov 2026, :class:`ForwardPass`): the object headers of a group of chunks sit
together and their data comes much later (ISH News ``sub_based``: headers at 1.81 GB, data at 3.65 GB), so the
per-chunk reads (header, data, next header ...) restored 3.5 times and re-inflated about 18 MB per 0.64 MB chunk
(measured on 25 real chunks: 9.3 MB of compressed input consumed and 180 MB downloaded per chunk with the
12-block read-ahead). The forward pass cuts every wanted range out of the stream as it goes by and keeps the parsed
headers in ``raw/openhands_raw/<member>.headers.jsonl`` so a restart resumes near where it stopped.
"""

from __future__ import annotations

import bisect
import io
import logging
import math
import re
import struct
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.features.spec import N_FACE, N_HAND, N_POSE
from app.isl.data.pose_sources.common import PoseClip
from app.isl.data.sources import MIN_FRAMES, MIN_HANDS_FRAC, _Base
from app.landmarks.raw import RawSequence

log = logging.getLogger(__name__)

RECORD = "https://zenodo.org/api/records/5558897"
URL = "https://zenodo.org/records/5558897"
LICENCE = "CC-BY-4.0 (Zenodo label; poses derived from third-party YouTube videos, pending owner decision)"
EXTRACTOR_VERSION = "openhands-raw-mp-holistic"
ASSUMED_FPS = 25.0
COMMON_ASPECTS = (9 / 16, 3 / 4, 1.0, 4 / 3, 16 / 9)
CITATION = ("Selvaraj et al., OpenHands: Making Sign Language Recognition Accessible with Pose-based Pretrained "
            "Models across Languages, ACL 2022 (aclanthology.org/2022.acl-long.150)")


# --------------------------------------------------------------------------------------------- HTTP

def range_fetch(url: str, start: int, end: int, retries: int = 10, timeout: float = 30.0, session=None,
                min_rate: float = 250e3, grace: float = 15.0) -> bytes:
    """Bytes [start, end] inclusive. Never downloads a whole file when a server ignores ``Range``. A transfer
    slower than ``min_rate`` bytes/s on average (after ``grace`` seconds) is abandoned and retried: Zenodo
    connections (about 0.8 MB/s each) sometimes stall without closing."""
    import requests
    from urllib3.exceptions import HTTPError as Urllib3Error

    s = session or _session()
    err = None
    want = end - start + 1
    for i in range(retries):
        try:
            with s.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(30.0, timeout), stream=True,
                       allow_redirects=True) as r:
                if r.status_code == 206:
                    deadline = time.time() + grace + want / min_rate
                    buf = bytearray()
                    for chunk in r.raw.stream(1 << 20, decode_content=False):
                        buf += chunk
                        if len(buf) > want:
                            break
                        if time.time() > deadline:
                            raise TimeoutError(f"stalled transfer ({len(buf)} of {want} bytes)")
                    data = bytes(buf)
                    if len(data) == want:
                        return data
                    err = f"short read {len(data)} of {want}"
                elif r.status_code == 200:
                    raise OSError("server ignored the Range header (refusing to download the whole file)")
                else:
                    err = f"HTTP {r.status_code}"
                    if r.status_code in (401, 403, 404, 416):
                        raise PermissionError(f"{err} for {url}")
                    if r.status_code == 429:
                        time.sleep(min(300, float(r.headers.get("Retry-After") or 30)))
        except (PermissionError, OSError) as e:
            if isinstance(e, PermissionError) or "ignored the Range" in str(e):
                raise
            err = f"{type(e).__name__}: {e}"
        except (requests.RequestException, Urllib3Error) as e:
            # urllib3 errors raised while streaming ``r.raw`` (IncompleteRead -> ProtocolError, read timeouts) are
            # not wrapped by requests: uncaught, one broke connection killed the job (25 Nov 2026 21:01 UTC)
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** i))
    raise OSError(f"range fetch failed after {retries} tries ({url} {start}-{end}): {err}")


_local = threading.local()


def _session():
    s = getattr(_local, "s", None)
    if s is None:
        import requests

        s = requests.Session()
        s.headers["User-Agent"] = "isl-pose-ingest/1.0"
        _local.s = s
    return s


class PrefetchReader:
    """Random-access reads over one remote file in aligned blocks, with parallel read-ahead when the access
    pattern is sequential. ``fetch(start, end_inclusive) -> bytes`` is injectable (tests use a local buffer)."""

    def __init__(self, fetch: Callable[[int, int], bytes], size: int, block: int = 8 << 20, ahead: int = 3,
                 keep: int = 6):
        self.fetch, self.size, self.block, self.ahead, self.keep = fetch, int(size), int(block), int(ahead), int(keep)
        self.pool = ThreadPoolExecutor(max(1, ahead)) if ahead > 0 else None
        self.futs: OrderedDict[int, object] = OrderedDict()
        self.last = -10
        self.fetched_bytes = 0  # bytes handed to callers
        self.downloaded_bytes = 0  # bytes actually downloaded (whole blocks, read-ahead included)
        self.lock = threading.Lock()
        self._dl_lock = threading.Lock()

    def _fetch_counted(self, s: int, e: int) -> bytes:
        b = self.fetch(s, e)
        with self._dl_lock:
            self.downloaded_bytes += len(b)
        return b

    def _submit(self, k: int):
        s = k * self.block
        e = min(self.size, s + self.block) - 1
        if self.pool is None:
            from concurrent.futures import Future

            f: Future = Future()
            f.set_result(self._fetch_counted(s, e))
            return f
        return self.pool.submit(self._fetch_counted, s, e)

    def _get(self, k: int) -> bytes:
        with self.lock:
            sequential = k in (self.last, self.last + 1)
            f = self.futs.get(k)
            if f is None or (f.done() and (f.cancelled() or f.exception() is not None)):
                # a block whose download failed (range_fetch gave up) is fetched again on the next read: kept,
                # it re-raised the same error forever, so a caller's retry could never succeed
                self.futs[k] = self._submit(k)
            self.futs.move_to_end(k)
            if sequential and self.pool is not None:
                for j in range(k + 1, k + 1 + self.ahead):
                    if j * self.block < self.size and j not in self.futs:
                        self.futs[j] = self._submit(j)
            self.last = k
            # evict least-recently-used blocks, sparing only the current block and its read-ahead window (an
            # index-based stop here kept every block after a backward seek: 6.9 GB RSS, OOM-killed on NIOS)
            while len(self.futs) > self.keep + self.ahead:
                old = next((j for j in self.futs if not (k <= j <= k + self.ahead)), None)
                if old is None:
                    break
                self.futs.pop(old)
            fut = self.futs[k]
        data = fut.result()
        return data

    def read_at(self, offset: int, n: int) -> bytes:
        if offset < 0 or n < 0:
            raise ValueError("negative read")
        n = max(0, min(n, self.size - offset))
        out = []
        pos = offset
        while pos < offset + n:
            k = pos // self.block
            b = self._get(k)
            lo = pos - k * self.block
            take = min(len(b) - lo, offset + n - pos)
            out.append(b[lo:lo + take])
            pos += take
            self.fetched_bytes += take
        return b"".join(out)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        self.futs.clear()


# --------------------------------------------------------------------------------------------- seekable inflate

class SeekableInflate(io.RawIOBase):
    """Read-only seekable file over a raw-deflate stream (zip method 8) of ``usize`` bytes, compressed bytes
    ``[cstart, cstart + csize)`` of ``reader``. Backward seeks restart from the nearest access point, taken about
    every ``every`` output bytes; requested 64 KB pages stay in a small LRU cache (h5py re-reads metadata).

    Access points (:mod:`~app.isl.data.pose_sources.zran`, ``libz`` through ``ctypes``) sit on deflate block
    boundaries and hold the last 32 KB of output, so they can be **persisted**: with ``points_file`` (a
    :class:`~app.isl.data.pose_sources.zran.PointFile`) the points saved by an earlier run are loaded and new
    ones are appended every ``persist_every`` output bytes, so a restarted job seeks straight to the point
    before its next chunk instead of re-inflating the member from its first byte. Without ``libz`` it falls
    back to in-memory ``zlib`` snapshots (no persistence)."""

    PAGE = 64 << 10

    def __init__(self, reader, cstart: int, csize: int, usize: int, every: int | None = None, chunk: int = 1 << 20,
                 cache_pages: int = 256, meta_cap: int = 256 << 20, points_file=None,
                 persist_every: int | None = None, native: bool | None = None):
        from app.isl.data.pose_sources import zran

        self.r, self.cstart, self.csize, self.usize = reader, int(cstart), int(csize), int(usize)
        self.every = int(every or max(4 << 20, self.usize // 3000))
        self.chunk = chunk
        self.native = zran.available() if native is None else bool(native)
        # access points: ck_out[i] = output offset; ck[i] = (input offset, state) where state is
        # ("w", bits, window) (native, restorable from disk) or ("z", zlib snapshot) (fallback, memory only)
        self.ck_out: list[int] = [0]
        self.ck: list[tuple[int, tuple]] = [(0, ("w", 0, b"") if self.native else ("z", zlib.decompressobj(-15)))]
        self.points_file = points_file if self.native else None
        self.persist_every = int(persist_every or max(64 << 20, self.every))
        self.persisted_out = 0
        self.points_loaded = 0
        if self.points_file is not None:
            for o, i, b, w in self.points_file.load():
                if 0 < o <= self.usize and i <= self.csize:
                    self._insert_point(o, i, ("w", b, w))
                    self.persisted_out = max(self.persisted_out, o)
                    self.points_loaded += 1
        self.win = b""  # last 32 KB of output (native mode: the window of the next access point)
        self.pages: OrderedDict[int, bytes] = OrderedDict()
        self.cache_pages = cache_pages
        self.pos = 0
        self.restores = 0
        # metadata retention: while inflating forward for the first time, 4 KB pages that do not look like
        # float data (or carry an HDF5 signature) are kept, so h5py's scattered metadata reads need no restart
        self.meta: dict[int, bytes] = {}
        self.meta_bytes = 0
        self.meta_cap = meta_cap
        self.scan_pos = 0
        self.stage = bytearray()
        self.zero_pages: set[int] = set()  # all-zero pages (served without a restart)
        self._prev: tuple[int, bytes, bool] | None = None  # previous scanned page (index, bytes, kept)
        self._restore(0)

    # -- inflate state
    def _insert_point(self, out: int, inp: int, state: tuple) -> None:
        i = bisect.bisect_left(self.ck_out, out)
        if i < len(self.ck_out) and self.ck_out[i] == out:
            return
        self.ck_out.insert(i, out)
        self.ck.insert(i, (inp, state))

    def _restore(self, i: int) -> None:
        from app.isl.data.pose_sources import zran

        inp, state = self.ck[i]
        if state[0] == "z":
            self.d = state[1].copy()
        else:
            _, bits, window = state
            prev = self.r.read_at(self.cstart + inp - 1, 1)[0] if bits else 0
            self.d = zran.Inflater.at_point(bits, prev, window)
            self.win = window
        self.out = self.ck_out[i]
        self.inp = inp  # compressed offset (relative to cstart) of the next byte to feed
        self.buf, self.bpos = b"", 0  # fetched compressed bytes starting at self.inp - self.bpos
        self.restores += 1

    def _limit(self) -> int:
        if self.native:
            return 1 << 30  # native points sit on block boundaries, not on exact multiples
        if self.out >= self.ck_out[-1]:
            nxt = (self.out // self.every + 1) * self.every
            return max(1, nxt - self.out)
        return 1 << 30

    def _maybe_checkpoint(self) -> None:
        if not self.native:
            if self.out > self.ck_out[-1] and self.out % self.every == 0:
                self.ck_out.append(self.out)
                self.ck.append((self.inp, ("z", self.d.copy())))
            return
        if not self.d.at_block_end or self.out <= 0:
            return
        i = bisect.bisect_right(self.ck_out, self.out)
        prev = self.ck_out[i - 1]
        nxt = self.ck_out[i] if i < len(self.ck_out) else None
        if self.out - prev < self.every or (nxt is not None and nxt - self.out < self.every):
            return
        window = self.win[-zran_window():]
        self._insert_point(self.out, self.inp, ("w", self.d.bits, window))
        if self.points_file is not None and self.out >= self.persisted_out + self.persist_every:
            try:
                self.points_file.add(self.out, self.inp, self.d.bits, window)
                self.persisted_out = self.out
            except OSError as e:  # a full disk must not stop the stream
                log.warning("could not persist an access point: %s", e)

    FEED = 16 << 10  # small input pieces: a snapshot keeps a reference to the unconsumed input tail

    def _step(self, limit: int) -> bytes:
        if self.bpos >= len(self.buf):
            if self.inp >= self.csize:
                raise EOFError(f"deflate stream ended at {self.out} of {self.usize}")
            n = min(self.chunk, self.csize - self.inp)
            self.buf, self.bpos = self.r.read_at(self.cstart + self.inp, n), 0
        piece = self.buf[self.bpos:self.bpos + self.FEED]
        out = self.d.decompress(piece, min(limit, self._limit()))
        used = len(piece) - len(self.d.unconsumed_tail)
        self.bpos += used
        self.inp += used
        if out and self.out + len(out) > self.scan_pos and self.meta_cap > 0:
            self._scan(out[max(0, self.scan_pos - self.out):])
        self.out += len(out)
        if self.native and out:
            w = zran_window()
            self.win = out[-w:] if len(out) >= w else (self.win + out)[-w:]
        self._maybe_checkpoint()
        if not out and self.d.eof:
            raise EOFError("deflate stream ended early")
        if not out and not used and not getattr(self.d, "at_block_end", False):
            raise EOFError(f"deflate stream made no progress at {self.out} of {self.usize}")
        return out

    MPAGE = 4096

    def _scan(self, new: bytes) -> None:
        self.stage += new
        self.scan_pos += len(new)
        n = len(self.stage) // self.MPAGE * self.MPAGE
        if not n:
            return
        first = (self.scan_pos - len(self.stage)) // self.MPAGE
        zero = bytes(self.MPAGE)
        for j in range(n // self.MPAGE):
            k = first + j
            pg = bytes(self.stage[j * self.MPAGE:(j + 1) * self.MPAGE])
            if pg == zero:
                if len(self.zero_pages) < 4_000_000:
                    self.zero_pages.add(k)
                kept = False
            else:
                kept = looks_like_metadata(pg)
                # metadata blocks straddle page boundaries: keep the neighbours of a kept page too
                if kept or (self._prev is not None and self._prev[2] and self._prev[0] == k - 1):
                    self._keep(k, pg)
                if kept and self._prev is not None and self._prev[0] == k - 1 and not self._prev[2]:
                    self._keep(k - 1, self._prev[1])
            self._prev = (k, pg, kept)
        del self.stage[:n]

    def _keep(self, k: int, pg: bytes) -> None:
        if k in self.meta or k in self.zero_pages or self.meta_bytes >= self.meta_cap:
            return
        self.meta[k] = pg
        self.meta_bytes += len(pg)

    def _from_meta(self, p: int, n: int) -> bytes | None:
        if p + n > self.scan_pos - len(self.stage):
            return None
        a, b = p // self.MPAGE, (p + n - 1) // self.MPAGE
        if any(k not in self.meta and k not in self.zero_pages for k in range(a, b + 1)):
            return None
        zero = bytes(self.MPAGE)
        blob = b"".join(self.meta.get(k, zero) for k in range(a, b + 1))
        return blob[p - a * self.MPAGE: p - a * self.MPAGE + n]

    def _goto(self, p: int) -> None:
        if p < self.out or p - self.out > self.every:
            i = bisect.bisect_right(self.ck_out, p) - 1
            if p < self.out or self.ck_out[i] > self.out:
                self._restore(i)
        while self.out < p:
            self._step(p - self.out)

    def read_range(self, offset: int, n: int) -> bytes:
        """Bytes [offset, offset + n) bypassing the page cache (bulk sequential reads)."""
        n = max(0, min(n, self.usize - offset))
        self._goto(offset)
        parts, got = [], 0
        while got < n:
            b = self._step(n - got)
            parts.append(b)
            got += len(b)
        return b"".join(parts)

    def _page(self, k: int) -> bytes:
        b = self.pages.get(k)
        if b is not None:
            self.pages.move_to_end(k)
            return b
        b = self.read_range(k * self.PAGE, self.PAGE)
        self.pages[k] = b
        while len(self.pages) > self.cache_pages:
            self.pages.popitem(last=False)
        return b

    # -- io.RawIOBase
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.usize + offset
        else:
            raise ValueError("bad whence")
        self.pos = max(0, self.pos)
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), max(0, self.usize - self.pos))
        if n <= 0:
            return 0
        hit = self._from_meta(self.pos, n) if n <= 4 * self.PAGE else None
        if hit is not None:
            data = hit
        elif n > 4 * self.PAGE:
            data = self.read_range(self.pos, n)
        else:
            parts = []
            p = self.pos
            while p < self.pos + n:
                k = p // self.PAGE
                pg = self._page(k)
                lo = p - k * self.PAGE
                take = min(len(pg) - lo, self.pos + n - p)
                parts.append(pg[lo:lo + take])
                p += take
            data = b"".join(parts)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def zran_window() -> int:
    from app.isl.data.pose_sources.zran import WINDOW

    return WINDOW


_SIGNATURES = re.compile(rb"TREE|HEAP|SNOD|GCOL|OHDR|OCHK|FRHP|FSHD|BTHD|BTIN|BTLF|FHDB|FHIB")


def looks_like_metadata(page: bytes) -> bool:
    """True for a page that is probably HDF5 metadata, not coordinates: it carries an HDF5 block signature, or
    fewer than 99 % of its 8-byte words read as plain float64 values (0 or 1e-6 < |v| < 1e3). Addresses and
    lengths read as denormals, names as wild exponents. Verified on Sign_Library's 1-minute file: 195 of
    357,076 pages flagged; with their neighbours (446 pages) and the all-zero pages h5py lists all 748 datasets
    without a single restart (``SeekableInflate`` keeps both)."""
    if _SIGNATURES.search(page):
        return True
    a = np.frombuffer(page[: len(page) // 8 * 8], dtype="<f8")
    if not len(a):
        return False
    with np.errstate(invalid="ignore", over="ignore"):
        av = np.abs(a)
        ok = (a == 0) | ((av > 1e-6) & (av < 1e3))
    return float(ok.mean()) < 0.99


def zip_member_span(reader, info) -> tuple[int, int, int, int]:
    """(data start, compressed size, uncompressed size, method) of a zip member from its local header."""
    head = reader.read_at(info.header_offset, 30)
    sig, _v, _f, method, _t, _d, _crc, _cs, _us, nlen, xlen = struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise OSError("bad zip local header")
    return info.header_offset + 30 + nlen + xlen, info.compress_size, info.file_size, method


def safe_member(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return not (name.startswith(("/", "\\")) or ".." in parts or ":" in parts[0])


# --------------------------------------------------------------------------------------------- HDF5 object header

class HeaderError(ValueError):
    pass


def parse_v1_header(buf: bytes) -> dict:
    """Minimal HDF5 version-1 object header parser for a plain dataset: ``{"shape", "itemsize", "dclass",
    "little", "layout", "address", "size"}``. Raises :class:`HeaderError` for anything unexpected
    (continuation blocks, version 2 headers, chunked layout ...): the caller then falls back to h5py.
    A malformed header (a message shorter than its fields, an absurd rank ...) is a :class:`HeaderError` too,
    never a bare ``struct.error`` / ``IndexError``, so the job records that chunk as failed instead of crashing."""
    try:
        return _parse_v1_header(buf)
    except HeaderError:
        raise
    except (struct.error, IndexError, TypeError, OverflowError) as e:
        raise HeaderError(f"malformed header ({type(e).__name__}: {e})"[:200]) from None


def _parse_v1_header(buf: bytes) -> dict:
    if len(buf) < 16:
        raise HeaderError("short header")
    ver, _res, nmsg, _ref, hsize = struct.unpack_from("<BBHII", buf, 0)
    if ver != 1:
        raise HeaderError(f"object header version {ver}")
    pos, end = 16, 16 + hsize
    if end > len(buf):
        raise HeaderError("header longer than buffer")
    out: dict = {}
    for _ in range(nmsg):
        if pos + 8 > end:
            break
        mtype, msize, _flags = struct.unpack_from("<HHB", buf, pos)
        data = buf[pos + 8: pos + 8 + msize]
        pos += 8 + msize
        if mtype == 0x0001:  # dataspace
            dv, rank = data[0], data[1]
            off = 8 if dv == 1 else 4
            out["shape"] = tuple(struct.unpack_from(f"<{rank}Q", data, off))
        elif mtype == 0x0003:  # datatype
            out["dclass"] = data[0] & 0x0F
            out["little"] = not (data[1] & 0x01)
            out["itemsize"] = struct.unpack_from("<I", data, 4)[0]
        elif mtype == 0x0008:  # layout
            lv = data[0]
            if lv != 3:
                raise HeaderError(f"layout message version {lv}")
            out["layout"] = data[1]
            if data[1] == 1:
                out["address"], out["size"] = struct.unpack_from("<QQ", data, 2)
        elif mtype == 0x0010:
            raise HeaderError("continuation block")
        elif mtype == 0x000B:
            raise HeaderError("filter pipeline")
    if out.get("layout") != 1 or "shape" not in out or "itemsize" not in out:
        raise HeaderError("not a contiguous dataset")
    return out


def data_nbytes(hdr: dict) -> int:
    """Byte length of a contiguous float dataset; :class:`HeaderError` for anything else."""
    if hdr.get("dclass") != 1 or hdr["itemsize"] not in (4, 8):
        raise HeaderError(f"unexpected datatype class {hdr.get('dclass')} size {hdr.get('itemsize')}")
    n = math.prod(int(d) for d in hdr["shape"]) * hdr["itemsize"]  # Python ints: np.prod wraps (2**40 x 2**40 -> 0)
    if hdr["size"] != n:
        raise HeaderError(f"layout size {hdr['size']} != {n}")
    return n


def array_from(hdr: dict, raw: bytes) -> np.ndarray:
    dt = np.dtype(("<" if hdr["little"] else ">") + ("f8" if hdr["itemsize"] == 8 else "f4"))
    return np.frombuffer(raw, dtype=dt).reshape(hdr["shape"])


def read_contiguous(stream: SeekableInflate, hdr: dict) -> np.ndarray:
    n = data_nbytes(hdr)
    return array_from(hdr, stream.read_range(hdr["address"], n))


# --------------------------------------------------------------------------------------------- forward pass

class HeaderCache:
    """Parsed object headers of one member (``address -> header dict`` or a :class:`HeaderError` message),
    persisted append-only as JSON lines so a restarted forward pass does not re-read headers it has already
    passed. A line cut by a kill is ignored; a file of another member (first line) is replaced."""

    def __init__(self, path, key: str, usize: int, flush_every: int = 500):
        from pathlib import Path

        self.path, self.key, self.usize = Path(path) if path else None, key, int(usize)
        self.items: dict[int, dict | str] = {}
        self._buf: list[str] = []
        self.flush_every = flush_every
        self._head = json_dumps({"key": key, "usize": self.usize})
        self._valid = False
        self._released = False
        self.loaded = 0  # headers read from the file by load()

    def load(self) -> HeaderCache:
        import json

        if self.path is None or not self.path.exists():
            return self
        lines = self.path.read_bytes().split(b"\n")
        # the last element is b"" when the file ends with a newline, else a line cut by a kill (even one cut just
        # before its newline: kept, the next append would run on into it and lose every later line)
        if len(lines) < 2 or lines[0].decode("utf-8", "replace") != self._head:
            return self
        self._valid = True
        good = len(lines[0]) + 1
        for ln in lines[1:-1]:
            if not ln:
                good += 1
                continue
            try:
                r = json.loads(ln)
                a = int(r["a"])
                v = r["e"] if "e" in r else dict(r["h"], shape=tuple(r["h"]["shape"]))
            except (ValueError, KeyError, TypeError):
                break
            self.items[a] = v
            good += len(ln) + 1
        self.loaded = len(self.items)
        size = self.path.stat().st_size
        if good < size:  # cut the damaged tail so later appends stay readable
            with self.path.open("r+b") as fh:
                fh.truncate(good)
        return self

    def get(self, addr: int):
        return self.items.get(int(addr))

    def release(self) -> None:
        """Drop the loaded headers from memory (about 0.9 KB each: 70 MB for ISH News, several hundred MB for a
        NewzHook-sized member, reloaded on every restart); later :meth:`add` calls are only written to the file."""
        self.loaded = max(self.loaded, len(self.items))
        self.items = {}
        self._released = True

    def add(self, addr: int, value: dict | str) -> None:
        addr = int(addr)
        if addr in self.items:
            return
        if not self._released:
            self.items[addr] = value
        if self.path is None:
            return
        self._buf.append(json_dumps({"a": addr, "e": value} if isinstance(value, str)
                                    else {"a": addr, "h": dict(value, shape=list(value["shape"]))}))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self.path is None or not self._buf:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self._valid:
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(self._head + "\n")
                tmp.replace(self.path)
                self._valid = True
            with self.path.open("a") as fh:
                fh.write("\n".join(self._buf) + "\n")
            self._buf.clear()
        except OSError as e:  # a full disk must not stop the stream (the cache only saves work on restart)
            log.warning("could not persist object headers: %s", e)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"), sort_keys=True)


class _Req:
    __slots__ = ("s", "e", "kind", "name", "role", "buf", "hdr")

    def __init__(self, s, e, kind, name, role, hdr=None):
        self.s, self.e, self.kind, self.name, self.role, self.hdr = int(s), int(e), kind, name, role, hdr
        self.buf = bytearray()


class ForwardPass:
    """Read many chunks of one member in **one forward pass** over the inflated stream.

    Why: in the OpenHands files the object headers of a group of chunks sit together and their data comes much
    later (ISH News ``sub_based``: headers at 1.81 GB, data at 3.65 GB), so reading chunk by chunk (header,
    data, next header ...) jumped back and forth: 3.5 restores and about 18 MB of re-inflation per 0.64 MB
    chunk. Here every wanted byte range (each pending header's 4 KB, then each dataset once its header is
    parsed) waits in a heap ordered by offset and is cut out of the stream as it goes by. Headers always come
    before their data in these files; a range that lies behind the kept ``window`` (or anything unusual) is
    read afterwards with the old random-access path (``fallback``), so the result is always the same bytes.

    ``pairs`` = [(name, keypoints header address, visibility header address or None)]. Yields
    ``(name, keypoints, visibility, error)``: arrays exactly as :func:`read_contiguous` returns them, or the
    exception the old per-chunk path would have raised (``HeaderError`` / ``OSError`` / ``EOFError`` /
    ``ValueError``). Chunks come in data order, not header order. Parsed headers go to ``headers``
    (:class:`HeaderCache`); a long gap with nothing wanted is skipped by restoring at a persisted access point.
    """

    HEADER_BYTES = 4096

    def __init__(self, stream: SeekableInflate, pairs, headers: HeaderCache | None = None, window: int = 16 << 20,
                 jump: int = 64 << 20, step: int = 1 << 20, max_stream_errors: int = 5, forgive: int = 256 << 20):
        import heapq

        self._heapq = heapq
        self.st = stream
        self.headers = headers if headers is not None else HeaderCache(None, "", stream.usize)
        self.window, self.jump, self.step, self.max_stream_errors = int(window), int(jump), int(step), max_stream_errors
        self.forgive = int(forgive)  # bytes of clean progress after which earlier stream errors are forgotten
        self.heap: list = []
        self._seq = 0
        self.active: list[_Req] = []
        self.chunks: dict[str, dict] = {}
        self.ready: list = []
        self.fallback: list[str] = []
        self.pairs = {}
        self.win = bytearray()
        self.win_start = stream.out
        self.stats = {"chunks": 0, "fallback": 0, "jumps": 0, "stream_errors": 0, "header_cache_hits": 0,
                      "inflated": 0}
        for name, kp_addr, vis_addr in pairs:
            self.pairs[name] = (kp_addr, vis_addr)
            c = {"kp": None, "vis": None if vis_addr is not None else ("ok", None), "done": False}
            self.chunks[name] = c
            for role, addr in (("kp", kp_addr), ("vis", vis_addr)):
                if addr is None:
                    continue
                h = self.headers.get(addr)
                if h is None:
                    self._push(_Req(addr, min(int(addr) + self.HEADER_BYTES, stream.usize), "hdr", name, role))
                else:
                    self.stats["header_cache_hits"] += 1
                    self._got_header(name, role, h)
                if c["done"]:
                    break
        self.headers.release()  # every cached header this pass needs is queued above

    # -- bookkeeping
    def _push(self, r: _Req) -> None:
        self._seq += 1
        self._heapq.heappush(self.heap, (r.s, self._seq, r))

    def _to_fallback(self, name: str) -> None:
        c = self.chunks[name]
        if not c["done"]:
            c["done"] = True
            c["kp"] = c["vis"] = None  # the old path reads both again
            self.fallback.append(name)

    def _resolve(self, name: str, role: str, state: tuple) -> None:
        c = self.chunks[name]
        if c["done"]:
            return
        c[role] = state
        kp, vis = c["kp"], c["vis"]
        if kp is not None and kp[0] == "err":  # the old path stops at the first keypoints error
            c["done"] = True
            self.ready.append((name, None, None, kp[1]))
        elif kp is not None and vis is not None:
            c["done"] = True
            if vis[0] == "err":
                self.ready.append((name, None, None, vis[1]))
            else:
                self.ready.append((name, kp[1], vis[1], None))
        if c["done"]:
            self.stats["chunks"] += 1
            c["kp"] = c["vis"] = None  # the arrays now live only in ``ready`` (kept here they leaked ~0.3 MB a chunk)

    def _got_header(self, name: str, role: str, h) -> None:
        if isinstance(h, str):
            self._resolve(name, role, ("err", HeaderError(h)))
            return
        try:
            n = data_nbytes(h)
        except HeaderError as e:
            self._resolve(name, role, ("err", e))
            return
        a = int(h["address"])
        if a < 0 or a + n > self.st.usize:  # odd layout: let the old path produce its exact result
            self._to_fallback(name)
            return
        self._push(_Req(a, a + n, "data", name, role, hdr=h))

    def _complete(self, r: _Req) -> None:
        if self.chunks[r.name]["done"]:
            return
        if r.kind == "hdr":
            try:
                h = parse_v1_header(bytes(r.buf))
            except HeaderError as e:
                self.headers.add(r.s, str(e))
                self._resolve(r.name, r.role, ("err", e))
                return
            except Exception:  # noqa: BLE001 - anything unexpected: the old path decides
                self._to_fallback(r.name)
                return
            self.headers.add(r.s, h)
            self._got_header(r.name, r.role, h)
        else:
            try:
                arr = array_from(r.hdr, bytes(r.buf))
            except ValueError:
                self._to_fallback(r.name)
                return
            self._resolve(r.name, r.role, ("ok", arr))

    # -- stream
    def _activate(self) -> None:
        out = self.st.out
        while self.heap and self.heap[0][0] <= out:
            _, _, r = self._heapq.heappop(self.heap)
            if self.chunks[r.name]["done"]:
                continue
            if r.s < self.win_start:
                self._to_fallback(r.name)
                continue
            hi = min(r.e, out)
            r.buf += self.win[r.s - self.win_start:hi - self.win_start]
            if r.s + len(r.buf) >= r.e:
                self._complete(r)
            else:
                self.active.append(r)

    def _feed(self, o: int, piece: bytes) -> None:
        self.win += piece
        end = o + len(piece)
        active, self.active = self.active, []
        for r in active:
            if self.chunks[r.name]["done"]:
                continue
            need = r.s + len(r.buf)  # by position: bytes streamed twice (after a re-positioning) are not doubled
            lo, hi = max(need, o), min(r.e, end)
            if hi > lo:
                r.buf += piece[lo - o:hi - o]
            if r.s + len(r.buf) >= r.e:
                self._complete(r)
            else:
                self.active.append(r)
        self._activate()  # ranges starting in this piece (and data ranges of headers just parsed), before trimming
        if len(self.win) > 2 * self.window:
            cut = len(self.win) - self.window
            del self.win[:cut]
            self.win_start += cut

    def _restore_before(self, p: int) -> None:
        i = bisect.bisect_right(self.st.ck_out, p) - 1
        self.st._restore(i)
        self.win = bytearray()
        self.win_start = self.st.out

    def _maybe_jump(self) -> None:
        if self.active or not self.heap:
            return
        target = self.heap[0][0]
        if target - self.st.out < self.jump:
            return
        i = bisect.bisect_right(self.st.ck_out, target) - 1
        if self.st.ck_out[i] > self.st.out + self.window:
            self._restore_before(target)
            self.stats["jumps"] += 1

    def __iter__(self):
        try:
            yield from self._run()
        finally:
            self.headers.flush()

    def _run(self):
        st = self.st
        st.pages.clear()
        st.meta.clear()
        st.meta_bytes, st.meta_cap = 0, 0  # no metadata scanning: headers are read from the stream directly
        # start at the access point before the first wanted byte (a restart: the point saved near where the last
        # run stopped), unless the stream already stands between that point and the byte
        first = self.heap[0][0] if self.heap else st.out
        i = bisect.bisect_right(st.ck_out, first) - 1
        if first < st.out or st.ck_out[i] > st.out:
            self._restore_before(first)
        else:
            self.win, self.win_start = bytearray(), st.out
        errors, err_at, redo = 0, 0, None

        def stream_error(e: OSError, at: int) -> None:
            nonlocal errors, err_at
            errors += 1
            err_at = max(err_at, at)
            self.stats["stream_errors"] += 1
            if errors > self.max_stream_errors:
                raise e
            log.warning("forward pass: stream error at %d (%s); restoring (%d of %d)", at, e, errors,
                        self.max_stream_errors)
            time.sleep(min(60, 5 * errors))

        while self.heap or self.active:
            # restores read from the network too: a failed one is retried like a failed step, never fatal at once
            try:
                if redo is not None:
                    self._restore_before(redo)
                    redo = None
                self._maybe_jump()
            except OSError as e:
                stream_error(e, st.out)
                continue
            self._activate()
            while self.ready:
                yield self.ready.pop(0)
            if not self.heap and not self.active:
                break
            if st.out >= st.usize:
                break
            o = st.out
            try:
                piece = st._step(self.step)
            except EOFError:
                break
            except OSError as e:  # network gave up after its retries: re-position and try again
                stream_error(e, o)
                redo = o
                continue
            if errors and st.out > err_at + self.forgive:  # an outage long ago does not count against a later one
                errors = 0
            self.stats["inflated"] += len(piece)
            self._feed(o, piece)
            while self.ready:
                yield self.ready.pop(0)
        # anything still wanted (stream ended early) is left to the old path
        for _, _, r in self.heap:
            self._to_fallback(r.name)
        for r in self.active:
            self._to_fallback(r.name)
        self.heap, self.active = [], []
        while self.ready:
            yield self.ready.pop(0)
        self.stats["fallback"] = len(self.fallback)
        if self.fallback:
            log.info("forward pass: %d chunks left to the random-access path", len(self.fallback))
        for name in self.fallback:
            kp_addr, vis_addr = self.pairs[name]
            try:
                kp = read_contiguous(st, parse_v1_header(st.read_range(kp_addr, self.HEADER_BYTES)))
                vis = (read_contiguous(st, parse_v1_header(st.read_range(vis_addr, self.HEADER_BYTES)))
                       if vis_addr is not None else None)
            except (OSError, EOFError, ValueError) as e:  # HeaderError is a ValueError
                yield name, None, None, e
                continue
            yield name, kp, vis, None


def list_datasets(h5file, groups: tuple[str, ...] = ("keypoints", "visibility")) -> list[tuple[int, str, str]]:
    """(object header address, group, name) of every dataset in ``groups``, sorted by address."""
    out = []
    for g in groups:
        if g not in h5file:
            continue
        gid = h5file[g].id
        for name in h5file[g].keys():
            info = gid.links.get_info(name.encode("utf-8"))
            out.append((int(info.u), g, name))
    out.sort()
    return out


# --------------------------------------------------------------------------------------------- conversion

def snap_aspect(a: float | None, tol: float = 0.2) -> float | None:
    if a is None:
        return None
    best = min(COMMON_ASPECTS, key=lambda c: abs(np.log(a / c)))
    return best if abs(np.log(a / best)) < tol else a


def openhands_to_raw(kp: np.ndarray, vis: np.ndarray | None, aspect: float = 16 / 9,
                     fps: float = ASSUMED_FPS) -> RawSequence:
    """(T, 75, 3) keypoints + (T, 75) visibility -> RawSequence (face all missing; width/height carry
    ``aspect`` at height 1080)."""
    kp = np.asarray(kp, dtype=np.float64)
    T = kp.shape[0]
    if kp.ndim != 3 or kp.shape[1] != 75 or kp.shape[2] < 3:
        raise ValueError(f"expected (T, 75, 3) keypoints, got {kp.shape}")
    vis = np.ones((T, 75)) if vis is None else np.asarray(vis, dtype=np.float64)[:T]
    finite = np.isfinite(kp).all(axis=2)
    pose = np.full((T, N_POSE, 4), np.nan, np.float32)
    pz = np.all(kp[:, :33] == 0, axis=(1, 2)) | ~finite[:, :33].all(axis=1)
    ok = ~pz
    pose[ok, :, :3] = kp[ok, :33, :3]
    pose[ok, :, 3] = vis[ok, :33]
    hands = []
    for lo in (33, 54):
        h = np.full((T, N_HAND, 3), np.nan, np.float32)
        hz = np.all(kp[:, lo:lo + 21] == 0, axis=(1, 2)) | ~finite[:, lo:lo + 21].all(axis=1)
        h[~hz] = kp[~hz, lo:lo + 21, :3]
        hands.append(h)
    face = np.full((T, N_FACE, 3), np.nan, np.float32)
    H = 1080
    W = int(round(H * (aspect or 16 / 9)))
    t_ms = np.arange(T, dtype=np.float64) * 1000.0 / fps
    return RawSequence(pose, hands[0], hands[1], face, t_ms, W, H, EXTRACTOR_VERSION)


def hands_nan(kp: np.ndarray) -> list[np.ndarray]:
    """Left / right hands of (T, 75, 3) keypoints with missing (all-zero) frames as NaN (for ``hand_aspect``)."""
    out = []
    for lo in (33, 54):
        h = np.asarray(kp[:, lo:lo + 21, :3], dtype=np.float64).copy()
        h[np.all(h == 0, axis=(1, 2))] = np.nan
        out.append(h)
    return out


def video_key(name: str) -> str:
    """``0-LgpNKQtNgJob_Announcement-3`` -> ``0-LgpNKQtNgJob_Announcement`` (chunk index dropped)."""
    head, sep, tail = name.rpartition("-")
    return head if sep and tail.isdigit() else name


def iter_pairs(entries: list[tuple[int, str, str]]) -> Iterator[tuple[str, tuple[int, str, str] | None,
                                                                   tuple[int, str, str] | None]]:
    """Group (address, group, name) entries into per-name (keypoints, visibility) pairs, ordered by the
    keypoints' address (each pair is emitted once both, or the last of them, have been seen)."""
    by_name: dict[str, dict[str, tuple[int, str, str]]] = {}
    for e in entries:
        by_name.setdefault(e[2], {})[e[1]] = e
    order = sorted(by_name.items(), key=lambda kv: min(x[0] for x in kv[1].values()))
    for name, d in order:
        yield name, d.get("keypoints"), d.get("visibility")


class OpenHandsRawAdapter(_Base):
    """Unlabelled chunks for self-supervised encoder pre-training. ``group = openhands:<channel>:<video>``."""

    name = "openhands_raw"
    licence = LICENCE
    commercial = False
    manifest_name = "openhands_raw"
    profiles = ("research",)
    pretraining_only = True

    def clips(self, *, only_ok: bool = True) -> Iterator[PoseClip]:
        for rec in self.records(only_ok=only_ok):
            if only_ok and (int(rec.get("n_frames") or 0) < MIN_FRAMES
                            or float(rec.get("hands_frac") or 0.0) < MIN_HANDS_FRAC):
                continue
            yield PoseClip(
                clip_id=rec["id"], source="openhands_raw", raw_gloss="", concept_id=None, variant_id=None,
                category=rec.get("channel"), path=self.paths.resolve(rec["path"]),
                group=f"openhands:{rec.get('channel')}:{rec.get('video')}", fps=rec.get("fps"), licence=self.licence,
                profiles=self.profiles,
                quality={k: rec.get(k) for k in ("n_frames", "hands_frac", "aspect_est", "member", "dtype", "stride",
                                                 "fps_assumed", "extractor_version")},
            )
