"""Remote access helpers for gated Hugging Face datasets (docs/ISL_DESIGN.md section 5.7).

* :func:`hf_access` tells whether this machine may read a gated dataset. A dataset card (``README.md``)
  is public even without access, so the probe requests one byte of a **data file** instead.
* :class:`RangeFile` is a seekable, read-only file over one or more HTTP URLs treated as one concatenated
  byte stream (split archives ``part_aa``, ``part_ab``, ...). It reads with HTTP Range requests, retries with
  backoff and keeps a small LRU block cache, so ``zipfile``/``tarfile`` can read single members of a
  multi-GB archive without downloading it.
* :class:`ConcatStream` is the non-seekable alternative: the parts streamed one after another through a
  pipe (for compressed tar archives that cannot be read at random).

Authentication: on this machine the egress proxy injects the Hugging Face credentials. Elsewhere the
standard ``HF_TOKEN`` environment variable is used if set. Tokens are never printed, logged or stored.
"""

from __future__ import annotations

import io
import logging
import os
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable, Iterator

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets"
HF_API = "https://huggingface.co/api/datasets"
PROBE_FILES = {"Exploration-Lab/CISLR": "dataset.csv", "Exploration-Lab/iSign": "iSign_v1.1.csv"}

Fetch = Callable[[str, int, int], bytes]  # (url, start, end_inclusive) -> bytes


def hf_url(repo: str, path: str, revision: str = "main") -> str:
    return f"{HF_BASE}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def _session():
    import requests

    s = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    s.headers["User-Agent"] = "sign-intelligence-studio-isl-ingest/1.0"
    return s


_SESSION = None


def session():
    global _SESSION
    if _SESSION is None:
        _SESSION = _session()
    return _SESSION


def hf_access(repo: str, probe: str | None = None, timeout: float = 30.0, get=None) -> str:
    """'granted' | 'pending' | 'error:<code or exception>'. Sends a 1-byte Range GET for a data file.

    ``get`` (tests) replaces ``session().get``; it receives (url, headers=..., timeout=..., allow_redirects=True)."""
    path = probe or PROBE_FILES.get(repo)
    if not path:
        return "error:no-probe-file"
    get = get or session().get
    try:
        r = get(hf_url(repo, path), headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True)
        code = int(getattr(r, "status_code", 0))
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"
    if code in (200, 206):
        return "granted"
    if code in (401, 403):
        return "pending"
    return f"error:{code}"


def hf_tree(repo: str, path: str = "", get=None) -> list[dict]:
    """File listing (path, size) from the HF tree API, following pagination."""
    get = get or session().get
    url = f"{HF_API}/{repo}/tree/main/{urllib.parse.quote(path)}" if path else f"{HF_API}/{repo}/tree/main"
    out = []
    while url:
        r = get(url, timeout=60)
        r.raise_for_status()
        out += r.json()
        url = None
        link = r.headers.get("Link", "")
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.find("<") + 1: part.find(">")]
    return out


def remote_size(url: str, get=None) -> int:
    """Total size from a 1-byte range request (``Content-Range: bytes 0-0/TOTAL``)."""
    get = get or session().get
    r = get(url, headers={"Range": "bytes=0-0"}, timeout=60, allow_redirects=True)
    cr = r.headers.get("Content-Range", "")
    r.close()
    if "/" in cr:
        return int(cr.rsplit("/", 1)[1])
    raise OSError(f"no Content-Range for {url} (status {r.status_code})")


def http_range_fetch(url: str, start: int, end: int, retries: int = 6, timeout: float = 120.0) -> bytes:
    """Bytes [start, end] (inclusive) of ``url`` with retries and exponential backoff."""
    err = None
    for i in range(retries):
        try:
            r = session().get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout, allow_redirects=True)
            if r.status_code in (200, 206):
                data = r.content
                if r.status_code == 200:  # server ignored the range
                    data = data[start: end + 1]
                if len(data) == end - start + 1:
                    return data
                err = f"short read {len(data)} of {end - start + 1}"
            else:
                err = f"HTTP {r.status_code}"
                if r.status_code in (401, 403, 404):
                    raise PermissionError(f"{err} for {url.split('?')[0]}")
        except PermissionError:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** i))
    raise OSError(f"range fetch failed after {retries} tries: {err}")


class RangeFile(io.RawIOBase):
    """Seekable read-only view of ``urls`` concatenated (sizes from the tree API or :func:`remote_size`)."""

    def __init__(self, urls: list[str], sizes: list[int], block: int = 8 << 20, cache_blocks: int = 8,
                 fetch: Fetch | None = None):
        if len(urls) != len(sizes) or not urls:
            raise ValueError("urls and sizes must be non-empty and of equal length")
        self.urls, self.sizes = list(urls), [int(s) for s in sizes]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.size = self.offsets[-1]
        self.block = int(block)
        self.cache_blocks = int(cache_blocks)
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0
        self._fetch = fetch or http_range_fetch
        self.bytes_fetched = 0
        self.requests = 0

    # -- io.RawIOBase
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError("bad whence")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def _read_span(self, start: int, end: int) -> bytes:
        """Bytes [start, end) across part boundaries."""
        out = bytearray()
        for i, url in enumerate(self.urls):
            a, b = self.offsets[i], self.offsets[i + 1]
            lo, hi = max(start, a), min(end, b)
            if lo < hi:
                out += self._fetch(url, lo - a, hi - a - 1)
                self.requests += 1
        self.bytes_fetched += len(out)
        return bytes(out)

    def _get_block(self, k: int) -> bytes:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        start = k * self.block
        data = self._read_span(start, min(self.size, start + self.block))
        self._cache[k] = data
        while len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return data

    def readinto(self, b) -> int:
        if self._pos >= self.size:
            return 0
        n = min(len(b), self.size - self._pos)
        mv = memoryview(b)
        got = 0
        # large reads bypass the cache
        if n >= 2 * self.block:
            data = self._read_span(self._pos, self._pos + n)
            mv[:n] = data
            self._pos += n
            return n
        while got < n:
            k = (self._pos) // self.block
            blk = self._get_block(k)
            off = self._pos - k * self.block
            take = min(n - got, len(blk) - off)
            if take <= 0:
                break
            mv[got: got + take] = blk[off: off + take]
            got += take
            self._pos += take
        return got

    def read_at(self, offset: int, n: int) -> bytes:
        self.seek(offset)
        return self.read(n)


class ConcatStream(io.RawIOBase):
    """Non-seekable stream of several URLs one after another (``curl part_aa; curl part_ab; ...`` as a pipe).

    ``start`` skips that many bytes of the concatenated stream with a Range request (resume of an
    uncompressed archive at a recorded member offset)."""

    def __init__(self, urls: list[str], sizes: list[int] | None = None, start: int = 0,
                 opener: Callable[[str, int], Iterator[bytes]] | None = None, chunk: int = 1 << 20):
        self.urls = list(urls)
        self.sizes = list(sizes) if sizes else None
        self.chunk = chunk
        self._opener = opener or self._http_iter
        self._parts = self._iter_parts(start)
        self._buf = b""
        self.position = start

    def _http_iter(self, url: str, offset: int) -> Iterator[bytes]:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        r = session().get(url, headers=headers, stream=True, timeout=120, allow_redirects=True)
        if r.status_code not in (200, 206):
            raise OSError(f"HTTP {r.status_code} for {url.split('?')[0]}")
        if offset and r.status_code == 200:
            raise OSError("server ignored the Range header; cannot resume")
        yield from r.iter_content(self.chunk)

    def _iter_parts(self, start: int) -> Iterator[bytes]:
        skip = start
        for i, url in enumerate(self.urls):
            if self.sizes is not None and skip >= self.sizes[i]:
                skip -= self.sizes[i]
                continue
            yield from self._opener(url, skip)
            skip = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            try:
                self._buf = next(self._parts)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.position += n
        return n


def sniff_format(head: bytes) -> str:
    """Archive format from its first bytes: tar | zip | gzip | zstd | bzip2 | xz | unknown."""
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"\x28\xb5\x2f\xfd":
        return "zstd"
    if head[:3] == b"BZh":
        return "bzip2"
    if head[:6] == b"\xfd7zXZ\x00":
        return "xz"
    return "unknown"
