"""Fetch one source item (zip member, URL, image bytes) into a temp file, hashing it on the way.

Rules for untrusted archives: member names are only used as labels, never as output paths (the temp file name
comes from ``mkstemp``); names that are absolute, contain ``..`` or a drive letter are rejected anyway, so a later
extraction step can never be tricked. Oversized members and zip bombs (huge declared size for a tiny compressed
size) are refused before reading. Every file descriptor is closed with a ``with`` block: an fd left open on a
deleted temp file keeps its disk space allocated (the ISLRTC job once leaked 23 GB that way).
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

MAX_MEMBER_BYTES = 2 << 30  # 2 GB: no single sign clip or image is anywhere near that
MAX_RATIO = 200  # declared / compressed size above this (and above 64 MB) looks like a zip bomb
CHUNK = 1 << 20


class UnsafeMember(ValueError):
    pass


def safe_member_name(name: str) -> str:
    """Return ``name`` (normalised to ``/``) if it is a harmless relative path, else raise :class:`UnsafeMember`."""
    n = name.replace("\\", "/")
    if not n or n.startswith("/") or re.match(r"^[A-Za-z]:", n) or "\x00" in n:
        raise UnsafeMember(f"absolute or empty member name: {name!r}")
    if any(p == ".." for p in PurePosixPath(n).parts):
        raise UnsafeMember(f"path traversal in member name: {name!r}")
    return n


def check_member(info: zipfile.ZipInfo, max_bytes: int = MAX_MEMBER_BYTES) -> None:
    safe_member_name(info.filename)
    if info.file_size > max_bytes:
        raise UnsafeMember(f"member too large ({info.file_size} bytes): {info.filename!r}")
    if info.file_size > (64 << 20) and info.compress_size and info.file_size / info.compress_size > MAX_RATIO:
        raise UnsafeMember(f"suspicious compression ratio: {info.filename!r}")


def temp_path(tmp_dir: Path, prefix: str, suffix: str) -> Path:
    """A fresh temp file name; the descriptor ``mkstemp`` opens is closed immediately."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=tmp_dir)
    os.close(fd)
    return Path(name)


def copy_hashed(src, dest: Path, limit: int = MAX_MEMBER_BYTES) -> tuple[str, int]:
    """Copy a binary stream to ``dest``; returns (sha256 hex, bytes). The destination fd is always closed."""
    h = hashlib.sha256()
    n = 0
    with dest.open("wb") as out:
        while True:
            buf = src.read(CHUNK)
            if not buf:
                break
            n += len(buf)
            if n > limit:
                raise UnsafeMember(f"stream exceeds {limit} bytes")
            h.update(buf)
            out.write(buf)
    return h.hexdigest(), n


def zip_member_to_file(zf: zipfile.ZipFile, member: str, dest: Path) -> tuple[str, int, int]:
    """Stream one member to ``dest``; returns (sha256, bytes, crc32 from the central directory)."""
    info = zf.getinfo(member)
    check_member(info)
    with zf.open(info) as src:
        sha, n = copy_hashed(src, dest)
    return sha, n, info.CRC


def url_to_file(url: str, dest: Path, get=None, retries: int = 5, timeout: float = 300.0,
                sleep: Callable[[float], None] = time.sleep) -> tuple[str, int]:
    """GET ``url`` (redirects followed) into ``dest`` with retries; returns (sha256, bytes)."""
    if get is None:
        from app.isl.data.remote import session

        get = session().get
    err = ""
    for i in range(retries):
        try:
            r = get(url, stream=True, timeout=timeout, allow_redirects=True)
            try:
                if r.status_code == 200:
                    return copy_hashed(_IterReader(r.iter_content(CHUNK)), dest)
                err = f"HTTP {r.status_code}"
                if r.status_code in (401, 403, 404, 410):
                    raise PermissionError(f"{err} for {url.split('?')[0]}")
            finally:
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass
        except (PermissionError, UnsafeMember):
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:200]
        sleep(min(60, 2 ** i))
    raise OSError(f"download failed after {retries} tries: {err}")


class _IterReader:
    """``read(n)`` over an iterator of byte chunks (requests ``iter_content``)."""

    def __init__(self, it):
        self._it = iter(it)
        self._buf = b""

    def read(self, n: int = -1) -> bytes:
        while n < 0 or len(self._buf) < n:
            try:
                chunk = next(self._it)
            except StopIteration:
                break
            if chunk:
                self._buf += chunk
        if n < 0:
            out, self._buf = self._buf, b""
        else:
            out, self._buf = self._buf[:n], self._buf[n:]
        return out


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()
