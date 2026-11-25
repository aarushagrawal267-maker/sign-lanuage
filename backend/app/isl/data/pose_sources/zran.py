"""Raw-deflate inflater with **restorable access points** (zlib's ``zran.c`` technique through ``ctypes``).

Python's ``zlib.decompressobj`` cannot report deflate block boundaries or be serialised, so a long streamed
member (OpenHands NIOS: 30 GB inflated from 11 GB of HTTP range requests) had to be re-inflated from its first
byte after every restart. Here ``libz`` is called directly:

* :class:`Inflater` decodes with ``Z_BLOCK`` and reports when it stands on a block boundary
  (``at_block_end``) and how many bits of the last consumed byte belong to the next block (``bits``).
* An access point is ``(out, inp, bits, window)``: uncompressed offset, compressed offset (bytes consumed),
  unused bits, and the last 32 KB of output. :meth:`Inflater.at_point` rebuilds a decoder there with
  ``inflatePrime`` (the partial byte) and ``inflateSetDictionary`` (the window), exactly like ``zran.c``.
* :class:`PointFile` persists points (append-only, one CRC per record, so a record cut by a kill is ignored)
  for one member, keyed by its compressed/uncompressed sizes.

``available()`` is False when ``libz`` cannot be loaded (callers then fall back to ``zlib`` snapshots that live
in memory only).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import struct
import threading
import zlib
from pathlib import Path

WINDOW = 32768
Z_OK, Z_STREAM_END, Z_NEED_DICT, Z_BUF_ERROR = 0, 1, 2, -5
Z_NO_FLUSH, Z_BLOCK = 0, 5


class _ZStream(ctypes.Structure):
    _fields_ = [
        ("next_in", ctypes.c_void_p), ("avail_in", ctypes.c_uint), ("total_in", ctypes.c_ulong),
        ("next_out", ctypes.c_void_p), ("avail_out", ctypes.c_uint), ("total_out", ctypes.c_ulong),
        ("msg", ctypes.c_char_p), ("state", ctypes.c_void_p),
        ("zalloc", ctypes.c_void_p), ("zfree", ctypes.c_void_p), ("opaque", ctypes.c_void_p),
        ("data_type", ctypes.c_int), ("adler", ctypes.c_ulong), ("reserved", ctypes.c_ulong),
    ]


_LIB = None
_LIB_LOCK = threading.Lock()


def _lib():
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            name = ctypes.util.find_library("z") or "libz.so.1"
            lib = ctypes.CDLL(name)
            p = ctypes.POINTER(_ZStream)
            lib.zlibVersion.restype = ctypes.c_char_p
            lib.inflateInit2_.argtypes = [p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
            lib.inflate.argtypes = [p, ctypes.c_int]
            lib.inflateEnd.argtypes = [p]
            lib.inflatePrime.argtypes = [p, ctypes.c_int, ctypes.c_int]
            lib.inflateSetDictionary.argtypes = [p, ctypes.c_char_p, ctypes.c_uint]
            for f in (lib.inflateInit2_, lib.inflate, lib.inflateEnd, lib.inflatePrime, lib.inflateSetDictionary):
                f.restype = ctypes.c_int
            _LIB = lib
        return _LIB


def available() -> bool:
    try:
        _lib()
        return True
    except OSError:
        return False


class InflateError(OSError):
    pass


class Inflater:
    """Raw deflate decoder (``wbits = -15``) that stops at block boundaries. Same calling pattern as
    ``zlib.decompressobj``: ``decompress(data, max_length)`` + ``unconsumed_tail`` + ``eof``."""

    OUT = 1 << 20

    def __init__(self):
        self.lib = _lib()
        self.z = _ZStream()
        ver = self.lib.zlibVersion()
        rc = self.lib.inflateInit2_(ctypes.byref(self.z), -15, ver, ctypes.sizeof(_ZStream))
        if rc != Z_OK:
            raise InflateError(f"inflateInit2 failed ({rc})")
        self._open = True
        self.out_buf = ctypes.create_string_buffer(self.OUT)
        self.unconsumed_tail = b""
        self.eof = False
        self.at_block_end = False
        self.bits = 0
        self.last_block = False

    @classmethod
    def at_point(cls, bits: int, prev_byte: int, window: bytes) -> Inflater:
        """Decoder positioned at an access point: ``bits`` unused bits of ``prev_byte`` (the last consumed byte)
        start the next block; ``window`` = the up to 32 KB of output before the point."""
        d = cls()
        if bits:
            rc = d.lib.inflatePrime(ctypes.byref(d.z), int(bits), int(prev_byte) >> (8 - int(bits)))
            if rc != Z_OK:
                raise InflateError(f"inflatePrime failed ({rc})")
        if window:
            rc = d.lib.inflateSetDictionary(ctypes.byref(d.z), bytes(window), len(window))
            if rc != Z_OK:
                raise InflateError(f"inflateSetDictionary failed ({rc})")
        return d

    def decompress(self, data: bytes, max_length: int = 0) -> bytes:
        """Inflate ``data``; returns at most ``max_length`` bytes (1 MB per call at most) and stops early at the
        end of a deflate block. Unused input is left in ``unconsumed_tail``."""
        if self.eof:
            self.unconsumed_tail = bytes(data)
            return b""
        data = bytes(data)
        n_out = min(self.OUT, max_length) if max_length and max_length > 0 else self.OUT
        z = self.z
        keep = ctypes.c_char_p(data)  # keeps ``data`` alive and gives its address
        z.next_in = ctypes.cast(keep, ctypes.c_void_p)
        z.avail_in = len(data)
        z.next_out = ctypes.addressof(self.out_buf)
        z.avail_out = n_out
        rc = self.lib.inflate(ctypes.byref(z), Z_BLOCK)
        produced = n_out - z.avail_out
        used = len(data) - z.avail_in
        z.next_in = None
        if rc == Z_STREAM_END:
            self.eof = True
        elif rc not in (Z_OK, Z_BUF_ERROR):
            msg = z.msg.decode("utf-8", "replace") if z.msg else ""
            raise InflateError(f"inflate failed ({rc}{': ' + msg if msg else ''})")
        dt = int(z.data_type)
        self.at_block_end = bool(dt & 128) and not (dt & 64) and not self.eof
        self.bits = dt & 7
        self.last_block = bool(dt & 64)
        self.unconsumed_tail = data[used:]
        return ctypes.string_at(self.out_buf, produced) if produced else b""

    def close(self) -> None:
        if getattr(self, "_open", False):
            self.lib.inflateEnd(ctypes.byref(self.z))
            self._open = False

    def __del__(self):  # noqa: D105
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class PointFile:
    """Append-only access points of one deflate member: a JSON header line (``key``, ``csize``, ``usize``) and
    records ``<QQBI out inp bits len>`` + zlib-compressed window + CRC32. A file for another member (different
    key or sizes) is ignored and replaced on the first :meth:`add`."""

    MAGIC = b"ZRANPT1\n"
    HEAD = struct.Struct("<QQBI")

    def __init__(self, path: Path, key: str, csize: int, usize: int):
        self.path, self.key, self.csize, self.usize = Path(path), key, int(csize), int(usize)
        self._valid_header = False

    def _header(self) -> bytes:
        return self.MAGIC + json.dumps({"key": self.key, "csize": self.csize, "usize": self.usize}).encode() + b"\n"

    def load(self) -> list[tuple[int, int, int, bytes]]:
        """[(out, inp, bits, window)] sorted by ``out``; stops at the first damaged or partial record."""
        if not self.path.exists():
            return []
        blob = self.path.read_bytes()
        head = self._header()
        if not blob.startswith(head):
            return []
        self._valid_header = True
        pos, out = len(head), []
        while pos + self.HEAD.size <= len(blob):
            o, i, b, n = self.HEAD.unpack_from(blob, pos)
            end = pos + self.HEAD.size + n + 4
            if end > len(blob) or b > 7 or i > self.csize or o > self.usize:
                break
            body = blob[pos:pos + self.HEAD.size + n]
            (crc,) = struct.unpack_from("<I", blob, pos + self.HEAD.size + n)
            if zlib.crc32(body) != crc:
                break
            try:
                win = zlib.decompress(blob[pos + self.HEAD.size:pos + self.HEAD.size + n])
            except zlib.error:
                break
            out.append((o, i, b, win))
            pos = end
        if pos != len(blob):  # cut the damaged tail so later appends stay readable
            with self.path.open("r+b") as fh:
                fh.truncate(pos)
        out.sort(key=lambda p: p[0])
        return out

    def add(self, out: int, inp: int, bits: int, window: bytes) -> None:
        cw = zlib.compress(bytes(window), 1)
        body = self.HEAD.pack(int(out), int(inp), int(bits), len(cw)) + cw
        rec = body + struct.pack("<I", zlib.crc32(body))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._valid_header:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_bytes(self._header())
            tmp.replace(self.path)
            self._valid_header = True
        with self.path.open("ab") as fh:
            fh.write(rec)
