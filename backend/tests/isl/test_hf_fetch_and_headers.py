"""Offline tests: HF alphabet sets taken in full (no per-class cap), the engine's ``hfrow`` fetch through
``hf_listing.RowPages`` (one paced, backed-off request per 100-row page, ``Retry-After``, signed-URL expiry, remote
refusal stops the run) and OpenHands ``parse_v1_header`` turning malformed headers into ``HeaderError``. No network."""

from __future__ import annotations

import importlib.util
import json
import struct

import pytest

from app.core.config import REPO_ROOT
from app.isl.data.manifest import read_manifest
from app.isl.data.pose_sources import openhands
from app.isl.data.video_sources import hf_listing as H
from app.isl.data.video_sources import listings as L
from app.isl.data.video_sources.catalog import SOURCES, get_source
from app.isl.data.video_sources.engine import Fetcher, run
from app.isl.paths import isl_paths

NAMES = ["1", "a", "B"]
P = H.Policy(min_interval=2.0, base=5.0, max_wait=60.0, max_failures=4, max_retry_after=900.0, jitter=0.0)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t, self.waits = t, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.waits.append(round(s, 3))
        self.t += s


class Resp:
    def __init__(self, code, doc=None, headers=None, body=b""):
        self.status_code, self._doc, self.headers, self._b = code, doc, headers or {}, body

    def json(self):
        if self._doc is None:
            raise ValueError("no JSON")
        return self._doc

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


class FakeHF:
    """Rows API over ``total`` rows + the image URLs it hands out. ``script``: answers for the next rows requests
    (int code, (code, headers), an exception or "ok"). ``expires(i)`` = the ``Expires`` of row i's URL (None: none).
    ``img_code(url)``: status of an image download (default 200)."""

    def __init__(self, total, script=(), expires=lambda i: None, img_code=lambda url: 200):
        self.total, self.script, self.expires, self.img_code = total, list(script), expires, img_code
        self.pages: list[int] = []
        self.images: list[str] = []
        self.version = 0

    def _url(self, i):
        e = self.expires(i)
        return f"https://x/{i}.jpg?v={self.version}" + (f"&Expires={int(e)}&Signature=s&Key-Pair-Id=k" if e else "")

    def __call__(self, url, params=None, timeout=None, **kw):
        if params is not None:
            assert url == H.HF_ROWS and params["length"] == H.PAGE and timeout
            self.pages.append(int(params["offset"]))
            if self.script:
                a = self.script.pop(0)
                if isinstance(a, Exception):
                    raise a
                if isinstance(a, tuple):
                    return Resp(a[0], None, a[1])
                if a != "ok":
                    return Resp(a)
            self.version += 1
            off = int(params["offset"])
            rows = [{"row_idx": i, "row": {"image": {"src": self._url(i)}, "label": i % 3}}
                    for i in range(off, min(self.total, off + 100))]
            return Resp(200, {"rows": rows, "num_rows_total": self.total})
        self.images.append(url)
        return Resp(self.img_code(url), body=b"jpeg:" + url.split("?")[0].encode())


def _pages(get, clock, **kw):
    return H.RowPages(lambda: get, sleep=clock.sleep, clock=clock, wall=lambda: 5_000_000.0, policy=P,
                      rnd=lambda: 0.0, **kw)


# ------------------------------------------------------------------------------------------- all rows

def test_hf_sets_have_no_per_class_cap():
    for name in ("hf_hemg_alphabet", "hf_akrit_alphabet"):
        assert SOURCES[name].extra["per_class"] is None


def test_hf_rows_items_takes_every_row_in_class_then_row_order():
    rows = [(i, (i * 7) % 3) for i in range(250)]
    items = L.hf_rows_items("hf_x", "u/d", "train", rows, NAMES, per_class=None)
    assert len(items) == 250 and len({i.id for i in items}) == 250
    labs = [int(i.meta["row"]) * 7 % 3 for i in items]
    assert labs == sorted(labs)  # class order
    for lab in range(3):
        rs = [i.meta["row"] for i in items if i.meta["row"] * 7 % 3 == lab]
        assert rs == sorted(rs)  # then row order
    assert items == L.hf_rows_items("hf_x", "u/d", "train", rows, NAMES, per_class=0)  # 0 = all as well
    assert items == L.hf_rows_items("hf_x", "u/d", "train", list(reversed(rows)), NAMES, per_class=None)
    assert len(L.hf_rows_items("hf_x", "u/d", "train", rows, NAMES, per_class=4)) == 12  # a cap still samples


def test_build_hf_items_uses_finished_labels_and_takes_all(tmp_path):
    spec = get_source("hf_akrit_alphabet")
    for split, n in (("train", 230), ("test", 70)):
        (tmp_path / f"labels_{split}.json").write_text(json.dumps(
            {"dataset": spec.extra["dataset"], "split": split, "names": NAMES, "num_rows_total": n,
             "rows": [[i, i % 3] for i in range(n)]}))

    def no_network(*a, **k):
        raise AssertionError("finished label lists must not be re-listed")

    items = H.build_hf_items(spec, tmp_path, get=no_network)
    assert len(items) == 300
    assert sum(1 for i in items if i.meta["split"] == "test") == 70
    assert len(H.build_hf_items(spec, tmp_path, per_class=10, get=no_network)) == 2 * 3 * 10  # --per-class N


# ------------------------------------------------------------------------------------------- RowPages

def test_one_request_per_page_paced():
    get, clock = FakeHF(250), Clock()
    rp = _pages(get, clock)
    urls = [rp.src("d", "train", r) for r in range(250)]
    assert get.pages == [0, 100, 200] and len(set(urls)) == 250
    assert clock.waits == [] or all(w <= 2.0 for w in clock.waits)
    # requests back to back are paced to >= 2 s apart
    get2, clock2 = FakeHF(1000), Clock()
    rp2 = _pages(get2, clock2)
    for off in (0, 100, 200):
        rp2.src("d", "train", off)
    assert get2.pages == [0, 100, 200] and clock2.waits == [2.0, 2.0]
    # two splits interleaved (akrit train / test in row order) still cost one request per page
    get3, clock3 = FakeHF(300), Clock()
    rp3 = _pages(get3, clock3)
    for r in range(150):
        rp3.src("d", "train", r)
        rp3.src("d", "test", r)
    assert get3.pages == [0, 0, 100, 100]


def test_backoff_and_retry_after_like_the_listing():
    get, clock = FakeHF(150, script=[(429, {"Retry-After": "90"}), 503, ConnectionError("reset")]), Clock()
    rp = _pages(get, clock)
    assert rp.src("d", "train", 5).startswith("https://x/5.jpg")
    assert get.pages == [0, 0, 0, 0]
    assert clock.waits == [90.0, 10.0, 20.0]  # Retry-After 90 (> 5 s backoff), then backoff 10, 20
    assert rp.stats["retries"] == 3 and rp.stats["pages"] == 1


def test_keeps_refusing_raises_remote_refused():
    get, clock = FakeHF(150, script=[503] * 4), Clock()
    with pytest.raises(H.RemoteRefused) as ei:
        _pages(get, clock).src("d", "train", 5)
    assert not ei.value.permanent and "4 failed tries" in str(ei.value) and len(get.pages) == 4
    # a Retry-After above max_retry_after stops at once (no hour-long sleep)
    get, clock = FakeHF(150, script=[(429, {"Retry-After": "7200"})]), Clock()
    with pytest.raises(H.RemoteRefused, match="Retry-After 7200"):
        _pages(get, clock).src("d", "train", 5)
    assert clock.waits == [] and len(get.pages) == 1
    # a 404 is permanent
    get, clock = FakeHF(150, script=[404]), Clock()
    with pytest.raises(H.RemoteRefused) as ei:
        _pages(get, clock).src("d", "train", 5)
    assert ei.value.permanent and len(get.pages) == 1


def test_page_lives_until_its_urls_expire_and_refresh_rules():
    wall = 5_000_000.0
    get, clock = FakeHF(150, expires=lambda i: wall + 3600), Clock()
    rp = _pages(get, clock, expiry_margin=300.0)
    u1 = rp.src("d", "train", 1)
    clock.t += 3000  # 50 min later: still valid (expires in 60 min, margin 5 min)
    assert rp.src("d", "train", 2) and get.pages == [0]
    clock.t += 400  # 56.7 min: inside the margin -> the page is requested again
    u1b = rp.src("d", "train", 1)
    assert get.pages == [0, 0] and u1b != u1
    # refresh of a page fetched seconds ago does not re-request it (that image's own failure)
    assert rp.src("d", "train", 1, refresh=True) == u1b and get.pages == [0, 0]
    clock.t += 200
    assert rp.src("d", "train", 1, refresh=True) != u1b and get.pages == [0, 0, 0]
    # no Expires: the fallback ttl applies
    get, clock = FakeHF(150), Clock()
    rp = _pages(get, clock, ttl=600.0)
    rp.src("d", "train", 1)
    clock.t += 601
    rp.src("d", "train", 1)
    assert get.pages == [0, 0]


def test_missing_row_and_lru():
    get, clock = FakeHF(150), Clock()
    rp = _pages(get, clock, max_pages=1)
    with pytest.raises(OSError, match="no image URL"):
        rp.src("d", "train", 160)  # past the end of the split: page 100 holds rows 100-149 only
    rp.src("d", "train", 1)
    rp.src("d", "train", 101)
    rp.src("d", "train", 2)  # page 0 was evicted (max_pages 1)
    assert get.pages == [100, 0, 100, 0]
    rp.clear()
    assert rp.pages == {}


# ------------------------------------------------------------------------------------------- engine

def _items(n, split="train"):
    return [L.Item(f"hf_hemg/{split}/{r}", "A", {"kind": "hfrow", "dataset": "d", "split": split, "row": r}, ".jpg",
                   order=r) for r in range(n)]


def _fake_extract(mode, src, out, step_fps, max_s):
    return {"ok": True, "n_frames": 1, "hands_frac": 1.0}


def test_engine_fetches_each_page_once_and_image_failures_stay_retry_pending(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("hf_hemg_alphabet")
    get = FakeHF(250, img_code=lambda url: 503 if url.startswith("https://x/7.jpg") else 200)
    clock = Clock()
    f = Fetcher(get=get, sleep=clock.sleep, clock=clock, hf_policy=P, rnd=lambda: 0.0)
    st = run(spec, list(reversed(_items(250))), paths, f, extract=_fake_extract, min_free_gb=0)
    assert get.pages == [0, 100, 200]  # sorted by row: one request per page, not per item
    assert st["ok"] == 249 and st["fetch_failed"] == 1 and st["stopped"] is None
    bad = [r for r in read_manifest(paths.manifest("hf_hemg_alphabet"), only_ok=False) if not r["ok"]]
    assert len(bad) == 1 and bad[0]["id"] == "hf_hemg/train/7" and bad[0]["error"].startswith("fetch: OSError")
    # the next run retries only the failed row (and its page once)
    get.img_code = lambda url: 200
    st2 = run(spec, _items(250), paths, f, extract=_fake_extract, min_free_gb=0)
    assert st2["todo"] == 1 and st2["ok"] == 1


def test_engine_stops_on_remote_refusal_without_failing_every_item(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("hf_hemg_alphabet")
    get = FakeHF(400, script=["ok"] + [503] * 4)  # page 0 answers, page 100 is refused 4 times in a row
    clock = Clock()
    f = Fetcher(get=get, sleep=clock.sleep, clock=clock, hf_policy=P, rnd=lambda: 0.0)
    st = run(spec, _items(400), paths, f, extract=_fake_extract, min_free_gb=0)
    assert st["stopped"] == "remote" and not st["remote_permanent"] and "503" in st["remote_error"]
    assert st["ok"] == 100 and st["fetch_failed"] == 1
    rows = read_manifest(paths.manifest("hf_hemg_alphabet"), only_ok=False)
    assert len(rows) == 101  # rows 101-399 were not touched
    failed = [r for r in rows if not r["ok"]]
    assert failed[0]["id"] == "hf_hemg/train/100" and failed[0]["error"].startswith("fetch: RemoteRefused")
    m = _script()
    assert m.remote_rc(st) == m.EXIT_MORE  # progress this run: the queue goes on with the next chunk
    assert m.remote_rc({**st, "ok": 0}) == m.EXIT_ERR  # none: the queue counts a try and waits
    assert m.remote_rc({**st, "remote_permanent": True}) == m.EXIT_ERR


def test_memory_guard_drops_the_page_cache(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("hf_hemg_alphabet")
    get, clock = FakeHF(50), Clock()
    f = Fetcher(get=get, sleep=clock.sleep, clock=clock, hf_policy=P, rnd=lambda: 0.0)
    st = run(spec, _items(3), paths, f, extract=_fake_extract, min_free_gb=0, max_rss_gb=1.0, rss=lambda pid: 2.0)
    assert st["stopped"] == "memory" and f.rows.pages == {}


def _script():
    spec = importlib.util.spec_from_file_location("ingest_isl_video_hf2", REPO_ROOT / "scripts" / "ingest_isl_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------- OpenHands headers

def _msg(mtype: int, data: bytes) -> bytes:
    return struct.pack("<HHB3x", mtype, len(data), 0) + data


def _header(*msgs: bytes, pad: int = 64) -> bytes:
    body = b"".join(msgs)
    return struct.pack("<BBHII", 1, 0, len(msgs), 1, len(body)) + b"\0" * 4 + body + b"\0" * pad


def _good() -> list[bytes]:
    space = bytes([1, 2, 0, 0]) + b"\0" * 4 + struct.pack("<2Q", 10, 3)
    dtype = bytes([0x11, 0x20, 0x1F, 0x00]) + struct.pack("<I", 4)
    layout = bytes([3, 1]) + struct.pack("<QQ", 4096, 120)
    return [_msg(0x0001, space), _msg(0x0003, dtype), _msg(0x0008, layout)]


def test_parse_v1_header_good_header_still_parses():
    h = openhands.parse_v1_header(_header(*_good()))
    assert h["shape"] == (10, 3) and h["itemsize"] == 4 and h["address"] == 4096 and h["size"] == 120
    assert openhands.data_nbytes(h) == 120


@pytest.mark.parametrize("bad", [
    _header(_msg(0x0001, b"")),  # empty dataspace message -> IndexError
    _header(_msg(0x0001, bytes([1, 200, 0, 0]) + b"\0" * 12)),  # rank 200, 2 dims of data -> struct.error
    _header(_msg(0x0003, bytes([0x11, 0x20]))),  # datatype message without its size field
    _header(_msg(0x0008, bytes([3, 1, 0, 0]))),  # contiguous layout cut short
    _header(_msg(0x0008, b"")),  # empty layout message
])
def test_parse_v1_header_malformed_is_header_error(bad):
    with pytest.raises(openhands.HeaderError, match="malformed header"):
        openhands.parse_v1_header(bad)


def test_data_nbytes_does_not_wrap_on_absurd_shapes():
    space = bytes([1, 2, 0, 0]) + b"\0" * 4 + struct.pack("<2Q", 2 ** 40, 2 ** 40)  # np.prod would wrap to 0
    msgs = _good()
    msgs[0] = _msg(0x0001, space)
    msgs[2] = _msg(0x0008, bytes([3, 1]) + struct.pack("<QQ", 4096, 0))
    h = openhands.parse_v1_header(_header(*msgs))
    with pytest.raises(openhands.HeaderError, match="layout size 0"):
        openhands.data_nbytes(h)
