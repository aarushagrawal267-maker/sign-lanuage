"""Offline tests of the Hugging Face alphabet listing (``app.isl.data.video_sources.hf_listing``): backoff,
``Retry-After``, partial progress saved after every page, resume, validation, status file and the script's exit
codes. A fake rows API stands in for datasets-server; no network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.core.config import REPO_ROOT
from app.isl.data.video_sources import hf_listing as H
from app.isl.data.video_sources.catalog import RESEARCH, SOURCES
from app.isl.profiles import COMMERCIAL, LETTER_SOURCES, PROFILES

NAMES = ["1", "a", "B"]
FAST = H.Policy(min_interval=2.0, base=5.0, max_wait=60.0, max_failures=4, max_retry_after=900.0, jitter=0.0)


class Resp:
    def __init__(self, code: int, doc: dict | None = None, headers: dict | None = None):
        self.status_code, self._doc, self.headers = code, doc, headers or {}

    def json(self):
        if self._doc is None:
            raise ValueError("no JSON")
        return self._doc


class FakeRows:
    """datasets-server /rows over ``total`` rows (label = row % 3). ``script`` = answers to give first, in order:
    an int status code, a (code, headers) pair, an exception instance or "ok"; then every request succeeds."""

    def __init__(self, total: int, script=(), names=NAMES, label=lambda i: i % 3, bad_idx_at: int | None = None):
        self.total, self.script, self.names, self.label = total, list(script), names, label
        self.calls: list[int] = []
        self.bad_idx_at = bad_idx_at

    def page(self, off: int, length: int) -> dict:
        rows = [{"row_idx": i, "row": {"image": {"src": f"https://x/{i}.jpg"}, "label": self.label(i)}}
                for i in range(off, min(self.total, off + length))]
        if self.bad_idx_at is not None and off == self.bad_idx_at and rows:
            rows[0]["row_idx"] += 1
            self.bad_idx_at = None
        return {"features": [{"name": "image", "type": {"_type": "Image"}},
                             {"name": "label", "type": {"names": self.names, "_type": "ClassLabel"}}],
                "rows": rows, "num_rows_total": self.total, "num_rows_per_page": 100, "partial": False}

    def __call__(self, url, params=None, timeout=None):
        assert url == H.HF_ROWS and params["length"] == H.PAGE and timeout
        self.calls.append(int(params["offset"]))
        if self.script:
            a = self.script.pop(0)
            if isinstance(a, Exception):
                raise a
            if isinstance(a, tuple):
                return Resp(a[0], None, a[1])
            if a != "ok":
                return Resp(a)
        return Resp(200, self.page(int(params["offset"]), int(params["length"])))


class Clock:
    """Fake monotonic clock: ``sleep`` advances it and records every wait."""

    def __init__(self):
        self.t, self.waits = 1000.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.waits.append(round(s, 3))
        self.t += s


def _list(tmp_path: Path, get, clock=None, policy=FAST, status=None):
    clock = clock or Clock()
    return H.list_split("u/d", "train", tmp_path / "labels_train.json", tmp_path / "labels_train.partial.json",
                        get=get, sleep=clock.sleep, clock=clock, policy=policy, status=status, rnd=lambda: 0.0)


# ------------------------------------------------------------------------------------------- helpers

def test_retry_after_parsing():
    assert H.retry_after_s({"Retry-After": "120"}) == 120.0
    assert H.retry_after_s({"retry-after": " 7 "}) == 7.0
    assert H.retry_after_s({"Retry-After": "Wed, 25 Nov 2026 20:00:30 GMT"},
                           now=H.email.utils.parsedate_to_datetime("Wed, 25 Nov 2026 20:00:00 GMT").timestamp()) == 30
    assert H.retry_after_s({"Retry-After": "Wed, 25 Nov 2026 20:00:00 GMT"}, now=4e9) == 0.0  # in the past
    assert H.retry_after_s({"Retry-After": "soon"}) is None
    assert H.retry_after_s({}) is None and H.retry_after_s(None) is None


def test_backoff_is_exponential_capped_and_never_below_retry_after():
    p = H.Policy(base=5.0, max_wait=60.0, jitter=0.2)
    assert [H.backoff_s(k, None, p, rnd=lambda: 0.0) for k in (1, 2, 3, 4, 5)] == [5, 10, 20, 40, 60]
    assert H.backoff_s(1, None, p, rnd=lambda: 1.0) == pytest.approx(6.0)  # jitter adds at most 20 %
    assert H.backoff_s(1, 300.0, p, rnd=lambda: 0.0) == 300.0  # the server's Retry-After wins


# ------------------------------------------------------------------------------------------- listing

def test_lists_all_pages_at_a_polite_rate(tmp_path):
    get, clock = FakeRows(250), Clock()
    lab = _list(tmp_path, get, clock)
    assert lab["names"] == NAMES and lab["num_rows_total"] == 250
    assert lab["rows"] == [[i, i % 3] for i in range(250)]
    assert get.calls == [0, 100, 200]
    assert clock.waits == [2.0, 2.0]  # min_interval between requests, nothing else
    assert (tmp_path / "labels_train.json").exists() and not (tmp_path / "labels_train.partial.json").exists()
    # a finished list is reused without any request
    get2 = FakeRows(250, script=[500] * 10)
    assert _list(tmp_path, get2)["rows"] == lab["rows"] and get2.calls == []


def test_429_and_503_back_off_and_honour_retry_after(tmp_path):
    get, clock = FakeRows(150, script=["ok", (429, {"Retry-After": "90"}), 503, ConnectionError("reset")]), Clock()
    lab = _list(tmp_path, get, clock)
    assert len(lab["rows"]) == 150
    assert get.calls == [0, 100, 100, 100, 100]
    # min interval, then Retry-After 90 (> backoff 5), then backoff 10 and 20 (failures 2 and 3)
    assert clock.waits == [2.0, 90.0, 10.0, 20.0]


def test_progress_is_saved_after_every_page_and_resumed(tmp_path):
    status = H.StatusFile(tmp_path / H.STATUS_FILE, "hf_x", "u/d")
    get = FakeRows(450, script=["ok", "ok", 503, 503, 503, 503])  # page 3 refused max_failures times
    with pytest.raises(H.ListingError) as ei:
        _list(tmp_path, get, status=status)
    e = ei.value
    assert e.progressed and not e.permanent and e.rows == 200 and e.total == 450
    part = json.loads((tmp_path / "labels_train.partial.json").read_text())
    assert part["rows"] == [[i, i % 3] for i in range(200)] and part["names"] == NAMES
    st = json.loads((tmp_path / H.STATUS_FILE).read_text())
    assert st["state"] == "incomplete" and st["splits"]["train"]["rows"] == 200
    assert "HTTP 503" in st["splits"]["train"]["last_error"]
    assert not (tmp_path / "labels_train.json").exists()

    # the next run starts at row 200, not 0
    get2 = FakeRows(450)
    status2 = H.StatusFile(tmp_path / H.STATUS_FILE, "hf_x", "u/d")
    lab = _list(tmp_path, get2, status=status2)
    assert get2.calls == [200, 300, 400] and lab["rows"] == [[i, i % 3] for i in range(450)]
    st = json.loads((tmp_path / H.STATUS_FILE).read_text())
    assert st["state"] == "complete" and st["splits"]["train"] == {**st["splits"]["train"], "rows": 450, "total": 450}


def test_no_progress_is_reported_as_such(tmp_path):
    with pytest.raises(H.ListingError) as ei:
        _list(tmp_path, FakeRows(100, script=[503] * 4))
    assert not ei.value.progressed and ei.value.rows == 0


def test_huge_retry_after_stops_at_once_keeping_progress(tmp_path):
    get, clock = FakeRows(300, script=["ok", (429, {"Retry-After": "7200"})]), Clock()
    with pytest.raises(H.ListingError, match="Retry-After 7200"):
        _list(tmp_path, get, clock)
    assert get.calls == [0, 100] and 7200.0 not in clock.waits
    assert len(json.loads((tmp_path / "labels_train.partial.json").read_text())["rows"]) == 100


def test_client_error_is_permanent(tmp_path):
    with pytest.raises(H.ListingError) as ei:
        _list(tmp_path, FakeRows(100, script=[404]))
    assert ei.value.permanent
    st_get = FakeRows(100, script=[404])
    status = H.StatusFile(tmp_path / "s.json", "hf_x", "u/d")
    with pytest.raises(H.ListingError):
        _list(tmp_path, st_get, status=status)
    assert json.loads((tmp_path / "s.json").read_text())["state"] == "failed"


def test_inconsistent_page_is_retried_not_stored(tmp_path):
    get = FakeRows(200, bad_idx_at=100)  # page 2 starts at row 101 once
    lab = _list(tmp_path, get)
    assert get.calls == [0, 100, 100] and lab["rows"] == [[i, i % 3] for i in range(200)]


def test_label_out_of_range_never_stored(tmp_path):
    get = FakeRows(100, label=lambda i: 7)
    with pytest.raises(H.ListingError):
        _list(tmp_path, get)
    assert json.loads((tmp_path / "labels_train.partial.json").read_text())["rows"] == []


def test_corrupt_partial_keeps_only_a_clean_prefix(tmp_path):
    p = tmp_path / "labels_train.partial.json"
    rows = [[i, i % 3] for i in range(100)] + [[150, 0]] + [[i, 0] for i in range(101, 120)]
    p.write_text(json.dumps({"dataset": "u/d", "split": "train", "names": NAMES, "rows": rows}))
    assert H.load_partial(p, "u/d", "train")["rows"] == rows[:100]
    p.write_text("{not json")
    assert H.load_partial(p, "u/d", "train")["rows"] == []
    p.write_text(json.dumps({"dataset": "other/d", "split": "train", "names": NAMES, "rows": rows[:100]}))
    assert H.load_partial(p, "u/d", "train")["rows"] == []
    # old-format partial (no dataset/split keys, written by the previous script code) is still resumed
    p.write_text(json.dumps({"names": NAMES, "rows": rows[:100]}))
    get = FakeRows(150)
    assert len(_list(tmp_path, get)["rows"]) == 150 and get.calls == [100]


def test_old_final_file_without_total_is_reused(tmp_path):
    (tmp_path / "labels_train.json").write_text(json.dumps({"names": NAMES, "rows": [[0, 1], [1, 2]]}))
    get = FakeRows(2, script=[500] * 10)
    assert _list(tmp_path, get)["rows"] == [[0, 1], [1, 2]] and get.calls == []
    (tmp_path / "labels_train.json").write_text(json.dumps({"names": NAMES, "rows": [[0, 1], [5, 2]]}))
    assert len(_list(tmp_path, FakeRows(3))["rows"]) == 3  # invalid final file: listed again


def test_class_list_change_restarts(tmp_path):
    p = tmp_path / "labels_train.partial.json"
    p.write_text(json.dumps({"dataset": "u/d", "split": "train", "names": ["x", "y"], "rows": [[0, 0]]}))
    get = FakeRows(120)
    lab = _list(tmp_path, get)
    assert lab["names"] == NAMES and get.calls[:2] == [1, 0] and len(lab["rows"]) == 120


# ------------------------------------------------------------------------------------------- whole source

def test_list_source_tries_every_split_and_builds_items(tmp_path):
    spec = SOURCES["hf_akrit_alphabet"]

    class Two(FakeRows):
        test_refusals: list[int] = []

        def __call__(self, url, params=None, timeout=None):
            if params["split"] == "test" and self.test_refusals:
                return Resp(self.test_refusals.pop(0))
            return super().__call__(url, params, timeout)

    get = Two(130)
    get.test_refusals = [503] * 4
    clock = Clock()
    with pytest.raises(H.ListingError) as ei:
        H.build_hf_items(spec, tmp_path, per_class=5, get=get, sleep=clock.sleep, clock=clock, policy=FAST)
    assert not ei.value.progressed  # train was finished, test made no progress
    assert (tmp_path / "labels_train.json").exists()
    st = H.listing_status(tmp_path)
    assert st["splits"]["train"]["state"] == "complete" and st["splits"]["test"]["state"] == "incomplete"
    assert st["state"] == "incomplete"

    items = H.build_hf_items(spec, tmp_path, per_class=5, get=Two(130), sleep=clock.sleep, clock=clock, policy=FAST)
    assert len(items) == 2 * 3 * 5  # 2 splits x 3 classes x 5 per class
    assert {i.fetch["kind"] for i in items} == {"hfrow"}
    assert all(i.id.startswith("hf_akrit/") for i in items)
    assert H.listing_status(tmp_path)["state"] == "complete"


# ------------------------------------------------------------------------------------------- script

def _script():
    spec = importlib.util.spec_from_file_location("ingest_isl_video_hf", REPO_ROOT / "scripts" / "ingest_isl_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exit_codes_for_refused_listings():
    m = _script()
    assert m._listing_rc(H.ListingError("x", progressed=True)) == m.EXIT_MORE  # resumes next chunk, no try counted
    assert m._listing_rc(H.ListingError("x", progressed=False)) == m.EXIT_ERR  # the queue counts it and backs off
    assert m._listing_rc(H.ListingError("x", progressed=True, permanent=True)) == m.EXIT_ERR
    assert m._listing_rc(OSError("disk")) is None


def test_list_command_writes_items_without_fetching(tmp_path, monkeypatch):
    from app.isl.data.video_sources.engine import Fetcher, load_items
    from app.isl.data.video_sources.listings import hf_rows_items

    m = _script()
    items = hf_rows_items("hf_akrit", "u/d", "train", [(i, i % 3) for i in range(30)], NAMES, per_class=4)
    calls = []

    def fake_build(name, paths, per_class=None):
        calls.append((name, per_class))
        return SOURCES[name], items, Fetcher()

    monkeypatch.setattr(m, "build_items", fake_build)
    monkeypatch.setattr(m, "_setup_logging", lambda paths, name: None)
    monkeypatch.setattr(m.os, "nice", lambda n: 0)
    root = tmp_path / "isl"
    assert m.main(["--root", str(root), "list", "hf_akrit_alphabet", "--per-class", "4"]) == 0
    assert calls == [("hf_akrit_alphabet", 4)]
    got = load_items(root / "manifests" / "hf_akrit_alphabet_items.json")
    assert [i.id for i in got] == [i.id for i in items]
    assert not (root / "manifests" / "hf_akrit_alphabet.jsonl").exists()  # nothing ingested

    def refused(name, paths, per_class=None):
        raise H.ListingError("HF 503", progressed=False)

    monkeypatch.setattr(m, "build_items", refused)
    assert m.main(["--root", str(root), "list", "hf_hemg_alphabet"]) == m.EXIT_ERR


# ------------------------------------------------------------------------------------------- licence / profiles

def test_hf_alphabets_are_research_only():
    for name in ("hf_akrit_alphabet", "hf_hemg_alphabet"):
        s = SOURCES[name]
        assert s.licence == "unknown" and s.profile == RESEARCH
        assert name in LETTER_SOURCES[RESEARCH] and name not in LETTER_SOURCES[COMMERCIAL]
        assert name not in PROFILES[COMMERCIAL].all_sources() and name in PROFILES[RESEARCH].all_sources()
