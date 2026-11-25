"""OpenHands raw: the one-pass forward reader (``openhands.ForwardPass``) returns exactly what the old per-chunk
random-access path returned, needs far less re-inflation on the real layout (object headers grouped, data much
later), and resumes after an interruption without redoing chunks. Synthetic zip + deflated HDF5, no network."""

from __future__ import annotations

import io
import json
import types
import zipfile

import numpy as np
import pytest

from app.isl.data.pose_sources import common, openhands, zran

h5py = pytest.importorskip("h5py")
NATIVE = zran.available()


def _make_h5(path, n_groups=6, per_group=5, seed=11):
    """OpenHands-like file: each group of chunks has its object headers written together (``create_dataset``
    without data: late allocation) and its data written afterwards, so data lies far behind its header, as in
    ISH News. Plus odd cases: keypoints only, an int dataset, a chunked dataset, data written next to its header."""
    rng = np.random.default_rng(seed)
    arrays = {}
    with h5py.File(path, "w") as f:
        k, v = f.create_group("keypoints"), f.create_group("visibility")
        for g in range(n_groups):
            names, made = [], []
            for i in range(per_group):
                name = f"vid{g}_t-{i}"
                T = int(rng.integers(20, 90))
                kp = np.cumsum(rng.normal(0, 0.01, (T, 75, 3)), axis=0) + 0.5
                kp[rng.random(T) < 0.1, 33:54] = 0
                vis = (rng.random((T, 75)) > 0.2).astype(np.float64)
                made.append((k.create_dataset(name, shape=kp.shape, dtype="f8"),
                             v.create_dataset(name, shape=vis.shape, dtype="f8"), kp, vis))
                names.append(name)
            for name, (dk, dv, kp, vis) in zip(names, made):
                dk[...] = kp
                dv[...] = vis
                arrays[name] = (kp, vis)
        kp = rng.uniform(0, 1, (30, 75, 3))
        k.create_dataset("solo-0", data=kp)  # keypoints without visibility
        arrays["solo-0"] = (kp, None)
        kp = rng.uniform(0, 1, (25, 75, 3))
        k.create_dataset("adjacent-0", data=kp)  # header and data side by side
        v.create_dataset("adjacent-0", data=np.ones((25, 75)))
        arrays["adjacent-0"] = (kp, np.ones((25, 75)))
        k.create_dataset("intkp-0", data=np.zeros((10, 75, 3), dtype=np.int32))
        v.create_dataset("intkp-0", data=np.ones((10, 75)))
        k.create_dataset("chunked-0", data=rng.uniform(0, 1, (40, 75, 3)), chunks=(8, 75, 3))
        v.create_dataset("chunked-0", data=np.ones((40, 75)))
        k.create_dataset("badvis-0", data=rng.uniform(0, 1, (12, 75, 3)))
        v.create_dataset("badvis-0", data=np.ones((12, 75), dtype=np.int16))
    return arrays


@pytest.fixture(scope="module")
def member(tmp_path_factory):
    d = tmp_path_factory.mktemp("oh")
    arrays = _make_h5(d / "x.h5")
    plain = (d / "x.h5").read_bytes()
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Chan/x.h5", plain)
    blob = b.getvalue()
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("Chan/x.h5")
    with h5py.File(d / "x.h5", "r") as hf:
        entries = openhands.list_datasets(hf)
    pairs = [(n, ke[0], ve[0] if ve else None) for n, ke, ve in openhands.iter_pairs(entries) if ke is not None]
    return types.SimpleNamespace(blob=blob, info=info, plain=plain, arrays=arrays, entries=entries, pairs=pairs)


def _stream(m, points=None, every=64 << 10, counter=None):
    def fetch(s, e):
        if counter is not None:
            counter["n"] += e - s + 1
        return m.blob[s:e + 1]

    reader = openhands.PrefetchReader(fetch, len(m.blob), block=16 << 10, ahead=0)
    start, csize, usize, method = openhands.zip_member_span(reader, m.info)
    assert method == 8
    pf = zran.PointFile(points, "Chan.zip:Chan/x.h5", csize, usize) if points else None
    return openhands.SeekableInflate(reader, start, csize, usize, every=every, chunk=16 << 10, points_file=pf,
                                     persist_every=128 << 10)


def _old_path(st, kp_addr, vis_addr):
    """The per-chunk reads of the job before the forward pass (header, data, header, data)."""
    try:
        kp = openhands.read_contiguous(st, openhands.parse_v1_header(st.read_range(kp_addr, 4096)))
        vis = (openhands.read_contiguous(st, openhands.parse_v1_header(st.read_range(vis_addr, 4096)))
               if vis_addr is not None else None)
    except (OSError, EOFError, ValueError) as e:
        return None, None, f"{type(e).__name__}: {e}"
    return kp, vis, None


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _check_identical(m, got, ref):
    assert set(got) == set(ref)
    for name, (kp, vis, err) in ref.items():
        gkp, gvis, gerr = got[name]
        assert gerr == err, name
        assert _same(gkp, kp) and _same(gvis, vis), name


def test_layout_puts_data_far_behind_headers(member):
    """The synthetic file reproduces the real pattern the old path paid for: a chunk's data starts after the
    headers of the following chunks."""
    st = _stream(member)
    behind = 0
    for name, kp_addr, _ in member.pairs:
        h = openhands.parse_v1_header(st.read_range(kp_addr, 4096)) if name.startswith("vid") else None
        if h is not None and any(a > kp_addr and a < h["address"] for _, a, _ in member.pairs):
            behind += 1
    assert behind >= 20


@pytest.mark.parametrize("window", [16 << 20, 1024])
def test_forward_pass_identical_to_old_path(member, window):
    st_old = _stream(member)
    ref = {n: _old_path(st_old, k, v) for n, k, v in member.pairs}
    assert ref["intkp-0"][2].startswith("HeaderError: unexpected datatype class 0")
    assert ref["chunked-0"][2] == "HeaderError: not a contiguous dataset"
    assert ref["badvis-0"][2].startswith("HeaderError: unexpected datatype class 0 size 2")
    assert ref["solo-0"][1] is None and ref["solo-0"][2] is None

    st = _stream(member)
    fp = openhands.ForwardPass(st, member.pairs, window=window, jump=1 << 30)
    got = {}
    for name, kp, vis, err in fp:
        assert name not in got
        got[name] = (kp, vis, None if err is None else f"{type(err).__name__}: {err}")
    _check_identical(member, got, ref)
    # finished chunks keep no arrays in the pass (a leak of 0.3 MB per chunk hit the live job's 1.5 GB cap)
    assert all(c["kp"] is None and c["vis"] is None for c in fp.chunks.values() if c["done"])
    assert not fp.ready and not fp.active and not fp.heap
    for name, (kp, vis) in member.arrays.items():
        np.testing.assert_array_equal(got[name][0], kp)
    if window > 1024:
        assert fp.stats["fallback"] == 0
        # one forward pass: about the member size inflated, versus several restores per chunk before
        assert fp.stats["inflated"] <= st.usize
        assert st.restores <= 2 < st_old.restores
    else:
        assert fp.stats["fallback"] >= 1  # side-by-side header + data behind a tiny window: old path, same bytes


@pytest.mark.skipif(not NATIVE, reason="no libz")
def test_forward_pass_jumps_over_unwanted_ranges(member, tmp_path):
    """A pass that wants only late chunks restores at a persisted access point instead of inflating from 0."""
    pts = tmp_path / "x.points"
    st0 = _stream(member, points=pts)
    st0.read_range(0, st0.usize)  # an earlier run (the listing pass) persisted points over the whole member
    late = member.pairs[-12:]
    st_old = _stream(member)
    ref = {n: _old_path(st_old, k, v) for n, k, v in late}
    st = _stream(member, points=pts)
    assert st.points_loaded >= 3
    fp = openhands.ForwardPass(st, late, window=4096, jump=32 << 10)
    got = {n: (kp, vis, None if e is None else f"{type(e).__name__}: {e}") for n, kp, vis, e in fp}
    _check_identical(member, got, ref)
    assert fp.stats["jumps"] >= 1
    first = min(min(k, v or k) for _, k, v in late)
    assert fp.stats["inflated"] < st.usize - first + st.persist_every


def test_header_cache_roundtrip_and_damaged_tail(tmp_path):
    p = tmp_path / "m.headers.jsonl"
    hc = openhands.HeaderCache(p, "Z.zip:a.h5", 1000, flush_every=1)
    hdr = {"shape": (3, 75, 3), "itemsize": 8, "dclass": 1, "little": True, "layout": 1, "address": 500, "size": 5400}
    hc.add(10, hdr)
    hc.add(20, "not a contiguous dataset")
    with p.open("a") as fh:
        fh.write('{"a": 30, "h": {"sha')  # line cut by a kill
    back = openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load()
    assert back.items == {10: hdr, 20: "not a contiguous dataset"}
    back.add(40, "x")
    back.flush()
    assert set(openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load().items) == {10, 20, 40}
    assert openhands.HeaderCache(p, "Z.zip:other.h5", 1000).load().items == {}  # another member: ignored


# --------------------------------------------------------------------------------------------- the job's member loop

def _script():
    import importlib.util

    from app.core.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location("ingest_isl_pose_fwd", REPO_ROOT / "scripts" / "ingest_isl_pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Guard:
    def __init__(self, stop_after=None):
        self.n, self.stop_after = 0, stop_after

    def check(self, force=False):
        self.n += 1
        if self.stop_after is not None and self.n > self.stop_after:
            raise common.DiskFull("test stop")


def _old_member_loop(mod, args, st, entries, channel, member, man, out_dir):
    """The job's per-chunk loop before the forward pass (reference for the records and files it wrote)."""
    stride = common.stride_for(openhands.ASSUMED_FPS, args.target_fps)
    for name, kp_e, vis_e in openhands.iter_pairs(entries):
        if kp_e is None:
            continue
        out = out_dir / channel / (common.safe_name(name) + ".npz")
        rec = {"id": f"openhands_raw/{channel}/{name}", "source": "openhands_raw", "gloss": "", "channel": channel,
               "video": openhands.video_key(name), "member": member, "path": mod.rel(out), "licence": openhands.LICENCE}
        try:
            kp = openhands.read_contiguous(st, openhands.parse_v1_header(st.read_range(kp_e[0], 4096)))
            vis = (openhands.read_contiguous(st, openhands.parse_v1_header(st.read_range(vis_e[0], 4096)))
                   if vis_e is not None else None)
            asp = common.hand_aspect(openhands.hands_nan(kp[:: max(1, len(kp) // 1500)]))
            aspect = openhands.snap_aspect(asp[0] if asp else None) or 16 / 9
            seq = openhands.openhands_to_raw(kp, vis, aspect=aspect)
            saved = common.save_compact(seq, out, dtype=args.dtype, stride=stride)
            rec.update(common.seq_stats(saved))
            rec.update({"ok": True, "fps": round(openhands.ASSUMED_FPS / stride, 3), "fps_native": openhands.ASSUMED_FPS,
                        "fps_assumed": True, "n_frames_native": len(seq), "stride": stride, "dtype": args.dtype,
                        "aspect_est": asp[0] if asp else None, "aspect": round(aspect, 4), "width": saved.width,
                        "height": saved.height, "extractor_version": openhands.EXTRACTOR_VERSION})
        except openhands.HeaderError as e:
            rec.update({"ok": False, "error": f"HeaderError: {e}"[:300]})
        except (OSError, EOFError, ValueError) as e:
            rec.update({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        rec["secs"] = 0.0
        man.add(rec)


def _records(path):
    out = {}
    for ln in path.read_text().splitlines():
        r = json.loads(ln)
        r.pop("secs")
        r["path"] = r["path"].split("/", 1)[1]  # old_out/... vs new_out/...
        out.setdefault(r["id"], []).append(r)
    return out


def _npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].tobytes() + str(z[k].dtype).encode() + str(z[k].shape).encode() for k in z.files}


def test_member_loop_identical_records_and_files_then_resume(member, tmp_path, monkeypatch):
    mod = _script()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "MEM", None)
    args = types.SimpleNamespace(target_fps=common.DEFAULT_TARGET_FPS, dtype="float16", convert_workers=2)
    # reference: the old loop
    old_man = mod.Manifest(tmp_path / "old.jsonl")
    _old_member_loop(mod, args, _stream(member), member.entries, "Chan", "Chan/x.h5", old_man, tmp_path / "old_out")

    # new loop, stopped after 9 chunks (DiskFull is how the job stops cleanly), then resumed
    pts, hdrs = tmp_path / "x.points", tmp_path / "x.headers.jsonl"
    new_path = tmp_path / "new.jsonl"
    made = []

    class Spy(openhands.ForwardPass):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(openhands, "ForwardPass", Spy)
    with pytest.raises(common.DiskFull):
        mod._openhands_member(args, _stream(member, points=pts), member.entries, "Chan", "Chan.zip:Chan/x.h5",
                              "Chan/x.h5", mod.Manifest(new_path), tmp_path / "new_out", _Guard(stop_after=9), {},
                              headers_file=hdrs)
    first = _records(new_path)
    assert 9 <= len(first) < len(member.pairs)
    assert hdrs.exists()
    man2 = mod.Manifest(new_path)
    n2 = mod._openhands_member(args, _stream(member, points=pts), member.entries, "Chan", "Chan.zip:Chan/x.h5",
                               "Chan/x.h5", man2, tmp_path / "new_out", _Guard(), {}, headers_file=hdrs)
    new, old = _records(new_path), _records(tmp_path / "old.jsonl")
    ok_first = {i for i, rs in first.items() if rs[-1]["ok"]}
    # nothing done twice: chunks finished before the stop are not redone (failed ones are retried, as before)
    assert all(len(rs) == 1 for i, rs in new.items() if i in ok_first)
    assert n2 == len(member.pairs) - len(ok_first)
    assert set(new) == set(old)
    for cid, rs in old.items():
        assert new[cid][-1] == rs[-1], cid  # same record, same key order
        assert list(new[cid][-1]) == list(rs[-1])
        if rs[-1]["ok"]:
            a = _npz(tmp_path / "old_out" / rs[-1]["path"])
            b = _npz(tmp_path / "new_out" / rs[-1]["path"])
            assert a == b, cid
    # the resumed pass reused the parsed headers and did not start from the member's first byte
    assert made[1].stats["header_cache_hits"] > 0
    assert made[1].stats["inflated"] < made[0].st.usize


def test_range_fetch_retries_a_broken_connection(monkeypatch):
    """A connection cut mid-body (urllib3 ``ProtocolError`` / IncompleteRead, not wrapped by requests) is retried;
    uncaught it killed the live job at 21:01 UTC on 25 Nov 2026."""
    from urllib3.exceptions import ProtocolError

    monkeypatch.setattr(openhands.time, "sleep", lambda s: None)
    body = bytes(range(256)) * 40
    calls = {"n": 0}

    class Raw:
        def __init__(self, broken):
            self.broken = broken

        def stream(self, n, decode_content=False):
            yield body[:100]
            if self.broken:
                raise ProtocolError("Connection broken: IncompleteRead(100 bytes read, 10140 more expected)")
            yield body[100:]

    class Resp:
        status_code = 206
        headers: dict = {}

        def __init__(self, broken):
            self.raw = Raw(broken)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Session:
        def get(self, url, **kw):
            calls["n"] += 1
            return Resp(broken=calls["n"] == 1)

    assert openhands.range_fetch("https://x/y.zip", 0, len(body) - 1, session=Session()) == body
    assert calls["n"] == 2


# --------------------------------------------------------------------------------------------- adversarial review

def test_prefetch_reader_refetches_a_block_whose_download_failed():
    """A failed download must not stay cached: with read-ahead the failed Future was kept, so every later read
    of that block re-raised the same error and the forward pass's 'restore and retry' could never succeed."""
    data = bytes(range(256)) * 64
    calls: dict[int, int] = {}

    def fetch(s, e):
        calls[s] = calls.get(s, 0) + 1
        if s == 1024 and calls[s] == 1:
            raise OSError("range fetch failed after 10 tries")
        return data[s:e + 1]

    r = openhands.PrefetchReader(fetch, len(data), block=1024, ahead=2)
    try:
        r.read_at(0, 10)  # sequential: blocks 1 and 2 are read ahead; block 1 fails
        with pytest.raises(OSError):
            r.read_at(1024, 10)
        assert r.read_at(1024, 10) == data[1024:1034]
        assert calls[1024] == 2
    finally:
        r.close()


@pytest.mark.parametrize("ahead", [0, 2])
def test_forward_pass_survives_network_outages(member, monkeypatch, ahead):
    """Downloads that fail (after range_fetch's own retries), including the reads of a restore, are retried by
    the pass with the read-ahead reader the job uses; the result is identical to the old path."""
    monkeypatch.setattr(openhands.time, "sleep", lambda s: None)
    ref = {n: _old_path(_stream(member), k, v) for n, k, v in member.pairs}
    seen: dict[int, int] = {}

    def fetch(s, e):
        seen[s] = seen.get(s, 0) + 1
        if (s // (16 << 10)) % 4 == 1 and seen[s] == 1:  # first download of every 4th block fails
            raise OSError("range fetch failed after 10 tries (simulated)")
        return member.blob[s:e + 1]

    span = openhands.zip_member_span(openhands.PrefetchReader(lambda s, e: member.blob[s:e + 1], len(member.blob),
                                                              block=16 << 10, ahead=0), member.info)
    reader = openhands.PrefetchReader(fetch, len(member.blob), block=16 << 10, ahead=ahead)
    try:
        start, csize, usize, _ = span
        st = openhands.SeekableInflate(reader, start, csize, usize, every=64 << 10, chunk=16 << 10)
        fp = openhands.ForwardPass(st, member.pairs, window=1 << 20, jump=1 << 30, max_stream_errors=2,
                                   forgive=64 << 10)
        got = {}
        for name, kp, vis, err in fp:
            got[name] = (kp, vis, None if err is None else f"{type(err).__name__}: {err}")
    finally:
        reader.close()
    if ahead == 0:  # (with read-ahead, how many failures reach the pass depends on thread timing)
        # more outages than allowed in a row, all survived: each was retried, clean progress forgave the last one
        assert fp.stats["stream_errors"] > fp.max_stream_errors
    _check_identical(member, got, ref)


def test_header_cache_line_cut_just_before_its_newline(tmp_path):
    """A kill between a line's last byte and its newline: the line is dropped, so the next append starts on a
    fresh line instead of running into it (which lost every later line on the following load)."""
    p = tmp_path / "m.headers.jsonl"
    hc = openhands.HeaderCache(p, "Z.zip:a.h5", 1000, flush_every=1)
    hc.add(10, "a")
    hc.add(20, "b")
    p.write_bytes(p.read_bytes()[:-1])  # '{"a":20,"e":"b"}' without its newline
    back = openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load()
    assert back.items == {10: "a"}
    back.add(30, "c")
    back.add(40, "d")
    back.flush()
    assert openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load().items == {10: "a", 30: "c", 40: "d"}
    p.write_bytes(p.read_bytes().split(b"\n")[0])  # only the head line, cut before its newline
    again = openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load()
    assert again.items == {}
    again.add(50, "e")
    again.flush()
    assert openhands.HeaderCache(p, "Z.zip:a.h5", 1000).load().items == {50: "e"}


def test_forward_pass_does_not_keep_the_header_cache_in_memory(member, tmp_path):
    """Loaded headers are queued and then dropped (0.9 KB each); new ones still reach the file."""
    hp = tmp_path / "x.headers.jsonl"
    st = _stream(member)
    half = member.pairs[: len(member.pairs) // 2]
    list(openhands.ForwardPass(st, half, headers=openhands.HeaderCache(hp, "k", st.usize).load()))
    hc = openhands.HeaderCache(hp, "k", st.usize).load()
    n_before = hc.loaded
    assert n_before > 0
    fp = openhands.ForwardPass(_stream(member), member.pairs, headers=hc)
    assert hc.items == {} and fp.stats["header_cache_hits"] > 0
    got = {name: err for name, _, _, err in fp}
    assert set(got) == {n for n, _, _ in member.pairs}
    assert len(openhands.HeaderCache(hp, "k", st.usize).load().items) > n_before
