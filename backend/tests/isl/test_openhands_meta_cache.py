"""OpenHands raw: ``SeekableInflate``'s metadata cache files every page under the offset it was inflated at, however
the stream moves (forward jumps through persisted restore points, backward restores, error-handler restores).

Regression for the 22:06 UTC crash of 25 Nov 2026 (ISL_DESIGN 15.10): a restarted job loaded persisted NewzHook
restore points, h5py's listing seeked far ahead (``_goto`` restored at a point beyond the scanned range) and the
newly inflated bytes were filed as metadata pages at the *old* scan position; h5py then read wrong bytes for early
metadata ("bad symbol table node signature"). Synthetic zip + deflated HDF5, no network."""

from __future__ import annotations

import io
import itertools
import random
import types
import zipfile

import numpy as np
import pytest

from app.isl.data.pose_sources import openhands, zran

h5py = pytest.importorskip("h5py")
NATIVE = zran.available()
needs_native = pytest.mark.skipif(not NATIVE, reason="libz (zran) not available")
KEY = "Chan.zip:Chan/x.h5"
EVERY = 32 << 10
MPAGE = openhands.SeekableInflate.MPAGE


def _make_h5(path, n_chunks=260, seed=5):
    """OpenHands-like layout with metadata spread over the whole file: datasets of ``keypoints`` / ``visibility``
    are created one chunk at a time with their data written in between, so the groups' symbol-table B-tree nodes
    and local heaps grow far from the superblock. Extra groups with attributes and sub-groups add more metadata;
    a few all-zero datasets give all-zero pages."""
    rng = np.random.default_rng(seed)
    arrays = {}
    with h5py.File(path, "w") as f:
        k, v = f.create_group("keypoints"), f.create_group("visibility")
        extra = [f.create_group(f"extra{i}") for i in range(6)]
        for c in range(n_chunks):
            name = f"yt{c:04d}_Some_Title_{rng.integers(1e6)}-{c % 7}"
            T = int(rng.integers(6, 22))
            kp = np.round(np.cumsum(rng.normal(0, 0.01, (T, 75, 3)), axis=0) + 0.5, 4)
            if c % 37 == 0:
                kp[:] = 0  # all-zero chunk
            vis = (rng.random((T, 75)) > 0.2).astype(np.float64)
            k.create_dataset(name, data=kp)
            v.create_dataset(name, data=vis)
            arrays[name] = (kp, vis)
            if c % 9 == 0:
                g = extra[c % len(extra)].create_group(f"sub{c}")
                g.attrs["note"] = f"chunk {c} " * int(rng.integers(1, 30))
                g.create_dataset("ids", data=rng.integers(0, 1 << 40, int(rng.integers(5, 200))))
    return arrays


@pytest.fixture(scope="module")
def member(tmp_path_factory):
    d = tmp_path_factory.mktemp("ohmeta")
    arrays = _make_h5(d / "x.h5")
    plain = (d / "x.h5").read_bytes()
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Chan/x.h5", plain)
    blob = b.getvalue()
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("Chan/x.h5")
    with h5py.File(d / "x.h5", "r") as hf:
        entries = openhands.list_datasets(hf)
    m = types.SimpleNamespace(blob=blob, info=info, plain=plain, arrays=arrays, entries=entries, dir=d,
                              points=[], n=itertools.count())
    if NATIVE:  # every access point of one full forward pass, persisted (what the crashed run had on disk)
        st = _stream(m, points=None, save_all=True)
        st.read_range(0, len(plain))
        m.points = zran.PointFile(d / "all.points", KEY, st.csize, st.usize).load()
        assert len(m.points) > 30  # one per deflate block boundary at least EVERY apart
    pages = [k for k in range(len(plain) // MPAGE) if openhands.looks_like_metadata(plain[k * MPAGE:(k + 1) * MPAGE])]
    m.meta_pages = pages
    assert len(pages) > 30 and pages[-1] * MPAGE > len(plain) // 2  # metadata really is spread over the file
    return m


def _stream(m, points=None, save_all=False, native=None, **kw):
    """A stream over the member; ``points`` = persisted access points to load (a list), None = none."""
    r = openhands.PrefetchReader(lambda s, e: m.blob[s:e + 1], len(m.blob), block=64 << 10, ahead=0)
    start, cs, us, _ = openhands.zip_member_span(r, m.info)
    pf = None
    if NATIVE and native is not False and (points is not None or save_all):
        path = m.dir / ("all.points" if save_all else f"p{next(m.n)}.points")
        pf = zran.PointFile(path, KEY, cs, us)
        for o, i, b, w in points or []:
            pf.add(o, i, b, w)
    kw.setdefault("every", EVERY)
    kw.setdefault("chunk", 16 << 10)
    return openhands.SeekableInflate(r, start, cs, us, points_file=pf, persist_every=EVERY if save_all else 1 << 40,
                                     native=native, **kw)


def _check_invariants(st, plain):
    """Every cached page holds the bytes of its own offset; the stage continues the stream exactly."""
    for k, pg in st.meta.items():
        assert pg == plain[k * MPAGE:(k + 1) * MPAGE], f"meta page {k} holds bytes of another offset"
    for k in st.zero_pages:
        assert plain[k * MPAGE:(k + 1) * MPAGE] == bytes(MPAGE), f"page {k} is not all zero"
    if st.stage_at is not None and st.meta_cap > 0:
        assert st.stage_at % MPAGE == 0 and st.stage_at + len(st.stage) == st.out
        assert bytes(st.stage) == plain[st.stage_at:st.out]
    assert all(a < b for a, b in zip(st._sc_lo, st._sc_hi))
    assert all(b < a for a, b in zip(st._sc_lo[1:], st._sc_hi[:-1]))  # disjoint, not touching (merged)


def _subsets(points):
    rnd = random.Random(7)
    n = len(points)
    return {
        "none": [],
        "all": points,
        "first_10pct": points[: n // 10],  # restart early in a listing (NewzHook: 158 points up to 16.8 of 159 GB)
        "first_half": points[: n // 2],
        "far_ahead_only": points[2 * n // 3:],
        "every_3rd": points[::3],
        "random": sorted(rnd.sample(points, n // 4)),
    }


def _listing(st):
    hf = h5py.File(st, "r")
    try:
        return openhands.list_datasets(hf)
    finally:
        hf.close()


@needs_native
def test_listing_with_persisted_points_matches_listing_without(member):
    base = _stream(member)
    ref = _listing(base)
    assert ref == member.entries
    _check_invariants(base, member.plain)
    for label, pts in _subsets(member.points).items():
        st = _stream(member, points=pts)
        assert st.points_loaded == len(pts), label
        assert _listing(st) == ref, label
        _check_invariants(st, member.plain)


@needs_native
def test_forward_jump_then_early_metadata_reads(member):
    """The scratch probe of 15.10: inflate the front, jump far ahead through a persisted point, then read early
    metadata pages that the cache serves (wrong bytes before the fix)."""
    plain = member.plain
    st = _stream(member, points=member.points)
    st.seek(0)
    assert st.read(4096) == plain[:4096]
    mid = len(plain) // 2
    st.seek(mid)
    assert st.read(4096) == plain[mid:mid + 4096]
    assert st.restores >= 2  # the forward seek restored at a persisted point ahead of the scanned range
    hits = 0
    for k in member.meta_pages:
        p = k * MPAGE + 7
        hits += st._from_meta(p, 300) is not None
        st.seek(p)
        assert st.read(300) == plain[p:p + 300], f"page {k}"
    assert hits  # some reads really came from the metadata cache
    _check_invariants(st, plain)


@needs_native
def test_random_metadata_reads_identical_with_and_without_points(member):
    plain = member.plain
    rnd = random.Random(11)
    a, b = _stream(member), _stream(member, points=member.points)
    for _ in range(400):
        if rnd.random() < 0.7:
            p = rnd.choice(member.meta_pages) * MPAGE + rnd.randrange(-600, MPAGE)
        else:
            p = rnd.randrange(len(plain))
        p = max(0, min(p, len(plain) - 1))
        n = rnd.choice((8, 16, 40, 328, 544, 4096, 9000, 3 * MPAGE + 5))
        a.seek(p)
        b.seek(p)
        ra, rb = a.read(n), b.read(n)
        assert ra == rb == plain[p:p + n], (p, n)
    _check_invariants(a, plain)
    _check_invariants(b, plain)


def _fuzz_one(member, seed, native):
    plain, usize = member.plain, len(member.plain)
    rnd = random.Random(seed)
    pts = []
    if native and member.points:
        kind = rnd.choice(("none", "all", "prefix", "suffix", "sample"))
        n = len(member.points)
        pts = {"none": [], "all": member.points, "prefix": member.points[: rnd.randrange(n)],
               "suffix": member.points[rnd.randrange(n):],
               "sample": sorted(rnd.sample(member.points, rnd.randrange(1, n)))}[kind]
    st = _stream(member, points=pts, native=native, every=rnd.choice((EVERY, 3 * EVERY, 97_000)),
                 chunk=rnd.choice((4 << 10, 16 << 10, 64 << 10)))
    for _ in range(60):
        op = rnd.random()
        cur = st.out
        # jump patterns: small / large forward, small / large backward, anywhere, near the end
        p = rnd.choice((cur + rnd.randrange(1, 3 * MPAGE), cur + rnd.randrange(EVERY, usize // 2 + EVERY + 1),
                        cur - rnd.randrange(1, 3 * MPAGE), cur - rnd.randrange(EVERY, usize // 2 + EVERY + 1),
                        rnd.randrange(usize), usize - rnd.randrange(1, 5 * MPAGE)))
        p = max(0, min(p, usize - 1))
        n = rnd.choice((1, 17, 333, MPAGE, MPAGE + 1, 5000, 20_000, 300_000))
        if op < 0.45:
            st.seek(p)
            got = st.read(n)  # readinto: metadata cache, page cache or bulk read
        elif op < 0.75:
            got = st.read_range(p, n)
        elif op < 0.85:
            st._goto(p)  # move the inflate position only
            got = None
            assert st.out == p
        elif op < 0.92:  # a restore as the error handlers do it (ForwardPass redo), at any point
            st._restore(rnd.randrange(len(st.ck_out)))
            got = None
        else:  # scanning switched off and on again (the job's memory relief turns it off)
            st.meta_cap = 0 if st.meta_cap else 256 << 20
            got = None
        if got is not None:
            assert got == plain[p:p + n], (seed, p, n)
        _check_invariants(st, plain)


@pytest.mark.parametrize("seed", range(24))
def test_fuzz_jump_patterns(member, seed):
    _fuzz_one(member, seed, native=NATIVE and seed % 4 != 3)


def test_fuzz_zlib_fallback(member):
    """Without libz (in-memory zlib snapshots, no persisted points) the same reads stay correct."""
    for seed in range(4):
        _fuzz_one(member, 1000 + seed, native=False)


@needs_native
def test_listing_with_points_then_forward_pass(member):
    """The job's path on a restart: listing with persisted points, then the data pass on the same stream."""
    st = _stream(member, points=member.points[: len(member.points) // 3])
    entries = _listing(st)
    assert entries == member.entries
    pairs = [(n, ke[0], ve[0] if ve else None) for n, ke, ve in openhands.iter_pairs(entries) if ke is not None]
    fp = openhands.ForwardPass(st, pairs, window=64 << 10, jump=128 << 10, step=16 << 10)
    got = {name: (kp, vis, err) for name, kp, vis, err in fp}
    assert set(got) == set(member.arrays)
    for name, (kp, vis) in member.arrays.items():
        gk, gv, err = got[name]
        assert err is None, name
        np.testing.assert_array_equal(gk, kp)
        np.testing.assert_array_equal(gv, vis)
    assert not st.zero_pages and st.meta_cap == 0


def test_forward_pass_window_restarts_when_the_stream_moved(member):
    """A piece that does not continue the kept window is never filed under the window's offsets."""
    st = _stream(member)
    fp = openhands.ForwardPass(st, [])
    fp.win, fp.win_start = bytearray(b"x" * 100), 0
    fp._feed(5000, b"abc")
    assert fp.win_start == 5000 and bytes(fp.win) == b"abc"
    fp._feed(5003, b"de")
    assert fp.win_start == 5000 and bytes(fp.win) == b"abcde"


def test_scanned_intervals_merge():
    st = types.SimpleNamespace(_sc_lo=[], _sc_hi=[])
    mark = openhands.SeekableInflate._mark_scanned
    uns = openhands.SeekableInflate._unscanned
    for a, b in ((10, 20), (30, 40), (20, 25), (5, 8), (8, 10), (26, 30)):
        mark(st, a, b)
    assert (st._sc_lo, st._sc_hi) == ([5, 26], [25, 40])
    assert uns(st, 0, 50) == [(0, 5), (25, 26), (40, 50)]
    assert uns(st, 6, 24) == [] and uns(st, 24, 27) == [(25, 26)]
    mark(st, 0, 100)
    assert (st._sc_lo, st._sc_hi) == ([0], [100])


def _flaky(st, rnd, p):
    """Make the stream's byte fetches fail with ``OSError`` at rate ``p`` (a network give-up)."""
    real = st.r.read_at

    def read_at(off, k):
        if rnd.random() < p:
            raise OSError("injected fetch failure")
        return real(off, k)

    st.r.read_at = read_at


@needs_native
@pytest.mark.parametrize("seed", range(4))
def test_fetch_failures_mid_restore_or_step_never_file_wrong_pages(member, seed):
    """Review of 3f2ad35: a fetch that fails inside ``_restore`` (the byte before a point) or ``_step`` leaves the
    inflate position, the scan stage and the metadata cache consistent; the next read returns the right bytes."""
    plain, usize = member.plain, len(member.plain)
    rnd = random.Random(500 + seed)
    st = _stream(member, points=sorted(rnd.sample(member.points, len(member.points) // 2)),
                 chunk=rnd.choice((4 << 10, 16 << 10)))
    _flaky(st, rnd, 0.15)
    failed = 0
    for _ in range(50):
        p = rnd.choice((rnd.randrange(usize), st.out + rnd.randrange(-3 * MPAGE, 3 * MPAGE), rnd.choice(st.ck_out)))
        p = max(0, min(p, usize - 1))
        n = rnd.choice((8, 300, MPAGE, 3 * MPAGE + 1, 300_000))
        st.seek(p)
        try:
            got = st.read(n)
        except OSError:
            failed += 1
        else:
            assert got == plain[p:p + n], (p, n)
        _check_invariants(st, plain)
    assert failed  # the failures really happened


@needs_native
def test_forward_pass_after_listing_with_points_survives_stream_errors(member, monkeypatch):
    """Listing with persisted points, then the forward pass on the same stream while fetches fail now and then
    (``redo`` restores and jumps through points): every chunk is exact or carries the fetch error."""
    monkeypatch.setattr(openhands.time, "sleep", lambda s: None)
    rnd = random.Random(3)
    st = _stream(member, points=member.points[::2])
    assert _listing(st) == member.entries
    _flaky(st, rnd, 0.03)
    pairs = [(n, ke[0], ve[0] if ve else None) for n, ke, ve in openhands.iter_pairs(member.entries) if ke is not None]
    fp = openhands.ForwardPass(st, pairs, window=64 << 10, jump=128 << 10, step=16 << 10, max_stream_errors=10_000,
                               forgive=1 << 40)
    got = {name: (kp, vis, err) for name, kp, vis, err in fp}
    assert set(got) == set(member.arrays) and fp.stats["stream_errors"]
    for name, (kp, vis) in member.arrays.items():
        gk, gv, err = got[name]
        if err is not None:
            assert isinstance(err, OSError), name  # only a fallback read can surface the injected failure
            continue
        np.testing.assert_array_equal(gk, kp)
        np.testing.assert_array_equal(gv, vis)
