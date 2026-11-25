"""Gated-dataset plumbing without network: access check, RangeFile/ConcatStream over in-memory "HTTP"
parts, CISLR ingestion end-to-end on a mocked RangeFile, iSign pose-format conversion and archive reading."""

from __future__ import annotations

import csv
import io
import tarfile
import zipfile

import numpy as np
import pytest

from app.features.spec import FACE_IDX, N_POSE
from app.isl.data import cislr as cislr_mod
from app.isl.data import isign as isign_mod
from app.isl.data.remote import ConcatStream, RangeFile, hf_access, sniff_format
from app.isl.data.sources import CislrAdapter, ISignAdapter
from app.isl.paths import isl_paths
from app.landmarks.raw import RawSequence

from .helpers import make_trim_sequence


class _Resp:
    def __init__(self, code):
        self.status_code = code
        self.headers = {}

    def close(self):
        pass


@pytest.mark.parametrize("code,expected", [(206, "granted"), (200, "granted"), (403, "pending"), (401, "pending"),
                                           (500, "error:500")])
def test_hf_access(code, expected):
    seen = {}

    def get(url, headers=None, timeout=None, allow_redirects=True):
        seen["url"], seen["headers"] = url, headers
        return _Resp(code)

    assert hf_access("Exploration-Lab/CISLR", get=get) == expected
    assert seen["url"].endswith("/Exploration-Lab/CISLR/resolve/main/dataset.csv")  # a data file, not README
    assert seen["headers"] == {"Range": "bytes=0-0"}


def test_hf_access_network_error():
    def get(*a, **k):
        raise ConnectionError("boom")

    assert hf_access("Exploration-Lab/iSign", get=get) == "error:ConnectionError"


def _split(data: bytes, n_parts: int):
    step = -(-len(data) // n_parts)
    parts = [data[i: i + step] for i in range(0, len(data), step)]
    store = {f"https://hf/part_{i}": p for i, p in enumerate(parts)}
    calls = []

    def fetch(url, start, end):
        calls.append((url, start, end))
        return store[url][start: end + 1]

    return list(store), [len(p) for p in parts], fetch, calls


def test_rangefile_reads_across_parts():
    data = bytes(range(256)) * 400  # 102,400 bytes
    urls, sizes, fetch, calls = _split(data, 3)
    rf = RangeFile(urls, sizes, block=4096, cache_blocks=4, fetch=fetch)
    assert rf.size == len(data)
    assert rf.read_at(34_000, 1000) == data[34_000:35_000]  # crosses the part_0 / part_1 boundary
    rf.seek(-10, io.SEEK_END)
    assert rf.read() == data[-10:]
    rf.seek(5)
    assert rf.read(3) == data[5:8]
    n = len(calls)
    rf.seek(6)
    rf.read(2)  # cached block, no new request
    assert len(calls) == n
    assert rf.read_at(0, 20_000) == data[:20_000]  # large read bypasses the cache


def test_concat_stream_and_resume():
    data = b"".join(bytes([i % 251]) for i in range(10_000))
    parts = [data[:3000], data[3000:7000], data[7000:]]
    urls = ["a", "b", "c"]

    def opener(url, offset):
        p = parts[urls.index(url)]
        for i in range(offset, len(p), 777):
            yield p[i: i + 777]

    assert ConcatStream(urls, opener=opener).read() == data
    s = ConcatStream(urls, [len(p) for p in parts], start=5000, opener=opener)
    assert s.read() == data[5000:]


def test_sniff_format():
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as tf:
        ti = tarfile.TarInfo("x.pose")
        ti.size = 3
        tf.addfile(ti, io.BytesIO(b"abc"))
    assert sniff_format(b.getvalue()[:512]) == "tar"
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a", "b")
    assert sniff_format(z.getvalue()[:512]) == "zip"
    assert sniff_format(b"\x1f\x8b\x08") == "gzip"
    assert sniff_format(b"hello") == "unknown"


# --------------------------------------------------------------------------------------------- CISLR

def _fake_extract(src: str, out: str) -> dict:
    """Stand-in for MediaPipe: the 'video' bytes encode a seed; writes a synthetic RawSequence."""
    seed = int(open(src, "rb").read().decode().split(":")[1])
    raw = make_trim_sequence([("rest", 3), ("sign", 15), ("rest", 3)], seed=seed)
    from pathlib import Path

    raw.save(Path(out))
    return {"ok": True, "n_frames": len(raw), "fps": 12.0, "hands_frac": 0.7}


def test_cislr_end_to_end_on_mocked_rangefile(tmp_path):
    P = isl_paths(tmp_path / "isl")
    raw_dir = P.raw / "cislr"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "dataset.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "word", "category"])
        w.writerows([["vidA", "apple", "food"], ["vidB", "Store or Shop", "places"], ["vidC", "pencil", "home"]])
    with (raw_dir / "prototype.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows([["uid", "word"], ["vidA", "apple"]])
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, uid in enumerate(["vidA", "vidB", "vidC"]):
            zf.writestr(f"CISLR_v1.5-a_videos/{uid}.mp4", f"video:{i}".encode() + b" " * 5000)
        zf.writestr("__MACOSX/._vidA.mp4", b"junk")
    urls, sizes, fetch, calls = _split(z.getvalue(), 2)
    rf = RangeFile(urls, sizes, block=2048, fetch=fetch)
    stats = cislr_mod.ingest_cislr(P, rf, extract=_fake_extract, limit=2)
    assert stats["members"] == 3 and stats["processed"] == 2 and stats["ok"] == 2
    stats = cislr_mod.ingest_cislr(P, rf, extract=_fake_extract)  # resume: only the third member
    assert stats["skipped_done"] == 2 and stats["processed"] == 1
    assert not list(P.tmp.glob("cislr_*"))  # temp videos deleted
    clips = list(CislrAdapter(P).clips())
    assert [c.clip_id for c in clips] == ["cislr/vidA", "cislr/vidB", "cislr/vidC"]
    assert clips[0].raw_gloss == "apple" and clips[0].quality["split"] == "prototype"
    assert len(clips[1].load()) == 21
    rows, split = cislr_mod.csv_index(P)
    assert rows["vidB"]["word"] == "Store or Shop" and split == {"vidA": "prototype"}


def test_normalise_cislr_row():
    assert cislr_mod.normalise_cislr_row({"Video_ID": "x.mp4", "Gloss": "Hello"}) == {"uid": "x", "word": "Hello", "category": None}
    assert cislr_mod.normalise_cislr_row({"foo": "bar"}) is None


# --------------------------------------------------------------------------------------------- iSign

COMPONENTS = [("POSE_LANDMARKS", 33), ("FACE_LANDMARKS", 468), ("LEFT_HAND_LANDMARKS", 21), ("RIGHT_HAND_LANDMARKS", 21)]


def _pose_bytes(F=10, W=1280, H=720, version=0.2, missing_left=(3, 4)):
    N = sum(n for _, n in COMPONENTS)
    rng = np.random.default_rng(0)
    data = np.zeros((F, 1, N, 3), np.float32)
    data[..., 0] = rng.uniform(0, W, (F, 1, N))
    data[..., 1] = rng.uniform(0, H, (F, 1, N))
    data[..., 2] = rng.uniform(-50, 50, (F, 1, N))
    conf = np.ones((F, 1, N), np.float32)
    lo = 33 + 468
    for f in missing_left:
        data[f, 0, lo: lo + 21] = 0
        conf[f, 0, lo: lo + 21] = 0
    return isign_mod.write_pose(None, COMPONENTS, data, conf, W, H, fps=25.0, version=version), data, conf


@pytest.mark.parametrize("version", [0.1, 0.2])
def test_pose_to_raw_mapping(version):
    buf, data, conf = _pose_bytes(version=version)
    raw = isign_mod.pose_to_raw(buf)
    assert len(raw) == 10 and (raw.width, raw.height) == (1280, 720)
    assert raw.extractor_version == "isign-pose-format"
    np.testing.assert_allclose(raw.pose[0, :, 0], data[0, 0, :N_POSE, 0] / 1280, rtol=1e-5)
    np.testing.assert_allclose(raw.pose[0, :, 1], data[0, 0, :N_POSE, 1] / 720, rtol=1e-5)
    np.testing.assert_allclose(raw.pose[0, :, 3], 1.0)
    face_src = data[2, 0, 33:33 + 468][FACE_IDX]
    np.testing.assert_allclose(raw.face[2, :, 0], face_src[:, 0] / 1280, rtol=1e-5)
    assert np.isnan(raw.left[3]).all() and np.isnan(raw.left[4]).all() and not np.isnan(raw.left[5]).any()
    np.testing.assert_allclose(raw.right[0, :, 2], data[0, 0, 33 + 468 + 21:, 2], rtol=1e-5)  # raw z
    np.testing.assert_allclose(np.diff(raw.t_ms), 40.0)
    swapped = isign_mod.pose_to_raw(buf, swap_hands=True)
    np.testing.assert_allclose(swapped.left[0], raw.right[0])
    # the converted sequence goes through the regular feature pipeline
    from app.features.build import build_features
    from app.features.spec import FeatureConfig

    assert build_features(raw, FeatureConfig(seq_len=16)).shape[0] == 16


def test_pose_format_errors():
    with pytest.raises(isign_mod.PoseFormatError):
        isign_mod.read_pose(b"\x00\x00\x00\x00")


def _pose_zip(uids):
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED) as zf:
        for u in uids:
            zf.writestr(f"poses/{u}.pose", _pose_bytes(F=8)[0])
        zf.writestr("poses/readme.txt", "x")
    return z.getvalue()


def test_isign_zip_random_access_subset_and_resume(tmp_path):
    P = isl_paths(tmp_path / "isl")
    urls, sizes, fetch, calls = _split(_pose_zip(["s1", "s2", "s3", "s4"]), 4)
    rf = RangeFile(urls, sizes, block=1 << 14, fetch=fetch)
    st = isign_mod.ingest_isign_zip(P, rf, wanted={"s2", "s4"}, sentences={"s2": "I go to school"}, limit=1)
    assert st["members"] == 4 and st["processed"] == 1
    st = isign_mod.ingest_isign_zip(P, rf, wanted={"s2", "s4"})
    assert st["skipped_done"] == 1 and st["processed"] == 1 and st["ok"] == 1
    clips = list(ISignAdapter(P).clips())
    assert [c.clip_id for c in clips] == ["isign/s2", "isign/s4"]
    assert clips[0].raw_gloss == "I go to school" and clips[0].licence == "CC-BY-NC-SA-4.0"
    assert isinstance(clips[0].load(), RawSequence)


def test_isign_stream_zip_and_tar(tmp_path):
    P = isl_paths(tmp_path / "a")
    data = _pose_zip(["s1", "s2"])
    st = isign_mod.ingest_isign_stream(P, io.BytesIO(data), "zip", wanted=None)
    assert st["ok"] == 2
    assert isign_mod.last_offset(P) > 0
    P2 = isl_paths(tmp_path / "b")
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as tf:
        for u in ("t1", "t2"):
            pb = _pose_bytes(F=6)[0]
            ti = tarfile.TarInfo(f"poses/{u}.pose")
            ti.size = len(pb)
            tf.addfile(ti, io.BytesIO(pb))
    b.seek(0)
    st = isign_mod.ingest_isign_stream(P2, b, "gzip", wanted={"t2"})
    assert st["processed"] == 1 and st["ok"] == 1


def test_select_subset():
    s = {"u1": "I like apples", "u2": "the pencil and the book", "u3": "nothing here", "u4": "Book!"}
    assert isign_mod.select_subset(s, ["pencil", "book", "apple pie"], n=10) == ["u2", "u4"]
    assert isign_mod.select_subset(s, ["book"], n=1) == ["u2"]


# --------------------------------------------------------------------------------------------- stage-2 boundary model

def test_boundary_model_smoke():
    pytest.importorskip("torch")
    from app.isl.data.boundary import decode, frame_features, make_example, refine, train_boundary
    from app.isl.data.trim import find_first_sign

    clean = [frame_features(make_trim_sequence([("rest", 3), ("sign", 14 + i % 5), ("rest", 2)], seed=i)) for i in range(8)]
    tails = [frame_features(make_trim_sequence([("expl", 30)], seed=i)) for i in range(4)]
    x, y, k = make_example(clean[0], tails[0], np.random.default_rng(0))
    assert x.shape[0] == len(clean[0]) + len(tails[0]) + (x.shape[0] - len(clean[0]) - len(tails[0])) and y[k - 1] == 1
    model, res = train_boundary(clean, tails, n_examples=64, epochs=2, threads=1)
    assert 0.0 <= res.val_within_2 <= 1.0 and res.train_examples == 64
    assert decode(np.array([0.1, 0.2, 0.9, 0.3]), 0) == 3
    raw = make_trim_sequence([("rest", 6), ("sign", 20), ("rest", 6), ("expl", 30)])
    st1 = find_first_sign(raw, "max_s")
    out = refine(raw, st1, model)
    assert out.method in (st1.method, "learned") and out.start == st1.start
