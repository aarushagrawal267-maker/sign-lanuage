"""Pose-native ISL sources (GloSL, ISL500, OpenHands raw / fingerspelling): converters, storage, streaming and
adapters. Synthetic data only, no network."""

from __future__ import annotations

import io
import json
import zipfile
import zlib

import numpy as np
import pytest

from app.features.spec import FACE_IDX, N_FACE, N_HAND, N_POSE
from app.isl.data.isign import write_pose
from app.isl.data.pose_sources import POSE_ADAPTERS, PROFILE_SOURCES, register
from app.isl.data.pose_sources import common, fingerspell, glosl, isl500, openhands
from app.isl.data.pose_sources.posefmt import holistic_pose_to_raw
from app.isl.paths import IslPaths
from app.landmarks.raw import RawSequence

HOLISTIC = [("POSE_LANDMARKS", 33), ("FACE_LANDMARKS", 468), ("LEFT_HAND_LANDMARKS", 21), ("RIGHT_HAND_LANDMARKS", 21)]


def _hand(wrist, scale=0.05, rot=0.0):
    """21 points of a planar hand of fixed bone lengths, rotated in-plane by ``rot``."""
    pts = np.zeros((21, 3))
    c, s = np.cos(rot), np.sin(rot)
    for f in range(5):
        ang = np.deg2rad(-50 + 25 * f)
        for j in range(4):
            r = scale * (j + 1) / 4
            v = np.array([np.sin(ang) * r, -np.cos(ang) * r])
            pts[1 + 4 * f + j, :2] = [c * v[0] - s * v[1], s * v[0] + c * v[1]]
    pts[:, :2] += wrist
    return pts


def _glosl_pose(T=10, W=640, H=480, fps=25.0):
    """Synthetic Holistic pose-format file in pixels; frame 3 has no left hand, frame 5 no body, 7 no face."""
    rng = np.random.default_rng(0)
    norm = np.zeros((T, 1, 543, 3))
    conf = np.ones((T, 1, 543))
    norm[..., 0] = rng.uniform(0.2, 0.8, (T, 1, 543))
    norm[..., 1] = rng.uniform(0.2, 0.9, (T, 1, 543))
    norm[..., 2] = rng.uniform(-0.5, 0.3, (T, 1, 543))  # raw MediaPipe z
    data = norm.copy()
    data[..., 0] *= W
    data[..., 1] *= H  # z stays raw (GloSL card / pose-format holistic writer)
    for t, lo, hi in ((3, 501, 522), (5, 0, 33), (7, 33, 501)):
        data[t, 0, lo:hi] = 0
        conf[t, 0, lo:hi] = 0
    buf = write_pose(None, HOLISTIC, data.astype(np.float32), conf.astype(np.float32), W, H, fps=fps)
    return buf, norm


# --------------------------------------------------------------------------------------------- GloSL / pose-format

def test_glosl_pose_matches_extractor_layout():
    buf, norm = _glosl_pose()
    seq, hdr = holistic_pose_to_raw(buf, glosl.EXTRACTOR_VERSION)
    assert hdr["width"] == 640 and hdr["height"] == 480 and hdr["fps"] == 25.0
    assert seq.pose.shape == (10, N_POSE, 4) and seq.left.shape == (10, N_HAND, 3)
    assert seq.face.shape == (10, N_FACE, 3) and seq.extractor_version == "glosl-pose-format"
    assert (seq.width, seq.height) == (640, 480)
    np.testing.assert_allclose(seq.t_ms, np.arange(10) * 40.0)
    # x/y normalised by the header size, z copied raw (NOT divided by the width)
    np.testing.assert_allclose(seq.pose[0, :, :3], norm[0, 0, :33], atol=1e-4)
    np.testing.assert_allclose(seq.left[0], norm[0, 0, 501:522], atol=1e-4)
    np.testing.assert_allclose(seq.right[0], norm[0, 0, 522:543], atol=1e-4)
    np.testing.assert_allclose(seq.face[0], norm[0, 0, 33 + np.array(FACE_IDX)], atol=1e-4)
    assert np.all(seq.pose[0, :, 3] == 1.0)  # confidence -> visibility channel
    # missing parts are all-NaN frames, as HolisticExtractor writes them
    pr = seq.presence()
    assert not pr["left"][3] and pr["right"][3]
    assert not pr["pose"][5] and np.isnan(seq.pose[5]).all()
    assert not pr["face"][7] and pr["face"][6]


def test_glosl_z_width_mode_and_errors():
    buf, norm = _glosl_pose()
    seq, _ = holistic_pose_to_raw(buf, "x", z_mode="width")
    np.testing.assert_allclose(seq.pose[0, :, 2], norm[0, 0, :33, 2] / 640, atol=1e-6)
    with pytest.raises(ValueError):
        holistic_pose_to_raw(buf, "x", z_mode="pixels")


def test_save_compact_float16_stride_and_poseclip_upcast(tmp_path):
    buf, _ = _glosl_pose(T=11)
    seq, _ = holistic_pose_to_raw(buf, glosl.EXTRACTOR_VERSION)
    out = tmp_path / "a" / "c.npz"
    saved = common.save_compact(seq, out, dtype="float16", stride=2)
    assert len(saved) == 6 and not list(tmp_path.rglob("*.tmp.npz"))
    raw = RawSequence.load(out)  # plain loader reads it (float16 arrays)
    assert raw.pose.dtype == np.float16 and raw.t_ms.dtype == np.float64
    np.testing.assert_allclose(raw.t_ms, seq.t_ms[::2])
    up = common.load_float32(out)
    assert up.pose.dtype == np.float32
    assert np.array_equal(np.isnan(up.left), np.isnan(seq.left[::2]))
    np.testing.assert_allclose(up.right[0], seq.right[0], atol=2e-3)
    clip = common.PoseClip(clip_id="glosl/x", source="glosl", raw_gloss="", concept_id=None, variant_id=None,
                           category=None, path=out, start=1, end=4)
    got = clip.load()
    assert len(got) == 3 and got.left.dtype == np.float32
    assert len(clip.with_slice(0, None).load()) == 6
    with pytest.raises(ValueError):
        common.save_compact(seq, out, dtype="int8")


def test_stride_for():
    assert common.stride_for(25, 15) == 2 and common.stride_for(29, 15) == 2 and common.stride_for(12, 15) == 1
    assert common.stride_for(30, 0) == 1 and common.stride_for(0, 15) == 1


def test_hand_aspect_recovers_frame_aspect():
    rng = np.random.default_rng(1)
    true_aspect = 16 / 9
    frames = []
    for t in range(60):  # a rigid hand rotating in 3-D, observed in normalised image coordinates
        h = _hand(np.array([0.0, 0.0]), scale=0.08, rot=rng.uniform(0, 6.28))
        a = rng.uniform(-1.2, 1.2)
        h[:, 2] = h[:, 0] * np.sin(a)
        h[:, 0] = h[:, 0] * np.cos(a)
        img = h.copy()
        img[:, 0] = 0.5 + h[:, 0]  # x and z in width units
        img[:, 1] = 0.5 + h[:, 1] * true_aspect  # y in height units
        frames.append(img)
    hands = np.array(frames)
    est = common.hand_aspect([hands])
    assert est is not None and abs(est[0] - true_aspect) < 0.08
    assert common.hand_aspect([np.full((10, 21, 3), np.nan)]) is None


def test_active_window():
    seq = RawSequence.empty()
    T = 20
    left = np.full((T, 21, 3), np.nan, np.float32)
    left[8:12] = 0.5
    seq = RawSequence(np.zeros((T, 33, 4), np.float32), left, np.full((T, 21, 3), np.nan, np.float32),
                      np.zeros((T, N_FACE, 3), np.float32), np.arange(T) * 66.7, 640, 480)
    w = common.active_window(seq)
    assert (w["active_start"], w["active_end"]) == (6, 14) and w["hands_frac_active"] == 0.5


def test_disk_guard(tmp_path):
    g = common.DiskGuard(tmp_path, min_free_gb=1e9)
    with pytest.raises(common.DiskFull):
        g.check(force=True)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "f.bin").write_bytes(b"x" * 10000)
    g2 = common.DiskGuard(tmp_path, min_free_gb=0, budget_dir=tmp_path / "b", budget_gb=1e-9)
    with pytest.raises(common.DiskFull):
        g2.check(force=True)
    assert common.safe_name("a/b c?.d") == "a_b_c_.d"


def _write_manifest(paths: IslPaths, name: str, rows: list[dict]) -> None:
    paths.manifests.mkdir(parents=True, exist_ok=True)
    paths.manifest(name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_glosl_adapter_and_isign_dedup(tmp_path):
    paths = IslPaths(tmp_path / "isl")
    buf, _ = _glosl_pose(T=20)
    seq, _ = holistic_pose_to_raw(buf, glosl.EXTRACTOR_VERSION)
    common.save_compact(seq, paths.landmarks / "glosl" / "0000" / "v-1.npz", dtype="float16")
    _write_manifest(paths, "glosl", [
        {"id": "glosl/v-1", "ok": True, "path": "data/isl/landmarks/glosl/0000/v-1.npz", "n_frames": 20,
         "hands_frac": 0.95, "text": "Make it shorter.", "split": "train", "video_ref": "v", "isign_uid": "v-1"},
        {"id": "glosl/v-2", "ok": True, "path": "data/isl/landmarks/glosl/0000/v-2.npz", "n_frames": 3,
         "hands_frac": 1.0, "text": "too short", "split": "dev", "video_ref": "v", "isign_uid": "v-2"},
        {"id": "glosl/v-3", "ok": False, "error": "x"},
    ])
    ad = glosl.GloslAdapter(paths=paths)
    assert ad.available() and not ad.commercial and ad.profiles == ("research",)
    clips = list(ad.clips())
    assert [c.clip_id for c in clips] == ["glosl/v-1"]
    c = clips[0]
    assert c.raw_gloss == "Make it shorter." and c.group == "glosl-video:v" and c.quality["split"] == "train"
    assert c.load().pose.dtype == np.float32
    assert ad.uids() == {"v-1", "v-2"}
    assert glosl.isign_duplicates({"v-1", "zz-9"}, ad.uids()) == {"v-1"}


def test_glosl_annotation_readers(tmp_path):
    f = tmp_path / "annotations.csv"
    f.write_text('id,split,text,pose_file\na-1,train,"two\nlines",poses/0000/a-1.pose\n,train,x,\n', encoding="utf-8")
    rows = glosl.read_annotations(f)
    assert len(rows) == 1 and rows[0]["text"] == "two\nlines"
    s = tmp_path / "s.csv"
    s.write_text("id,split\na-1,dev\n")
    assert glosl.read_split_file(s) == {"a-1": "dev"} and glosl.read_split_file(tmp_path / "none.csv") == {}
    u = tmp_path / "u.csv"
    u.write_text("﻿uid,text\na-1,x\nb-2,y\n", encoding="utf-8")
    assert glosl.read_id_column(u, "uid") == {"a-1", "b-2"}


# --------------------------------------------------------------------------------------------- ISL500

def _isl500_clip(T=30, W=2560, H=1440):
    rng = np.random.default_rng(3)
    hands = np.zeros((T, 2, 21, 3), np.float32)
    wb = np.zeros((T, 133, 3), np.float32)
    wb[:, :, 2] = 0.9
    wb[:, :, 0] = rng.uniform(800, 1800, (T, 133))
    wb[:, :, 1] = rng.uniform(200, 1300, (T, 133))
    for t in range(T):
        if 5 <= t < 25:  # hands visible in the middle of the clip
            for k, off in ((0, isl500.WB_LHAND0), (1, isl500.WB_RHAND0)):
                h = _hand(np.array([0.4 + 0.2 * k, 0.6]), rot=t * 0.1)
                hands[t, k] = h
                wb[t, off:off + 21, 0] = h[:, 0] * W
                wb[t, off:off + 21, 1] = h[:, 1] * H
    wb[2, :, 2] = 0.05  # one frame without a body
    return hands, wb


def test_isl500_names():
    assert isl500.clip_key("_mnt_9a5_ALL_CLIPS_R2_Clips_R2_user001_Absent__session7__clip000") == "Absent__session7__clip000"
    assert isl500.clip_key("R2_28.03.26_All_user002_R2_clips_clips_R2_user002_Absent__s1") == "Absent__s1"
    assert isl500.clip_key("Absent__000001__c545daf6-s004-000_Absent") == "Absent__000001__c545daf6-s004-000_Absent"
    assert isl500.gloss_of("Absent__session7__clip000") == "Absent"
    assert isl500.gloss_of("R2_Thank_You__000001") == "Thank You"
    assert isl500.user_of("ISL_DATA_USER007") == "user007" and isl500.user_of("User015") == "user015"
    with pytest.raises(ValueError):
        isl500.user_of("nobody")


def test_isl500_conversion_and_size():
    hands, wb = _isl500_clip()
    assert isl500.estimate_size(hands, wb) == (2560, 1440)
    assert isl500.estimate_size(np.zeros_like(hands), wb) is None
    seq = isl500.isl500_to_raw(hands, wb, (2560, 1440), fps=30.0)
    assert seq.extractor_version == isl500.EXTRACTOR_VERSION and (seq.width, seq.height) == (2560, 1440)
    np.testing.assert_allclose(seq.t_ms[:3], [0, 1000 / 30, 2000 / 30])
    pr = seq.presence()
    assert not pr["pose"][2] and pr["pose"][3] and not pr["face"][2]
    assert not pr["left"][0] and pr["left"][10] and pr["right"][10]
    np.testing.assert_allclose(seq.left[10], hands[10, 0], atol=1e-6)
    # MediaPipe pose 11/12 (shoulders) <- COCO 5/6, 15 (left wrist) <- 9; normalised by the frame size
    for mp_i, coco_i in ((0, 0), (11, 5), (12, 6), (15, 9), (23, 11)):
        np.testing.assert_allclose(seq.pose[3, mp_i, :2], wb[3, coco_i, :2] / [2560, 1440], atol=1e-6)
    np.testing.assert_allclose(seq.pose[3, 17, :2], wb[3, isl500.WB_LHAND0 + 17, :2] / [2560, 1440], atol=1e-6)
    assert np.all(seq.pose[3, :, 2] == 0) and np.allclose(seq.pose[3, :, 3], 0.9)
    j = FACE_IDX.index(152)  # chin <- iBUG 8
    np.testing.assert_allclose(seq.face[3, j, :2], wb[3, isl500.WB_FACE0 + 8, :2] / [2560, 1440], atol=1e-6)
    # features can be built from it (shoulders present)
    from app.features.build import to_body_frame

    to_body_frame(seq)


def test_isl500_adapter_slices_and_links(tmp_path):
    from app.isl.vocab.build import VocabIndex
    from app.isl.vocab.schema import VocabEntry

    paths = IslPaths(tmp_path / "isl")
    hands, wb = _isl500_clip()
    seq = isl500.isl500_to_raw(hands, wb, (2560, 1440))
    saved = common.save_compact(seq, paths.landmarks / "isl500" / "user003" / "Accept__s1.npz", dtype="float16")
    rec = {"id": "isl500/user003/Accept__s1", "ok": True, "gloss": "Accept", "user": "user003",
           "path": "data/isl/landmarks/isl500/user003/Accept__s1.npz", **common.seq_stats(saved),
           **common.active_window(saved)}
    _write_manifest(paths, "isl500", [rec, {**rec, "id": "isl500/user004/Zebra__s1", "gloss": "Zebra", "user": "user004"}])
    vocab = VocabIndex([VocabEntry(concept_id="isl.accept", headword="accept", display={"en": "accept"})])
    ad = isl500.Isl500Adapter(paths=paths, vocab=vocab)
    clips = {c.clip_id: c for c in ad.clips()}
    c = clips["isl500/user003/Accept__s1"]
    assert c.group == "isl500-user003" and c.variant_id == "isl500:Accept" and c.licence == isl500.LICENCE
    assert c.concept_id == "isl.accept" and c.quality["link"] == "exact"
    assert (c.start, c.end) == (3, 27) and len(c.load()) == 24
    assert clips["isl500/user004/Zebra__s1"].concept_id is None
    assert isl500.Isl500Adapter(paths=paths).link("Accept") == (None, "none", False)


# --------------------------------------------------------------------------------------------- OpenHands raw

def _deflate(data: bytes) -> bytes:
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def test_seekable_inflate_random_access():
    rng = np.random.default_rng(5)
    plain = rng.integers(0, 8, 3_000_000, dtype=np.uint8).tobytes()
    comp = _deflate(plain)
    pre = b"HEAD" * 10
    blob = pre + comp + b"TAIL"
    reader = openhands.PrefetchReader(lambda s, e: blob[s:e + 1], len(blob), block=64 << 10, ahead=2)
    st = openhands.SeekableInflate(reader, len(pre), len(comp), len(plain), every=256 << 10, chunk=32 << 10)
    for off, n in ((0, 100), (2_900_000, 50_000), (5, 10), (1_000_000, 300_000), (2_999_990, 100), (123_456, 7)):
        st.seek(off)
        assert st.read(n) == plain[off:off + n]
    assert st.read_range(1_500_000, 1000) == plain[1_500_000:1_501_000]
    assert len(st.ck_out) >= 10 and st.restores >= 2
    reader.close()


def test_zip_member_span_and_safe_names():
    payload = b"abc" * 100_000
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Dir/x.h5", payload)
    blob = b.getvalue()
    reader = openhands.PrefetchReader(lambda s, e: blob[s:e + 1], len(blob), block=4096, ahead=0)
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("Dir/x.h5")
    start, csize, usize, method = openhands.zip_member_span(reader, info)
    assert method == 8 and usize == len(payload)
    st = openhands.SeekableInflate(reader, start, csize, usize)
    assert st.read_range(0, usize) == payload
    assert openhands.safe_member("Sign-Library/a.h5")
    assert not openhands.safe_member("../etc/passwd") and not openhands.safe_member("/abs.h5")
    assert not openhands.safe_member("a/../../b.h5")


def test_openhands_h5_streaming_pipeline(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(7)
    src = tmp_path / "x.h5"
    arrays = {}
    with h5py.File(src, "w") as f:
        k, v = f.create_group("keypoints"), f.create_group("visibility")
        for i, T in enumerate((40, 25, 60)):
            name = f"vidA_title-{i}" if i < 2 else "vidB-0"
            kp = rng.uniform(0, 1, (T, 75, 3))
            kp[::7, 33:54] = 0  # left hand missing in some frames
            vis = np.ones((T, 75))
            k.create_dataset(name, data=kp)
            v.create_dataset(name, data=vis)
            arrays[name] = kp
    plain = src.read_bytes()
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Chan/x.h5", plain)
    blob = b.getvalue()
    reader = openhands.PrefetchReader(lambda s, e: blob[s:e + 1], len(blob), block=16 << 10, ahead=2)
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("Chan/x.h5")
    start, csize, usize, _ = openhands.zip_member_span(reader, info)
    st = openhands.SeekableInflate(reader, start, csize, usize, every=8 << 10, chunk=4 << 10)
    hf = h5py.File(st, "r")
    try:
        entries = openhands.list_datasets(hf)
    finally:
        hf.close()
    assert len(entries) == 6 and entries == sorted(entries)
    pairs = list(openhands.iter_pairs(entries))
    assert {p[0] for p in pairs} == set(arrays)
    for name, kp_e, vis_e in pairs:
        hdr = openhands.parse_v1_header(st.read_range(kp_e[0], 4096))
        kp = openhands.read_contiguous(st, hdr)
        np.testing.assert_array_equal(kp, arrays[name])
        vis = openhands.read_contiguous(st, openhands.parse_v1_header(st.read_range(vis_e[0], 4096)))
        seq = openhands.openhands_to_raw(kp, vis, aspect=16 / 9)
        assert (seq.width, seq.height) == (1920, 1080) and np.isnan(seq.face).all()
        assert not seq.presence()["left"][0] and seq.presence()["right"][0]
        np.testing.assert_allclose(seq.right[1], kp[1, 54:75], atol=1e-6)
        np.testing.assert_allclose(seq.pose[1, :, :3], kp[1, :33], atol=1e-6)
    assert openhands.video_key("vidA_title-1") == "vidA_title" and openhands.video_key("plain") == "plain"
    with pytest.raises(openhands.HeaderError):
        openhands.parse_v1_header(b"\x02" + b"\x00" * 40)


def test_openhands_conversion_helpers():
    assert openhands.snap_aspect(1.7) == pytest.approx(16 / 9) and openhands.snap_aspect(None) is None
    assert openhands.snap_aspect(2.4) == 2.4
    kp = np.zeros((4, 75, 3))
    kp[1, :33] = 0.5
    seq = openhands.openhands_to_raw(kp, None)
    assert seq.presence()["pose"].tolist() == [False, True, False, False]
    with pytest.raises(ValueError):
        openhands.openhands_to_raw(np.zeros((4, 70, 3)), None)
    left, right = openhands.hands_nan(kp)
    assert np.isnan(left).all() and left.shape == (4, 21, 3)


# --------------------------------------------------------------------------------------------- fingerspelling, registry

def test_fingerspell_members_and_adapter(tmp_path):
    assert fingerspell.parse_member("Indian/videos/train/r/1.mp4") == {
        "split": "train", "gloss": "r", "symbol": "R", "kind": "letter", "n": "1"}
    assert fingerspell.parse_member("Indian/videos/test/seven/2.mp4")["symbol"] == "7"
    assert fingerspell.parse_member("Indian/pkl_poses/train/r/1.pkl") is None
    assert fingerspell.parse_member("Indian/videos/val/r/1.mp4") is None
    paths = IslPaths(tmp_path / "isl")
    _write_manifest(paths, "openhands_fs", [{"id": "openhands_fs/train/r/1", "ok": True, "symbol": "R", "kind": "letter",
                                             "split": "train", "path": "data/isl/landmarks/openhands_fs/train/r/1.npz",
                                             "n_frames": 10, "hands_frac": 1.0}])
    c = next(fingerspell.OpenHandsFingerspellAdapter(paths=paths).clips())
    assert c.raw_gloss == "R" and c.category == "letter" and c.quality["split"] == "train"


def test_registry_and_profiles():
    reg = register({"include": object})
    assert reg["include"] is object and set(POSE_ADAPTERS) <= set(reg)
    for name, cls in POSE_ADAPTERS.items():
        assert cls.name == name and cls.commercial is False and cls.manifest_name == name
        assert "research" in cls.profiles
    assert not PROFILE_SOURCES["commercial-safe"]["sources"] and not PROFILE_SOURCES["commercial-safe"]["pretraining"]
    research = set(PROFILE_SOURCES["research"]["sources"]) | set(PROFILE_SOURCES["research"]["pretraining"])
    assert research == set(POSE_ADAPTERS)


# --------------------------------------------------------------------------------------------- script helpers

def _script():
    import importlib.util

    from app.core.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location("ingest_isl_pose_script", REPO_ROOT / "scripts" / "ingest_isl_pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_manifest_and_isl500_splits(tmp_path):
    mod = _script()
    man = mod.Manifest(tmp_path / "m.jsonl")
    man.add({"id": "a", "ok": False, "error": "x"})
    man.add({"id": "a", "ok": True})
    man.add({"id": "b", "ok": False})
    assert mod.Manifest(tmp_path / "m.jsonl").done == {"a"}
    (tmp_path / "PersonDependentTrain.csv").write_text(
        "fold_0\r\n/mnt/x/ISL_Goa_Data_h5_30fps/ISL_DATA_USER001/_mnt_x_R2_Clips_R2_user001_Absent__session7__clip000.h5\r\n")
    (tmp_path / "PersonDependentTest.csv").write_text("fold_2\n/mnt/x/ISL_DATA_USER002/Accept__s1__c2.h5\nnot a path\n")
    sp = mod.read_pd_splits(tmp_path)
    assert sp == {("user001", "Absent__session7__clip000"): "train", ("user002", "Accept__s1__c2"): "test"}


def test_inflate_metadata_pages_serve_reads_without_restart():
    rng = np.random.default_rng(11)
    page = 4096
    data = rng.uniform(-1, 1, 200 * page // 8).astype("<f8").tobytes()  # float pages: dropped
    meta = b"SNOD" + bytes(range(256)) * 3  # metadata block straddling a page boundary
    plain = bytearray(data)
    off = 50 * page - 300
    plain[off:off + len(meta)] = meta
    plain[120 * page:121 * page] = bytes(page)  # an all-zero page
    plain = bytes(plain)
    comp = _deflate(plain)
    reader = openhands.PrefetchReader(lambda s, e: comp[s:e + 1], len(comp), block=1 << 20, ahead=0)
    st = openhands.SeekableInflate(reader, 0, len(comp), len(plain), every=1 << 20)
    st.read_range(0, len(plain))  # first forward pass (what h5py's listing triggers)
    restores = st.restores
    st.seek(off)
    assert st.read(len(meta)) == meta  # crosses into the next page: kept as a neighbour
    st.seek(120 * page + 10)
    assert st.read(100) == bytes(100)
    assert st.restores == restores and 120 in st.zero_pages and len(st.meta) <= 6
    st.seek(10 * page)
    assert st.read(64) == plain[10 * page:10 * page + 64]  # a dropped float page needs a restart
    assert st.restores == restores + 1
