"""My signs: enrolment rules, storage, user index, sign-pack export/import and its safety checks."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from app.dataset.synthetic import make_sequence
from app.features.spec import FeatureConfig
from app.isl.models.encoder import FeatureStatsEmbedder, RawEmbedder
from app.isl.mysigns.packfile import PackError, export_pack, import_pack, read_pack
from app.isl.mysigns.store import MySignsStore, take_quality
from app.landmarks.raw import RawSequence

FCFG = FeatureConfig(seq_len=24, canonical_hand=True)


def _emb():
    return RawEmbedder(FeatureStatsEmbedder(FCFG), FCFG)


def _takes(cls: int, n: int = 3, frames: int = 40):
    return [make_sequence(cls, signer=s, n_frames=frames, fps=25.0) for s in range(n)]


def test_take_quality_rules():
    ok = take_quality(make_sequence(1, 0, n_frames=40, fps=25.0))
    assert ok["ok"], ok
    too_short = take_quality(make_sequence(1, 0, n_frames=5, fps=25.0))
    assert not too_short["ok"] and any("duration" in p for p in too_short["problems"])
    no_hands = make_sequence(1, 0, n_frames=40, fps=25.0)
    no_hands.left[:] = np.nan
    no_hands.right[:] = np.nan
    assert not take_quality(no_hands)["ok"]


def test_add_dry_run_save_update_delete(tmp_path):
    st = MySignsStore(tmp_path, "isl")
    emb = _emb()
    r = st.add("Poha", _takes(2), embedder=emb, region="Indore, MP", dry_run=True)
    assert r.sign is None and r.consistency is not None and r.self_test["n"] == 3 and st.list() == []
    r = st.add("Poha", _takes(2), embedder=emb, region="Indore, MP", text_hi="पोहा")
    assert r.accepted and r.sign is not None and r.sign.n_takes == 3 and r.sign.theta is not None
    sid = r.sign.sign_id
    assert [s.text_en for s in st.list()] == ["Poha"] and len(st.takes(sid)) == 3
    row = json.loads(st.file.read_text().splitlines()[-1])
    assert row["gloss"] == "Poha" and row["takes"] == [f"takes/{sid}/{i}.npz" for i in range(3)]  # MySignsAdapter format
    st.update(sid, enabled=False, text_en="Poha\x07 jalebi")
    assert st.get(sid).enabled is False and st.get(sid).text_en == "Poha jalebi"
    assert st.user_index(emb) is None  # disabled signs are not searched
    st.update(sid, enabled=True)
    idx = st.user_index(emb)
    assert idx is not None and len(idx) == 3 and idx.concepts == [f"mysign:{sid}"]
    assert st.user_index(emb) is idx  # cached until signs.jsonl changes
    st.delete(sid)
    assert st.list() == [] and not (st.dir / "takes" / sid).exists()


def test_add_rejections(tmp_path):
    st = MySignsStore(tmp_path, "isl")
    emb = _emb()
    assert not st.add("x", _takes(1, n=2), embedder=emb).accepted  # too few takes
    assert not st.add("", _takes(1), embedder=emb).accepted
    bad = _takes(1)
    bad[1] = make_sequence(1, 0, n_frames=4)
    assert "could not be used" in st.add("x", bad, embedder=emb).message
    mixed = [make_sequence(c, 0, n_frames=40, fps=25.0) for c in (0, 5, 9)]
    r = st.add("x", mixed, embedder=emb, c_min=0.99)
    assert not r.accepted and "look different" in r.message


def test_conflict_with_existing_user_sign(tmp_path):
    st = MySignsStore(tmp_path, "isl")
    emb = _emb()
    assert st.add("first", _takes(4), embedder=emb).accepted
    r = st.add("second", _takes(4), embedder=emb, dry_run=True)
    assert r.conflicts and r.conflicts[0]["name"].startswith("first")


def test_signpack_roundtrip_without_face(tmp_path):
    a = MySignsStore(tmp_path / "a", "isl")
    emb = _emb()
    s1 = a.add("Sarafa", _takes(3), embedder=emb, region="Indore").sign
    data = export_pack(a, None, title="St. X School, Indore", region="Indore, Madhya Pradesh", licence="CC-BY-4.0")
    man, signs = read_pack(data)
    assert man["format"] == "sign-pack" and man["n_signs"] == 1 and "permission" in man["consent"]
    assert np.isnan(signs[0][1][0].face).all()  # face left out by default
    b = MySignsStore(tmp_path / "b", "isl")
    out = import_pack(b, data, "merge", encoder_version="featstats-1")
    assert out["imported"] == 1 and out["skipped"] == 0
    got = b.list()[0]
    assert got.sign_id == s1.sign_id and got.origin == "imported:St. X School, Indore" and got.region == "Indore"
    assert import_pack(b, data, "merge")["skipped"] == 1
    assert import_pack(b, data, "replace")["imported"] == 1 and len(b.list()) == 1


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return buf.getvalue()


def test_signpack_import_rejects_unsafe_content(tmp_path):
    st = MySignsStore(tmp_path, "isl")
    man = json.dumps({"format": "sign-pack", "version": 1, "pack": "isl", "title": "t"}).encode()
    with pytest.raises(PackError, match="unexpected file"):
        read_pack(_zip({"manifest.json": man, "signs.jsonl": b"", "../evil.py": b"x"}))
    with pytest.raises(PackError, match="not a sign pack"):
        read_pack(b"not a zip")
    sid = "12345678-1234-1234-1234-123456789abc"
    row = json.dumps({"sign_id": sid, "text_en": "x", "n_takes": 1}).encode()
    bad_npz = io.BytesIO()
    np.savez(bad_npz, pose=np.zeros((3, 2, 4)), left=np.zeros((3, 21, 3)), right=np.zeros((3, 21, 3)),
             face=np.zeros((3, 20, 3)), t_ms=np.arange(3.0), width=640, height=480)
    with pytest.raises(PackError, match="shape"):
        read_pack(_zip({"manifest.json": man, "signs.jsonl": row, f"takes/{sid}/0.npz": bad_npz.getvalue()}))
    obj_npz = io.BytesIO()
    np.savez(obj_npz, pose=np.array([{"a": 1}], dtype=object))
    with pytest.raises(PackError, match="not a valid landmark file"):
        read_pack(_zip({"manifest.json": man, "signs.jsonl": row, f"takes/{sid}/0.npz": obj_npz.getvalue()}))
    with pytest.raises(PackError, match="sign id"):
        read_pack(_zip({"manifest.json": man, "signs.jsonl": json.dumps({"sign_id": "../../x", "n_takes": 1}).encode()}))
    other = json.dumps({"format": "sign-pack", "version": 1, "pack": "asl-wlasl100", "title": "t"}).encode()
    good = RawSequence.from_frames([{"t": 0, "pose": None, "left": None, "right": None, "face": None}], 640, 480)
    buf = io.BytesIO()
    np.savez(buf, pose=good.pose, left=good.left, right=good.right, face=good.face, t_ms=good.t_ms, width=640, height=480)
    with pytest.raises(PackError, match="is for"):
        import_pack(st, _zip({"manifest.json": other, "signs.jsonl": row, f"takes/{sid}/0.npz": buf.getvalue()}))


# ------------------------------------------------------------------------------------- code-review fixes

def _good_npz(width=640) -> bytes:
    good = RawSequence.from_frames([{"t": 0}], 640, 480)
    buf = io.BytesIO()
    np.savez(buf, pose=good.pose, left=good.left, right=good.right, face=good.face, t_ms=good.t_ms, width=width,
             height=480)
    return buf.getvalue()


SID = "12345678-1234-1234-1234-123456789abc"
MAN = json.dumps({"format": "sign-pack", "version": 1, "pack": "isl", "title": "t"}).encode()


@pytest.mark.parametrize("n_takes", ["abc", [1], "1", 1.5, True, None, 9])
def test_signpack_bad_take_count_is_a_pack_error(n_takes):
    """Malformed rows must give a clear PackError (HTTP 422), never an unhandled ValueError/TypeError (500)."""
    row = json.dumps({"sign_id": SID, "text_en": "x", "n_takes": n_takes}).encode()
    with pytest.raises(PackError, match="takes"):
        read_pack(_zip({"manifest.json": MAN, "signs.jsonl": row, f"takes/{SID}/0.npz": _good_npz()}))


@pytest.mark.parametrize("width", [np.array([1, 2]), np.float64("nan"), "abc"])
def test_signpack_bad_frame_size_is_a_pack_error(width):
    row = json.dumps({"sign_id": SID, "text_en": "x", "n_takes": 1}).encode()
    with pytest.raises(PackError):
        read_pack(_zip({"manifest.json": MAN, "signs.jsonl": row, f"takes/{SID}/0.npz": _good_npz(width)}))


def test_signpack_rejects_compressed_bomb_inside_take():
    """A take's .npz is a zip of its own: a tiny member that expands to a huge array is refused before loading."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pose.npy", b"\0" * (20 * 1024 * 1024))
    row = json.dumps({"sign_id": SID, "text_en": "x", "n_takes": 1}).encode()
    data = _zip({"manifest.json": MAN, "signs.jsonl": row, f"takes/{SID}/0.npz": inner.getvalue()})
    assert len(data) < 200_000
    with pytest.raises(PackError, match="too large"):
        read_pack(data)


def test_imported_theta_is_recomputed_not_trusted(tmp_path):
    """A pack that claims theta=-1 would otherwise let any movement be accepted as that sign."""
    a = MySignsStore(tmp_path / "a", "isl")
    emb = _emb()
    s1 = a.add("Sarafa", _takes(3), embedder=emb).sign
    data = export_pack(a, None, title="evil")
    zin = zipfile.ZipFile(io.BytesIO(data))
    members = {n: zin.read(n) for n in zin.namelist()}
    row = json.loads(members["signs.jsonl"].decode().splitlines()[0])
    row["theta"] = -1.0
    members["signs.jsonl"] = (json.dumps(row) + "\n").encode()
    tampered = _zip(members)

    b = MySignsStore(tmp_path / "b", "isl")
    import_pack(b, tampered, "merge", embedder=emb)
    got = b.get(s1.sign_id).theta
    assert got is not None and got > 0.5 and abs(got - s1.theta) < 0.01  # same takes (minus face points), same embedder

    c = MySignsStore(tmp_path / "c", "isl")
    import_pack(c, tampered, "merge")  # no embedder: never accepted directly, only suggested
    assert c.get(s1.sign_id).theta is None


def test_replace_import_refreshes_cached_embeddings(tmp_path):
    """Re-importing a sign id with different takes must not reuse the old takes' cached embeddings."""
    emb = _emb()
    a = MySignsStore(tmp_path / "a", "isl")
    s = a.add("Sign", _takes(2), embedder=emb).sign
    old = export_pack(a, None)
    a.save_sign(s, _takes(8))  # same id, new recordings
    new = export_pack(a, None)
    b = MySignsStore(tmp_path / "b", "isl")
    import_pack(b, old, "merge", embedder=emb)
    first = b.user_index(emb).emb.copy()
    import_pack(b, new, "replace", embedder=emb)
    second = b.user_index(emb).emb
    fresh = MySignsStore(tmp_path / "c", "isl")
    import_pack(fresh, new, "merge", embedder=emb)
    assert not np.allclose(first, second) and np.allclose(second, fresh.user_index(emb).emb)


def test_delete_all_and_row_cache(tmp_path):
    st = MySignsStore(tmp_path, "isl")
    emb = _emb()
    ids = [st.add(f"w{i}", _takes(i), embedder=emb).sign.sign_id for i in range(3)]
    assert st.get(ids[0]).text_en == "w0"
    st.update(ids[0], text_en="changed")
    assert st.get(ids[0]).text_en == "changed"  # cache invalidated by the append
    st.update(ids[0], concept_id="isl.pen")
    assert st.get(ids[0]).concept_id == "isl.pen"
    st.update(ids[0], concept_id="")
    assert st.get(ids[0]).concept_id is None  # "" unlinks
    n_lines = len(st.file.read_text().splitlines())
    assert st.delete_all() == 3 and st.list() == []
    assert len(st.file.read_text().splitlines()) == n_lines + 3
    assert not any((st.dir / "takes" / i).exists() for i in ids)


def test_alphabet_store_survives_bad_file_and_parallel_adds(tmp_path):
    import threading

    from app.isl.fingerspell.features import LETTER_DIM
    from app.isl.fingerspell.prototypes import AlphabetStore

    (tmp_path / "alphabet.npz").write_bytes(b"not an npz")
    assert AlphabetStore(tmp_path).counts() == {} and AlphabetStore(tmp_path).model() is None
    (tmp_path / "alphabet.npz").unlink()
    letters = "ABCDEFGHIJ"

    def add(ch):  # a new store per call, like IslPack.alphabet()
        AlphabetStore(tmp_path).add(ch, np.full(LETTER_DIM, ord(ch), np.float32))

    threads = [threading.Thread(target=add, args=(ch,)) for ch in letters]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(AlphabetStore(tmp_path).counts()) == set(letters)
