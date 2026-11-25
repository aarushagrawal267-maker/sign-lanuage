"""Video/image source ingestion without network: safe zip members, hashing, listings, the resumable engine,
adapters and the queue helpers of scripts/ingest_isl_video.py."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.core.config import REPO_ROOT
from app.isl.data.manifest import read_manifest
from app.isl.data.video_sources import listings as L
from app.isl.data.video_sources.adapters import VIDEO_ADAPTERS, FingerspellingAdapter, WslpAdapter, letter_samples
from app.isl.data.video_sources.catalog import COMMERCIAL_SAFE, RESEARCH, SOURCES, get_source, profile_sources
from app.isl.data.video_sources.engine import Fetcher, load_items, dump_items, out_path, run, summarise
from app.isl.data.video_sources.extract import fill_hands, hand_results
from app.isl.data.video_sources.fetch import (
    UnsafeMember,
    check_member,
    copy_hashed,
    safe_member_name,
    temp_path,
    url_to_file,
    zip_member_to_file,
)
from app.isl.paths import isl_paths
from app.landmarks.raw import RawSequence

from .helpers import make_trim_sequence


# ------------------------------------------------------------------------------------------- catalogue

def test_catalogue_profiles_and_licences():
    for s in SOURCES.values():
        assert s.profile in (COMMERCIAL_SAFE, RESEARCH)
        assert s.licence and s.url.startswith("https://") and s.citation
        assert s.mode in ("full", "hands", "image")
        if "NC" in s.licence or s.licence == "unknown" or s.gated:
            assert s.profile == RESEARCH, s.name  # non-commercial or unknown licences never in commercial-safe
    safe = profile_sources(COMMERCIAL_SAFE)
    assert "wslp" not in safe and "isl_fingerspelling" not in safe and "cislr" not in safe
    assert {"isl_csltr", "mendeley_static", "mendeley_emergency", "mendeley_agri"} <= set(safe)
    assert set(safe) <= set(profile_sources(RESEARCH))
    with pytest.raises(ValueError):
        get_source("nope")
    assert set(VIDEO_ADAPTERS) <= set(SOURCES)


# ------------------------------------------------------------------------------------------- fetch safety

@pytest.mark.parametrize("bad", ["../x.mp4", "a/../../x.mp4", "/etc/passwd", "C:/x.mp4", "..\\x.mp4", ""])
def test_safe_member_name_rejects(bad):
    with pytest.raises(UnsafeMember):
        safe_member_name(bad)


def test_safe_member_name_ok():
    assert safe_member_name("train\\abc.mp4") == "train/abc.mp4"
    assert safe_member_name("ISL Images/अ/x (1).jpg") == "ISL Images/अ/x (1).jpg"


def test_check_member_limits():
    zi = zipfile.ZipInfo("a.mp4")
    zi.file_size, zi.compress_size = 100 << 20, 100_000  # 1000:1 at 100 MB
    with pytest.raises(UnsafeMember):
        check_member(zi)
    zi.file_size = 3 << 30
    with pytest.raises(UnsafeMember):
        check_member(zi)
    zi.file_size, zi.compress_size = 5 << 20, 4 << 20
    check_member(zi)


def _open_fds() -> int:
    return len(os.listdir("/proc/self/fd")) if Path("/proc/self/fd").exists() else 0


def test_temp_files_close_descriptors(tmp_path):
    before = _open_fds()
    for _ in range(20):
        p = temp_path(tmp_path, "v_", ".mp4")
        copy_hashed(io.BytesIO(b"x" * 1000), p)
        p.unlink()
    assert _open_fds() == before  # no leaked descriptors (deleted files would keep their disk space)


def test_zip_member_hash_and_crc(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("train/a.mp4", b"hello video")
    zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
    sha, n, crc = zip_member_to_file(zf, "train/a.mp4", tmp_path / "x")
    import hashlib
    import zlib

    assert sha == hashlib.sha256(b"hello video").hexdigest() and n == 11 and crc == zlib.crc32(b"hello video")


class _R:
    def __init__(self, code, body=b"", url="https://x", js=None):
        self.status_code, self._b, self.url, self._js = code, body, url, js

    def iter_content(self, n):
        for i in range(0, len(self._b), 4):
            yield self._b[i: i + 4]

    def json(self):
        return self._js

    def close(self):
        pass


def test_url_to_file_retries_then_fails_fast_on_403(tmp_path):
    calls = []

    def get(url, **kw):
        calls.append(url)
        return _R(500) if len(calls) == 1 else _R(200, b"abcdefghij")

    sha, n = url_to_file("https://h/f", tmp_path / "a", get=get, sleep=lambda s: None)
    assert n == 10 and len(calls) == 2 and (tmp_path / "a").read_bytes() == b"abcdefghij"
    with pytest.raises(PermissionError):
        url_to_file("https://h/g", tmp_path / "b", get=lambda u, **k: _R(403), sleep=lambda s: None)


# ------------------------------------------------------------------------------------------- listings

def test_safe_name():
    assert L.safe_name("abc/def ghi.x") == "abc__def_ghi.x"
    a, b = L.safe_name("Hindi/अ"), L.safe_name("Hindi/आ")
    assert a != b and all(c.isascii() for c in a)
    assert L.safe_name("x(1)") != L.safe_name("x[1]")


def test_wslp_items():
    csv_text = ",uid,word\n0,abc,Range\n1,def,Boy\n"
    items = L.wslp_items(csv_text, [("train/", 0), ("train/abc.mp4", 10), ("train/def.mp4", 99), ("train/zzz.mp4", 5)],
                         "https://a/train.zip")
    by = {i.id: i for i in items}
    assert by["wslp/abc"].gloss == "Range" and by["wslp/abc"].order == 10 and by["wslp/abc"].meta["labelled"]
    assert by["wslp/zzz"].gloss is None and not by["wslp/zzz"].meta["in_csv"]
    un = L.wslp_items("uid\n", [("test/q.mp4", 1)], "u", split="test", labelled=False)
    assert un[0].gloss is None and un[0].meta["split"] == "test"


def test_fingerspelling_items_keep_letter_timing():
    letters = ('"uid","char_index","letter","start_frame","end_frame"\n"v_seg000","1","i","10","19"\n'
               '"v_seg000","0","v","0","10"\n"v_seg000","2"," ","19","22"\n')
    ts = ("video_id,segment_index,uid,signer,signer_split,transcript,letter,abs_start_sec,abs_end_sec,start_frame,"
          "end_frame,fps,rel_start_sec,rel_end_sec,standard_split\n"
          "v,0,v_seg000,signer_01,test,vi,v,1,1.3,0,10,29.97,0,0.33,train\n")
    items = L.fingerspelling_items(letters, ts, ["videos/v_seg000.mp4", "videos/readme.txt"], lambda p: "https://h/" + p)
    assert len(items) == 1
    it = items[0]
    assert it.id == "islfs/v_seg000" and it.gloss == "vi " and it.fetch["url"] == "https://h/videos/v_seg000.mp4"
    assert [x["letter"] for x in it.meta["letters"]] == ["v", "i", " "]
    assert it.meta["group"] == "signer_01" and it.meta["src_fps"] == pytest.approx(29.97)


def test_csltr_items():
    members = [("ISL_CSLRT_Corpus/Videos_Sentence_Level/are you free today/free (2).MP4", 5),
               ("ISL_CSLRT_Corpus/Videos_Sentence_Level/are you free today/", 0),
               ("ISL_CSLRT_Corpus/Frames_Word_Level/A LOT/A LOT (1).jpg", 7),
               ("ISL_CSLRT_Corpus/Frames_Sentence_Level/x/y.jpg", 9)]
    gl = L.csltr_glosses("Sentence,SIGN GLOSSES\nare you free today,YOU FREE TODAY\n")
    s = L.csltr_sentence_items(members, "u", gl)
    assert len(s) == 1 and s[0].gloss == "are you free today" and s[0].meta["glosses"] == ["YOU", "FREE", "TODAY"]
    assert s[0].id == "csltr/are_you_free_today/free_(2)"
    w = L.csltr_word_items(members, "u")
    assert len(w) == 1 and w[0].gloss == "a lot" and w[0].meta["category"] == "word_frame"


def _folders():
    return [{"id": "r", "name": "ISL Images", "parent_id": None},
            {"id": "k", "name": "1. Kids ISL images", "parent_id": "r"},
            {"id": "kf", "name": "Kids ISL images in Full Sleeves", "parent_id": "k"},
            {"id": "kh", "name": "Kids ISL images in Half Sleeves", "parent_id": "k"},
            {"id": "kfa", "name": "English Alphabet", "parent_id": "kf"},
            {"id": "kha", "name": "English Alphabet", "parent_id": "kh"},
            {"id": "kfa_e1", "name": "E1", "parent_id": "kfa"},
            {"id": "kha_e1", "name": "E1", "parent_id": "kha"},
            {"id": "kfn", "name": "Numerals", "parent_id": "kf"},
            {"id": "kfn_9b", "name": "9b", "parent_id": "kfn"},
            {"id": "kfv", "name": "Hindi Vowels", "parent_id": "kf"},
            {"id": "kfv_a", "name": "अ", "parent_id": "kfv"}]


def _mfile(fid, folder, name="x.jpg"):
    return {"id": fid, "filename": name, "folder_id": folder, "size": 10,
            "content_details": {"download_url": f"https://m/{fid}", "sha256_hash": "ab" * 32}}


def test_mendeley_folder_paths_and_static_balanced_sample():
    fp = L.mendeley_folder_paths(_folders())
    assert fp["kfa_e1"] == "ISL Images/1. Kids ISL images/Kids ISL images in Full Sleeves/English Alphabet/E1"
    files = [_mfile(f"f{i}", "kfa_e1") for i in range(10)] + [_mfile(f"h{i}", "kha_e1") for i in range(3)]
    files += [_mfile("n1", "kfn_9b"), _mfile("v1", "kfv_a"), _mfile("bad", "kfv_a", "notes.txt")]
    items = L.static_items(files, fp, per_class=6)
    e1 = [i for i in items if i.meta["variant"] == "E1"]
    assert len(e1) == 6 and {i.gloss for i in e1} == {"E"}
    assert sum(1 for i in e1 if i.meta["sleeves"] == "half") == 3  # round-robin over the two groups
    assert items == L.static_items(files, fp, per_class=6)  # deterministic
    nine = next(i for i in items if i.meta["variant"] == "9b")
    assert nine.gloss == "9" and nine.meta["category"] == "digit" and nine.meta["age_group"] == "kids"
    vowel = next(i for i in items if i.meta["category"] == "hindi_vowel")
    assert vowel.gloss == "अ" and vowel.fetch["sha256"] == "ab" * 32
    assert len(items) == 8


def test_balanced_sample_caps_and_spreads():
    by = {"A": {"g1": [f"a{i}" for i in range(50)], "g2": ["b1", "b2"]}, "B": {"g1": ["c1"]}}
    out = L.balanced_sample(by, 5)
    assert len(out["A"]) == 5 and {"b1", "b2"} <= set(out["A"]) and out["B"] == ["c1"]


def test_agri_and_emergency_items():
    folders = [{"id": "d", "name": "DataSet", "parent_id": None}, {"id": "s1", "name": "Subject 1", "parent_id": "d"},
               {"id": "o", "name": "Original", "parent_id": "s1"}, {"id": "c", "name": "cropped1", "parent_id": "s1"}]
    fp = L.mendeley_folder_paths(folders)
    files = [_mfile("1", "o", "Cococnut_001_01.mp4"), _mfile("2", "c", "Coconut_001_01_crop1.avi")]
    items = L.agri_items(files, fp)
    assert len(items) == 1 and items[0].gloss == "coconut" and items[0].meta["group"] == "agri-subject1"
    efold = [{"id": "r", "name": "Raw_Data", "parent_id": None}, {"id": "c", "name": "Cropped_Data", "parent_id": None}]
    arch = L.emergency_archives([_mfile("a", "r", "call_Raw.zip"), _mfile("b", "c", "call_Cropped.zip")],
                                L.mendeley_folder_paths(efold))
    assert [a["word"] for a in arch] == ["call"]
    ei = L.emergency_items("call", "u", [("call_Raw/call_007_02.AVI", 3)])
    assert ei[0].id == "emergency/call/call_007_02" and ei[0].meta["group"] == "emergency-p007"


def test_hf_rows_items():
    rows = [(i, i % 3) for i in range(30)]
    items = L.hf_rows_items("hf_x", "u/d", "train", rows, ["1", "a", "B"], per_class=4)
    assert len(items) == 12
    assert {i.gloss for i in items} == {"1", "A", "B"}
    assert {i.meta["category"] for i in items} == {"digit", "letter"}
    assert items[0].fetch == {"kind": "hfrow", "dataset": "u/d", "split": "train", "row": items[0].meta["row"]}


# ------------------------------------------------------------------------------------------- engine

def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, v in members.items():
            zf.writestr(k, v)
    return buf.getvalue()


def _fake_extract(calls):
    def extract(mode, src, out, step_fps, max_s):
        data = Path(src).read_bytes()
        calls.append((mode, data, step_fps, max_s))
        if data == b"broken":
            return {"ok": False, "error": "ValueError: cannot open"}
        seq = make_trim_sequence([("sign", 10)])
        seq.save(Path(out))
        return {"ok": True, "n_frames": len(seq), "fps": 12.0, "hands_frac": 1.0}
    return extract


def test_engine_run_resume_and_cleanup(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("wslp")
    data = _zip_bytes({"train/a.mp4": b"clip-a", "train/b.mp4": b"broken", "train/c.mp4": b"clip-c",
                       "train/d.mp4": b"clip-a"})
    url = "https://h/train.zip"
    zf = zipfile.ZipFile(io.BytesIO(data))
    members = [(i.filename, i.header_offset) for i in zf.infolist()]
    items = L.wslp_items(",uid,word\n0,a,Apple\n1,b,Ball\n2,c,Cat\n3,d,Apple\n", members, url)
    calls = []
    st = run(spec, items, paths, Fetcher({url: zf}), extract=_fake_extract(calls), min_free_gb=0)
    assert st["ok"] == 2 and st["failed"] == 1 and st["todo"] == 4 and st["duplicates"] == 1
    assert len(calls) == 3  # the byte-identical d.mp4 is not extracted again
    rows = read_manifest(paths.manifest("wslp"), only_ok=False)
    by = {r["id"]: r for r in rows}
    assert by["wslp/a"]["gloss"] == "Apple" and by["wslp/a"]["sha256"] and by["wslp/a"]["bytes"] == 6
    assert by["wslp/a"]["path"] == "data/isl/landmarks/wslp/a.npz" and by["wslp/a"]["profile"] == RESEARCH
    assert paths.resolve(by["wslp/a"]["path"]).exists()
    assert not by["wslp/b"]["ok"]
    assert by["wslp/d"]["duplicate_of"] == "wslp/a" and by["wslp/d"]["path"] == by["wslp/a"]["path"]
    assert by["wslp/d"]["gloss"] == "Apple" and by["wslp/d"]["n_frames"] == by["wslp/a"]["n_frames"]
    assert calls[0][0] == "full" and calls[0][2] == 15.0
    assert not any((paths.tmp / "video_wslp").iterdir())  # temp videos deleted
    # resume: only the failed item is retried
    calls.clear()
    st2 = run(spec, items, paths, Fetcher({url: zipfile.ZipFile(io.BytesIO(data))}), extract=_fake_extract(calls),
              min_free_gb=0)
    assert st2["todo"] == 1 and [c[1] for c in calls] == [b"broken"]
    # the adapter sees the two good labelled clips
    clips = list(WslpAdapter(paths=paths).clips())
    assert sorted(c.raw_gloss for c in clips) == ["Apple", "Cat"] and all(c.licence == "CC-BY-NC-ND-4.0" for c in clips)
    s = summarise(paths.manifest("wslp"))
    assert s["ok"] == 3 and s["failed"] == 1 and s["glosses"] == 2 and s["duplicates"] == 1


def test_engine_disk_guard_and_time_limit(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("wslp")
    data = _zip_bytes({f"train/{k}.mp4": b"x" + k.encode() for k in "abcd"})
    zf = zipfile.ZipFile(io.BytesIO(data))
    items = L.wslp_items("uid\n", [(i.filename, i.header_offset) for i in zf.infolist()], "u", labelled=False)
    st = run(spec, items, paths, Fetcher({"u": zf}), extract=_fake_extract([]), min_free_gb=1e9)
    assert st["stopped"] == "disk" and st["ok"] == 0
    t = iter([0.0] + [0.0] * 3 + [1e6] * 100)
    st = run(spec, items, paths, Fetcher({"u": zipfile.ZipFile(io.BytesIO(data))}), extract=_fake_extract([]),
             min_free_gb=0, max_minutes=1, clock=lambda: next(t))
    assert st["stopped"] == "time" and 0 < st["ok"] < 4


def test_engine_url_fetch_sha_check_and_hfrow(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("mendeley_static")
    import hashlib

    good = hashlib.sha256(b"img1").hexdigest()
    items = [L.Item("isl_static/1", "A", {"kind": "url", "url": "https://m/1", "sha256": good}, ".jpg"),
             L.Item("isl_static/2", "B", {"kind": "url", "url": "https://m/2", "sha256": "00" * 32}, ".jpg")]
    body = {"https://m/1": b"img1", "https://m/2": b"img2"}
    f = Fetcher(get=lambda url, **k: _R(200, body[url]), sleep=lambda s: None)
    calls = []
    st = run(spec, items, paths, f, extract=_fake_extract(calls), min_free_gb=0)
    assert st["ok"] == 1 and st["fetch_failed"] == 1 and calls[0][0] == "image"
    bad = [r for r in read_manifest(paths.manifest("mendeley_static"), only_ok=False) if not r["ok"]][0]
    assert "sha256 mismatch" in bad["error"]

    def get(url, params=None, **k):
        if params:
            return _R(200, js={"rows": [{"row_idx": params["offset"] + 3, "row": {"image": {"src": "https://a/3.jpg"}}}]})
        return _R(200, b"jpegbytes")

    f2 = Fetcher(get=get, sleep=lambda s: None)
    info = f2.fetch(L.Item("hf/train/3", "A", {"kind": "hfrow", "dataset": "d", "split": "train", "row": 3}, ".jpg"),
                    tmp_path / "x.jpg")
    assert info["bytes"] == 9


def test_items_roundtrip(tmp_path):
    items = L.wslp_items(",uid,word\n0,a,A\n", [("train/a.mp4", 3)], "u")
    dump_items(tmp_path / "i.json", items)
    assert load_items(tmp_path / "i.json") == items


def test_out_path_uses_cislr_names(tmp_path):
    paths = isl_paths(tmp_path)
    it = L.cislr_items({"u 1": {"uid": "u 1", "word": "Hello"}}, {"u 1": "test"}, [("v/u 1.mp4", 0)], "z")[0]
    assert it.id == "cislr/u 1" and out_path(get_source("cislr"), paths, it).name == "u_1.npz"
    assert it.gloss == "Hello" and it.meta["split"] == "test"


# ------------------------------------------------------------------------------------------- adapters / extract

def test_letter_samples_map_frames(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    seq = make_trim_sequence([("sign", 30)])
    fps = 29.97
    seq = RawSequence(seq.pose, seq.left, seq.right, seq.face, np.arange(30) * 1000.0 / fps, 480, 480)
    p = paths.landmarks / "isl_fingerspelling" / "v.npz"
    seq.save(p)
    rec = {"id": "islfs/v", "source": "isl_fingerspelling", "gloss": "ab", "ok": True, "n_frames": 30, "hands_frac": 1.0,
           "path": "data/isl/landmarks/isl_fingerspelling/v.npz", "src_fps": fps, "group": "signer_00",
           "letters": [{"letter": "a", "start_frame": 0, "end_frame": 12}, {"letter": "b", "start_frame": 12,
                                                                            "end_frame": 30}]}
    paths.manifests.mkdir(parents=True)
    paths.manifest("isl_fingerspelling").write_text(json.dumps(rec) + "\n")
    out = list(letter_samples(FingerspellingAdapter(paths=paths)))
    assert len(out) == 1
    clip, letters = out[0]
    assert letters == [{"letter": "a", "start": 0, "end": 12}, {"letter": "b", "start": 12, "end": 30}]
    assert clip.group == "signer_00" and not FingerspellingAdapter.spec.commercial


class _Cat:
    def __init__(self, name, score=0.9):
        self.category_name, self.score = name, score


class _P:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class _Res:
    def __init__(self, hands):
        self.hand_landmarks = [[_P(x, y)] * 21 for _, x, y in hands]
        self.handedness = [[_Cat(n)] for n, _, _ in hands]


def test_fill_hands_only_fills_missing_and_counts_agreement():
    left = np.full((2, 21, 3), np.nan, np.float32)
    right = np.full((2, 21, 3), np.nan, np.float32)
    right[0, :, :2] = 0.5
    stats = {"filled": 0, "both": 0, "agree": 0}
    fill_hands(left, right, 0, hand_results(_Res([("Left", 0.2, 0.3), ("Right", 0.51, 0.5)])), stats)
    assert stats == {"filled": 1, "both": 1, "agree": 1}
    assert left[0, 0, 0] == pytest.approx(0.2) and right[0, 0, 0] == pytest.approx(0.5)  # Holistic hand kept
    assert np.isnan(left[1, 0, 0])
    # a mislabelled hand at the position of Holistic's right hand is not duplicated into "left"
    right[1, :, :2] = 0.5
    stats = {"filled": 0, "both": 0, "agree": 0}
    fill_hands(left, right, 1, hand_results(_Res([("Left", 0.5, 0.51)])), stats)
    assert stats == {"filled": 0, "both": 1, "agree": 0} and np.isnan(left[1, 0, 0])


# ------------------------------------------------------------------------------------------- CLI helpers

def _script():
    spec = importlib.util.spec_from_file_location("ingest_isl_video_t", REPO_ROOT / "scripts" / "ingest_isl_video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_auto_workers():
    m = _script()
    assert m.auto_workers(4, {"include": True, "islrtc": True}) == 1
    assert m.auto_workers(4, {"include": True, "islrtc": False}) == 1
    assert m.auto_workers(4, {"include": False, "islrtc": True}) == 1
    assert m.auto_workers(4, {"include": False, "islrtc": False}) == 3
    assert m.auto_workers(2, {}) == 1


def test_islrtc_full_targets():
    m = _script()
    index = [{"path": p} for p in ["A/Alphabet.mp4", "A/Ancient_Alphabets.mp4", "F/Fingerspell.mp4", "A/Apple.mp4",
                                   "Numbers/12_Twelve.mp4", "Numbers/Numbers_1-10_(Odisha).mp4"]]
    t = m.islrtc_full_targets(index, done={"islrtc/A/Alphabet.mp4"})
    assert t == ["A/Ancient_Alphabets.mp4", "F/Fingerspell.mp4", "Numbers/Numbers_1-10_(Odisha).mp4"]


def test_queue_script_is_safe():
    q = (REPO_ROOT / "scripts" / "isl_queue.sh").read_text()
    assert "nice -n 10" in q and "queue.log" in q and "MIN_FREE_GB" in q and "flock -n" in q
    code = "\n".join(ln for ln in q.splitlines() if not ln.lstrip().startswith("#"))
    for forbidden in ("pkill", "kill ", "git ", "rm -rf", "ingest_isl.py include", "ingest_isl.py islrtc"):
        assert forbidden not in code  # never touches the running INCLUDE / ISLRTC jobs or the checkout
