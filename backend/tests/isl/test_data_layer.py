"""Manifest reading, source adapters, trimming, splits and duplicates on synthetic fixtures."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.isl.data import splits as sp
from app.isl.data.dedup import duplicate_clusters, find_duplicates, fingerprint
from app.isl.data.manifest import read_manifest
from app.isl.data.sources import (
    Clip,
    IncludeAdapter,
    IslrtcAdapter,
    include_clip_id,
    include_key_from_parts,
    load_profile_clips,
)
from app.isl.data.trim import TrimConfig, activity_signal, find_first_sign, plot_trim, trim_record_for
from app.isl.paths import isl_paths
from app.isl.profiles import PROFILES, get_profile
from app.landmarks.raw import RawSequence

from .helpers import make_trim_sequence, write_manifest

# --------------------------------------------------------------------------------------------- manifests


def test_manifest_tolerates_partial_and_retries(tmp_path):
    p = tmp_path / "m.jsonl"
    rows = [{"id": "a", "ok": False, "error": "x"}, {"id": "b", "ok": True}, {"id": "a", "ok": True, "n": 2}]
    write_manifest(p, rows, partial_tail='{"id": "c", "ok": tr')
    out = read_manifest(p)
    assert [r["id"] for r in out] == ["a", "b"]
    assert out[0]["n"] == 2
    assert {r["id"] for r in read_manifest(p, only_ok=False)} == {"a", "b"}
    assert read_manifest(tmp_path / "missing.jsonl") == []


def test_paths_resolve(tmp_path):
    P = isl_paths(tmp_path / "isl")
    assert P.resolve("data/isl/landmarks/x.npz") == tmp_path / "isl" / "landmarks" / "x.npz"
    assert P.resolve("/abs/y.npz").as_posix() == "/abs/y.npz"


def test_profiles():
    assert get_profile("commercial-safe").commercial
    assert "isign" not in PROFILES["commercial-safe"].sources and "cislr" not in PROFILES["commercial-safe"].sources
    assert not get_profile("research").commercial
    with pytest.raises(ValueError):
        get_profile("nope")


# --------------------------------------------------------------------------------------------- adapters

def test_include_key_extra_folder():
    assert include_key_from_parts(("Adjectives", "5. Beautiful", "Extra", "MVI_9573.MOV")) == ("Adjectives/5. Beautiful", "MVI_9573")
    assert include_key_from_parts(("Adjectives", "79. short", "extra", "MVI_5192.MOV")) == ("Adjectives/79. short", "MVI_5192")
    assert include_key_from_parts(("Seasons", "63. Winter", "MVI_4997.MOV")) == ("Seasons/63. Winter", "MVI_4997")
    assert include_key_from_parts(("x.MOV",)) is None
    assert include_clip_id("Seasons/63. Winter", "MVI_4997") == "include/Seasons/63._Winter/MVI_4997"


def _fixture_data(tmp_path):
    P = isl_paths(tmp_path / "isl")
    (P.landmarks / "include").mkdir(parents=True)
    (P.landmarks / "islrtc").mkdir(parents=True)
    inc_rows = []
    for i, (member, legacy_id) in enumerate([
        ("Adjectives/1. loud/MVI_5177.MOV", "include/Adjectives/1._loud/MVI_5177"),
        ("Adjectives/1. loud/MVI_5178.MOV", "include/Adjectives/1._loud/MVI_5178"),
        ("Adjectives/5. Beautiful/Extra/MVI_9573.MOV", "include/5._Beautiful/Extra/MVI_9573"),
    ]):
        raw = make_trim_sequence([("rest", 4), ("sign", 20), ("rest", 4)], seed=i)
        path = P.landmarks / "include" / f"c{i}.npz"
        raw.save(path)
        inc_rows.append({"id": legacy_id, "member": member, "path": f"data/isl/landmarks/include/c{i}.npz",
                         "n_frames": len(raw), "hands_frac": 0.8, "fps": 12.0, "ok": True})
    inc_rows.append({"id": "include/x/y/z", "member": "Adjectives/2. quiet/MVI_1.MOV", "path": "data/isl/landmarks/include/none.npz",
                     "n_frames": 3, "hands_frac": 0.9, "ok": True})  # too short -> filtered
    write_manifest(P.manifest("include"), inc_rows)
    isl_rows = []
    for name, segs in [("A/Able.mp4", [("rest", 4), ("sign", 20), ("rest", 6), ("expl", 30)]),
                       ("A/Able_(Explanation).mp4", [("rest", 4), ("expl", 30)]),
                       ("P/Pencil_(Sign_2).mp4", [("rest", 4), ("sign", 20), ("rest", 6)])]:
        raw = make_trim_sequence(segs)
        fn = name.replace("/", "__").rsplit(".", 1)[0] + ".npz"
        raw.save(P.landmarks / "islrtc" / fn)
        isl_rows.append({"id": f"islrtc/{name}", "path": f"data/isl/landmarks/islrtc/{fn}", "n_frames": len(raw),
                         "hands_frac": 0.7, "fps": 12.0, "stop": "max_s" if "Able.mp4" in name else "eof", "ok": True})
    isl_rows.append({"id": "islrtc/A/Ab (Symbol of Line) (Math).mp4", "path": "data/isl/landmarks/islrtc/x.npz",
                     "n_frames": 0, "hands_frac": 0.0, "ok": True})
    write_manifest(P.manifest("islrtc"), isl_rows)
    return P


def test_include_adapter_canonical_ids(tmp_path):
    P = _fixture_data(tmp_path)
    clips = list(IncludeAdapter(P).clips())
    ids = [c.clip_id for c in clips]
    assert ids == ["include/Adjectives/1._loud/MVI_5177", "include/Adjectives/1._loud/MVI_5178",
                   "include/Adjectives/5._Beautiful/MVI_9573"]
    beautiful = clips[-1]
    assert beautiful.raw_gloss.lower() == "beautiful" and beautiful.category == "Adjectives"
    assert beautiful.quality["manifest_id"] == "include/5._Beautiful/Extra/MVI_9573"
    assert len(beautiful.load()) == 28


def test_islrtc_adapter_trims_and_filters(tmp_path):
    P = _fixture_data(tmp_path)
    a = IslrtcAdapter(P)
    ids = [c.clip_id for c in a.clips()]
    assert ids == ["islrtc/A/Able.mp4", "islrtc/P/Pencil_(Sign_2).mp4"]  # explanation and 0-frame rows skipped
    # trims from the sidecar are applied on load
    raw = RawSequence.load(P.landmarks / "islrtc" / "A__Able.npz")
    rec = trim_record_for("islrtc/A/Able.mp4", raw, "max_s", "word")
    write_manifest(P.trim_sidecar, [rec])
    able = next(c for c in IslrtcAdapter(P).clips() if c.clip_id == "islrtc/A/Able.mp4")
    assert (able.start, able.end) == (rec["start"], rec["end"])
    assert len(able.load()) == rec["end"] - rec["start"]
    assert able.quality["trim_method"] == rec["method"]
    assert len(list(IslrtcAdapter(P, include_non_index=True).clips())) == 3


def test_load_profile_clips(tmp_path):
    P = _fixture_data(tmp_path)
    clips = load_profile_clips("commercial-safe", P)
    assert {c.source for c in clips} == {"include", "islrtc"}
    assert all(c.licence for c in clips)


# --------------------------------------------------------------------------------------------- trimming

@pytest.mark.parametrize("gap_kind", ["rest", "low"])
def test_trim_sign_then_explanation(gap_kind):
    raw = make_trim_sequence([("rest", 10), ("sign", 24), (gap_kind, 8), ("expl", 40)])
    res = find_first_sign(raw, "max_s")
    assert res.method in ("rest", "dip")
    assert abs(res.end - 34) <= 2, res
    assert res.start <= 10
    assert res.confidence >= 0.7
    assert all(a > res.end for a in res.alternatives[:1])


def test_trim_pause_fallback():
    raw = make_trim_sequence([("rest", 6), ("sign", 20), ("hold", 10), ("expl", 40)])
    res = find_first_sign(raw, "max_s")
    assert res.method == "pause" and abs(res.end - 26) <= 4


def test_trim_bounded_clip_keeps_compound():
    raw = make_trim_sequence([("rest", 6), ("sign", 12), ("rest", 4), ("sign", 12), ("rest", 10)])
    res = find_first_sign(raw, "eof")
    assert res.method == "source_end"
    assert abs(res.end - 34) <= 2
    assert any(abs(a - 18) <= 2 for a in res.alternatives)  # the first part is kept as an alternative end


def test_trim_cap_and_empty():
    long = make_trim_sequence([("rest", 6), ("sign", 90)])
    res = find_first_sign(long, "max_s", TrimConfig())
    assert res.method == "cap" and res.confidence <= 0.3
    assert res.end - res.start <= int(4.0 * 12) + 3
    rest = find_first_sign(make_trim_sequence([("rest", 20)]), "max_s")
    assert rest.method == "empty"
    assert find_first_sign(RawSequence.empty(), None).method == "empty"


def test_trim_multiword_longer_minimum():
    raw = make_trim_sequence([("rest", 6), ("sign", 8), ("low", 3), ("sign", 14), ("rest", 6), ("expl", 30)])
    one = find_first_sign(raw, "max_s", n_words=1)
    two = find_first_sign(raw, "max_s", n_words=3)
    assert two.end > one.end


def test_trim_phrase_full_and_plot(tmp_path):
    pytest.importorskip("matplotlib")
    raw = make_trim_sequence([("rest", 6), ("sign", 20), ("rest", 6), ("expl", 20)])
    rec = trim_record_for("islrtc/A/x.mp4", raw, "max_s", "phrase")
    assert rec["method"] == "full" and rec["end"] == len(raw)
    sig = activity_signal(raw)
    assert set(sig) >= {"speed", "shape_change", "rest", "dip", "up_left", "wrist_y_right"}
    out = tmp_path / "p.png"
    plot_trim(raw, find_first_sign(raw, "max_s"), out, title="t")
    assert out.stat().st_size > 1000


# --------------------------------------------------------------------------------------------- splits

def _include_clips(n_words=6, signers=4, takes=3):
    clips, num = [], 1000
    for s in range(signers):
        for w in range(n_words):
            for _ in range(takes):
                word = f"Cat/{w + 1}. w{w}"
                clips.append(Clip(include_clip_id(word, f"MVI_{num}"), "include", f"w{w}", f"isl.w{w}", None, "Cat",
                                  path=None, quality={"word_dir": word}))  # type: ignore[arg-type]
                num += 1
            num += 1  # next word
        num += 200  # next signer's pass
    return clips


def test_include_official_mapping_reports_everything():
    clips = _include_clips(n_words=2, signers=1, takes=2)
    lists = {"train": ["Cat/1. w0/MVI_1000.MOV", "Cat/1. w0/Extra/MVI_1001.MOV"], "val": [],
             "test": ["Cat/2. w1/MVI_1003.MOV", "Cat/9. nope/MVI_1.MOV"]}
    assign, rep = sp.include_official(clips, lists)
    assert assign == {"include/Cat/1._w0/MVI_1000": "train", "include/Cat/1._w0/MVI_1001": "train",
                      "include/Cat/2._w1/MVI_1003": "test"}
    assert rep["not_ingested"]["test"] == 1 and rep["not_listed"] == 1
    assert rep["matched"] + sum(rep["not_ingested"].values()) == rep["list_paths"]


def test_sessions_and_grouped_kfold_no_leakage():
    clips = _include_clips()
    sessions = sp.include_sessions(clips)
    assert len(set(sessions.values())) == 6 * 4  # one session per (word, signer pass)
    class_of = {c.clip_id: c.concept_id for c in clips}
    dup = [[clips[0].clip_id, clips[-1].clip_id]]  # a duplicate pair across sessions must stay together
    for f in sp.grouped_kfold(sessions, k=5, seed=0, merge=dup):
        rep = sp.leakage_report(f, sessions, class_of, dup)
        assert rep["passed"], rep
        assert rep["group_overlap"] == {"train-val": 0, "train-test": 0, "val-test": 0}
        assert f[dup[0][0]] == f[dup[0][1]]


def test_pseudo_signers_from_descriptors():
    # 4 signers x 2 recording passes each; every pass = 5 words x 3 takes with consecutive camera numbers
    clips = _include_clips(n_words=5, signers=8, takes=3)
    signer_of = {c.clip_id: (i // 15) // 2 for i, c in enumerate(clips)}
    sessions = sp.include_sessions(clips)
    rng = np.random.default_rng(0)
    base = {s: rng.normal(0, 1, 11) * 0.5 + 1.5 for s in range(4)}
    desc = {c.clip_id: base[signer_of[c.clip_id]] + rng.normal(0, 0.01, 11) for c in clips}
    groups, rep = sp.include_pseudo_signers(clips, sessions, k_range=(2, 6), descriptors=desc)
    assert rep["blocks"] == 8  # one block per pass (camera numbers chained, a repeated word starts a new pass)
    assert rep["k"] == 4 and not rep["weak"]
    by_ps = {}
    for c in clips:
        by_ps.setdefault(groups[c.clip_id]["pseudo_signer"], set()).add(signer_of[c.clip_id])
    assert all(len(v) == 1 for v in by_ps.values())  # never mixes two signers
    assert len(by_ps) == 4


def test_oneshot_folds_stratified():
    concepts = [{"concept_id": f"isl.c{i}", "category": "a" if i % 2 else "b", "usable": i != 3} for i in range(20)]
    fold = sp.oneshot_folds(concepts, k=5, seed=0)
    assert "isl.c3" not in fold and len(fold) == 19
    counts = np.bincount(list(fold.values()), minlength=5)
    assert counts.max() - counts.min() <= 1
    clips = [Clip("include/a/1._c1/MVI_1", "include", "c1", "isl.c1", None, None, path=None)]  # type: ignore[arg-type]
    prot = sp.oneshot_protocol(fold, clips, 5)
    assert sum(len(f["queries"]) for f in prot["folds"]) == 1


def test_split_doc_written(tmp_path):
    P = isl_paths(tmp_path / "isl")
    doc = sp.split_doc("x", "session", {"a": "train"}, {"passed": True}, 0, tmp_path / "none.jsonl", "session")
    p = sp.write_split(P, doc)
    d = json.loads(p.read_text())
    assert d["label"] == sp.LABELS["session"] and d["assign"] == {"a": "train"}


# --------------------------------------------------------------------------------------------- dedup

def test_duplicates_found_and_clustered():
    a = make_trim_sequence([("rest", 4), ("sign", 20), ("rest", 4)], seed=1)
    b = make_trim_sequence([("rest", 4), ("sign", 20), ("rest", 4)], seed=1)  # identical video
    c = make_trim_sequence([("rest", 4), ("expl", 20), ("rest", 4)], seed=2)

    class C:
        def __init__(self, cid, raw):
            self.clip_id, self.raw = cid, raw

        def load(self):
            return self.raw

    pairs = find_duplicates([C("a", a), C("b", b), C("c", c)])
    assert [(x, y) for x, y, _ in pairs] == [("a", "b")]
    assert duplicate_clusters(pairs) == [["a", "b"]]
    f = fingerprint(a)
    assert f.dtype == np.float32 and abs(np.linalg.norm(f) - 1) < 1e-4
