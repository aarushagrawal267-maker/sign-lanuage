"""Vocabulary builder and linking on a small fixture (WordNet off: deterministic)."""

from __future__ import annotations

import json

import pytest

from app.isl.cli import main as cli_main
from app.isl.vocab import wordnet as wnx
from app.isl.vocab.build import BuildInputs, VocabIndex, build_vocabulary, read_vocabulary, write_vocabulary
from app.isl.vocab.normalize import load_overrides

from .helpers import make_trim_sequence, write_manifest

ISLRTC = [
    "P/Pencil.mp4", "P/Pencil_(Sign_2).mp4", "P/Pencil_(Explanation).mp4", "T/Tight_(Clothes).mp4",
    "C/Cheap_Inexpensive.mp4", "Y/You.mp4", "R/Race.mp4", "R/Race_(Competition).mp4", "T/Test.mp4",
    "T/Test_(Indore).mp4", "A/Alphabet.mp4", "Numbers/12_Twelve.mp4", "S/Second_(Time).mp4",
    "A/A_bird_in_the_hand_is_worth_two_in_the_bush.mp4",
]
INCLUDE = {"Home/33. Pencil": 18, "Adjectives/21. tight": 16, "Pronouns/41. you": 17, "Pronouns/46. you (plural)": 15,
           "Society/8. Race (ethnicity)": 14, "Adjectives/16. cheap": 16, "Animals/1. Dog": 20,
           "Days_and_Time/81. Second": 12}


@pytest.fixture
def no_wordnet():
    wnx.disable(True)
    yield
    wnx.disable(False)


def _cheap_synonym_list(entries):
    # without WordNet "Cheap Inexpensive" is a plain multiword name; mark it as a synonym list via overrides
    return [{"match": r"^islrtc/C/Cheap_Inexpensive\.mp4$", "set": {"kind": "synonym_list", "synonyms": ["cheap", "inexpensive"]}}]


def test_build_and_link(no_wordnet, tmp_path):
    trims = {"islrtc/P/Pencil.mp4": {"method": "rest", "confidence": 0.9}}
    res = build_vocabulary(BuildInputs(ISLRTC, INCLUDE, [], trims, load_overrides() + _cheap_synonym_list(None)))
    by = {e.concept_id: e for e in res.entries}
    assert [e.concept_id for e in res.entries] == sorted(by)
    pencil = by["isl.pencil"]
    assert [v.variant_id for v in pencil.variants] == ["islrtc:P/Pencil", "islrtc:P/Pencil_(Sign_2)",
                                                     "islrtc:P/Pencil_(Explanation)", "include:Home/33._Pencil"]
    assert [v.index for v in pencil.variants] == [True, True, False, True]
    assert pencil.variants[0].trim == {"method": "rest", "confidence": 0.9}
    assert pencil.in_classifier and pencil.category == "home" and pencil.fingerspell == "PENCIL"
    inc = pencil.variants[-1]
    assert inc.link == "exact" and not inc.needs_review and inc.clips_count == 18
    assert {s.licence for s in pencil.sources} == {"GODL-India", "CC-BY-4.0"}
    # qualified: the only dictionary entry is "Tight (Clothes)"
    t = by["isl.tight~clothes"]
    assert any(v.source == "include" and v.link == "qualified" and v.needs_review for v in t.variants)
    # synonym-list member
    assert any(v.source == "include" and v.link == "synonym" for v in by["isl.cheap-inexpensive"].variants)
    # a different sense is never merged: you (plural) is its own concept, suggested next to "you"
    assert "isl.you~plural" in by and by["isl.you~plural"].see_also == ["isl.you"]
    assert any(v.source == "include" for v in by["isl.you"].variants)
    assert "isl.race~ethnicity" in by and "isl.race" in by["isl.race~ethnicity"].see_also
    assert any(v.link == "exact" for v in by["isl.second~time"].variants if v.source == "include") is False
    # unlinked word -> new concept
    assert by["isl.dog"].variants[0].link == "new" and by["isl.dog"].in_classifier
    # regional variant and compilation
    test = by["isl.test"]
    assert {v.region for v in test.variants} == {None, "Indore, Madhya Pradesh"}
    assert by["isl.alphabet"].indexable is False
    assert by["isl.twelve"].number_value == 12 and by["isl.twelve"].category == "numbers"
    phrase = by["isl.a-bird-in-the-hand-is-worth-two-in-the-bush"]
    assert phrase.kind == "phrase" and phrase.variants[0].full_clip
    assert res.report["include"]["words"] == len(INCLUDE)
    # round trip + index
    out = tmp_path / "vocabulary.jsonl"
    write_vocabulary(res.entries, out)
    back = read_vocabulary(out)
    assert [e.concept_id for e in back] == [e.concept_id for e in res.entries]
    assert back[0].model_dump() == res.entries[0].model_dump()
    idx = VocabIndex(back)
    assert idx.concept_of_clip("islrtc/P/Pencil_(Sign_2).mp4") == ("isl.pencil", "islrtc:P/Pencil_(Sign_2)")
    assert idx.concept_of_variant("include:Home/33._Pencil") == "isl.pencil"
    assert idx.class_index["isl.pencil"] == [e.concept_id for e in back].index("isl.pencil")


def test_second_time_links_qualified(no_wordnet):
    res = build_vocabulary(BuildInputs(ISLRTC, {"Days_and_Time/81. Second": 12}, [], {}, []))
    by = {e.concept_id: e for e in res.entries}
    v = [v for v in by["isl.second~time"].variants if v.source == "include"]
    assert v and v[0].link == "qualified" and v[0].needs_review


def test_cli_vocab_trim_status_splits(no_wordnet, tmp_path, capsys):
    root = tmp_path / "isl"
    (root / "manifests").mkdir(parents=True)
    (root / "manifests" / "islrtc_index.json").write_text(json.dumps([{"path": p, "size": 1} for p in ISLRTC]))
    (root / "landmarks" / "islrtc").mkdir(parents=True)
    (root / "landmarks" / "include").mkdir(parents=True)
    rows = []
    for p in ("P/Pencil.mp4", "T/Test.mp4"):
        raw = make_trim_sequence([("rest", 4), ("sign", 20), ("rest", 6), ("expl", 24)])
        fn = p.replace("/", "__").rsplit(".", 1)[0] + ".npz"
        raw.save(root / "landmarks" / "islrtc" / fn)
        rows.append({"id": f"islrtc/{p}", "path": f"data/isl/landmarks/islrtc/{fn}", "n_frames": len(raw),
                     "hands_frac": 0.8, "stop": "max_s", "ok": True})
    write_manifest(root / "manifests" / "islrtc.jsonl", rows)
    inc = []
    for i in range(6):
        raw = make_trim_sequence([("rest", 3), ("sign", 16), ("rest", 3)], seed=i)
        raw.save(root / "landmarks" / "include" / f"p{i}.npz")
        inc.append({"id": f"include/Home/33._Pencil/MVI_{100 + i}", "member": f"Home/33. Pencil/MVI_{100 + i}.MOV",
                    "path": f"data/isl/landmarks/include/p{i}.npz", "n_frames": len(raw), "hands_frac": 0.8, "ok": True})
    write_manifest(root / "manifests" / "include.jsonl", inc)
    off = root / "splits" / "official"
    off.mkdir(parents=True)
    (off / "include_train.txt").write_text("\n".join(f"Home/33. Pencil/MVI_{100 + i}.MOV" for i in range(4)))
    (off / "include_val.txt").write_text("Home/33. Pencil/MVI_104.MOV")
    (off / "include_test.txt").write_text("Home/33. Pencil/MVI_105.MOV\nHome/34. Pen/MVI_200.MOV")
    vocab = tmp_path / "vocabulary.jsonl"
    assert cli_main(["--root", str(root), "vocab", "--out", str(vocab), "--no-wordnet"]) == 0
    assert vocab.exists()
    assert cli_main(["--root", str(root), "trim"]) == 0
    trims = [json.loads(x) for x in (root / "derived" / "islrtc_trim.jsonl").read_text().splitlines()]
    assert {t["id"] for t in trims} == {"islrtc/P/Pencil.mp4", "islrtc/T/Test.mp4"}
    assert all(t["method"] in ("rest", "dip") for t in trims)
    assert cli_main(["--root", str(root), "status", "--offline"]) == 0
    assert cli_main(["--root", str(root), "splits", "--no-fetch", "--vocab", str(vocab), "--k", "2"]) == 0
    off_split = json.loads((root / "splits" / "include-official.json").read_text())
    assert len(off_split["assign"]) == 6
    assert off_split["unmatched"]["not_ingested"]["test"] == 1
    rep = json.loads((root / "splits" / "splits_report.json").read_text())
    assert rep["include-official"]["matched"] == 6
    assert (root / "splits" / "oneshot_concepts.json").exists()
    assert cli_main(["--root", str(root), "profiles"]) == 0
    capsys.readouterr()
