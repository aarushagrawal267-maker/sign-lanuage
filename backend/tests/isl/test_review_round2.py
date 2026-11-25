"""Regression tests for the adversarial review of the phase-2 fix round (docs/ISL_DESIGN.md 15.4), offline:

* a re-extracted ISLRTC clip whose FIRST extraction had 0 frames (opening intro / false title card) reaches the
  training data (the adapter used to drop it before the recut row could replace it);
* the readiness guard sees ``ingest_isl_gated.py`` runs (iSign / CISLR) but not its ``--check`` probes;
* E3 / E4 (which use the classifier) are refused together with E1 when the classifier's split cannot be shown;
* Sign Library chunks that the pretrain-dedup step never checked are not used by pre-training;
* pack licence lists ignore stale sources that are no longer in the profile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.isl.training import pose_dedup as pd
from app.isl.training import readiness as rd
from tests.isl.helpers import write_manifest
from tests.isl.isl_world import make_world


# --------------------------------------------------------------------------------------------- ISLRTC recut rows

def test_recut_row_replaces_a_zero_frame_original(tmp_path):
    from app.isl.cli import main
    from app.isl.data.manifest import read_manifest
    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex
    from app.landmarks.raw import RawSequence

    w = make_world(tmp_path)
    P = w["paths"]
    rows = read_manifest(P.manifest("islrtc"), only_ok=False)
    cid = "islrtc/A/Apple.mp4"
    src = RawSequence.load(P.resolve(next(r for r in rows if r["id"] == cid)["path"]))
    # first extraction stopped at a (false) title card before any frame: 0 frames, ok
    rows = [{**r, "n_frames": 0, "hands_frac": 0.0, "stop": "card"} if r["id"] == cid else r for r in rows]
    write_manifest(P.manifest("islrtc"), rows)
    vocab = VocabIndex.load(w["vocab"])
    assert cid not in {c.clip_id for c in load_isl_data(P, vocab, "commercial-safe", w["split"]).islrtc}
    (P.landmarks / "islrtc_recut").mkdir()
    src.save(P.landmarks / "islrtc_recut" / "A__Apple.npz")
    write_manifest(P.manifest("islrtc_recut"), [{"id": cid, "path": "data/isl/landmarks/islrtc_recut/A__Apple.npz",
                                                 "n_frames": len(src), "hands_frac": 0.9, "fps": 12.0, "stop": "eof",
                                                 "ok": True, "recut_of": "islrtc", "card_detector": "card-v2"}])
    # before the trim step: used whole from the re-extracted file
    data = load_isl_data(P, vocab, "commercial-safe", w["split"])
    c = {x.clip_id: x for x in data.islrtc}[cid]
    assert "islrtc_recut" in str(c.path) and c.start == 0 and c.end is None and c.quality.get("recut")
    # after the trim step: trimmed on the re-extracted file
    assert main(["--root", str(P.root), "trim", "--learned-model", "none"]) == 0
    data = load_isl_data(P, vocab, "commercial-safe", w["split"])
    c = {x.clip_id: x for x in data.islrtc}[cid]
    assert "islrtc_recut" in str(c.path) and c.end is not None
    assert data.islrtc_stats.get("recut_used") == 1


# --------------------------------------------------------------------------------------------- readiness

def _proc(root: Path, pid: int, argv: list[str]) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_readiness_sees_gated_ingestion_but_not_its_probes(tmp_path):
    proc = tmp_path / "proc"
    _proc(proc, 101, ["/v/bin/python", "scripts/ingest_isl_gated.py", "isign", "--mode", "zip"])
    _proc(proc, 102, ["/v/bin/python", "scripts/ingest_isl_gated.py", "--check", "cislr"])
    _proc(proc, 103, ["/v/bin/python", "scripts/ingest_isl_gated.py"])  # no source = access check only
    _proc(proc, 104, ["/v/bin/python", "/r/scripts/ingest_isl_gated.py", "cislr", "--workers", "2"])
    _proc(proc, 105, ["/v/bin/python", "scripts/ingest_isl_gated.py", "cislr", "--check"])
    found = rd.ingestion_processes(proc, None, self_pid=1)
    assert [(p["pid"], p["subcommand"]) for p in found] == [(101, "isign"), (104, "cislr")]
    assert all(p["blocking"] for p in found)


def test_include_is_complete_once_every_zip_is_processed(tmp_path):
    from app.isl.paths import isl_paths

    P = isl_paths(tmp_path / "isl")
    P.manifests.mkdir(parents=True)
    write_manifest(P.manifest("include"), [{"id": f"include/x/{i}", "ok": True} for i in range(10)])
    st = rd.source_status(P, "include")
    assert st["complete"] is False and st["expected"] == rd.KNOWN_TOTALS["include"]
    (P.manifests / "include_zips_done.txt").write_text("".join(f"z{i}.zip\n" for i in range(rd.INCLUDE_ZIPS - 1)))
    assert rd.source_status(P, "include")["complete"] is False
    with (P.manifests / "include_zips_done.txt").open("a") as fh:
        fh.write("last.zip\n")
    st = rd.source_status(P, "include")
    assert st["complete"] is True and "zips" in st["expected_from"]


# --------------------------------------------------------------------------------------------- classifier guard

class _Cls:
    def __init__(self, meta):
        self.meta = meta
        self.class_names = ["a"]

    def predict(self, f):  # pragma: no cover - must never be called when refused
        raise AssertionError("a refused classifier must not be used")


def test_classifier_split_status_refuses_every_classifier_protocol():
    from app.isl.evaluation.runners import classifier_refusal, refused_entry

    assert classifier_refusal(None, "s") is None  # no classifier: nothing to refuse
    assert classifier_refusal(_Cls({"split": "s"}), "s") is None
    why = classifier_refusal(_Cls({}), "s")
    assert why and "no 'split'" in why
    assert "trained on split 'other'" in classifier_refusal(_Cls({"split": "other"}), "s")
    e = refused_entry("E3 fused", why)
    assert e["run"] is False and e["reason"].startswith("refused:") and e["leakage"] == why and e["n"] == 0


# --------------------------------------------------------------------------------------------- Sign Library

class _C:
    def __init__(self, cid, cat="Sign_Library", source="openhands_raw"):
        self.clip_id, self.category, self.source = cid, cat, source


def test_apply_dedup_drops_sign_library_chunks_the_dedupe_never_checked():
    clips = [_C("sl/1"), _C("sl/2"), _C("sl/new"), _C("nios/1", "NIOS"), _C("glosl/1", None, "glosl")]
    dd = {"reliable": True, "drop_ids": ["sl/2"], "drop_channels": [], "checked_ids": ["sl/1", "sl/2"]}
    keep, rep = pd.apply_dedup(clips, dd)
    assert [c.clip_id for c in keep] == ["sl/1", "nios/1", "glosl/1"]
    assert rep["unchecked_dropped"] == 1
    # a result without the checked-id list (older file) cannot vouch for any chunk: channel dropped
    keep, rep = pd.apply_dedup(clips, {"reliable": True, "drop_ids": [], "drop_channels": []})
    assert [c.clip_id for c in keep] == ["nios/1", "glosl/1"]
    # "skipped" (no Sign Library chunks when the step ran) vouches for none of the chunks that arrived later
    keep, _ = pd.apply_dedup(clips, {"skipped": True, "reliable": True, "drop_ids": [], "drop_channels": [],
                                     "checked_ids": []})
    assert [c.clip_id for c in keep] == ["nios/1", "glosl/1"]


def test_dedupe_result_lists_every_checked_chunk():
    rng = np.random.default_rng(0)

    class _SL:
        def __init__(self, cid, member, group, v):
            self.clip_id, self.quality, self.group, self.v = cid, {"member": member}, group, v
            self.category, self.source = "Sign_Library", "openhands_raw"

    def walk(n, seed):
        r = np.random.default_rng(seed)
        return np.cumsum(r.normal(0, 0.02, (n, 14)), 0).astype(np.float32) + 0.5

    sl = [_SL(f"one/{k}", "sign_library_1_min_splits.h5", f"v{k}", walk(80, k)) for k in range(6)]
    sl.append(_SL("sub/x", "sign_library_sub_based.h5", "x", walk(40, 99)))
    orig = pd._vectors_of
    pd._vectors_of = lambda item: item.v if hasattr(item, "v") else orig(item)
    try:
        res = pd.dedupe_sign_library(sl, [], n_calib=20, seed=int(rng.integers(10)))
    finally:
        pd._vectors_of = orig
    doc = res.to_json()
    assert sorted(doc["checked_ids"]) == sorted(c.clip_id for c in sl)


# --------------------------------------------------------------------------------------------- licences

def test_pack_licences_ignore_sources_outside_the_profile(tmp_path):
    from app.isl.training.pipeline import pack_licences

    (tmp_path / "sources.json").write_text(json.dumps({
        "wlasl": {"available": True, "licence": "C-UDA", "commercial": False, "used": {"words": 10}},
        "isl500": {"available": True, "licence": "CC-BY-NC", "commercial": False, "used": {"words": 5}},
        "mendeley_agri": {"available": True, "licence": "CC-BY-4.0", "commercial": True, "used": {"words": 2}}}))
    names = [x["source"] for x in pack_licences(tmp_path, profile="commercial-safe")]
    assert names == ["include", "islrtc", "mendeley_agri"]
    names = [x["source"] for x in pack_licences(tmp_path, profile="research")]
    assert "isl500" in names and "wlasl" not in names
