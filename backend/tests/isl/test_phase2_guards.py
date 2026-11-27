"""Phase-2 honesty guards (docs/ISL_DESIGN.md 15.4), offline and synthetic:

1. no "95 % interval" from fewer than 5 signer groups (JSON null + reason, markdown says why);
2. the partial-data guard (process table + manifest completeness, ``--allow-partial`` marks the work folder, reports
   and packs say PARTIAL DATA; the driver refuses with exit 4);
3. a classifier without a recorded split is refused; the pre-training leakage flag is computed from the real list;
4. one source of truth for profiles, registered adapters, licence lists naming only contributing sources;
5. the Sign Library pose-content dedupe; 6. learned trims + re-extracted / truncated ISLRTC clips;
7. the E2 label; 8. the "My signs" neutral classifier term chosen on validation data.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.isl.data.sources import Clip
from app.isl.evaluation import report as rp
from app.isl.models.fusion import FusionInputs, FusionParams, fuse
from app.isl.training import multisource as ms
from app.isl.training import pose_dedup as pd
from app.isl.training import readiness as rd
from tests.isl.helpers import write_manifest
from tests.isl.isl_world import make_world

REPO = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------------------------- 1. intervals

def test_no_interval_from_fewer_than_five_groups():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    two = ["a", "a", "a", "b", "b", "b"]
    assert rp.grouped_bootstrap(vals, two) is None
    assert rp.interval_reason(vals, two) == "too few signer groups (2)"
    five = ["a", "b", "c", "d", "e", "e"]
    ci = rp.grouped_bootstrap(vals, five)
    assert ci is not None and ci[0] <= ci[1]
    f = rp.interval_fields({"top1": (vals, two), "top5": (vals, five), "empty": ([], [])})
    assert f["intervals"]["top1"] is None and f["interval_reasons"]["top1"] == "too few signer groups (2)"
    assert f["intervals"]["top5"] and "top5" not in f["interval_reasons"] and f["interval_groups"]["top5"] == 5
    assert f["interval_reasons"]["empty"] == "no values"
    assert rp.format_interval(None, 2, "too few signer groups (2)") == "none: too few signer groups (2)"
    assert rp.format_interval([0.5, 0.75], 7) == "0.500-0.750 (7 groups)"


def _doc(**kw):
    p = {"run": True, "n": 30, "n_groups": 2, "label": "E1b test", "metrics": {"top1": 0.5, "top5": 0.8},
         **rp.interval_fields({"top1": ([1.0, 0.0] * 15, ["a", "b"] * 15)})}
    return {"profile": "commercial-safe", "label": "x", "created": "now", "smoke": False, "protocols": {"E1b": p},
            "leakage_guards": {"pretraining_excludes_eval_clips": None, "pretraining_check": {"reason": "none"}}, **kw}


def test_markdown_report_says_why_there_is_no_interval_and_prints_group_counts():
    from app.isl.evaluation.runners import markdown_report

    md = markdown_report(_doc())
    assert "none: too few signer groups (2)" in md and "95% CI" in md
    doc = _doc()
    doc["protocols"]["E1b"].update(rp.interval_fields({"top1": ([1.0, 0.0] * 15, list("abcdef") * 5)}))
    assert "groups)" in markdown_report(doc)
    md_t = rp.markdown_table({"protocols": {"E1": {"label": "l", "n": 4, "metrics": {"top1": 0.5},
                                                   **rp.interval_fields({"top1": ([1.0, 0.0], ["a", "b"])})}}})
    assert "none: too few signer groups (2)" in md_t


# --------------------------------------------------------------------------------------------- 2. partial data

def _proc(root: Path, pid: int, argv: list[str]) -> None:
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


def test_ingestion_processes_found_and_queue_polling_ignored(tmp_path):
    proc = tmp_path / "proc"
    _proc(proc, 101, [".venv/bin/python", "scripts/ingest_isl.py", "islrtc", "--workers", "3"])
    _proc(proc, 102, ["python", "scripts/ingest_isl_pose.py", "status"])  # read-only: ignored
    _proc(proc, 103, ["bash", "scripts/isl_queue.sh"])
    _proc(proc, 104, ["/bin/bash", "-c", "echo scripts/isl_queue.sh is mentioned in a string"])  # not the script
    _proc(proc, 105, ["python", "-m", "pytest"])
    log = tmp_path / "queue.log"
    log.write_text("2026-11-25 queue start (pid 103): wslp\n")
    got = rd.ingestion_processes(proc, log, self_pid=1)
    assert [(p["pid"], p["blocking"]) for p in got] == [(101, True), (103, True)]
    log.write_text(log.read_text() + "2026-11-26 all listed sources processed; status:\n")
    got = rd.ingestion_processes(proc, log, self_pid=1)
    assert [(p["pid"], p["blocking"]) for p in got] == [(101, True), (103, False)]  # queue only polls CISLR now
    log.write_text(log.read_text() + "2026-11-27 queue start (pid 200): wslp\n")  # restarted: blocks again
    assert all(p["blocking"] for p in rd.ingestion_processes(proc, log, self_pid=1))


def _complete_world(tmp_path) -> rd.IslPaths:
    """Manifests of every commercial-safe source, complete against their item lists."""
    from app.isl.paths import isl_paths

    P = isl_paths(tmp_path / "isl")
    for name in ("include", "islrtc", "mendeley_emergency", "mendeley_agri", "isl_csltr", "mendeley_static"):
        ids = [f"{name}/{i}" for i in range(3)]
        write_manifest(P.manifest(name), [{"id": i, "ok": True, "n_frames": 9} for i in ids])
        (P.manifests / f"{name}_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    return P


def test_readiness_checks_processes_and_every_manifest(tmp_path):
    P = _complete_world(tmp_path)
    empty = tmp_path / "proc_empty"
    empty.mkdir()
    rep = rd.check(P, "commercial-safe", proc_root=empty)
    assert rep["ready"], rep["reasons"]
    # an incomplete manifest blocks
    (P.manifests / "mendeley_agri_items.json").write_text(json.dumps([{"id": f"mendeley_agri/{i}"} for i in range(5)]))
    rep = rd.check(P, "commercial-safe", proc_root=empty)
    assert not rep["ready"] and any("mendeley_agri 3/5" in r for r in rep["reasons"])
    # a running ingestion job blocks
    (P.manifests / "mendeley_agri_items.json").write_text(json.dumps([{"id": f"mendeley_agri/{i}"} for i in range(3)]))
    proc = tmp_path / "proc"
    _proc(proc, 300, ["python", "scripts/ingest_isl_video.py", "run", "mendeley_agri"])  # a commercial-safe source
    rep = rd.check(P, "commercial-safe", proc_root=proc)
    assert not rep["ready"] and rep["reasons"] == ["ingestion job running: pid 300 ingest_isl_video.py run"]
    # gated sources without access are unavailable, not incomplete (research profile)
    st = rd.source_status(P, "cislr")
    assert st["complete"] is None and "gated" in st["note"]
    # failed rows count as attempted; the published size is used when no item list exists
    write_manifest(P.manifest("openhands_fs"), [{"id": f"x{i}", "ok": i % 2 == 0} for i in range(172)])
    st = rd.source_status(P, "openhands_fs")
    assert st["complete"] and st["failed"] == 86


def test_partial_flag_labels_reports_and_packs(tmp_path, monkeypatch):
    monkeypatch.delenv("SLR_ISL_PARTIAL_DATA", raising=False)
    work = tmp_path / "w"
    assert not rd.partial_data_flag(work)
    rd.mark_partial(work, {"checked": "t0", "reasons": ["ingestion job running: pid 1"]})
    assert rd.partial_data_flag(work)
    doc = rp.write_results(tmp_path / "r.json", {}, profile="commercial-safe", smoke=False, data={}, partial_data=True)
    assert doc["partial_data"] and "PARTIAL DATA" in doc["label"] and "Measured on held-out" not in doc["label"]
    assert "PARTIAL DATA" in rp.markdown_table(doc)
    from app.isl.evaluation.runners import markdown_report

    md = markdown_report(_doc(partial_data=True, label=rp.PARTIAL_DATA_LABEL))
    assert md.splitlines()[0].endswith("- PARTIAL DATA") and "Measured on held-out" not in md


def _driver(tmp_path, *args, proc=None, isl_root=None):
    env = {**os.environ, "PY": sys.executable, "SLR_ISL_DIR": str(isl_root), "PHASE2_WORK_BASE": str(tmp_path / "art"),
           "SLR_ISL_PROC_ROOT": str(proc), "SLR_TORCH_THREADS": "1"}
    env.pop("SLR_ISL_PARTIAL_DATA", None)
    return subprocess.run(["bash", str(REPO / "scripts" / "phase2_train.sh"), "--profile", "commercial-safe",
                           "--check-only", *args], capture_output=True, text=True, env=env, timeout=300)


def test_driver_refuses_full_run_on_partial_data_unless_allowed(tmp_path):
    P = _complete_world(tmp_path)
    proc = tmp_path / "proc"
    _proc(proc, 400, ["python", "scripts/ingest_isl.py", "include", "--workers", "1"])
    r = _driver(tmp_path, proc=proc, isl_root=P.root)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "refusing" in r.stdout + r.stderr
    marker = tmp_path / "art" / "commercial-safe" / "partial_data.json"
    assert not marker.exists()
    r = _driver(tmp_path, "--allow-partial", proc=proc, isl_root=P.root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(marker.read_text())["partial_data"] is True
    # ready data, no job: starts, and the old partial mark stays (sticky) until a rebuild --from trim
    empty = tmp_path / "proc_empty"
    empty.mkdir()
    assert _driver(tmp_path, proc=empty, isl_root=P.root).returncode == 0 and marker.exists()
    assert _driver(tmp_path, "--from", "trim", proc=empty, isl_root=P.root).returncode == 0 and not marker.exists()


# --------------------------------------------------------------------------------------------- 3. guards

def test_classifier_without_split_is_refused():
    from app.isl.evaluation.runners import LeakageError, check_classifier_split, e1_closed_set

    with pytest.raises(LeakageError, match="no 'split'"):
        check_classifier_split({}, "a")
    with pytest.raises(LeakageError):
        check_classifier_split(None, "a")

    class _Clf:
        meta = {"class_names": ["x"]}
        class_names = ["x"]

    class _Rec:
        classifier = _Clf()

    out = e1_closed_set(_Rec(), [], "E1b", "a", "primary")
    assert out["run"] is False and "refused" in out["reason"] and "split" in out["leakage"]


class _Data:
    def __init__(self, include, assign, islrtc=()):
        self.include, self.assign, self.extra, self.extra_assign = include, assign, [], {}
        self.islrtc = list(islrtc)
        self.dedup = ms.DedupIndex()


def test_pretraining_guard_is_computed_from_the_clip_list(tmp_path):
    from app.isl.evaluation.runners import pretraining_guard

    mk = lambda i, s="include": Clip(i, s, "w", "c", None, None, Path("/x"))  # noqa: E731
    data = _Data([mk("inc/a"), mk("inc/b"), mk("inc/c")], {"inc/a": "train", "inc/b": "test", "inc/c": "val"},
                 islrtc=[mk("islrtc/A/A.mp4", "islrtc")])
    data.dedup.union("inc/b", "wslp/copy-of-b", "same_video")
    work = tmp_path / "w"
    work.mkdir()
    assert pretraining_guard(work, data)["excludes_eval_clips"] is None  # no pre-training at all
    (work / "pretrained_encoder.pt").write_bytes(b"x")
    g = pretraining_guard(work, data)
    assert g["excludes_eval_clips"] is None and "missing" in g["reason"]
    ms.write_pretrain_list(work, [mk("inc/a"), mk("islrtc/A/A.mp4", "islrtc"), mk("glosl/1", "glosl")])
    g = pretraining_guard(work, data)
    assert g["excludes_eval_clips"] is True and g["overlap"] == 0 and g["islrtc_clips_seen_unlabelled"] == 1
    # a duplicate of a test clip (other source, other id) in the list is caught through the duplicate group
    ms.write_pretrain_list(work, [mk("wslp/copy-of-b", "wslp")], resume=True)
    g = pretraining_guard(work, data)
    assert g["excludes_eval_clips"] is False and g["overlap"] == 1 and "wslp/copy-of-b" in g["overlap_examples"]
    assert len(ms.read_pretrain_list(work)) == 4  # resume keeps the union


# --------------------------------------------------------------------------------------------- 4. profiles

def test_profiles_are_the_multisource_lists_and_adapters_are_registered():
    from app.isl import profiles
    from app.isl.data import sources
    from app.isl.data.pose_sources import PROFILE_SOURCES
    from app.isl.data.video_sources.catalog import SOURCES, profile_sources

    for name, prof in profiles.PROFILES.items():
        assert prof.sources == ms.WORD_SOURCES[name] and prof.pretraining == ms.PRETRAIN_SOURCES[name]
        assert prof.letters == ms.LETTER_SOURCES[name]
        assert "wlasl" not in prof.all_sources() and "isl_csltr_words" not in prof.all_sources()
    assert {"wlasl", "isl_csltr_words"} <= set(profiles.NOT_USED)
    reg = sources.all_adapters()
    assert {"glosl", "isl500", "openhands_raw", "openhands_fs", "wslp", "isl_csltr", "mendeley_static"} <= set(reg)
    fac = ms._factories()
    for prof in profiles.PROFILES.values():
        for n in prof.all_sources():
            assert n in fac, n  # every listed source has an adapter (wslp_unlabelled = filtered WSLP view)
    # commercial-safe lists agree with the data agents' own profile tables
    com = set(profiles.PROFILES["commercial-safe"].all_sources())
    video = {s.manifest_name for s in SOURCES.values()}
    assert com & video <= set(profile_sources("commercial-safe"))
    assert not com & set(PROFILE_SOURCES["research"]["sources"] + PROFILE_SOURCES["research"]["pretraining"])


def test_pack_licences_name_only_contributing_sources(tmp_path):
    from app.isl.training.pipeline import pack_licences

    work = tmp_path / "w"
    work.mkdir()
    assert [x["source"] for x in pack_licences(work)] == ["include", "islrtc"]  # phase-1 build
    import argparse  # noqa: F401  (cli helper below)

    from app.isl.cli import _record_sources

    _record_sources(work, {"cislr": {"available": False}, "isign": {"available": False},
                           "isl500": {"available": True, "clips": 900, "licence": "research-only", "commercial": False},
                           "wslp": {"available": True, "clips": 50, "licence": "CC-BY-NC-ND-4.0", "commercial": False}},
                    "words", {"isl500": 700, "wslp": 0})
    _record_sources(work, {"glosl": {"available": True, "clips": 100, "licence": "CC-BY-NC-4.0", "commercial": False}},
                    "pretraining", {"glosl": 40})
    lic = {x["source"]: x for x in pack_licences(work)}
    assert set(lic) == {"include", "islrtc", "isl500", "glosl"}  # no cislr / isign / wslp (0 clips used)
    assert lic["isl500"]["clips"] == {"words": 700} and lic["glosl"]["role"] == "pretraining"
    assert lic["include"]["role"] == "vocabulary"


# --------------------------------------------------------------------------------------------- 5. Sign Library dedupe

def _walk(rng, n=80):
    """(n, 14) smooth random pose trajectories (like signing)."""
    steps = rng.normal(0, 0.01, (n, 14)).cumsum(0)  # ~0.04 between frames, like the real 12.5 fps data
    return (0.5 + 0.1 * np.sin(np.linspace(0, 6, n))[:, None] + steps).astype(np.float32)


def test_frame_matcher_finds_recut_copies_and_not_other_videos():
    rng = np.random.default_rng(0)
    refs = [(f"r{i}", f"v{i // 2}", _walk(rng)) for i in range(20)]
    idx = pd.FrameIndex([(a, v) for a, _, v in refs])
    copy = pd._resample_half(refs[3][2][10:50], rng, 0.002)
    rid, sc = idx.match(copy)
    assert rid == "r3" and sc > 0.6
    other = _walk(np.random.default_rng(99))
    assert idx.match(other)[1] < 0.2
    cal = pd.calibrate(refs, n=60, seed=1)
    assert cal["reliable"] and cal["recall"] >= 0.95 and cal["false_positive_rate"] <= 0.01


def test_apply_dedup_drops_sign_library_without_a_reliable_check():
    mk = lambda i, ch: Clip(i, "openhands_raw", "", None, None, ch, Path("/x"))  # noqa: E731
    clips = [mk("a", "Sign_Library"), mk("b", "Sign_Library"), mk("c", "NIOS"), Clip("g", "glosl", "", None, None,
                                                                                      None, Path("/y"))]
    keep, rep = pd.apply_dedup(clips, None)
    assert [c.clip_id for c in keep] == ["c", "g"] and rep["dropped"] == 2
    keep, _ = pd.apply_dedup(clips, {"reliable": True, "drop_ids": ["b"], "drop_channels": [], "checked_ids": ["a", "b"]})
    assert [c.clip_id for c in keep] == ["a", "c", "g"]
    keep, _ = pd.apply_dedup(clips, {"reliable": False, "drop_ids": [], "drop_channels": ["Sign_Library"]})
    assert [c.clip_id for c in keep] == ["c", "g"]


def test_dedupe_sign_library_end_to_end(tmp_path):
    from app.landmarks.raw import RawSequence

    rng = np.random.default_rng(3)

    def save(name, vec):
        n = len(vec)
        pose = np.zeros((n, 33, 4), np.float32)
        pose[:, list(pd.POSE_POINTS), :2] = vec.reshape(n, len(pd.POSE_POINTS), 2)
        pose[:, :, 3] = 1.0
        raw = RawSequence(pose, np.zeros((n, 21, 3), np.float32), np.zeros((n, 21, 3), np.float32),
                          np.zeros((n, 478, 3), np.float32), np.arange(n) * 80.0, 1920, 1080)
        p = tmp_path / f"{name}.npz"
        raw.save(p)
        return p

    one, other = [], []
    vids = [_walk(rng, 120) for _ in range(12)]
    for i, v in enumerate(vids):
        for j in range(2):
            p = save(f"m{i}_{j}", v[60 * j:60 * j + 60])
            one.append(Clip(f"sl/m{i}_{j}", "openhands_raw", "", None, None, "Sign_Library", p, group=f"v{i}",
                            quality={"member": "Sign-Library/sign_library_1_min_splits.h5"}))
    dup = save("dup", pd._resample_half(vids[4][20:60], rng, 0.002))
    new = save("new", _walk(np.random.default_rng(77), 60))
    for i, p in (("sl/dup", dup), ("sl/new", new)):
        other.append(Clip(i, "openhands_raw", "", None, None, "Sign_Library", p, group=i,
                          quality={"member": "Sign-Library/sign_library_sub_based.h5"}))
    ev = save("eval", pd._resample_half(vids[7][62:100], rng, 0.002))  # an "evaluation clip" inside chunk m7_1
    res = pd.dedupe_sign_library(one + other, [Clip("include/x", "include", "w", "c", None, None, ev)], n_calib=60)
    assert res.reliable, res.report["calibration"]
    assert "sl/dup" in res.drop_ids and "sl/new" not in res.drop_ids and "sl/m7_1" in res.drop_ids
    assert res.report["n_eval_matches"] == 1


# --------------------------------------------------------------------------------------------- 6. trimming

def test_trim_uses_learned_model_and_prefers_reextracted_islrtc_files(tmp_path, monkeypatch):
    from app.isl.cli import main
    from app.isl.data.boundary import _model
    from app.isl.data.manifest import read_manifest
    from app.isl.data.trim_learned import MODEL_VERSION, save
    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex
    from app.landmarks.raw import RawSequence

    w = make_world(tmp_path)
    P = w["paths"]
    model = tmp_path / "trim_boundary.pt"
    save(_model(), model, {"note": "untrained, test only"})
    assert main(["--root", str(P.root), "trim", "--learned-model", str(model)]) == 0
    trims = {r["id"]: r for r in read_manifest(P.trim_sidecar, only_ok=False)}
    assert {r["trim_version"] for r in trims.values()} == {f"trim-2+{MODEL_VERSION}"}
    # rules only (no model): every clip is redone and says so
    assert main(["--root", str(P.root), "trim", "--learned-model", "none"]) == 0
    trims = {r["id"]: r for r in read_manifest(P.trim_sidecar, only_ok=False)}
    assert {r["trim_version"] for r in trims.values()} == {"trim-2"}
    # a re-extracted (recut) file replaces the original; it is re-trimmed on the new file
    cid = "islrtc/A/Apple.mp4"
    src = RawSequence.load(P.resolve(trims[cid]["landmarks"]))
    (P.landmarks / "islrtc_recut").mkdir()
    src.save(P.landmarks / "islrtc_recut" / "A__Apple.npz")
    write_manifest(P.manifest("islrtc_recut"), [{"id": cid, "path": "data/isl/landmarks/islrtc_recut/A__Apple.npz",
                                                 "n_frames": len(src), "hands_frac": 0.9, "fps": 12.0, "stop": "eof",
                                                 "ok": True, "recut_of": "islrtc"}])
    assert main(["--root", str(P.root), "trim", "--learned-model", "none"]) == 0
    trims = {r["id"]: r for r in read_manifest(P.trim_sidecar, only_ok=False)}
    assert trims[cid]["landmarks"].endswith("islrtc_recut/A__Apple.npz") and trims[cid]["recut"]
    # a truncated clip without a re-extraction is kept out of training; the recut one is used from its new file
    rows = [{**r, "method": "truncated"} if r["id"] in (cid, "islrtc/B/Book.mp4") else r for r in trims.values()]
    write_manifest(P.trim_sidecar, rows)
    vocab = VocabIndex.load(w["vocab"])
    data = load_isl_data(P, vocab, "commercial-safe", w["split"])
    ids = {c.clip_id: c for c in data.islrtc}
    assert "islrtc/B/Book.mp4" not in ids and cid not in ids  # recut clip whose trim on the new file is truncated
    assert data.islrtc_stats["truncated_excluded"] == 2
    rows = [{**r, "method": "rest"} if r["id"] == cid else r for r in rows]
    write_manifest(P.trim_sidecar, rows)
    data = load_isl_data(P, vocab, "commercial-safe", w["split"])
    ids = {c.clip_id: c for c in data.islrtc}
    assert ids[cid].path.name == "A__Apple.npz" and "islrtc_recut" in str(ids[cid].path) and ids[cid].quality["recut"]
    assert data.summary()["islrtc_reextraction"]["recut_used"] == 1


# --------------------------------------------------------------------------------------------- 8. My signs fusion

def test_neutral_user_classifier_term():
    cls = np.log(np.array([[0.7, 0.2, 0.1]] * 3, np.float32))
    inp = FusionInputs(cls_logp=cls, cls_ids=["a", "b", "c"], ret=np.full((3, 3), 0.5, np.float32),
                       ret_ids=["a", "b", "c"], user=np.full((3, 1), 0.5, np.float32), user_ids=["mysign:u"])
    pen = fuse(inp, FusionParams(beta=0.0, gamma=0.0))
    neu = fuse(inp, FusionParams(beta=0.0, gamma=0.0, user_cls_mode="neutral"))
    p_pen = dict(zip(pen.keys, pen.probs))["mysign:u"]
    p_neu = dict(zip(neu.keys, neu.probs))["mysign:u"]
    assert p_pen < 0.01 and p_neu == pytest.approx(dict(zip(neu.keys, neu.probs))["a"], rel=1e-5)
    assert FusionParams.from_json(FusionParams(user_cls_mode="neutral").to_json()).user_cls_mode == "neutral"
    with pytest.raises(ValueError):
        FusionParams(user_cls_mode="bonus")


class _FakeRec:
    """A recogniser stand-in for E4: retrieval evidence close to the clip's own class, a classifier that is
    unsure between two words."""

    def __init__(self):
        from app.isl.training.common import ISL_FEATURES

        self.feature_config = ISL_FEATURES
        self.fusion = FusionParams()
        self.encoder = None

    def evidence(self, f, use_my_signs=False):
        return FusionInputs(cls_logp=np.log(np.full((len(f), 2), 0.5, np.float32)), cls_ids=["x", "y"]), {}


def test_fit_fusion_keeps_penalty_without_enough_validation_data():
    from app.isl.training.fit_fusion import choose_user_mode

    p, rep = choose_user_mode(_FakeRec(), [], FusionParams())
    assert p.user_cls_mode == "penalty" and "insufficient" in rep["reason"]
    one_group = [Clip(f"v/{w}/{t}", "include", w, f"isl.{w}", None, None, Path("/x"), group="s1")
                 for w in ("a", "b") for t in range(4)]
    p, rep = choose_user_mode(_FakeRec(), one_group, FusionParams())  # one signer group: never compared
    assert p.user_cls_mode == "penalty" and "1 signer groups" in rep["reason"]


def test_fit_fusion_chooses_user_mode_on_validation_e4(monkeypatch):
    from app.isl.training import fit_fusion as ff

    calls = []

    def fake_e4(rec, clips, ks, params, cache, **kw):
        calls.append(params.user_cls_mode)
        good = params.user_cls_mode == "neutral"
        return {"run": True, "by_k": {"k3": {"n": 40, "n_groups": 3, "top1_with_enrolment": 0.9 if good else 0.6,
                                             "gain": 0.3, "user_sign_false_accept_rate": 0.02, "n_other_words": 50}}}

    import app.isl.evaluation.runners as runners

    monkeypatch.setattr(runners, "e4_my_signs", fake_e4)
    val = [Clip(f"v/{g}/{w}/{t}", "isl500", w, f"isl.{w}", None, None, Path("/x"), group=g)
           for g in ("u1", "u2") for w in ("a", "b") for t in range(4)]
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "neutral" and rep["k"] == 3 and set(calls) == {"penalty", "neutral"}

    def worse_fa(rec, clips, ks, params, cache, **kw):
        r = fake_e4(rec, clips, ks, params, cache)
        if params.user_cls_mode == "neutral":
            r["by_k"]["k3"]["user_sign_false_accept_rate"] = 0.2
        return r

    monkeypatch.setattr(runners, "e4_my_signs", worse_fa)
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and rep["false_accept_increase_neutral"] > 0.1


def test_sources_json_counts_only_sampled_clips(tmp_path):
    from collections import Counter

    from app.isl.cli import _record_sources

    work = tmp_path / "w"
    work.mkdir()
    _record_sources(work, {"wslp": {"available": True, "clips": 1100, "licence": "x", "commercial": False},
                           "glosl": {"available": True, "clips": 5000, "licence": "y", "commercial": False}},
                    "pretraining", Counter({"glosl": 7}))
    doc = json.loads((work / "sources.json").read_text())
    assert doc["wslp"]["used"] == {"pretraining": 0} and not doc["wslp"]["contributed"]
    assert doc["glosl"]["used"] == {"pretraining": 7} and doc["glosl"]["clips_read"] == 5000


def test_smoke_islrtc_cap_keeps_dictionary_clips_of_include_concepts(tmp_path):
    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex

    w = make_world(tmp_path)
    data = load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "commercial-safe", w["split"], max_islrtc=4)
    inc = {c.concept_id for c in data.include}
    assert len(data.islrtc) == 4 and all(c.concept_id in inc for c in data.islrtc)


# --------------------------------------------------------------------------------------------- 9. retrains retrain

STEPS_DONE = ("trim", "dedup", "vocab", "encoder", "classifier", "cv", "e1a", "pack")


def _fake_py(tmp_path) -> tuple[Path, Path]:
    """A stand-in for the venv python: logs its arguments and succeeds (guard, cli steps, the cv helper)."""
    log = tmp_path / "fake_py.log"
    py = tmp_path / "fake_py"
    py.write_text('#!/usr/bin/env bash\necho "$*" >> "$FAKE_PY_LOG"\nexit 0\n')
    py.chmod(0o755)
    return py, log


def _run_driver(tmp_path, *args, work_base=None, tag=None, isl_root=None, py=None):
    proc = tmp_path / "proc_none"
    proc.mkdir(exist_ok=True)
    fake, flog = _fake_py(tmp_path)
    env = {**os.environ, "PY": str(py or fake), "FAKE_PY_LOG": str(flog), "SLR_ISL_DIR": str(isl_root),
           "SLR_ISL_PROC_ROOT": str(proc), "SLR_TORCH_THREADS": "1"}
    for k in ("SLR_ISL_PARTIAL_DATA", "PHASE2_WORK_BASE", "PHASE2_TAG"):
        env.pop(k, None)
    if work_base is not None:
        env["PHASE2_WORK_BASE"] = str(work_base)
    if tag is not None:
        env["PHASE2_TAG"] = tag
    return subprocess.run(["bash", str(REPO / "scripts" / "phase2_train.sh"), "--profile", "commercial-safe", *args],
                          capture_output=True, text=True, env=env, timeout=120)


def _old_run(work: Path, stamps: Path) -> list[Path]:
    """A finished run: every stamp, and the training state files of the classifier, cv folds, e1a, encoder."""
    stamps.mkdir(parents=True, exist_ok=True)
    for s in STEPS_DONE:
        (stamps / f"{s}.done").write_text("old\n")
    files = [work / "encoder.pt.state", work / "classifier.pt.state", work / "cv" / "fold1" / "classifier.pt.state",
             work / "cv" / "fold3" / "classifier.pt.state", work / "e1a" / "classifier.pt.state",
             work / "oneshot" / "fold0" / "encoder.pt.state"]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"finished state")
    return files


def test_driver_dry_run_from_classifier_removes_state_files(tmp_path):
    isl = tmp_path / "isl"
    work = tmp_path / "art" / "commercial-safe"
    files = _old_run(work, work / "driver" / "phase2_commercial-safe.stamps")
    r = _run_driver(tmp_path, "--from", "classifier", "--dry-run", work_base=tmp_path / "art", isl_root=isl)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    removed = {ln.split(" ", 2)[2] for ln in out.splitlines() if ln.startswith("would remove ")}
    assert str(work / "classifier.pt.state") in removed
    assert str(work / "cv" / "fold1" / "classifier.pt.state") in removed
    assert str(work / "cv" / "fold3" / "classifier.pt.state") in removed
    assert str(work / "e1a" / "classifier.pt.state") in removed  # a later step
    assert str(work / "encoder.pt.state") not in removed and str(work / "oneshot" / "fold0" / "encoder.pt.state") \
        not in removed  # earlier steps keep their state
    assert "would skip encoder (done old)" in out and "would run classifier" in out and "would run pack" in out
    assert all(f.exists() for f in files)  # a dry run writes nothing
    assert not (tmp_path / "fake_py.log").exists()  # and runs nothing (no data guard, no cli)


def test_driver_from_classifier_really_removes_state_and_retrains(tmp_path):
    isl = tmp_path / "isl"
    work = tmp_path / "art" / "commercial-safe"
    stamps = work / "driver" / "phase2_commercial-safe.stamps"
    files = _old_run(work, stamps)
    r = _run_driver(tmp_path, "--from", "classifier", work_base=tmp_path / "art", isl_root=isl)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (work / "encoder.pt.state").exists() and (work / "oneshot" / "fold0" / "encoder.pt.state").exists()
    gone = [f for f in files if "encoder" not in f.name]
    assert not any(f.exists() for f in gone), [str(f) for f in gone if f.exists()]
    calls = (tmp_path / "fake_py.log").read_text().splitlines()
    assert any(" train-classifier " in c and "--resume" in c for c in calls)  # resumes nothing: the state is gone
    assert not any(" train-encoder " in c for c in calls)  # stamped earlier step skipped
    assert (stamps / "classifier.done").read_text() != "old\n" and (stamps / "encoder.done").read_text() == "old\n"


def test_driver_force_takes_a_comma_separated_list(tmp_path):
    isl = tmp_path / "isl"
    work = tmp_path / "art" / "commercial-safe"
    _old_run(work, work / "driver" / "phase2_commercial-safe.stamps")
    r = _run_driver(tmp_path, "--force", "encoder,cv", "--dry-run", work_base=tmp_path / "art", isl_root=isl)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for f in (work / "encoder.pt.state", work / "cv" / "fold1" / "classifier.pt.state",
              work / "driver" / "phase2_commercial-safe.stamps" / "encoder.done"):
        assert f"would remove {f}" in out
    assert f"would remove {work / 'classifier.pt.state'}" not in out
    assert "would run encoder" in out and "would run cv" in out and "would skip classifier (done old)" in out
    bad = _run_driver(tmp_path, "--force", "encoder,nope", "--dry-run", work_base=tmp_path / "art", isl_root=isl)
    assert bad.returncode == 2 and "nope" in bad.stderr
    assert _run_driver(tmp_path, "--from", "classifer", "--dry-run", work_base=tmp_path / "art",
                       isl_root=isl).returncode == 2  # a typo is refused, not "run everything"


def test_scratch_run_never_touches_the_real_runs_stamps(tmp_path):
    isl = tmp_path / "isl"
    real = isl / "logs" / "phase2_commercial-safe.stamps"
    real_work = tmp_path / "real_work"  # stands in for artifacts/isl/commercial-safe (never written by this test)
    _old_run(real_work, real)
    before = {p.name: p.read_text() for p in real.iterdir()}
    # the real run (no tag, default work folder) keeps the old stamp place
    r = _run_driver(tmp_path, "--dry-run", isl_root=isl)
    assert r.returncode == 0 and f"stamps: {real}" in r.stdout and "would skip classifier (done old)" in r.stdout
    assert "pack: default" in r.stdout
    # a scratch run (own work base + tag) does not read, skip on or delete them
    scratch = tmp_path / "scratch"
    r = _run_driver(tmp_path, "--from", "classifier", work_base=scratch, tag="abl1", isl_root=isl)
    assert r.returncode == 0, r.stdout + r.stderr
    own = scratch / "commercial-safe" / "driver" / "phase2_commercial-safe_abl1.stamps"
    assert (own / "trim.done").exists() and (own / "pack.done").exists()  # it ran trim: no real stamp was used
    assert {p.name: p.read_text() for p in real.iterdir()} == before
    assert not (isl / "logs" / "phase2_commercial-safe.log").exists()  # nor its log
    assert (scratch / "commercial-safe" / "driver" / "phase2_commercial-safe_abl1.log").exists()
    calls = (tmp_path / "fake_py.log").read_text()
    assert f"--out {scratch / 'commercial-safe' / 'pack'}" in calls  # pack and eval report stay in the scratch work
    # a work base without a tag is just as separate
    r = _run_driver(tmp_path, "--dry-run", work_base=tmp_path / "other", isl_root=isl)
    assert f"stamps: {tmp_path / 'other' / 'commercial-safe' / 'driver'}" in r.stdout
    # a tag alone gets its own work folder; a tag on the real work folder is refused
    r = _run_driver(tmp_path, "--dry-run", tag="abl2", isl_root=isl)
    assert r.returncode == 0 and "artifacts/isl/runs/abl2/commercial-safe" in r.stdout
    r = _run_driver(tmp_path, "--tag", "abl3", "--dry-run", work_base=REPO / "artifacts" / "isl", isl_root=isl)
    assert r.returncode == 2 and "real work folder" in r.stderr
    assert _run_driver(tmp_path, "--tag", "../x", "--dry-run", isl_root=isl).returncode == 2


def test_run_loop_refuses_a_finished_state_of_another_config(tmp_path):
    import torch
    from torch import nn

    from app.isl.training.loop import LoopConfig, StateMismatch, run_loop

    def go(cfg, fresh=False):
        torch.manual_seed(0)
        m = nn.Linear(3, 2)
        def batches(e):
            return iter([(torch.ones(4, 3), torch.zeros(4, dtype=torch.long))] * cfg.steps_per_epoch)

        return run_loop(m, batches, lambda mm, b: nn.functional.cross_entropy(mm(b[0]), b[1]), cfg,
                        tmp_path / "m.pt.state", fresh=fresh)

    cfg = LoopConfig(epochs=2, steps_per_epoch=2, extra={"kind": "classifier", "class_power": 0.5})
    assert go(cfg, fresh=True).finished
    again = go(cfg)
    assert again.finished and again.resumed_from["finished"] and again.seconds == 0.0  # the old weights, untrained
    with pytest.raises(StateMismatch):
        go(LoopConfig(epochs=2, steps_per_epoch=2, extra={"kind": "classifier", "class_power": 0.0}))
    with pytest.raises(StateMismatch):
        go(LoopConfig(epochs=2, steps_per_epoch=2, extra={"kind": "classifier", "class_power": 0.5, "quota": 1}))
    # patience / time budget may change between parts of one run
    assert go(LoopConfig(epochs=2, steps_per_epoch=2, patience=3, max_minutes=5,
                         extra={"kind": "classifier", "class_power": 0.5})).finished


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    from app.isl.training.common import load_isl_data
    from app.isl.vocab.build import VocabIndex

    w = make_world(tmp_path_factory.mktemp("sig_world"))
    return load_isl_data(w["paths"], VocabIndex.load(w["vocab"]), "commercial-safe", w["split"])


def test_classifier_and_encoder_signatures_cover_training_options(tiny_data, tmp_path):
    from dataclasses import replace

    import torch

    from app.isl.models.encoder import SignEncoder
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.loop import StateMismatch
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder
    from app.training.trainer import TrainConfig

    cfg = TrainConfig(hidden=16, layers=1, epochs=1, batch_size=8, patience=1, lr=3e-3)

    def cls(**kw):
        a = {"threads": 1, "steps_per_epoch": 2, "smoke": True, **kw}
        return train_classifier(tiny_data, tmp_path / "cls.pt", ISL_FEATURES, a.pop("cfg", cfg), **a)

    cls()
    assert cls(resume=True)["finished"]
    for kw in ({"class_power": 0.0}, {"source_weights": {"include": 3.0}}, {"islrtc_oversample": 1},
               {"cfg": replace(cfg, label_smoothing=0.0)}, {"cfg": replace(cfg, dropout=0.1)},
               {"cfg": replace(cfg, batch_size=4)}, {"use_extra": False}):
        with pytest.raises(StateMismatch):
            cls(resume=True, **kw)
    assert cls(resume=True, cfg=replace(cfg, patience=5))["finished"]  # patience may change

    def enc(**kw):
        return train_encoder(tiny_data, tmp_path / "enc.pt", ISL_FEATURES, cfg, emb_dim=16, p=4, k=2, threads=1,
                             batches_per_epoch=2, smoke=True, **kw)

    init = tmp_path / "pre.pt"
    torch.save({"state_dict": SignEncoder(ISL_FEATURES.dim, hidden=16, layers=1, emb_dim=16).state_dict(),
                "meta": {"kind": "isl-encoder", "feature_config": ISL_FEATURES.to_dict()}}, init)
    enc(init_from=init)
    assert enc(init_from=init, resume=True)["resumed_from"]["finished"]
    for kw in ({"arc_weight": 0.0}, {"source_weights": {"isl500": 0.1}}, {"max_val_queries": 3}):
        with pytest.raises(StateMismatch):
            enc(init_from=init, resume=True, **kw)
    torch.save({"state_dict": SignEncoder(ISL_FEATURES.dim, hidden=16, layers=1, emb_dim=16).state_dict(),
                "meta": {"kind": "isl-encoder", "feature_config": ISL_FEATURES.to_dict(), "retrained": 1}}, init)
    with pytest.raises(StateMismatch):  # same path, new pre-training: not resumed on the old warm start
        enc(init_from=init, resume=True)
