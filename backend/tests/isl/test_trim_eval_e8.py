"""E8 describes exactly the trimmer the run used (ISL_DESIGN 15.8): ``cli trim-eval`` evaluates the same trimmer as
``cli trim`` (learned end model included) and records its trim_version, model file and time; ``e8_trim`` reads the
work folder's report and refuses (status ``mismatch`` / ``missing``) one that describes another trimmer; the
phase-2 driver runs the ``trim_eval`` step before ``evaluate``. Offline, synthetic data only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.isl import cli
from app.isl.data import trim_eval as te
from app.isl.data import trim_learned as tl
from app.isl.data.manifest import read_manifest
from app.isl.evaluation.runners import e8_trim, markdown_report
from app.isl.paths import isl_paths
from tests.isl.helpers import write_manifest
from tests.isl.isl_world import make_world

REPO = Path(__file__).resolve().parents[3]
LEARNED = "trim-2+trim-learned-1"


_UNSET = object()


def _sidecar(P, version: str, n: int = 4, methods=("cap", "rest"), sha=_UNSET) -> None:
    """``sha``: the ``trim_model_sha256`` of every record (a list: one per record; unset: field left out, as in
    sidecars written before 25 Nov 2026)."""
    rows = [{"id": f"islrtc/x{i}", "start": 0, "end": 10, "method": methods[i % len(methods)], "confidence": 0.3,
             "trim_version": version} for i in range(n)]
    if sha is not _UNSET:
        for i, r in enumerate(rows):
            r["trim_model_sha256"] = sha[i] if isinstance(sha, list) else sha
    P.trim_sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(P.trim_sidecar, rows)


def _report(path: Path, version: str | None, **extra) -> None:
    doc = {"synthetic": {"n": 5, "within_300ms": 0.8, "median_abs_err_ms": 40.0, "late_frac": 0.1,
                         "early_frac": 0.1, "learned_ends": 2},
           "include_active_kept": {"n": 3, "mean": 1.0, "frac_ge_90pct": 1.0}, "learned": version != "trim-2",
           "written_at": "2026-11-25T23:00:00Z", **extra}
    if version is not None:
        doc["trim_version"] = version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _doc(e8: dict) -> dict:
    return {"language": "isl", "profile": "research", "label": "x", "created": "t", "git_commit": "abc",
            "protocols": {"E8": e8}}


# --------------------------------------------------------------------------------------------- check / E8

def test_check_report_statuses(tmp_path):
    rep = tmp_path / "trim_eval.json"
    c = te.check_report(rep, {LEARNED: 4})
    assert c["status"] == "missing" and "trim_eval" in c["reason"]
    _report(rep, None)  # the 11:00 rules-only report: no trim_version recorded
    c = te.check_report(rep, {LEARNED: 4})
    assert c["status"] == "mismatch" and "no trim_version" in c["reason"] and "doc" not in c
    _report(rep, "trim-2")
    c = te.check_report(rep, {LEARNED: 4})
    assert c["status"] == "mismatch" and "evaluated trim-2" in c["reason"] and LEARNED in c["reason"]
    c = te.check_report(rep, {"trim-2": 3, LEARNED: 1})
    assert c["status"] == "mismatch" and "mixes" in c["reason"]
    _report(rep, LEARNED)
    c = te.check_report(rep, {LEARNED: 4})
    assert c["status"] == "ok" and c["reason"] is None and c["doc"]["trim_version"] == LEARNED
    assert c["written_at"] == "2026-11-25T23:00:00Z" and c["file_mtime_utc"].endswith("Z")
    # the learned model file changed (or vanished) after trim-eval: not the evaluated trimmer any more
    m = tmp_path / "trim_boundary.pt"
    m.write_bytes(b"model-a")
    _report(rep, LEARNED, learned_model=te.file_info(m))
    assert te.check_report(rep, {LEARNED: 4})["status"] == "ok"
    m.write_bytes(b"model-b")
    c = te.check_report(rep, {LEARNED: 4})
    assert c["status"] == "mismatch" and "changed after trim-eval" in c["reason"]
    m.unlink()
    assert "no longer exists" in te.check_report(rep, {LEARNED: 4})["reason"]


def test_e8_reads_work_report_and_refuses_another_trimmer(tmp_path):
    P = isl_paths(tmp_path / "isl")
    _sidecar(P, LEARNED)
    work = tmp_path / "work"
    art = tmp_path / "art"
    _report(art / "reports" / "trim_eval.json", "trim-2")  # the old global report is never read with a work folder
    e8 = e8_trim(P, art, work)
    assert e8["trim_eval_check"]["status"] == "missing" and "trim_eval" not in e8
    assert "NOT SHOWN (MISSING)" in e8["label"] and e8["run"] is True and e8["trim_version"] == LEARNED
    _report(work / "trim_eval.json", "trim-2")  # rules-only report vs learned trims
    e8 = e8_trim(P, art, work)
    chk = e8["trim_eval_check"]
    assert chk["status"] == "mismatch" and chk["trim_version"] == "trim-2" and chk["run_trim_versions"] == {LEARNED: 4}
    assert "trim_eval" not in e8 and "NOT SHOWN (MISMATCH)" in e8["label"]
    md = markdown_report(_doc(e8))
    assert "trim-eval: **mismatch**" in md and "evaluated trim-2" in md
    m = tmp_path / "trim_boundary.pt"
    m.write_bytes(b"model-a")
    _report(work / "trim_eval.json", LEARNED, learned_model=te.file_info(m))
    e8 = e8_trim(P, art, work)  # the sidecar was written before trim recorded model hashes: refused
    assert e8["trim_eval_check"]["status"] == "mismatch" and "record no learned-model hash" in e8["label"]
    assert e8["trim_model_hashes"] == {te.UNRECORDED: 4}
    _sidecar(P, LEARNED, sha=te.file_info(m)["sha256"])
    e8 = e8_trim(P, art, work)
    assert e8["trim_eval_check"]["status"] == "ok" and e8["trim_eval"]["synthetic"]["within_300ms"] == 0.8
    assert LEARNED in e8["label"] and "2026-11-25T23:00:00Z" in e8["label"] and "NOT SHOWN" not in e8["label"]
    md = markdown_report(_doc(e8))
    assert "trim-eval: **ok**" in md and "within 300 ms 0.800" in md
    assert te.file_info(m)["sha256"][:12] in md  # the model hash evaluated and the run's hashes are printed
    # without a sidecar and without a report: not run, with the reason
    P2 = isl_paths(tmp_path / "isl2")
    e8 = e8_trim(P2, art, tmp_path / "work2")
    assert e8["run"] is False and "trim-eval" in e8["reason"]


def test_check_report_compares_the_records_model_hashes(tmp_path):
    """E8 hardening (15.9 open issue 1): the single model hash of the run's trim records must be the model trim-eval
    evaluated; records without a hash, mixed hashes or another model's hash are a mismatch."""
    rep = tmp_path / "trim_eval.json"
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    a.write_bytes(b"model-a")
    b.write_bytes(b"model-b")
    ha, hb = te.file_info(a)["sha256"], te.file_info(b)["sha256"]
    _report(rep, LEARNED, learned_model=te.file_info(a))
    assert te.check_report(rep, {LEARNED: 4}, {ha: 4})["status"] == "ok"
    c = te.check_report(rep, {LEARNED: 4}, {te.UNRECORDED: 1, ha: 3})
    assert c["status"] == "mismatch" and "1 of 4 trims record no learned-model hash" in c["reason"]
    assert c["run_model_hashes"] == {te.UNRECORDED: 1, ha: 3} and "doc" not in c
    c = te.check_report(rep, {LEARNED: 4}, {ha: 3, hb: 1})  # two models under one trim_version
    assert c["status"] == "mismatch" and "mixes learned-model hashes" in c["reason"] and hb[:12] in c["reason"]
    c = te.check_report(rep, {LEARNED: 4}, {hb: 4})  # retrained and re-trimmed, but trim-eval saw the old model
    assert c["status"] == "mismatch" and f"used learned model {hb[:12]}" in c["reason"] and ha[:12] in c["reason"]
    c = te.check_report(rep, {LEARNED: 4}, {te.RULES_ONLY: 4})
    assert c["status"] == "mismatch" and "used learned model none" in c["reason"]
    _report(rep, LEARNED)  # a learned report without its model hash cannot be matched
    c = te.check_report(rep, {LEARNED: 4}, {ha: 4})
    assert c["status"] == "mismatch" and "records no model hash" in c["reason"]
    _report(rep, "trim-2", learned_model=None)  # rules only: records carry null
    assert te.check_report(rep, {"trim-2": 4}, {te.RULES_ONLY: 4})["status"] == "ok"
    c = te.check_report(rep, {"trim-2": 4}, {te.UNRECORDED: 4})
    assert c["status"] == "mismatch" and "re-run the trim step" in c["reason"]
    assert te.trim_model_key({}) == te.UNRECORDED and te.trim_model_key({"trim_model_sha256": None}) == te.RULES_ONLY
    assert te.trim_model_key({"trim_model_sha256": ha}) == ha


# --------------------------------------------------------------------------------------------- trim-eval = trim

class _FakeModel:
    pass


@pytest.fixture
def fake_learned(monkeypatch, tmp_path):
    """A 'learned end model' that moves every cap end 2 frames earlier (and says so in the notes, as refine_cap)."""
    from app.isl.data.trim import TrimResult

    calls = []

    def refine_cap(raw, res, model, n_words=1, cfg=None, threshold=0.5):
        calls.append(res.method)
        if res.method != "cap" or res.end - res.start < 4:
            return res
        end = res.end - 2
        return TrimResult(res.start, end, res.t_start_ms, float(raw.t_ms[end - 1]), "cap", res.confidence,
                          list(res.alternatives), res.notes + [f"{tl.MODEL_VERSION}: end {end}"])

    monkeypatch.setattr(tl, "refine_cap", refine_cap)
    monkeypatch.setattr(tl, "load", lambda path: (_FakeModel(), {}))
    path = get_settings().artifacts_dir / "isl" / "trim_boundary.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake model")
    yield calls
    path.unlink()


def _world_with_tails(tmp_path):
    w = make_world(tmp_path)
    P = isl_paths(tmp_path / "isl")
    rows = read_manifest(P.manifest("islrtc"), only_ok=False)
    for i, r in enumerate(rows):  # half of the clips run on into an explanation (max_s): the synthetic tails
        r["stop"] = "max_s" if i % 2 else "eof"
    write_manifest(P.manifest("islrtc"), rows)
    return w, P


def test_trim_eval_evaluates_the_trimmer_trim_used(tmp_path, fake_learned, capsys):
    _, P = _world_with_tails(tmp_path)
    root = ["--root", str(P.root)]
    assert cli.main(root + ["trim"]) == 0  # auto: the learned model exists -> trim-2+trim-learned-1
    capsys.readouterr()
    side = {r["id"]: r for r in read_manifest(P.trim_sidecar, only_ok=False)}
    assert {r["trim_version"] for r in side.values()} == {LEARNED}
    # trim_one (used by trim-eval) gives exactly the sidecar's cuts
    from app.isl.data.trim import TrimConfig
    from app.isl.vocab.normalize import parse_islrtc_name
    from app.landmarks.raw import RawSequence

    for r in cli._islrtc_records(P):
        pn = parse_islrtc_name(r["id"].removeprefix("islrtc/"))
        res = te.trim_one(RawSequence.load(P.resolve(r["path"])), r.get("stop"), pn.kind, pn.text, TrimConfig(),
                          _FakeModel())
        assert (res.start, res.end, res.method) == (side[r["id"]]["start"], side[r["id"]]["end"],
                                                    side[r["id"]]["method"]), r["id"]
    # force synthetic examples: some clean (source_end) clips and some explanation tails
    rows = sorted(side.values(), key=lambda r: r["id"])
    for i, r in enumerate(rows):
        r["method"] = "source_end" if i % 2 == 0 else "rest"
        r["end"] = max(8, int(r["end"]) - 6) if i % 2 else int(r["end"])
    write_manifest(P.trim_sidecar, rows)
    out = tmp_path / "work" / "trim_eval.json"
    fake_learned.clear()
    assert cli.main(root + ["trim-eval", "--n", "20", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["trim_version"] == LEARNED and doc["learned"] is True and doc["schema"] == te.SCHEMA
    assert doc["learned_model"]["sha256"] and doc["learned_model"]["path"].endswith("trim_boundary.pt")
    assert doc["sidecar"]["versions"] == {LEARNED: len(rows)} and doc["written_at"].endswith("Z")
    assert doc["synthetic"]["n"] > 0 and doc["include_active_kept"]["n"] > 0
    assert fake_learned, "trim-eval never applied the learned end model"
    assert te.check_report(out, doc["sidecar"]["versions"])["status"] == "ok"
    # rules only: a different trim_version, and E8 would refuse it for these trims
    fake_learned.clear()
    assert cli.main(root + ["trim-eval", "--n", "20", "--learned-model", "none", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["trim_version"] == "trim-2" and doc["learned"] is False and doc["learned_model"] is None
    assert not fake_learned
    assert te.check_report(out, {LEARNED: len(rows)})["status"] == "mismatch"


def test_trim_records_model_hash_and_retrims_when_it_changes(tmp_path, fake_learned, capsys):
    """``cli trim`` stores the learned model's SHA-256 in every record and re-trims a clip when it changes (same
    trim_version); records without a hash are re-trimmed once; E8 follows the hashes."""
    _, P = _world_with_tails(tmp_path)
    root = ["--root", str(P.root)]
    model = get_settings().artifacts_dir / "isl" / "trim_boundary.pt"
    h1 = te.file_info(model)["sha256"]

    def trim(*extra) -> dict:
        capsys.readouterr()
        assert cli.main(root + ["trim", *extra]) == 0
        return json.loads(capsys.readouterr().out)

    def side() -> dict:
        return {r["id"]: r for r in read_manifest(P.trim_sidecar, only_ok=False)}

    first = trim()
    n = first["total"]
    assert n > 0 and first["trimmed_now"] == n and first["learned_model_sha256"] == h1
    assert first["model_hashes"] == {h1: n} and {r["trim_model_sha256"] for r in side().values()} == {h1}
    assert trim()["trimmed_now"] == 0  # same model: nothing to do
    # a sidecar from before the hash was recorded: those records are re-trimmed once, then never again
    rows = sorted(side().values(), key=lambda r: r["id"])
    for r in rows[:2]:
        del r["trim_model_sha256"]
    write_manifest(P.trim_sidecar, rows)
    again = trim()
    assert again["trimmed_now"] == 2 and again["model_hashes"] == {h1: n}
    assert trim()["trimmed_now"] == 0
    # trim-eval of this trimmer: E8 ok
    work = tmp_path / "work"
    assert cli.main(root + ["trim-eval", "--n", "10", "--out", str(work / "trim_eval.json")]) == 0
    doc = json.loads((work / "trim_eval.json").read_text())
    assert doc["learned_model"]["sha256"] == h1 and doc["sidecar"]["model_hashes"] == {h1: n}
    assert e8_trim(P, tmp_path / "art", work)["trim_eval_check"]["status"] == "ok"
    # the model is retrained (same trim_version): a partial re-trim mixes two models -> E8 mismatch
    model.write_bytes(b"retrained model")
    h2 = te.file_info(model)["sha256"]
    part = trim("--limit", "2")
    assert part["trimmed_now"] == 2 and part["model_hashes"] == {h1: n - 2, h2: 2}
    assert part["versions"] == {LEARNED: n}  # the version alone cannot tell the two models apart
    chk = e8_trim(P, tmp_path / "art", work)["trim_eval_check"]
    assert chk["status"] == "mismatch" and "mixes learned-model hashes" in chk["reason"]
    # every clip re-trimmed with the new model: still a mismatch until trim-eval evaluates that model
    rest = trim()
    assert rest["trimmed_now"] == n - 2 and rest["model_hashes"] == {h2: n}
    chk = e8_trim(P, tmp_path / "art", work)["trim_eval_check"]
    assert chk["status"] == "mismatch" and f"used learned model {h2[:12]}" in chk["reason"]
    assert cli.main(root + ["trim-eval", "--n", "10", "--out", str(work / "trim_eval.json")]) == 0
    assert e8_trim(P, tmp_path / "art", work)["trim_eval_check"]["status"] == "ok"
    # rules only (--learned-model none): another version and null hashes; every clip redone once
    rules = trim("--learned-model", "none")
    assert rules["trimmed_now"] == n and rules["model_hashes"] == {te.RULES_ONLY: n}
    assert {r["trim_model_sha256"] for r in side().values()} == {None}


def test_trim_eval_records_the_hash_of_the_model_it_loaded(tmp_path):
    """A model file replaced while trim-eval runs: the report keeps the hash of the bytes it evaluated and says the
    file changed, so E8's file check refuses it."""
    P = isl_paths(tmp_path / "isl")
    m = tmp_path / "trim_boundary.pt"
    m.write_bytes(b"new bytes on disk")
    doc = te.run(P, n=0, model=None, model_path=m, version=LEARNED, records={}, model_sha256="ab" * 32)
    assert doc["learned_model"]["sha256"] == "ab" * 32 and doc["learned_model"]["changed_during_eval"] is True
    m.unlink()
    doc = te.run(P, n=0, model=None, model_path=m, version=LEARNED, records={}, model_sha256="ab" * 32)
    assert doc["learned_model"]["gone_after_eval"] is True and doc["learned_model"]["sha256"] == "ab" * 32


# --------------------------------------------------------------------------------------------- driver

def test_driver_runs_trim_eval_into_the_work_folder(tmp_path):
    make_world(tmp_path)
    P = isl_paths(tmp_path / "isl")
    for name in ("include", "islrtc"):
        ids = [r["id"] for r in read_manifest(P.manifest(name), only_ok=False)]
        (P.manifests / f"{name}_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    for name in ("mendeley_emergency", "mendeley_agri", "isl_csltr", "mendeley_static"):
        ids = [f"{name}/{i}" for i in range(2)]
        write_manifest(P.manifest(name), [{"id": i, "ok": True, "n_frames": 9} for i in ids])
        (P.manifests / f"{name}_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    proc = tmp_path / "proc"
    proc.mkdir()
    stamps = P.root / "logs" / "phase2_commercial-safe.stamps"
    stamps.mkdir(parents=True)
    (stamps / "evaluate.done").write_text("old\n")
    (stamps / "pack.done").write_text("old\n")
    env = {**os.environ, "PY": sys.executable, "SLR_ISL_DIR": str(P.root), "PHASE2_WORK_BASE": str(tmp_path / "art"),
           "SLR_ISL_PROC_ROOT": str(proc), "SLR_TORCH_THREADS": "1"}
    r = subprocess.run(["bash", str(REPO / "scripts" / "phase2_train.sh"), "--profile", "commercial-safe",
                        "--only", "trim_eval"], capture_output=True, text=True, env=env, timeout=300)
    log = (P.root / "logs" / "phase2_commercial-safe.log").read_text() if (P.root / "logs").exists() else ""
    assert r.returncode == 0, r.stdout + r.stderr + log
    doc = json.loads((tmp_path / "art" / "commercial-safe" / "trim_eval.json").read_text())
    assert doc["trim_version"] == "trim-2" and doc["learned"] is False  # no model in the test artifacts dir
    assert (stamps / "trim_eval.done").exists()
    assert not (stamps / "evaluate.done").exists() and not (stamps / "pack.done").exists()
    steps = subprocess.run(["bash", str(REPO / "scripts" / "phase2_train.sh"), "--profile", "commercial-safe",
                            "--list"], capture_output=True, text=True, env=env, timeout=60).stdout
    names = [ln.split()[0] for ln in steps.splitlines() if ln.strip() and not ln.startswith("work folder")]
    assert names.index("fusion") < names.index("trim_eval") < names.index("evaluate")
