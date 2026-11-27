"""Final reality check before the full phase-2 run (docs/ISL_DESIGN.md 15.6), offline.

The readiness guard must not call a source complete when ingestion has not really attempted every item:

* a row whose last record is a download / fetch failure or a ``superseded`` marker is retried by the next ingestion
  run, so it is missing, not attempted (permanent failures such as ISL500's missing WholeBody files still count);
* a public source with no rows and no item list (its Hugging Face listing failed with 429 / 503 and the queue gave
  up) is incomplete; only the gated sources (CISLR, iSign) are "unavailable".
"""

from __future__ import annotations

import json

from app.isl.training import readiness as rd
from tests.isl.helpers import write_manifest


def _paths(tmp_path):
    from app.isl.paths import isl_paths

    P = isl_paths(tmp_path / "isl")
    P.manifests.mkdir(parents=True)
    return P


def test_retryable_failures_are_not_attempted(tmp_path):
    P = _paths(tmp_path)
    ids = [f"wslp/{i}" for i in range(4)]
    (P.manifests / "wslp_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    write_manifest(P.manifest("wslp"), [
        {"id": ids[0], "ok": True},
        {"id": ids[1], "ok": False, "error": "extract: no frames"},  # permanent: attempted
        {"id": ids[2], "ok": False, "error": "fetch: HTTPError: 429 Client Error"},  # retried next run
        {"id": ids[3], "ok": False, "error": "superseded: benchmark row extracted before the fix; re-extract"},
    ])
    st = rd.source_status(P, "wslp")
    assert st["complete"] is False and st["done"] == 2 and st["retry_pending"] == 2
    assert "to retry" in st["note"]
    # the retried rows succeed: complete
    with P.manifest("wslp").open("a") as fh:
        for i in ids[2:]:
            fh.write(json.dumps({"id": i, "ok": True}) + "\n")
    st = rd.source_status(P, "wslp")
    assert st["complete"] is True and st["retry_pending"] == 0


def test_download_failure_blocks_a_published_size_source(tmp_path):
    P = _paths(tmp_path)
    rows = [{"id": f"x{i}", "ok": True} for i in range(171)]
    rows.append({"id": "x171", "ok": False, "error": "download failed: https://example.invalid/x171"})
    write_manifest(P.manifest("openhands_fs"), rows)
    st = rd.source_status(P, "openhands_fs")
    assert st["complete"] is False and st["retry_pending"] == 1
    # permanent failures (no download error) still count as attempted, as before
    rows[-1] = {"id": "x171", "ok": False, "error": "LookupError: no MMPose WholeBody file"}
    write_manifest(P.manifest("openhands_fs"), rows)
    assert rd.source_status(P, "openhands_fs")["complete"] is True


def test_public_source_with_nothing_ingested_is_incomplete(tmp_path):
    P = _paths(tmp_path)
    st = rd.source_status(P, "hf_akrit_alphabet")
    assert st["complete"] is False and "listing failed" in st["note"]
    assert rd.source_status(P, "cislr")["complete"] is None  # gated: unavailable, not incomplete
    empty = tmp_path / "proc"
    empty.mkdir()
    rep = rd.check(P, "research", proc_root=empty)
    assert not rep["ready"]
    assert any(r.startswith("manifest incomplete: hf_akrit_alphabet") for r in rep["reasons"])
    assert not any("cislr" in r or "isign" in r for r in rep["reasons"])


def test_user_mode_needs_more_than_a_noise_gain(monkeypatch):
    """Review open item: ``choose_user_mode`` picked ``neutral`` on any top-1 gain (one lucky take of 30). Since
    docs 15.27 the rule is a one-sided paired sign test (p < 0.05) on the enrolled takes plus a false-accept cap."""
    from pathlib import Path

    import app.isl.evaluation.runners as runners
    from app.isl.models.fusion import FusionParams
    from app.isl.training import fit_fusion as ff
    from app.isl.training.common import Clip

    def fake(won, lost=0, n=30, base=18, fa_neutral=0.0):
        """penalty gets ``base`` takes right; neutral wins ``won`` of the others and loses ``lost`` of those."""
        pen = [1.0] * base + [0.0] * (n - base)
        neu = list(pen)
        for i in range(won):
            neu[base + i] = 1.0
        for i in range(lost):
            neu[i] = 0.0

        def e4(rec, clips, ks, params, cache, **kw):
            h = neu if params.user_cls_mode == "neutral" else pen
            fa = fa_neutral if params.user_cls_mode == "neutral" else 0.0
            return {"run": True, "by_k": {"k3": {"n": n, "n_groups": 3, "top1_with_enrolment": sum(h) / n,
                                                 "gain": 0.1, "user_sign_false_accept_rate": 0.0,
                                                 "user_sign_false_accept_rate_threshold_rule": fa,
                                                 "n_other_words": 50, "_with": h}}}
        return e4

    val = [Clip(f"v/{g}/{w}/{t}", "isl500", w, f"isl.{w}", None, None, Path("/x"), group=g)
           for g in ("u1", "u2") for w in ("a", "b") for t in range(4)]
    monkeypatch.setattr(runners, "e4_my_signs", fake(1))  # n = 30, one more correct take: p = 0.5
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and "within noise" in rep["reason"]
    assert rep["sign_test"]["neutral_only_correct"] == 1 and rep["sign_test"]["p_one_sided"] == 0.5
    monkeypatch.setattr(runners, "e4_my_signs", fake(3))  # +3 of 30: p = 0.125, still noise
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and rep["extra_correct_neutral"] == 3
    monkeypatch.setattr(runners, "e4_my_signs", fake(5))  # +5 of 30, none lost: p = 1/32 < 0.05
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "neutral"
    # research validation (QA plan 0.5): 4,689 takes, +84 correct, none lost, no false-accept change -> neutral.
    # The old absolute rule (gain >= 0.02) kept penalty: 84 / 4689 = 0.018.
    monkeypatch.setattr(runners, "e4_my_signs", fake(84, n=4689, base=3000))
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "neutral" and rep["top1_gain_neutral"] < 0.02 and rep["sign_test"]["p_one_sided"] < 1e-20
    # the same net +84, but neutral wins 1,500 takes and loses 1,416: p = 0.06, no evidence -> penalty (the net
    # gain alone cannot tell these two cases apart; the paired counts can)
    monkeypatch.setattr(runners, "e4_my_signs", fake(1500, lost=1416, n=4689, base=3000))
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and rep["extra_correct_neutral"] == 84
    # a clear gain with more user-sign false accepts (threshold rule, the app's own rate is 0 when confirm-only)
    monkeypatch.setattr(runners, "e4_my_signs", fake(84, n=4689, base=3000, fa_neutral=0.02))
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and "false accepts" in rep["reason"]


def test_sign_test_p_values():
    from app.isl.training.fit_fusion import sign_test_p

    assert sign_test_p(0, 0) == 1.0 and sign_test_p(1, 0) == 0.5 and sign_test_p(5, 0) == 1 / 32
    assert abs(sign_test_p(5, 1) - 7 / 64) < 1e-12  # P(X >= 5), X ~ Bin(6, 1/2)
    assert sign_test_p(0, 3) == 1.0

def test_pack_serves_the_phase2_report_not_a_stale_results_json(tmp_path):
    """A full run reuses ``artifacts/isl/<profile>/``, which still held the phase-1 smoke ``results.json``
    (11:57, 25 Nov). ``build-pack`` copied it into the pack, so the app would have served phase-1 smoke results for
    a phase-2 pack; and a phase-2 pack without that file served no results at all."""
    from app.isl.paths import REPO_ROOT
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack

    vocab = tmp_path / "vocabulary.jsonl"
    with (REPO_ROOT / "models" / "packs" / "isl" / "vocabulary.jsonl").open(encoding="utf-8") as fh:
        vocab.write_text("".join(next(fh) for _ in range(5)), encoding="utf-8")
    work = tmp_path / "w"
    work.mkdir()
    (work / "results.json").write_text(json.dumps({"schema": 1, "smoke": True, "label": "phase-1 smoke",
                                                   "protocols": {"E1": {}}}))
    r = build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "p1", smoke=True, vocab_path=vocab)
    assert json.loads((tmp_path / "p1" / "results.json").read_text())["label"] == "phase-1 smoke"  # phase-1 path
    (work / "eval").mkdir()
    rep = {"schema": 1, "smoke": False, "partial_data": False, "label": "phase-2 report", "protocols": {"E2": {}}}
    (work / "eval" / "report.json").write_text(json.dumps(rep))
    r = build_pack(work, "commercial-safe", ISL_FEATURES, tmp_path / "p2", smoke=True, vocab_path=vocab)
    pack = tmp_path / "p2"
    assert json.loads((pack / "results.json").read_text())["label"] == "phase-2 report"
    assert (pack / "eval" / "report.json").exists() and "results.json" in r["files"]
    assert json.loads((pack / "pack.json").read_text())["results"] == "results.json"


def test_a_job_blocks_only_the_profiles_that_read_its_source(tmp_path):
    """The OpenHands raw job (research only) reads ~10 chunks / min of 86k on 25 Nov: it must not hold back the
    commercial-safe run, which never reads OpenHands. The queue and unknown jobs still block every profile."""
    P = _paths(tmp_path)
    for name in ("include", "islrtc", "mendeley_emergency", "mendeley_agri", "isl_csltr", "mendeley_static"):
        ids = [f"{name}/{i}" for i in range(3)]
        write_manifest(P.manifest(name), [{"id": i, "ok": True} for i in ids])
        (P.manifests / f"{name}_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    proc = tmp_path / "proc"

    def add(pid, argv):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")

    add(201, ["/v/python", "/r/scripts/ingest_isl_pose.py", "openhands-raw", "--workers", "12"])
    add(202, ["python", "scripts/ingest_isl_video.py", "run", "wslp", "--workers", "auto"])
    add(203, ["python", "scripts/ingest_isl_gated.py", "isign", "--mode", "zip"])
    rep = rd.check(P, "commercial-safe", proc_root=proc)
    assert rep["ready"], rep["reasons"]
    assert {p["pid"]: p["source"] for p in rep["processes"]} == {201: "openhands_raw", 202: "wslp", 203: "isign"}
    assert all("does not read" in p["note"] for p in rep["processes"])
    rep = rd.check(P, "research", proc_root=proc)
    assert {r for r in rep["reasons"] if r.startswith("ingestion job")} == {
        "ingestion job running: pid 201 ingest_isl_pose.py openhands-raw",
        "ingestion job running: pid 202 ingest_isl_video.py run",
        "ingestion job running: pid 203 ingest_isl_gated.py isign"}
    # jobs of commercial-safe sources, the queue and unknown sub-commands block commercial-safe
    add(204, ["python", "scripts/ingest_isl.py", "islrtc-recut", "--follow"])
    add(205, ["python", "scripts/ingest_isl_video.py", "run", "mendeley_agri"])
    add(206, ["bash", "scripts/isl_queue.sh"])
    add(207, ["python", "scripts/ingest_isl_video.py", "something-new"])
    rep = rd.check(P, "commercial-safe", proc_root=proc)
    assert sorted(int(r.split()[4]) for r in rep["reasons"] if r.startswith("ingestion job")) == [204, 205, 206, 207]
