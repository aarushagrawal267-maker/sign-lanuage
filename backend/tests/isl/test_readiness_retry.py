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
    """Review open item: ``choose_user_mode`` picked ``neutral`` on any top-1 gain (one lucky take of 30)."""
    from pathlib import Path

    import app.isl.evaluation.runners as runners
    from app.isl.models.fusion import FusionParams
    from app.isl.training import fit_fusion as ff
    from app.isl.training.common import Clip

    def fake(top1_neutral, n=30):
        def e4(rec, clips, ks, params, cache, **kw):
            t = top1_neutral if params.user_cls_mode == "neutral" else 0.6
            return {"run": True, "by_k": {"k3": {"n": n, "n_groups": 3, "top1_with_enrolment": t, "gain": 0.1,
                                                 "user_sign_false_accept_rate": 0.02, "n_other_words": 50}}}
        return e4

    val = [Clip(f"v/{g}/{w}/{t}", "isl500", w, f"isl.{w}", None, None, Path("/x"), group=g)
           for g in ("u1", "u2") for w in ("a", "b") for t in range(4)]
    monkeypatch.setattr(runners, "e4_my_signs", fake(0.6 + 1 / 30))  # one more correct take of 30
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty" and "within noise" in rep["reason"]
    monkeypatch.setattr(runners, "e4_my_signs", fake(0.6 + 3 / 30))  # three more: a real gain
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "neutral" and rep["extra_correct_neutral"] == 3.0
    monkeypatch.setattr(runners, "e4_my_signs", fake(0.615, n=400))  # 6 takes but only +0.015
    p, rep = ff.choose_user_mode(None, val, FusionParams())
    assert p.user_cls_mode == "penalty"


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
