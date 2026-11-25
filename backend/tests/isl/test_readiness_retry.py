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
