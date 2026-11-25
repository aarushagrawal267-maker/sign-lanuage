"""Adversarial review of the readiness changes (commit 0b54985), offline.

* Option values before the source (``ingest_isl_pose.py --budget-gb 26 openhands-raw``, ``ingest_isl_video.py run
  --workers auto wslp``; argparse accepts both) were read as the source ("26", "auto"): a source no profile reads, so
  the job blocked nothing (fail open). Option values are skipped and only known names count; an unknown name blocks
  every profile.
* OpenHands raw (budget-limited) counted as complete as soon as its job was gone, also after a crash (25 Nov 22:06 UTC:
  h5py "bad symbol table node signature" during the NewzHook listing). It is complete only when its log's last line
  is a normal end (all members done, disk-guard stop, deliberate SIGINT).
"""

from __future__ import annotations

from app.isl.training import readiness as rd
from tests.isl.helpers import write_manifest
from tests.isl.test_readiness_queue import DEFAULT, World


def _job_reasons(rep: dict, pid: int) -> list[str]:
    return [r for r in rep["reasons"] if f"pid {pid} " in r]


def test_option_values_before_the_source_are_not_the_source():
    cases = {
        ("ingest_isl_pose.py", ("--budget-gb", "26", "openhands-raw")): "openhands_raw",
        ("ingest_isl_pose.py", ("--workers", "4", "isl500")): "isl500",
        ("ingest_isl_pose.py", ("isl500-links",)): "isl500",  # writes derived/isl500_links.jsonl for ISL500
        ("ingest_isl_video.py", ("run", "--workers", "auto", "wslp")): "wslp",
        ("ingest_isl_video.py", ("list", "--per-class", "5", "hf_hemg_alphabet")): "hf_hemg_alphabet",
        ("ingest_isl_video.py", ("--root", "/x", "run", "mendeley-agri")): "mendeley_agri",
        ("ingest_isl.py", ("--workers", "3", "islrtc")): "islrtc",
        ("ingest_isl.py", ("islrtc-recut", "--follow")): "islrtc",
        # unknown names are unknown (None blocks every profile), never "a source nobody reads"
        ("ingest_isl_video.py", ("run", "bogus")): None,
        ("ingest_isl_pose.py", ("--budget-gb", "26")): None,
        ("ingest_isl.py", ("--tmp", "/t", "--workers", "2")): None,
    }
    for (script, rest), want in cases.items():
        assert rd.process_source(script, list(rest)) == want, (script, rest)


def test_pose_job_with_options_first_blocks_research_only(tmp_path):
    w = World(tmp_path)
    w.add(400, ["/v/python", "scripts/ingest_isl_pose.py", "--budget-gb", "26", "openhands-raw", "--workers", "12"])
    (w.P.manifests / "openhands_raw.lock").write_text("400")
    research, cs = w.check("research"), w.check("commercial-safe")
    assert _job_reasons(research, 400), research["reasons"]  # was: source "26", blocking nobody
    assert not _job_reasons(cs, 400)
    assert next(p for p in cs["processes"] if p["pid"] == 400)["source"] == "openhands_raw"


def test_video_run_with_options_first_blocks_its_profiles_and_unknown_blocks_all(tmp_path):
    w = World(tmp_path)
    w.add(410, ["/v/python", "scripts/ingest_isl_video.py", "run", "--workers", "auto", "mendeley_agri"])
    assert _job_reasons(w.check("commercial-safe"), 410)  # was: source "auto"
    w.drop(410)
    w.add(411, ["/v/python", "scripts/ingest_isl_video.py", "run", "--workers", "3", "hf_akrit_alphabet"])
    assert not _job_reasons(w.check("commercial-safe"), 411)  # research-only source: commercial-safe unaffected
    assert _job_reasons(w.check("research"), 411)
    w.drop(411)
    w.add(412, ["/v/python", "scripts/ingest_isl_video.py", "run", "some_new_source"])
    assert _job_reasons(w.check("commercial-safe"), 412)  # unknown: fail closed


def test_queue_child_with_options_first_is_a_known_child(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "scripts/isl_queue.sh"])
    w.add(301, ["/v/python", "scripts/ingest_isl_video.py", "run", "--workers", "auto", "wslp"], ppid=300)
    w.logs(f"queue start (pid 300): {DEFAULT}", "islrtc-full: finished", "wslp: start chunk (free disk 30 GB)")
    q = next(p for p in w.check("commercial-safe")["processes"] if p["pid"] == 300)["queue"]
    assert q["error"] is None and q["children"] == ["wslp"]


def _openhands(w: World, *log_lines: str | None) -> dict:
    write_manifest(w.P.manifest("openhands_raw"), [{"id": f"openhands_raw/{i}", "ok": True} for i in range(5)])
    log_f = w.P.root / "logs" / "ingest_pose_openhands_raw.log"
    log_f.parent.mkdir(parents=True, exist_ok=True)
    if log_lines == (None,):
        log_f.unlink(missing_ok=True)
    else:
        log_f.write_text("".join(f"{ln}\n" for ln in log_lines))
    return w.check("research")["sources"]["openhands_raw"]


def test_crashed_budget_limited_job_is_not_complete(tmp_path):
    w = World(tmp_path)
    crash = _openhands(w, "2026-11-25 22:02:30,542 INFO openhands-raw: NewzHook.zip: listing datasets",
                       "Traceback (most recent call last):", '  File "x.py", line 1, in <module>',
                       "RuntimeError: Unable to get group info (bad symbol table node signature)", "")
    assert crash["complete"] is False and not crash["job_end"]["clean"]
    assert "did not end normally" in crash["note"] and "bad symbol table" in crash["note"]
    rep = w.check("research")
    assert any(r.startswith("manifest incomplete: openhands_raw") for r in rep["reasons"])

    killed = _openhands(w, "2026-11-25 22:10:00,000 INFO openhands-raw: x: 100 chunks this run (1.0/s)")
    assert killed["complete"] is False  # SIGKILL / OOM: no end line
    assert _openhands(w, None)["complete"] is False  # no log: cannot confirm


def test_budget_limited_job_that_ended_normally_is_complete(tmp_path):
    w = World(tmp_path)
    for last in ("2026-11-26 01:00:00,000 INFO openhands-raw: all members done (12 new chunks this run)",
                 "2026-11-26 01:00:00,000 ERROR openhands-raw: stopping: landmark budget 26.0 GB reached",
                 "KeyboardInterrupt"):  # the documented stop: kill -INT (ISL_DESIGN 15.6, command 4)
        st = _openhands(w, "2026-11-26 00:00:00,000 INFO openhands-raw: start", last)
        assert st["complete"] is True and st["job_end"]["clean"], last
    assert not any("openhands_raw" in r for r in w.check("research")["reasons"])


def test_running_budget_limited_job_is_incomplete_without_reading_the_log(tmp_path):
    w = World(tmp_path)
    w.add(420, ["/v/python", "scripts/ingest_isl_pose.py", "openhands-raw", "--budget-gb", "26"])
    (w.P.manifests / "openhands_raw.lock").write_text("420")
    st = _openhands(w, "2026-11-26 01:00:00,000 INFO openhands-raw: all members done (0 new chunks this run)")
    assert st["complete"] is False and "job_end" not in st
