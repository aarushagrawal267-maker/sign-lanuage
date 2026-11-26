"""The data guard and the video queue, per profile (docs/ISL_DESIGN.md 15.6, command 1), offline.

``scripts/isl_queue.sh`` blocked every profile until its log said "all listed sources processed". Its list ends with
research-only sources (the two HF alphabet sets, WSLP unlabelled), so the commercial-safe run waited hours for data it
never reads. Now the queue blocks only the profiles that read a source it still has to process, and anything the
guard cannot read or that does not agree blocks both (fail closed). Also: ``ingest_isl_video.py list <source>`` blocks
only the profiles of that source, and a gated source (CISLR, iSign) left partly ingested by a stopped job is reported
as "partial (gated; job not running)": visible, not blocking research.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.isl.training import readiness as rd
from tests.isl.helpers import write_manifest

DEFAULT = "islrtc-full wslp mendeley_static isl_csltr_words hf_akrit_alphabet wslp_unlabelled"
COMMERCIAL_SOURCES = ("include", "islrtc", "mendeley_emergency", "mendeley_agri", "isl_csltr", "mendeley_static")


class World:
    def __init__(self, tmp_path: Path, default: str = DEFAULT):
        from app.isl.paths import isl_paths

        self.P = isl_paths(tmp_path / "isl")
        self.P.manifests.mkdir(parents=True)
        for name in COMMERCIAL_SOURCES:  # every commercial-safe manifest complete
            ids = [f"{name}/{i}" for i in range(3)]
            write_manifest(self.P.manifest(name), [{"id": i, "ok": True} for i in ids])
            (self.P.manifests / f"{name}_items.json").write_text(json.dumps([{"id": i} for i in ids]))
        self.proc = tmp_path / "proc"
        self.proc.mkdir()
        self.log = tmp_path / "queue.log"
        self.script = tmp_path / "isl_queue.sh"
        self.script.write_text('set -u\nCHUNK_MIN="${CHUNK_MIN:-45}"\n'
                               f'SOURCES="${{SOURCES:-{default}}}"\nlog "queue start (pid $$): $SOURCES"\n')

    def add(self, pid: int, argv: list[str], ppid: int = 1, env: dict | None = None) -> None:
        d = self.proc / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "status").write_text(f"Name:\tx\nPid:\t{pid}\nPPid:\t{ppid}\n")
        (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in
                                               {"PATH": "/bin", **(env or {})}.items()) + b"\0")

    def drop(self, pid: int) -> None:
        for f in (self.proc / str(pid)).iterdir():
            f.unlink()
        (self.proc / str(pid)).rmdir()

    def logs(self, *lines: str) -> None:
        with self.log.open("a") as fh:
            for i, ln in enumerate(lines):
                fh.write(f"2026-11-26 0{i % 10}:00:00 {ln}\n")

    def check(self, profile: str) -> dict:
        return rd.check(self.P, profile, proc_root=self.proc, queue_log=self.log, queue_script=self.script)


def queue_reasons(rep: dict) -> list[str]:
    return [r for r in rep["reasons"] if "isl_queue.sh" in r]


def queue_of(rep: dict) -> dict:
    return next(p for p in rep["processes"] if p["script"] == "isl_queue.sh")


def test_queue_blocks_only_profiles_that_read_a_remaining_source(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "scripts/isl_queue.sh"])
    w.add(301, ["/v/python", "scripts/ingest_isl_video.py", "run", "wslp", "--workers", "auto"], ppid=300)
    w.add(302, ["bash", "scripts/isl_queue.sh"], ppid=300)  # a "$(free_gb)" subshell: same queue
    w.logs(f"queue start (pid 300): {DEFAULT}", "islrtc-full: start chunk (free disk 30 GB)", "islrtc-full: finished",
           "wslp: start chunk (free disk 30 GB)", "wslp: chunk done, more to do", "wslp: start chunk (free disk 29 GB)")
    rep = w.check("commercial-safe")
    assert [p["pid"] for p in rep["processes"]] == [300, 301]  # the subshell is not a second queue
    q = queue_of(rep)
    assert q["queue"]["current"] == "wslp" and q["queue"]["children"] == ["wslp"]
    assert q["queue"]["remaining"] == ["wslp", "mendeley_static", "isl_csltr_words", "hf_akrit_alphabet",
                                       "wslp_unlabelled"]
    assert rep["reasons"] == [("ingestion job running: pid 300 isl_queue.sh (still to process for commercial-safe: "
                               "mendeley_static)")]
    assert not rep["ready"]
    # mendeley_static done; the queue is on isl_csltr_words (no profile reads it): commercial-safe is ready
    w.drop(301)
    w.add(303, ["/v/python", "scripts/ingest_isl_video.py", "run", "isl_csltr_words", "--workers", "auto"], ppid=300)
    w.logs("wslp: finished", "mendeley_static: start chunk (free disk 29 GB)", "mendeley_static: finished",
           "isl_csltr_words: start chunk (free disk 29 GB)")
    rep = w.check("commercial-safe")
    assert rep["ready"], rep["reasons"]
    assert "reads none" in queue_of(rep)["note"] and not queue_of(rep)["blocking"]
    # research still waits for the queue (HF alphabet, WSLP unlabelled = WSLP manifest, and CISLR / iSign may start)
    rep = w.check("research")
    q = queue_of(rep)
    assert q["blocking"] and q["sources"] == ["isl_csltr_words", "hf_akrit_alphabet", "wslp", "wslp_unlabelled",
                                              "cislr", "isign"]
    assert "hf_akrit_alphabet, wslp, wslp_unlabelled, cislr, isign" in queue_reasons(rep)[0]
    # gated datasets that the queue finished in this run are no longer pending
    w.logs("cislr: finished", "isign: finished")
    assert w.check("research")["processes"][0]["queue"]["gated_pending"] == []
    # list done: the queue only polls -> blocks nothing
    w.drop(303)
    w.logs("isl_csltr_words: finished", "hf_akrit_alphabet: giving up for now", "wslp_unlabelled: finished",
           "all listed sources processed; status:")
    for prof in ("commercial-safe", "research"):
        rep = w.check(prof)
        assert not queue_reasons(rep) and queue_of(rep)["queue"]["polling_only"]


def test_terminal_lines_and_out_of_order_logs(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "/repo/scripts/isl_queue.sh"])
    # mendeley_static gave up (gated / memory / errors): behind the queue. wslp never logged an end although a later
    # source started (an odd log): it still counts as remaining (fail closed)
    w.logs(f"queue start (pid 300): {DEFAULT}", "islrtc-full: gated dataset not accessible yet",
           "wslp: start chunk (free disk 30 GB)", "mendeley_static: too many memory stops; giving up for now",
           "isl_csltr_words: start chunk (free disk 30 GB)")
    q = queue_of(w.check("commercial-safe"))["queue"]
    assert q["remaining"] == ["wslp", "isl_csltr_words", "hf_akrit_alphabet", "wslp_unlabelled"]
    assert q["current"] == "isl_csltr_words" and q["error"] is None
    assert w.check("commercial-safe")["ready"]
    # a source that ran out of fetch-failure retry passes is behind the queue too (its rows still count as missing
    # in the manifest check); a CISLR run that moved on is not done: the queue probes and re-runs it
    w.log.write_text("")
    w.logs(f"queue start (pid 300): {DEFAULT}", "islrtc-full: finished", "wslp: finished",
           "mendeley_static: fetch failures remain after 3 retry passes; moving on",
           "cislr: fetch failures remain after 3 retry passes; moving on", "isign: finished",
           "isl_csltr_words: finished", "hf_akrit_alphabet: start chunk (free disk 30 GB)")
    q = queue_of(w.check("commercial-safe"))
    assert q["queue"]["remaining"] == ["hf_akrit_alphabet", "wslp_unlabelled"] and q["queue"]["error"] is None
    assert q["queue"]["gated_pending"] == ["cislr"] and not q["blocking"]
    assert w.check("commercial-safe")["ready"]
    assert rd._terminal("mendeley_static: fetch failures remain after 5 retry passes; moving on")
    # no source started yet: the whole list remains
    w.log.write_text("")
    w.logs(f"queue start (pid 300): {DEFAULT}")
    q = queue_of(w.check("commercial-safe"))
    assert q["queue"]["remaining"] == DEFAULT.split() and q["blocking"]
    assert "islrtc, mendeley_static" in w.check("commercial-safe")["reasons"][0]


def _both_blocked(w: World, why: str) -> None:
    for prof in ("commercial-safe", "research"):
        rep = w.check(prof)
        q = queue_of(rep)
        assert q["blocking"] and q["sources"] is None, (prof, q)
        assert why in q["queue"]["error"], q["queue"]["error"]
        assert any("progress unknown" in r for r in queue_reasons(rep)), rep["reasons"]


def test_unreadable_or_inconsistent_queue_state_blocks_every_profile(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "scripts/isl_queue.sh"])
    _both_blocked(w, "queue log not readable")
    w.logs("2026-11-26 queue stopped")
    _both_blocked(w, "no 'queue start")
    w.logs(f"queue start (pid 299): {DEFAULT}")  # another (older / newer) queue's run
    _both_blocked(w, "pid 299, not this queue (pid 300)")
    w.logs("queue start (pid 300): islrtc-full wslp_unlabelled")  # differs from the script's default list
    _both_blocked(w, "differs from the SOURCES default")
    w.logs("queue start (pid 300): islrtc-full wslp new_source")
    w.script.write_text(w.script.read_text().replace(DEFAULT, "islrtc-full wslp new_source"))
    _both_blocked(w, "unknown sources: ['new_source']")
    w.script.unlink()
    _both_blocked(w, "could not read the SOURCES default")
    w.logs("queue start (pid 300): wslp wslp")
    _both_blocked(w, "repeated names")


def test_environment_sources_override_the_script_default(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "scripts/isl_queue.sh"], env={"SOURCES": "hf_akrit_alphabet wslp_unlabelled"})
    w.logs("queue start (pid 300): hf_akrit_alphabet wslp_unlabelled", "hf_akrit_alphabet: start chunk (x)")
    rep = w.check("commercial-safe")
    assert rep["ready"], rep["reasons"]
    assert queue_of(rep)["queue"]["list_checked_against"] == "SOURCES in the queue's environment"
    assert queue_of(w.check("research"))["blocking"]
    # an empty SOURCES falls back to the default in bash ("${SOURCES:-...}"): so does the guard
    w2 = World(tmp_path / "b")
    w2.add(300, ["bash", "scripts/isl_queue.sh"], env={"SOURCES": ""})
    w2.logs("queue start (pid 300): hf_akrit_alphabet wslp_unlabelled")
    _both_blocked(w2, "differs from the SOURCES default")


def test_a_queue_child_with_an_unknown_or_unlisted_source(tmp_path):
    w = World(tmp_path)
    w.add(300, ["bash", "scripts/isl_queue.sh"])
    w.logs(f"queue start (pid 300): {DEFAULT}", "islrtc-full: finished", "wslp: finished",
           "mendeley_static: finished", "isl_csltr_words: finished", "hf_akrit_alphabet: start chunk (x)")
    assert w.check("commercial-safe")["ready"]
    # a child the log does not explain (a CISLR chunk started by the gated probe) still counts for its profiles
    w.add(310, ["/v/python", "scripts/ingest_isl_video.py", "run", "mendeley_agri"], ppid=300)
    rep = w.check("commercial-safe")
    assert "mendeley_agri" in queue_of(rep)["sources"] and len(queue_reasons(rep)) == 1
    w.drop(310)
    w.add(311, ["/v/python", "scripts/ingest_isl_video.py", "something-new"], ppid=300)
    _both_blocked(w, "child job of the queue writes an unknown source")


def test_list_subcommand_blocks_only_profiles_of_its_source(tmp_path):
    w = World(tmp_path)
    w.add(400, ["/v/python", "scripts/ingest_isl_video.py", "list", "hf_akrit_alphabet"])
    assert w.check("commercial-safe")["ready"]
    assert "pid 400 ingest_isl_video.py list" in w.check("research")["reasons"][0]
    w.add(401, ["/v/python", "scripts/ingest_isl_video.py", "--root", "/data/isl", "list", "mendeley_static"])
    rep = w.check("commercial-safe")
    assert [r.split()[4] for r in rep["reasons"]] == ["401"]
    assert {p["pid"]: p["source"] for p in rep["processes"]} == {400: "hf_akrit_alphabet", 401: "mendeley_static"}
    w.add(402, ["/v/python", "scripts/ingest_isl_video.py", "list"])  # no source: unknown -> every profile
    assert "402" in " ".join(w.check("commercial-safe")["reasons"])
    # read-only sub-commands after a global option are still read-only
    w.add(403, ["/v/python", "scripts/ingest_isl_video.py", "--root", "/data/isl", "status"])
    assert 403 not in [p["pid"] for p in w.check("research")["processes"]]
    # a job of WSLP unlabelled writes the WSLP manifest: research only
    w.add(404, ["/v/python", "scripts/ingest_isl_video.py", "run", "wslp_unlabelled"])
    assert 404 not in [int(r.split()[4]) for r in w.check("commercial-safe")["reasons"]]


def test_partial_gated_source_is_visible_but_does_not_block_research(tmp_path):
    w = World(tmp_path)
    ids = [f"cislr/{i}" for i in range(10)]
    (w.P.manifests / "cislr_items.json").write_text(json.dumps([{"id": i} for i in ids]))
    write_manifest(w.P.manifest("cislr"), [{"id": i, "ok": True} for i in ids[:4]])
    write_manifest(w.P.manifest("isign"), [{"id": "isign/a", "ok": True}])  # no item list: total unknown
    rep = w.check("research")
    assert not any("cislr" in r or "isign" in r for r in rep["reasons"]), rep["reasons"]
    st = rep["sources"]["cislr"]
    assert st["complete"] is None and st["state"] == rd.GATED_PARTIAL and st["done"] == 4 and st["expected"] == 10
    assert rep["sources"]["isign"]["state"] == rd.GATED_PARTIAL
    assert set(rep["partial_gated"]) == {"cislr", "isign"}
    assert [x.split(":")[0] for x in rep["warnings"]] == ["cislr", "isign"]
    assert "4/10 rows" in rep["warnings"][0] and "not blocking" in rep["warnings"][0]
    # commercial-safe never reads them: no warning there
    assert w.check("commercial-safe")["warnings"] == []
    # the job is running again: it blocks (process and manifest), no longer a warning
    w.add(500, ["/v/python", "scripts/ingest_isl_video.py", "run", "cislr", "--workers", "auto"])
    rep = w.check("research")
    assert rep["sources"]["cislr"]["state"] == rd.GATED_RUNNING and rep["sources"]["cislr"]["complete"] is False
    assert any(r.startswith("manifest incomplete: cislr 4/10") for r in rep["reasons"])
    assert any(r.startswith("ingestion job running: pid 500") for r in rep["reasons"])
    assert [x.split(":")[0] for x in rep["warnings"]] == ["isign"]
    # complete gated data is just complete; no rows at all is "unavailable", as before
    w.drop(500)
    write_manifest(w.P.manifest("cislr"), [{"id": i, "ok": True} for i in ids])
    assert w.check("research")["sources"]["cislr"]["complete"] is True
    w.P.manifest("isign").unlink()
    st = w.check("research")["sources"]["isign"]
    assert st["complete"] is None and "no access" in st["note"] and not st.get("partial_gated")


def test_report_data_section_shows_the_guard_warnings(tmp_path):
    """data_readiness.json (the driver's --out) -> evaluation report: data.readiness + a markdown section."""
    from app.isl.evaluation.runners import markdown_report

    w = World(tmp_path)
    write_manifest(w.P.manifest("cislr"), [{"id": "cislr/1", "ok": True}])
    rep = w.check("research")
    work = tmp_path / "work"
    work.mkdir()
    assert rd.readiness_summary(work) is None
    (work / "data_readiness.json").write_text(json.dumps(rep, indent=1))
    s = rd.readiness_summary(work)
    assert s["partial_gated"]["cislr"]["state"] == rd.GATED_PARTIAL and s["warnings"] == rep["warnings"]
    doc = {"profile": "research", "label": "x", "created": "t", "protocols": {}, "leakage_guards": {},
           "data": {"readiness": s}}
    md = markdown_report(doc)
    assert "## Data readiness" in md and "partial (gated; job not running)" in md
