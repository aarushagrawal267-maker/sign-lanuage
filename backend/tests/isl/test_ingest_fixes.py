"""Ingestion fixes of 25 Nov 2026 (no network): ISLRTC title-card detector card-v2 and the re-extraction
targets, exact job detection, per-manifest locks, the video engine's memory stop, the single CISLR path and the
iSign GloSL skip."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile

import numpy as np
import pytest

from app.core.config import REPO_ROOT
from app.isl.data.video_sources import listings as L
from app.isl.data.video_sources.catalog import get_source
from app.isl.data.video_sources.engine import Fetcher, run
from app.isl.paths import isl_paths

cv2 = pytest.importorskip("cv2")


def _load(name: str):
    if str(REPO_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ing():
    return _load("ingest_isl")


# --------------------------------------------------------------------------------------------- card detector

GREY = (128, 128, 128)


def _signer_frame(bg=GREY, x=0.5, label=False, picture=False, h=360, w=640):
    """Plain background, a dark shirt + skin head around ``x`` (fraction of the width), optional word label."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = bg
    cx = int(x * w)
    cv2.rectangle(f, (cx - 70, int(h * 0.35)), (cx + 70, h), (20, 20, 20), -1)  # shirt
    cv2.circle(f, (cx, int(h * 0.22)), 40, (120, 160, 200), -1)  # head
    if label:
        cv2.rectangle(f, (20, 60), (160, 90), (90, 90, 90), -1)
        cv2.putText(f, "word", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if picture:
        cv2.rectangle(f, (30, 40), (150, 150), (60, 140, 60), -1)
    return f


def _card_frame(colour=(160, 30, 60), h=360, w=640):
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = colour
    cv2.putText(f, "Explanation", (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return f


def _stop_ms(ing, frames, fps=12.0):
    det = ing.CardDetector()
    for i, f in enumerate(frames):
        if det.update(f, i * 1000.0 / fps):
            return i * 1000.0 / fps
    return None


def _old_rule_fires(frames, fps=12.0):
    run = 0
    for i, f in enumerate(frames):
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA)
        if float(cv2.cvtColor(small[:, :60], cv2.COLOR_BGR2GRAY).std()) < 6.0:
            run += 1
            if run >= 3 and i * 1000.0 / fps > 800:
                return True
        else:
            run = 0
    return False


def test_card_v2_ignores_signer_on_plain_background(ing):
    """The misfire of the old rule: plain grey background, signer in the centre, word label gone after 1 s."""
    frames = [_signer_frame(label=i < 12, x=0.5 + 0.03 * np.sin(i / 3)) for i in range(120)]
    assert _old_rule_fires(frames)
    assert _stop_ms(ing, frames) is None


def test_card_v2_ignores_cut_to_second_signer(ing):
    frames = [_signer_frame(picture=True, x=0.6) for _ in range(60)] + [_signer_frame(x=0.45) for _ in range(60)]
    assert _stop_ms(ing, frames) is None


def test_card_v2_stops_at_fade_to_black(ing):
    frames = [_signer_frame() for _ in range(40)]
    frames += [(f * a).astype(np.uint8) for a, f in zip((0.6, 0.3), [_signer_frame()] * 2)]
    frames += [np.zeros_like(frames[0]) for _ in range(10)]
    t = _stop_ms(ing, frames)
    assert t is not None and 42 * 1000 / 12 <= t <= 46 * 1000 / 12


def test_card_v2_stops_at_coloured_explanation_card(ing):
    frames = [_signer_frame(label=True) for _ in range(30)] + [_card_frame() for _ in range(12)]
    t = _stop_ms(ing, frames)
    assert t is not None and 30 * 1000 / 12 <= t <= 33 * 1000 / 12


def test_card_v2_stops_at_light_plain_card_and_respects_min_time(ing):
    frames = [_signer_frame(bg=(200, 200, 200)) for _ in range(20)] + [_card_frame((205, 205, 205))] * 8
    assert _stop_ms(ing, frames) is not None
    assert _stop_ms(ing, [np.zeros((360, 640, 3), np.uint8)] * 8) is None  # black opening (< 800 ms) only


def test_cut_at_card_generator_keeps_frames_before_the_card(ing):
    frames = [(_signer_frame(), i * 1000 / 12) for i in range(30)] + [(_card_frame(), (30 + i) * 1000 / 12)
                                                                       for i in range(10)]
    stop = {"reason": "max_s"}
    kept = list(ing._cut_at_card(iter(frames), stop))
    assert stop["reason"] == "card" and 30 <= len(kept) <= 32


# --------------------------------------------------------------------------------------------- re-extraction targets

def test_recut_targets(tmp_path, ing):
    main = tmp_path / "islrtc.jsonl"
    rows = [{"id": "islrtc/A/a.mp4", "ok": True, "stop": "card"},
            {"id": "islrtc/B/b.mp4", "ok": True, "stop": "card+posegap"},
            {"id": "islrtc/C/c.mp4", "ok": True, "stop": "max_s"},
            {"id": "islrtc/D/d.mp4", "ok": True, "stop": "card", "card_detector": ing.CARD_DETECTOR},
            {"id": "islrtc/E/e.mp4", "ok": False, "stop": "card"}]
    main.write_text("".join(json.dumps(r) + "\n" for r in rows))
    trim = tmp_path / "trim.jsonl"
    trim.write_text(json.dumps({"id": "islrtc/F/f.mp4", "method": "truncated"}) + "\n"
                    + json.dumps({"id": "islrtc/C/c.mp4", "method": "cap"}) + "\n")
    assert ing.recut_targets(main, trim, done=set()) == ["A/a.mp4", "B/b.mp4", "F/f.mp4"]
    assert ing.recut_targets(main, None, done={"islrtc/A/a.mp4"}) == ["B/b.mp4"]


def test_running_video_jobs_is_exact():
    m = _load("ingest_isl_video")
    table = [(11, [".venv/bin/python", "scripts/ingest_isl.py", "islrtc-recut", "--workers", "1"]),
             (12, [".venv/bin/python", "scripts/ingest_isl.py", "include", "--workers", "1"])]
    assert m.running_video_jobs(table) == {"include": True, "islrtc": False}
    table.append((13, [".venv/bin/python", "scripts/ingest_isl.py", "islrtc", "--workers", "3"]))
    assert m.running_video_jobs(table) == {"include": True, "islrtc": True}


# --------------------------------------------------------------------------------------------- locks, memory stop

def test_video_source_lock_is_exclusive(tmp_path):
    m = _load("ingest_isl_video")
    paths = isl_paths(tmp_path / "isl")
    a = m.source_lock(paths, get_source("wslp"), "wslp", table=[])
    assert a is not None
    assert m.source_lock(paths, get_source("wslp_unlabelled"), "wslp_unlabelled", table=[]) is None  # same manifest
    assert m.source_lock(paths, get_source("cislr"), "cislr", table=[]) is not None
    a.close()
    assert m.source_lock(paths, get_source("wslp"), "wslp", table=[]) is not None
    old_run = [(77, [".venv/bin/python", "scripts/ingest_isl_video.py", "run", "wslp", "--workers", "auto"])]
    assert m.source_lock(paths, get_source("wslp"), "wslp", table=old_run) is None  # unlocked older copy


def test_engine_stops_for_memory_and_cleans_temp(tmp_path):
    paths = isl_paths(tmp_path / "isl")
    spec = get_source("wslp")
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as zf:
        for k in "abcdef":
            zf.writestr(f"train/{k}.mp4", b"x" + k.encode())
    zf = zipfile.ZipFile(io.BytesIO(b.getvalue()))
    items = L.wslp_items("uid\n", [(i.filename, i.header_offset) for i in zf.infolist()], "u", labelled=False)

    def extract(mode, src, out, step_fps, max_s):
        return {"ok": True, "n_frames": 10, "hands_frac": 1.0}

    st = run(spec, items, paths, Fetcher({"u": zf}), extract=extract, min_free_gb=0, max_rss_gb=1.0,
             rss=lambda pid=None: 3.0)
    assert st["stopped"] == "memory" and 1 <= st["ok"] < 6
    assert not any((paths.tmp / "video_wslp").iterdir())
    st2 = run(spec, items, paths, Fetcher({"u": zipfile.ZipFile(io.BytesIO(b.getvalue()))}), extract=extract,
              min_free_gb=0, max_rss_gb=1.0, rss=lambda pid=None: 0.2)
    assert st2["stopped"] is None and st["ok"] + st2["ok"] == 6


# --------------------------------------------------------------------------------------------- gated sources

def test_gated_cislr_delegates_to_the_queue_path(monkeypatch):
    g = _load("ingest_isl_gated")
    import ingest_isl_video

    seen = {}
    monkeypatch.setattr(ingest_isl_video, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    ns = type("A", (), {"workers": 5, "limit": 7, "root": None})()
    assert g.run_cislr(ns, None) == 0
    assert seen["argv"] == ["run", "cislr", "--workers", "2", "--limit", "7"]


def test_isign_skips_glosl_uids(tmp_path):
    g = _load("ingest_isl_gated")
    sentences = {"u1": "a", "u2": "b", "u3": "c"}
    f = tmp_path / "glosl_isign_uids.txt"
    f.write_text("u1\nu3\n\n")
    skip = g.read_uid_file(f)
    assert skip == {"u1", "u3"} and g.read_uid_file(tmp_path / "missing.txt") == set()
    assert g.isign_wanted(sentences, skip, None) == {"u2"}
    assert g.isign_wanted(sentences, skip, ["u1", "u2"]) == {"u2"}


def test_queue_handles_memory_busy_and_gated_probes():
    q = (REPO_ROOT / "scripts" / "isl_queue.sh").read_text()
    assert "--max-rss-gb" in q and "5) mem=" in q and "6) log" in q
    assert "ingest_isl_gated.py --check cislr" in q and "ingest_isl_gated.py --check isign" in q
    assert "9>&-" in q  # children never hold the queue lock


def test_opening_intro_is_skipped_and_timestamps_rebased(ing):
    """ "Academic Terms" videos open with ~6.5 s of animated title (no signer): both card rules used to stop at
    0.88 s with no sign. The intro is now skipped and max_s counts from the first signer frame."""
    fps = 12.0
    intro = [_card_frame((150, 90, 10)) for _ in range(78)]  # 6.5 s
    signer = [_signer_frame(bg=(230, 200, 170)) for _ in range(200)]
    frames = [(f, i * 1000 / fps) for i, f in enumerate(intro + signer)]
    stop = {"reason": "max_s"}
    kept = list(ing._cut_at_card(iter(frames), stop, max_s=10.0))
    assert stop["intro_ms"] == pytest.approx(6500.0) and stop["reason"] == "max_s"
    assert kept[0][1] == 0.0 and kept[-1][1] <= 10_000.0 and len(kept) == 121
    # no signer within INTRO_MAX_S: stop as a card, nothing kept
    stop = {"reason": "max_s"}
    assert list(ing._cut_at_card(iter([(f, i * 1000 / fps) for i, f in enumerate(intro * 3)]), stop)) == []
    assert stop["reason"] == "card"
    # a video that starts on the signer is unchanged (no intro, original timestamps)
    stop = {"reason": "max_s"}
    kept = list(ing._cut_at_card(iter([(f, i * 1000 / fps) for i, f in enumerate(signer[:30])]), stop))
    assert len(kept) == 30 and stop.get("intro_ms") == 0.0 and kept[5][1] == pytest.approx(5000 / 12)
