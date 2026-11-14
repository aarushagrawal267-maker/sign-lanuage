from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.dataset import acquisition as acq


def test_url_classification():
    assert acq.classify_url("https://www.youtube.com/watch?v=0UsjUE-TXns") == "youtube"
    assert acq.classify_url("https://youtu.be/0UsjUE-TXns") == "youtube"
    assert acq.classify_url("http://www.aslpro.com/main/b/book.swf") == "aslpro"
    assert acq.classify_url("https://media.spreadthesign.com/video/mp4/13/93.mp4") == "http"
    assert acq.youtube_id("https://www.youtube.com/watch?v=0UsjUE-TXns&t=3") == "0UsjUE-TXns"


def test_clip_range_is_one_based_inclusive():
    v = SimpleNamespace(frame_start=1, frame_end=-1)
    assert acq.clip_range(v, 50) == (0, 50)
    v = SimpleNamespace(frame_start=11, frame_end=20)
    assert acq.clip_range(v, 50) == (10, 20)
    v = SimpleNamespace(frame_start=40, frame_end=90)
    assert acq.clip_range(v, 50) == (39, 50)


def _write_video(path, n=20, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i * 10 % 255, np.uint8))
    vw.release()


def test_probe_valid_and_corrupt(tmp_path):
    good = tmp_path / "good.mp4"
    _write_video(good)
    r = acq.probe_video(good)
    assert r.ok and r.frame_count == 20 and r.width == 64 and abs(r.fps - 25) < 0.5
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"<html>not a video</html>")
    assert not acq.probe_video(bad).ok


class _Resp:
    def __init__(self, status, ctype="video/mp4", body=b""):
        self.status_code, self.headers, self._body = status, {"content-type": ctype}, body

    def iter_content(self, n):
        yield self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_http_errors_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(acq.requests, "get", lambda *a, **k: _Resp(404))
    with pytest.raises(acq.DownloadError) as e:
        acq._http_get("https://x.org/v.mp4", tmp_path / "v.mp4")
    assert e.value.http_status == 404
    monkeypatch.setattr(acq.requests, "get", lambda *a, **k: _Resp(200, "text/html"))
    with pytest.raises(acq.DownloadError):
        acq._http_get("https://x.org/v.mp4", tmp_path / "v.mp4")
    assert not (tmp_path / "v.mp4").exists()


def test_acquire_records_failures_and_validates(tmp_path, monkeypatch):
    """End-to-end acquisition bookkeeping with the network mocked out."""
    from tests.conftest import fresh_data_dir

    fresh_data_dir("acquire")
    from app.db.models import DownloadAttempt, Gloss, Video
    from app.db.session import session_scope

    with session_scope() as db:
        db.add(Gloss(class_index=0, gloss="book"))
        for vid, url in (("00001", "https://ok.org/a.mp4"), ("00002", "https://dead.org/b.mp4")):
            db.add(Video(video_id=vid, class_index=0, gloss="book", signer_id=1, split="train", source="x", url=url,
                         annotated_fps=25, frame_start=1, frame_end=-1))
    src = tmp_path / "src.mp4"
    _write_video(src)

    def fake_get(url, **k):
        if "dead" in url:
            return _Resp(410)
        return _Resp(200, body=src.read_bytes())

    monkeypatch.setattr(acq.requests, "get", fake_get)
    monkeypatch.setattr(acq.time, "sleep", lambda s: None)
    counts = acq.acquire()
    assert counts["valid"] == 1 and counts["failed"] == 1
    with session_scope() as db:
        ok, dead = db.get(Video, "00001"), db.get(Video, "00002")
        assert ok.status == "valid" and ok.frame_count == 20 and ok.sha256
        assert dead.status == "failed" and "410" in dead.error
        assert db.query(DownloadAttempt).count() == 2
    # re-running does not re-download valid videos
    assert acq.acquire()["requested"] == 0
