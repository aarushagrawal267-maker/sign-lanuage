"""In-process background jobs (download, validation, extraction, training).

Status and progress are persisted in the ``jobs`` table so the UI shows real
progress. Long jobs check ``should_stop`` and can be cancelled.
"""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import select

from app.db.models import Job, utcnow
from app.db.session import session_scope

log = logging.getLogger(__name__)

JobFn = Callable[[Callable[[float, str], None], Callable[[], bool]], dict]


class JobManager:
    def __init__(self, workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")
        self._stop: dict[int, threading.Event] = {}
        self._kind_lock: dict[str, threading.Lock] = {}

    def submit(self, kind: str, params: dict, fn: JobFn, on_error: Callable[[str], None] | None = None) -> int:
        with session_scope() as db:
            running = db.scalar(select(Job).where(Job.kind == kind, Job.status.in_(("queued", "running"))))
            if running is not None:
                raise RuntimeError(f"a '{kind}' job (#{running.id}) is already running")
            job = Job(kind=kind, params=params, status="queued")
            db.add(job)
            db.flush()
            jid = job.id
        ev = threading.Event()
        self._stop[jid] = ev
        self._pool.submit(self._run, jid, fn, ev, on_error)
        return jid

    def _update(self, jid: int, **kw) -> None:
        with session_scope() as db:
            j = db.get(Job, jid)
            for k, v in kw.items():
                setattr(j, k, v)

    def _run(self, jid: int, fn: JobFn, ev: threading.Event, on_error) -> None:
        self._update(jid, status="running")

        def progress(frac: float, msg: str) -> None:
            self._update(jid, progress=float(max(0.0, min(1.0, frac))), message=msg)

        try:
            result = fn(progress, ev.is_set)
            status = "cancelled" if ev.is_set() else "completed"
            kw = {"status": status, "result": result or {}, "finished_at": utcnow()}
            if status == "completed":
                kw["progress"] = 1.0
            self._update(jid, **kw)
        except Exception as e:  # noqa: BLE001 - recorded and surfaced to the UI
            from app.training.trainer import Cancelled

            if isinstance(e, Cancelled):
                self._update(jid, status="cancelled", finished_at=utcnow(), message="cancelled by user")
                if on_error:
                    on_error("cancelled")
                return
            log.exception("job %s failed", jid)
            self._update(jid, status="failed", message=f"{type(e).__name__}: {e}", finished_at=utcnow(),
                         result={"traceback": traceback.format_exc()[-4000:]})
            if on_error:
                on_error(f"{type(e).__name__}: {e}")
        finally:
            self._stop.pop(jid, None)

    def cancel(self, jid: int) -> bool:
        ev = self._stop.get(jid)
        if ev is None:
            return False
        ev.set()
        return True


def recover_stale_jobs() -> None:
    """Jobs marked running when the server starts were interrupted by a restart."""
    from app.db.models import Experiment

    with session_scope() as db:
        for j in db.scalars(select(Job).where(Job.status.in_(("queued", "running")))):
            j.status = "failed"
            j.message = "interrupted (server restarted)"
            j.finished_at = utcnow()
        for e in db.scalars(select(Experiment).where(Experiment.status.in_(("queued", "running")))):
            e.status = "failed"
            e.error = "interrupted (server restarted)"


_manager: JobManager | None = None


def get_jobs() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
