"""Label listing of the Hugging Face alphabet image sets (``hf_akrit_alphabet``, ``hf_hemg_alphabet``).

The datasets-server rows API (``/rows``, at most 100 rows per page) is the only way to get each row's label without
downloading the parquet files (no parquet reader in the venv). It rate-limits (429) and has 503 spells, which is why
the first listings of both sets failed (docs/DATASETS_ISL.md 3.6). This module lists one split politely and
resumably:

* at most one request every ``min_interval`` seconds (default 2 s);
* on 429 / 5xx / connection errors: exponential backoff with jitter, and never shorter than the server's
  ``Retry-After`` (seconds or an HTTP date); a ``Retry-After`` above ``max_retry_after`` stops the listing at once
  (progress kept) instead of sleeping for hours;
* the rows listed so far are saved after **every** page (``labels_<split>.partial.json``, atomic replace), so a
  killed or refused run resumes where it stopped; only a contiguous, validated prefix is ever kept;
* a status file (``listing_status.json``) says per split ``complete`` / ``listing`` / ``incomplete`` / ``failed``,
  rows done / total and the last error, for the queue log and the main session;
* the finished list is checked (row ids 0..n-1, n = ``num_rows_total``, labels inside the class list) before it is
  written as ``labels_<split>.json``.

Everything takes ``get`` / ``sleep`` / ``clock`` so the logic is unit-tested offline. The image of a row is fetched
later by the engine (``Fetcher``, kind ``hfrow``) through :class:`RowPages`, which uses the same pacing, backoff and
``Retry-After`` rules and fetches each 100-row page once (its signed image URLs are cached until shortly before they
expire); item ids are ``<prefix>/<split>/<row>`` (``listings.hf_rows_items``).
"""

from __future__ import annotations

import email.utils
import json
import logging
import random
import threading
import time
import urllib.parse
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

HF_ROWS = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the rows API maximum
STATUS_FILE = "listing_status.json"


class ListingError(OSError):
    """The listing cannot go on now. ``progressed`` = new rows were saved in this call (worth retrying soon);
    ``permanent`` = the server refused for good (4xx other than 429) - retrying will not help."""

    def __init__(self, msg: str, *, progressed: bool = False, permanent: bool = False, rows: int = 0,
                 total: int | None = None):
        super().__init__(msg)
        self.progressed, self.permanent, self.rows, self.total = progressed, permanent, rows, total


@dataclass(frozen=True)
class Policy:
    min_interval: float = 2.0  # seconds between two requests
    base: float = 5.0  # first backoff wait
    max_wait: float = 600.0  # backoff cap (one wait)
    max_failures: int = 8  # consecutive failed requests before giving up (about 20 min of waiting)
    max_retry_after: float = 1800.0  # a longer Retry-After stops the listing (progress kept) instead of sleeping
    jitter: float = 0.2  # +0..20 % random extra wait
    timeout: float = 120.0


def retry_after_s(headers, now: float | None = None) -> float | None:
    """Seconds from a ``Retry-After`` header (delta-seconds or an HTTP date); None if absent or unreadable."""
    if not headers:
        return None
    v = None
    for k in headers:
        if str(k).lower() == "retry-after":
            v = str(headers[k]).strip()
            break
    if not v:
        return None
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return max(0.0, dt.timestamp() - (time.time() if now is None else now))


def backoff_s(failures: int, retry_after: float | None, policy: Policy,
              rnd: Callable[[], float] = random.random) -> float:
    """Wait before the next try after ``failures`` consecutive failures (1 = the first)."""
    wait = min(policy.max_wait, policy.base * 2 ** max(0, failures - 1)) * (1.0 + policy.jitter * rnd())
    return max(wait, retry_after or 0.0)


def _atomic_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(path)


def _valid_prefix(doc: dict | None, n_names: int | None = None) -> list[list[int]]:
    """The longest prefix of ``doc["rows"]`` that is ``[[0, l0], [1, l1], ...]`` with labels in range."""
    out: list[list[int]] = []
    if not isinstance(doc, dict):
        return out
    for i, r in enumerate(doc.get("rows") or []):
        try:
            idx, lab = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            break
        if idx != i or lab < 0 or (n_names is not None and lab >= n_names):
            break
        out.append([idx, lab])
    return out


def load_partial(path: Path, dataset: str, split: str) -> dict:
    """Resume state from a partial file: a clean, contiguous prefix (or an empty state)."""
    empty = {"dataset": dataset, "split": split, "names": None, "rows": [], "num_rows_total": None}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(doc, dict) or doc.get("dataset", dataset) != dataset or doc.get("split", split) != split:
        return empty
    names = doc.get("names")
    if not (isinstance(names, list) and names):
        return empty
    rows = _valid_prefix(doc, len(names))
    total = doc.get("num_rows_total")
    return {"dataset": dataset, "split": split, "names": list(names), "rows": rows,
            "num_rows_total": int(total) if isinstance(total, int) else None}


def check_complete(doc: dict) -> None:
    """Raise ValueError unless ``doc`` is a whole, consistent label list."""
    names, rows, total = doc.get("names"), doc.get("rows") or [], doc.get("num_rows_total")
    if not (isinstance(names, list) and names):
        raise ValueError("no class names")
    if not isinstance(total, int) or total <= 0:
        raise ValueError(f"unknown row total {total!r}")
    if len(_valid_prefix(doc, len(names))) != len(rows) or len(rows) != total:
        raise ValueError(f"{len(rows)} rows listed, {total} expected (or rows out of order / labels out of range)")


def _label_names(features) -> list[str] | None:
    for f in features or []:
        if f.get("name") == "label":
            names = (f.get("type") or {}).get("names")
            return list(names) if isinstance(names, list) and names else None
    return None


def _page_rows(d: dict, off: int, n_names: int) -> list[list[int]]:
    """``[[row_idx, label]]`` of one page; ValueError if ids are not exactly off, off+1, ... or a label is bad."""
    out = []
    for k, x in enumerate(d.get("rows") or []):
        idx, lab = int(x["row_idx"]), int((x.get("row") or {})["label"])
        if idx != off + k:
            raise ValueError(f"page at offset {off}: row {k} has row_idx {idx}")
        if not 0 <= lab < n_names:
            raise ValueError(f"row {idx}: label {lab} outside the {n_names} classes")
        out.append([idx, lab])
    return out


class StatusFile:
    """``listing_status.json`` next to the label files: one entry per split plus the last error."""

    def __init__(self, path: Path | None, source: str, dataset: str, clock: Callable[[], float] = time.time):
        self.path, self.clock = path, clock
        self.doc = {"source": source, "dataset": dataset, "splits": {}}
        if path is not None and path.exists():
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(old, dict) and old.get("dataset") == dataset:
                    self.doc["splits"] = dict(old.get("splits") or {})
            except (OSError, ValueError):
                pass

    def update(self, split: str, **kw) -> None:
        e = self.doc["splits"].setdefault(split, {})
        e.update(kw)
        e["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock()))
        states = [s.get("state") for s in self.doc["splits"].values()]
        self.doc["state"] = ("complete" if states and all(s == "complete" for s in states)
                             else "failed" if "failed" in states else "incomplete" if "incomplete" in states
                             else "listing")
        if self.path is not None:
            _atomic_json(self.path, self.doc)


def list_split(dataset: str, split: str, final_path: Path, partial_path: Path, *, get, sleep=time.sleep,
               clock=time.monotonic, policy: Policy = Policy(), status: StatusFile | None = None,
               rnd: Callable[[], float] = random.random, config: str = "default") -> dict:
    """List one split's labels; returns ``{"names", "rows": [[row_idx, label]], "num_rows_total", ...}``.

    A finished ``final_path`` is reused as is (after validation). Otherwise the listing resumes from
    ``partial_path`` and saves it after every page. Raises :class:`ListingError` when the server keeps refusing;
    everything listed so far stays in ``partial_path``."""
    if final_path.exists():
        try:
            doc = json.loads(final_path.read_text(encoding="utf-8"))
            if "num_rows_total" not in doc:  # files written before this module: rows are the whole split
                doc["num_rows_total"] = len(doc.get("rows") or [])
            check_complete(doc)
            if status:
                status.update(split, state="complete", rows=len(doc["rows"]), total=doc["num_rows_total"])
            return doc
        except (OSError, ValueError) as e:
            log.warning("%s %s: %s is not a valid finished list (%s); listing again", dataset, split, final_path, e)

    st = load_partial(partial_path, dataset, split)
    start_rows = len(st["rows"])
    failures, last_req, last_err, name_changes = 0, None, "", 0

    def _save(state: str, **extra) -> None:
        _atomic_json(partial_path, st)
        if status:
            status.update(split, state=state, rows=len(st["rows"]), total=st["num_rows_total"], **extra)

    while st["num_rows_total"] is None or len(st["rows"]) < st["num_rows_total"]:
        if last_req is not None:
            gap = policy.min_interval - (clock() - last_req)
            if gap > 0:
                sleep(gap)
        off = len(st["rows"])
        last_req = clock()
        ra = None
        try:
            r = get(HF_ROWS, params={"dataset": dataset, "config": config, "split": split, "offset": off,
                                     "length": PAGE}, timeout=policy.timeout)
            code = int(r.status_code)
            if code == 200:
                d = r.json()
                names = _label_names(d.get("features"))
                if names is None:
                    raise ValueError("no ClassLabel 'label' feature in the rows API answer")
                if st["names"] is None:
                    st["names"] = names
                elif names != st["names"]:  # the dataset changed under us: start again
                    name_changes += 1
                    if name_changes > 2:  # answers flip between class lists: back off like any bad answer
                        raise ValueError(f"class list changed {name_changes} times during the listing")
                    log.warning("%s %s: class list changed; restarting the listing", dataset, split)
                    st.update(names=names, rows=[], num_rows_total=None)
                    start_rows = 0
                    continue
                total = int(d.get("num_rows_total") or 0)
                if total <= 0:
                    raise ValueError("num_rows_total missing")
                st["num_rows_total"] = total
                page = _page_rows(d, off, len(names))
                if not page and off < total:
                    raise ValueError(f"empty page at offset {off} of {total}")
                st["rows"] += page[: total - off]
                failures, last_err = 0, ""
                _save("listing")
                continue
            last_err = f"HTTP {code}"
            ra = retry_after_s(getattr(r, "headers", None))
            if code != 429 and code < 500:
                _save("failed", last_error=last_err)
                raise ListingError(f"{dataset} {split}: rows API {last_err} (not retryable)", permanent=True,
                                   progressed=len(st["rows"]) > start_rows, rows=len(st["rows"]),
                                   total=st["num_rows_total"])
        except ListingError:
            raise
        except Exception as e:  # noqa: BLE001 - connection errors, bad JSON, inconsistent pages: retry
            last_err = f"{type(e).__name__}: {e}"[:300]
        failures += 1
        wait = backoff_s(failures, ra, policy, rnd)
        if failures >= policy.max_failures or (ra is not None and ra > policy.max_retry_after):
            why = (f"Retry-After {ra:.0f} s" if ra is not None and ra > policy.max_retry_after
                   else f"{failures} failed tries in a row")
            _save("incomplete", last_error=f"{last_err} ({why})")
            raise ListingError(f"{dataset} {split}: listing stopped at {len(st['rows'])}/{st['num_rows_total']} rows "
                               f"after {why}; last error {last_err}; progress saved in {partial_path.name}",
                               progressed=len(st["rows"]) > start_rows, rows=len(st["rows"]),
                               total=st["num_rows_total"])
        _save("listing", last_error=last_err)
        log.info("%s %s: %s at offset %d; waiting %.0f s (try %d of %d)", dataset, split, last_err, off, wait,
                 failures, policy.max_failures)
        sleep(wait)

    check_complete(st)
    _atomic_json(final_path, st)
    partial_path.unlink(missing_ok=True)
    if status:
        status.update(split, state="complete", rows=len(st["rows"]), total=st["num_rows_total"], last_error="")
    log.info("%s %s: %d rows listed (%d classes)", dataset, split, len(st["rows"]), len(st["names"]))
    return st


def list_source(name: str, extra: dict, raw: Path, *, get, sleep=time.sleep, clock=time.monotonic,
                policy: Policy = Policy(), rnd: Callable[[], float] = random.random) -> dict[str, dict]:
    """All splits of one HF alphabet source (``extra`` = the catalog's ``dataset`` / ``splits``): {split: labels}.
    Every split is attempted (a refused split does not stop the next one from making progress); the first
    :class:`ListingError` is re-raised at the end, with ``progressed`` set if any split made progress."""
    ds = extra["dataset"]
    status = StatusFile(raw / STATUS_FILE, name, ds)
    out: dict[str, dict] = {}
    errors: list[ListingError] = []
    for split in extra["splits"]:
        try:
            out[split] = list_split(ds, split, raw / f"labels_{split}.json", raw / f"labels_{split}.partial.json",
                                    get=get, sleep=sleep, clock=clock, policy=policy, status=status, rnd=rnd)
        except ListingError as e:
            errors.append(e)
    if errors:
        e = errors[0]
        raise ListingError(str(e) + (f" (+{len(errors) - 1} more split(s) incomplete)" if len(errors) > 1 else ""),
                           progressed=any(x.progressed for x in errors), permanent=all(x.permanent for x in errors),
                           rows=sum(x.rows for x in errors), total=e.total)
    return out


def build_hf_items(spec, raw: Path, per_class: int | None = None, *, get=None, sleep=time.sleep,
                   clock=time.monotonic, policy: Policy = Policy()) -> list:
    """Items of an HF alphabet source (``listings.hf_rows_items``): every row when neither ``per_class`` nor the
    catalog's ``per_class`` is set (the default for both sets since 25 Nov 2026), else a sample per class."""
    from app.isl.data.video_sources.listings import hf_rows_items

    if get is None:
        from app.isl.data.remote import session

        get = session().get
    labels = list_source(spec.name, spec.extra, raw, get=get, sleep=sleep, clock=clock, policy=policy)
    items = []
    for split in spec.extra["splits"]:
        lab = labels[split]
        items += hf_rows_items(spec.prefix, spec.extra["dataset"], split, [tuple(x) for x in lab["rows"]],
                               lab["names"], per_class=per_class or spec.extra.get("per_class"))
    return items


class RemoteRefused(OSError):
    """The rows API keeps refusing (``max_failures`` failed requests in a row, or a ``Retry-After`` above
    ``max_retry_after``), or refused for good (``permanent``: a 4xx other than 429). The engine stops the run: the
    item that hit it gets a ``fetch:`` row (retry_pending), the items after it are not touched."""

    def __init__(self, msg: str, *, permanent: bool = False):
        super().__init__(msg)
        self.permanent = permanent


def url_expiry(url: str) -> float | None:
    """Epoch seconds of a signed URL's ``Expires`` query parameter (the datasets-server ``cached-assets`` URLs are
    CloudFront-signed: ``?Expires=...&Signature=...&Key-Pair-Id=...``); None when absent."""
    try:
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        return float(q["Expires"][0])
    except (KeyError, IndexError, ValueError, TypeError):
        return None


class RowPages:
    """Image URLs of HF dataset rows for ``engine.Fetcher`` (kind ``hfrow``), one rows-API page (100 rows) at a time.

    * Each page is requested once and kept (LRU, ``max_pages``) until shortly before its signed URLs expire
      (``Expires`` - ``expiry_margin``; ``ttl`` when the URLs carry no expiry), so items fetched in row order cost one
      request per 100 rows, not one per item.
    * Requests are paced and retried exactly like the listing (:class:`Policy`: >= ``min_interval`` between two
      requests, exponential backoff with jitter, never shorter than ``Retry-After``). When the server keeps refusing
      it raises :class:`RemoteRefused` instead of an endless wait.
    * ``refresh`` re-requests a page whose URL was refused (expired signature), but not a page fetched less than
      ``fresh_s`` ago: a refused image of a fresh page is that image's own failure, not an expired page.

    ``get_fn`` returns the ``get`` function (lazy, so the HTTP session is created on first use)."""

    def __init__(self, get_fn: Callable[[], Callable], *, sleep=time.sleep, clock=time.monotonic, wall=time.time,
                 policy: Policy = Policy(), rnd: Callable[[], float] = random.random, max_pages: int = 8,
                 ttl: float = 1800.0, expiry_margin: float = 300.0, fresh_s: float = 120.0, config: str = "default"):
        self.get_fn, self.sleep, self.clock, self.wall, self.policy, self.rnd = get_fn, sleep, clock, wall, policy, rnd
        self.max_pages, self.ttl, self.expiry_margin, self.fresh_s, self.config = (max_pages, ttl, expiry_margin,
                                                                                   fresh_s, config)
        self.pages: OrderedDict[tuple[str, str, int], dict] = OrderedDict()
        self.last_req: float | None = None
        self.stats = {"requests": 0, "pages": 0, "retries": 0, "waited_s": 0.0}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self.pages.clear()

    def src(self, dataset: str, split: str, row: int, *, refresh: bool = False) -> str:
        """The image URL of one row. ``refresh``: the cached URL was refused; re-request its page unless fresh."""
        key = (dataset, split, row - row % PAGE)
        with self._lock:
            page = self.pages.get(key)
            now = self.clock()
            if page is None or now >= page["valid_until"] or (refresh and now - page["fetched"] >= self.fresh_s):
                page = self._fetch(*key)
                self.pages[key] = page
                while len(self.pages) > self.max_pages:
                    self.pages.popitem(last=False)
            self.pages.move_to_end(key)
            url = page["src"].get(row)
        if url is None:
            raise OSError(f"rows API: no image URL for {dataset} {split} row {row}")
        return url

    def _pace(self) -> None:
        if self.last_req is not None:
            gap = self.policy.min_interval - (self.clock() - self.last_req)
            if gap > 0:
                self.sleep(gap)
                self.stats["waited_s"] += gap

    def _fetch(self, dataset: str, split: str, off: int) -> dict:
        failures, err = 0, ""
        while True:
            self._pace()
            self.last_req = self.clock()
            self.stats["requests"] += 1
            ra = None
            try:
                r = self.get_fn()(HF_ROWS, params={"dataset": dataset, "config": self.config, "split": split,
                                                   "offset": off, "length": PAGE}, timeout=self.policy.timeout)
                code = int(r.status_code)
                if code == 200:
                    src: dict[int, str] = {}
                    for x in r.json().get("rows") or []:
                        img = (x.get("row") or {}).get("image")
                        if isinstance(img, dict) and img.get("src"):
                            src[int(x["row_idx"])] = str(img["src"])
                    if not src:
                        raise ValueError(f"page at offset {off} has no image URLs")
                    exp = [e for e in (url_expiry(u) for u in src.values()) if e is not None]
                    now = self.clock()
                    life = (min(exp) - self.wall() - self.expiry_margin) if exp else self.ttl
                    self.stats["pages"] += 1
                    return {"src": src, "fetched": now, "valid_until": now + max(0.0, min(life, self.ttl * 4))}
                err = f"HTTP {code}"
                ra = retry_after_s(getattr(r, "headers", None))
                if code != 429 and code < 500:
                    raise RemoteRefused(f"rows API {err} for {dataset} {split} offset {off} (not retryable)",
                                        permanent=True)
            except RemoteRefused:
                raise
            except Exception as e:  # noqa: BLE001 - connection errors, bad JSON, empty pages: retry
                err = f"{type(e).__name__}: {e}"[:300]
            failures += 1
            if failures >= self.policy.max_failures or (ra is not None and ra > self.policy.max_retry_after):
                why = (f"Retry-After {ra:.0f} s" if ra is not None and ra > self.policy.max_retry_after
                       else f"{failures} failed tries in a row")
                raise RemoteRefused(f"rows API refused {dataset} {split} offset {off} after {why}; last error {err}")
            wait = backoff_s(failures, ra, self.policy, self.rnd)
            self.stats["retries"] += 1
            self.stats["waited_s"] += wait
            log.info("%s %s: rows page %d: %s; waiting %.0f s (try %d of %d)", dataset, split, off, err, wait,
                     failures, self.policy.max_failures)
            self.sleep(wait)


def listing_status(raw: Path) -> dict | None:
    """The saved status of a source's listing (None if it never ran)."""
    try:
        return json.loads((raw / STATUS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
