"""Trim evaluation of exactly the trimmer a run uses, and the check that a trim-eval report describes it (E8).

``cli trim-eval`` (the phase-2 driver's ``trim_eval`` step) runs two checks on the trimmer that ``cli trim``
applies: stage 1 (:func:`app.isl.data.trim.find_first_sign`) plus, when the learned end model
(``artifacts/isl/trim_boundary.pt``, :mod:`app.isl.data.trim_learned`) is used, its ``refine_cap`` on every clip
that is not a phrase (phrases keep the full clip) - the same calls as ``cmd_trim``. The report records the
``trim_version`` it evaluated (``trim-2`` or ``trim-2+trim-learned-1``), the model file (path, SHA-256, time), the
trim sidecar's versions and when it was written.

:func:`check_report` is what E8 uses: the report's ``trim_version`` must equal the single ``trim_version`` of the
run's trim sidecar (and the model file must be unchanged), otherwise E8 says ``mismatch`` with the reason instead
of showing numbers that describe another trimmer.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 2
LEGACY_STAGE2 = "boundary-stage2"  # the old ``--learned`` model (artifacts/isl/boundary.pt): not used by any run


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_info(path: Path | None) -> dict | None:
    """Path, SHA-256 and modification time (UTC) of a model file."""
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "mtime_utc": _utc(p.stat().st_mtime)}


def trim_one(raw, stop: str | None, kind: str | None, text: str, cfg, model=None):
    """The run's trimmer on one sequence (mirrors ``cli trim``): stage 1, then the learned end for non-phrases;
    phrases keep the full clip from the first activity."""
    from app.isl.data.trim import METHOD_CONFIDENCE, TrimResult, find_first_sign, words_for

    nw = words_for(kind, text)
    res = find_first_sign(raw, stop, cfg, n_words=nw)
    if kind == "phrase":
        if res.method != "empty":
            last = len(raw)
            res = TrimResult(res.start, last, res.t_start_ms, float(raw.t_ms[last - 1]), "full",
                             METHOD_CONFIDENCE["full"], [], ["phrase: full clip"])
        return res
    if model is not None:
        from app.isl.data.trim_learned import refine_cap

        res = refine_cap(raw, res, model, nw, cfg)
    return res


def _learned_end(res) -> bool:
    return any("trim-learned" in str(x) for x in (res.notes or []))


def run(paths, *, n: int = 300, seed: int = 0, model=None, model_path: Path | None = None, version: str,
        legacy_model=None, records: dict | None = None, log=print) -> dict:
    """Synthetic end-error check + INCLUDE active-frame check of the trimmer ``version`` (``model`` = the learned
    end model or None; ``legacy_model`` = the old stage-2 boundary model, only for ``--learned``)."""
    import numpy as np

    from app.isl.data.manifest import read_manifest
    from app.isl.data.sources import IncludeAdapter
    from app.isl.data.trim import TrimConfig, activity_signal
    from app.isl.data.trim_synth import align_concat
    from app.isl.vocab.normalize import parse_islrtc_name
    from app.landmarks.raw import RawSequence

    cfg = TrimConfig()
    rng = random.Random(seed)
    recs = records or {}
    side = read_manifest(paths.trim_sidecar, only_ok=False) if paths.trim_sidecar.exists() else []
    trims = {t["id"]: t for t in side if t["id"] in recs}
    clean = [t for t in trims.values() if t["method"] == "source_end"]
    tails = [t for t in trims.values() if (recs[t["id"]].get("stop") == "max_s") and t["method"] in ("dip", "rest", "cap")]
    rng.shuffle(clean)
    errs, methods, within, n_learned = [], Counter(), 0, 0
    for t in clean[: min(n, len(clean))]:
        a_full = RawSequence.load(paths.resolve(recs[t["id"]]["path"]))
        a = a_full.slice(0, t["end"])
        bt = rng.choice(tails) if tails else None
        if bt is None:
            break
        b_full = RawSequence.load(paths.resolve(recs[bt["id"]]["path"]))
        L = int(rng.uniform(2.0, 6.0) * 12)
        b = b_full.slice(bt["end"], min(len(b_full), bt["end"] + L))
        if len(b) < 6:
            continue
        x = align_concat(a, b, gap_frames=rng.randint(0, 3))
        pn = parse_islrtc_name(t["id"].removeprefix("islrtc/"))
        res = trim_one(x, "max_s", pn.kind, pn.text, cfg, model)
        if legacy_model is not None:
            from app.isl.data.boundary import refine

            res = refine(x, res, legacy_model, cfg)
        n_learned += _learned_end(res)
        err = float(x.t_ms[min(len(x), res.end) - 1] - float(a.t_ms[-1]))
        errs.append(err)
        methods[res.method] += 1
        within += abs(err) <= 300
    e = np.array(errs, dtype=float)
    out: dict = {
        "schema": SCHEMA, "trim_version": version, "learned": model is not None or legacy_model is not None,
        "learned_model": file_info(model_path),
        "synthetic": {
            "description": "clean ISLRTC clips (bounded source_end trims) + 2-6 s explanation-like tail of another "
                           "clip (after its trimmed end), aligned to the same body frame; the run's trimmer "
                           f"({version}) run as if stop=max_s",
            "n": len(errs), "within_300ms": round(within / max(1, len(errs)), 3),
            "median_abs_err_ms": round(float(np.median(np.abs(e))), 1) if len(e) else None,
            "late_frac": round(float(np.mean(e > 300)), 3) if len(e) else None,
            "early_frac": round(float(np.mean(e < -300)), 3) if len(e) else None,
            "learned_ends": n_learned, "methods": dict(methods)}}
    # INCLUDE: one sign per clip; the trimmer (bounded) should keep >= 90% of the active frames
    keep = []
    for c in list(IncludeAdapter(paths).clips())[:n]:
        try:
            raw = c.load()
            act = ~activity_signal(raw, cfg)["dip"]
        except Exception:  # noqa: BLE001 - malformed clips are skipped, as before
            continue
        if act.sum() == 0:
            continue
        res = trim_one(raw, "eof", "word", "", cfg, model)
        keep.append(float(act[res.start: res.end].sum() / act.sum()))
    out["include_active_kept"] = {"n": len(keep), "mean": round(float(np.mean(keep)), 3) if keep else None,
                                  "frac_ge_90pct": round(float(np.mean(np.array(keep) >= 0.9)), 3) if keep else None}
    out["methods_all"] = dict(Counter(t["method"] for t in trims.values()))
    out["sidecar"] = {"path": str(paths.trim_sidecar), "records": len(side),
                      "versions": dict(Counter(t.get("trim_version") for t in side)),
                      "mtime_utc": _utc(paths.trim_sidecar.stat().st_mtime) if paths.trim_sidecar.exists() else None}
    out["written_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def check_report(path: Path, run_versions: dict) -> dict:
    """Does the trim-eval report at ``path`` describe the trimmer of the run whose trim sidecar has
    ``run_versions`` ({trim_version: clips})? Returns {status: ok|mismatch|missing, reason, file, written_at,
    file_mtime_utc, trim_version, run_trim_versions, learned_model} plus ``doc`` (the report, only when ok)."""
    path = Path(path)
    run_versions = {str(k): v for k, v in (run_versions or {}).items()}
    out: dict = {"file": str(path), "run_trim_versions": run_versions, "status": "ok", "reason": None}
    if not path.exists():
        out.update(status="missing", reason=f"no trim-eval report at {path} (the driver's trim_eval step writes it: "
                                            "python -m app.isl.cli trim-eval --out <work>/trim_eval.json)")
        return out
    out["file_mtime_utc"] = _utc(path.stat().st_mtime)
    try:
        doc = json.loads(path.read_text())
    except ValueError as e:
        out.update(status="mismatch", reason=f"unreadable trim-eval report: {e}")
        return out
    ver = doc.get("trim_version")
    out.update(written_at=doc.get("written_at"), trim_version=ver, learned_model=doc.get("learned_model"))
    reasons = []
    if not ver:
        reasons.append("the report records no trim_version (written by an older trim-eval; learned="
                       f"{doc.get('learned')}), so it cannot be matched to the run's trims")
    if not run_versions:
        reasons.append("the run has no trim sidecar to compare with")
    elif len(run_versions) > 1:
        reasons.append(f"the run's trim sidecar mixes trim versions {run_versions}: re-run the trim step")
    elif ver and ver not in run_versions:
        reasons.append(f"trim-eval evaluated {ver} but the run's trims are {next(iter(run_versions))}")
    lm = doc.get("learned_model") or {}
    if lm.get("path") and lm.get("sha256"):
        now = file_info(Path(lm["path"]))
        if now is None:
            reasons.append(f"the learned model {lm['path']} evaluated by trim-eval no longer exists")
        elif now["sha256"] != lm["sha256"]:
            reasons.append(f"the learned model {lm['path']} changed after trim-eval (now {now['mtime_utc']})")
    if reasons:
        out.update(status="mismatch", reason="; ".join(reasons))
    else:
        out["doc"] = doc
    return out
