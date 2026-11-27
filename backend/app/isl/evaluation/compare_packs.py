"""``compare-packs``: which research pack to ship (QA plan "Final model choice logic", critic 9; docs 15.44).

Candidates are work folders (``artifacts/isl/research``, a scratch run such as ``runs/fallback-csenc/research``, a
retrained run). The reference is the commercial-safe work folder. Two stages, never mixed:

1. **Selection on validation / out-of-fold data only.** For every candidate, the four gates are checked on numbers
   that no test clip touched:
   (a) CV out-of-fold top-1 on the folds other than the primary one (their test parts are not E1b's test clips);
   (b) the encoder probe (INCLUDE validation -> ISLRTC dictionary, paired on the common queries) and E2 one-shot on
   the held-out-concept queries of the validation part;
   (c) the certified worst-source strict lower bound of ``fusion.json`` (fitted on validation or out-of-fold clips);
   (d) ISL500 validation top-1 against the ISL500 baseline (same validation clips only).
   A candidate with a failed selection gate is rejected. Among the rest, one is selected by a fixed rule (most
   passed selection checks, then the CV difference, then the probe difference).
2. **One confirmation on test**, for the selected candidate only: (a) E1b on the common test clips (and CV pooled),
   (b) E2_seen and E2 one-shot (test-part queries), (c) the strict certificate holds on test for every major source,
   (d) ISL500 test top-1 (classifier alone and fused) is no worse than the baseline's. The other candidates' test
   numbers are not looked at. Every confirmation is logged in ``<out>/test_looks.jsonl``; a second look from the same
   folder is reported, because it is no longer a single confirmation.

Paired checks: both systems on the same clip ids. McNemar's exact test on the discordant clips; the paired difference
with a clip-level (Agresti-Min) interval and, with at least ``MIN_INTERVAL_GROUPS`` signer groups, a bootstrap over signer groups
(the wider interval counts). A check passes when the interval's lower end is above ``-margin`` (non-inferiority),
fails when its upper end is below ``-margin``, and is inconclusive otherwise. Without per-clip records
(``evaluate --all --per-clip``) a check falls back to an unpaired comparison of the two rates, said so.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from app.isl.evaluation.report import MIN_INTERVAL_GROUPS

SCHEMA = 1
DEFAULT_MARGINS = {"top1": 0.02, "oneshot": 0.01, "isl500": 0.01, "lower_bound": 0.0}
GATES = {"a": "word classifier on INCLUDE (E1b / CV)", "b": "dictionary retrieval (E2_seen, E2 one-shot)",
         "c": "certified worst-source strict precision", "d": "ISL500 no regression"}
# worst first: a gate is as good as its worst required check
SEVERITY = {"fail": 0, "inconclusive": 1, "not_available": 2, "pass": 3, "not_applicable": 4}
MIN_CONFIRM_ACCEPTS = 30  # gate (c) on test: fewer strict accepts of a source cannot confirm its certificate
LEDGER = "test_looks.jsonl"
COMPONENTS = ("classifier.pt", "encoder.pt", "fusion.json", "index/meta.json")


# --------------------------------------------------------------------------------------------- statistics

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: ``b`` clips only the candidate gets right, ``c`` only the reference."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def _verdict(lo: float, hi: float, margin: float) -> str:
    if lo >= -margin:
        return "pass"
    if hi < -margin:
        return "fail"
    return "inconclusive"


def group_bootstrap_diff(d: np.ndarray, groups: list[str], n_boot: int = 2000, seed: int = 0) -> list[float]:
    """95 % interval of the mean paired difference, resampling signer groups (not clips)."""
    keys = sorted(set(groups))
    idx = {k: i for i, k in enumerate(keys)}
    gi = np.array([idx[g] for g in groups])
    sums = np.bincount(gi, weights=d, minlength=len(keys))
    cnts = np.bincount(gi, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(keys), (n_boot, len(keys)))
    means = sums[pick].sum(1) / np.maximum(1.0, cnts[pick].sum(1))
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def paired_difference(cand: dict[str, int], ref: dict[str, int], groups: dict[str, str], margin: float,
                      n_boot: int = 2000, seed: int = 0, min_groups: int = MIN_INTERVAL_GROUPS) -> dict:
    """Candidate minus reference hit@1 on the clip ids both measured (strict labels)."""
    ids = sorted(set(cand) & set(ref))
    n = len(ids)
    if not n:
        return {"paired": True, "common": 0, "status": "not_available", "reason": "no common clip ids"}
    x = np.array([int(cand[i]) for i in ids], float)
    y = np.array([int(ref[i]) for i in ids], float)
    d = x - y
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    delta = (b - c) / n
    # Agresti-Min interval for a paired difference (0.5 added to each cell): never zero-width, so 40 clips that all
    # agree do not "prove" non-inferiority
    ba, ca, na = b + 0.5, c + 0.5, n + 2.0
    da = (ba - ca) / na
    se = math.sqrt(max(0.0, (ba + ca) - (ba - ca) ** 2 / na)) / na
    clip_ci = [round(da - 1.96 * se, 4), round(da + 1.96 * se, 4)]
    g = [groups.get(i) or "?" for i in ids]
    per_group: dict[str, dict] = {}
    for gg, xi, yi in zip(g, x, y):
        r = per_group.setdefault(gg, {"n": 0, "cand": 0.0, "ref": 0.0})
        r["n"] += 1
        r["cand"] += xi
        r["ref"] += yi
    for r in per_group.values():
        r["delta"] = round((r["cand"] - r["ref"]) / r["n"], 4)
        r["cand"], r["ref"] = round(r["cand"] / r["n"], 4), round(r["ref"] / r["n"], 4)
    ng = len(per_group)
    boot = group_bootstrap_diff(d, g, n_boot, seed) if ng >= min_groups else None
    lo, hi = clip_ci
    if boot is not None:
        lo, hi = min(lo, boot[0]), max(hi, boot[1])
    p = mcnemar_exact(b, c)
    return {"paired": True, "common": n, "cand": round(float(x.mean()), 4), "ref": round(float(y.mean()), 4),
            "delta": round(delta, 4), "cand_only": b, "ref_only": c, "p_mcnemar": round(p, 6),
            "clip_interval": clip_ci, "group_interval": boot,
            "group_interval_reason": None if boot is not None else f"too few signer groups ({ng})",
            "interval": [round(lo, 4), round(hi, 4)], "n_groups": ng, "per_group": per_group,
            "groups_worse": sum(1 for r in per_group.values() if r["delta"] < 0),
            "significantly_worse": bool(p < 0.05 and delta < 0), "significantly_better": bool(p < 0.05 and delta > 0),
            "margin": margin, "status": _verdict(lo, hi, margin)}


def unpaired_difference(cand: tuple[float, int], ref: tuple[float, int], margin: float, note: str = "") -> dict:
    """Fallback without per-clip records: the two rates as independent proportions (wider than a paired interval on
    the same clips; not like for like when the clip sets differ)."""
    (p1, n1), (p2, n2) = cand, ref
    if not n1 or not n2 or p1 is None or p2 is None:
        return {"paired": False, "status": "not_available", "reason": "a rate or its n is missing"}
    delta = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    return {"paired": False, "cand": round(p1, 4), "ref": round(p2, 4), "n_cand": n1, "n_ref": n2,
            "delta": round(delta, 4), "interval": [round(lo, 4), round(hi, 4)], "margin": margin,
            "status": _verdict(lo, hi, margin),
            "note": "unpaired: aggregate rates only (run evaluate --all --per-clip for a paired test)"
                    + (f"; {note}" if note else "") + ("" if n1 == n2 else f"; different n ({n1} vs {n2})")}


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(mid - half, 4), round(mid + half, 4)]


# --------------------------------------------------------------------------------------------- inputs

def _sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _same_sha(a: str | None, b: str | None) -> bool | None:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def _parse_utc(stamp) -> float | None:
    import calendar

    try:
        return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) if stamp else None
    except (TypeError, ValueError):
        return None


def include_signer_groups(splits_dir: Path | None) -> dict[str, str]:
    """INCLUDE clip id -> pseudo-signer (``splits/include_groups.json``), the group the INCLUDE adapter gives a clip
    by default: the fallback signer groups of a report without per-clip records."""
    d = _json(Path(splits_dir) / "include_groups.json") if splits_dir else None
    return {k: str(v.get("pseudo_signer") or v.get("session")) for k, v in ((d or {}).get("clips") or {}).items()
            if isinstance(v, dict) and (v.get("pseudo_signer") or v.get("session"))}


def clip_source_group(cid: str, include_groups: dict[str, str]) -> tuple[str, str | None]:
    """Source and signer group of a clip id when a report did not record them (``include/...`` -> its pseudo-signer,
    ``isl500/userNNN/...`` -> that user; anything else -> its first path part, group unknown)."""
    parts = cid.split("/")
    if parts[0] == "include":
        return "include", include_groups.get(cid)
    if parts[0] == "isl500" and len(parts) > 2:
        return "isl500", parts[1]
    return parts[0], None


class Side:
    """One work folder: its report, per-clip records, fusion.json, encoder probe and classifier meta."""

    def __init__(self, name: str, work: Path, include_groups: dict[str, str] | None = None):
        self.name, self.work = name, Path(work)
        self.include_groups = include_groups or {}
        self.notes: list[str] = []
        ev = self.work / "eval"
        self.report = _json(ev / "report.json")
        self.fusion = _json(self.work / "fusion.json")
        self.per_clip = None
        pc = _json(ev / "per_clip.json")
        if pc is not None:
            if self.report is not None and pc.get("report_created") == self.report.get("created"):
                self.per_clip = pc.get("protocols") or {}
            else:
                self.notes.append("per_clip.json belongs to another evaluate run than report.json: not used")
        if self.report is None:
            self.notes.append(f"no {ev / 'report.json'}: run evaluate --all")
        self.components = {k: _sha(self.work / k) for k in COMPONENTS}
        self.stale = self._staleness()
        self.probe = self._probe(ev / "encoder_probe.json")
        self._cls_meta: dict | None | bool = False

    def _staleness(self) -> list[str]:
        out = []
        if self.report is None:
            return out
        enc_rep = ((self.report.get("leakage_guards") or {}).get("index") or {}).get("encoder_sha256")
        if _same_sha(enc_rep, self.components.get("encoder.pt")) is False:
            out.append("report.json measured another encoder.pt")
        made = _parse_utc(self.report.get("created"))
        for k in ("classifier.pt", "encoder.pt", "fusion.json"):
            p = self.work / k
            if made is not None and p.is_file() and p.stat().st_mtime > made + 1:
                out.append(f"{k} is newer than report.json")
        self.notes += [f"STALE: {s}: re-run evaluate --all" for s in out]
        return out

    def _probe(self, f: Path) -> dict | None:
        p = _json(f)
        if p is None:
            return None
        if _same_sha(p.get("encoder_sha256"), self.components.get("encoder.pt")) is False:
            self.notes.append("encoder_probe.json probed another encoder.pt: not used")
            return None
        return p

    @property
    def cls_meta(self) -> dict | None:
        if self._cls_meta is False:
            self._cls_meta = None
            f = self.work / "classifier.pt"
            if f.is_file():
                try:
                    import torch

                    self._cls_meta = torch.load(f, map_location="cpu", weights_only=False).get("meta") or {}
                except Exception as e:  # noqa: BLE001 - an unreadable checkpoint is a note, not a crash
                    self.notes.append(f"classifier.pt meta unreadable: {type(e).__name__}")
        return self._cls_meta

    def protocol(self, key: str) -> dict:
        return ((self.report or {}).get("protocols") or {}).get(key) or {}

    def records(self, key: str, split: str | None = None) -> dict | None:
        """Per-clip record (``runners.clip_record`` shape) of a protocol: from per_clip.json, else E1b / E1c from
        report.json's clip ids and hit bits (sources and groups from the clip ids)."""
        if self.per_clip is not None:
            r = self.per_clip.get(key)
            if key == "E1b_cv":
                r = (r or {}).get(split) if split else None
            if r:
                return r
        if key in ("E1b", "E1c"):
            e = self.protocol(key)
            ids, bits = e.get("clip_ids"), (e.get("per_clip") or {}).get("hit1")
            if ids and bits and len(ids) == len(bits):
                return {"clip_ids": ids, "hit1": bits, "groups": None, "sources": None, "from_report": True}
        return None

    def hits(self, key: str, *, source: str | None = None, part: str | None = None,
             split: str | None = None) -> tuple[dict[str, int], dict[str, str]] | None:
        r = self.records(key, split)
        if r is None:
            return None
        ids, bits = r["clip_ids"], r["hit1"]
        grp, src, prt = r.get("groups"), r.get("sources"), r.get("parts")
        hits, groups = {}, {}
        for j, cid in enumerate(ids):
            s0, g0 = clip_source_group(cid, self.include_groups)
            s = src[j] if src and src[j] else s0
            if source is not None and s != source:
                continue
            if part is not None and (not prt or prt[j] != part):
                continue
            hits[cid] = int(bits[j] == "1")
            groups[cid] = (grp[j] if grp and grp[j] else None) or g0 or "?"
        return hits, groups


# --------------------------------------------------------------------------------------------- checks

def _check(gate: str, name: str, stage: str, result: dict, role: str = "gate", **extra) -> dict:
    return {"gate": gate, "check": name, "stage": stage, "role": role, **extra, **result}


def _na(reason: str) -> dict:
    return {"status": "not_available", "reason": reason}


def _paired_or_unpaired(c_hits, r_hits, c_rate, r_rate, margin, opts, note="") -> dict:
    if c_hits is not None and r_hits is not None:
        groups = {**r_hits[1], **c_hits[1]}
        res = paired_difference(c_hits[0], r_hits[0], groups, margin, opts["boot"], opts["seed"])
        if res.get("common"):
            return res
    return unpaired_difference(c_rate, r_rate, margin, note)


def _cv_folds(side: Side) -> dict[str, dict]:
    """The candidate's CV folds other than the primary split: {split: {"n", "top1"}} from report.json."""
    e1 = side.protocol("E1b")
    primary = e1.get("split")
    cv = e1.get("cv") or {}
    out = {}
    for f in cv.get("per_fold") or []:
        s = str(f.get("split") or "")
        if not s or s == primary or s.startswith("E1b") or f.get("top1") is None:
            continue
        out[s] = {"n": f.get("n"), "top1": f.get("top1")}
    return out


def selection_checks(cand: Side, ref: Side, base: Side | None, m: dict, opts: dict) -> list[dict]:
    """Validation / out-of-fold checks only (no test clip of the primary split)."""
    out = []
    # (a) CV out-of-fold, folds other than the primary one
    cf, rf = _cv_folds(cand), _cv_folds(ref)
    common = sorted(set(cf) & set(rf))
    if not common:
        out.append(_check("a", "CV out-of-fold top-1 (folds other than the primary)", "validation",
                          _na("no common CV fold besides the primary split in both reports")))
    else:
        ch, rh, groups, paired = {}, {}, {}, True
        for s in common:
            a, b = cand.hits("E1b_cv", split=s), ref.hits("E1b_cv", split=s)
            if a is None or b is None:
                paired = False
                break
            ch.update(a[0])
            rh.update(b[0])
            groups.update({k: f"{s}:{v}" for k, v in {**b[1], **a[1]}.items()})
        cn = sum(cf[s]["n"] for s in common)
        rn = sum(rf[s]["n"] for s in common)
        crate = sum(cf[s]["n"] * cf[s]["top1"] for s in common) / cn
        rrate = sum(rf[s]["n"] * rf[s]["top1"] for s in common) / rn
        if paired:
            res = paired_difference(ch, rh, groups, m["top1"], opts["boot"], opts["seed"])
        else:
            res = unpaired_difference((crate, cn), (rrate, rn), m["top1"],
                                      "each side on its own classes' clips of these folds")
        out.append(_check("a", "CV out-of-fold top-1 (folds other than the primary)", "validation", res,
                          folds=common))
    # (a, info) classifier INCLUDE validation top-1: the early-stopping set, so optimistic for both
    cm, rm = cand.cls_meta or {}, ref.cls_meta or {}
    ci = (cm.get("val_by_source") or {}).get("include")
    ri = (rm.get("val_by_source") or {}).get("include")
    if ci and ri:
        res = unpaired_difference((ci["top1"], ci["n"]), (ri["top1"], ri["n"]), m["top1"],
                                  "the classifiers were early-stopped on these clips")
        out.append(_check("a", "classifier INCLUDE validation top-1", "validation", res, role="info"))
    # (b) encoder probe: INCLUDE validation -> ISLRTC dictionary, query by query
    if cand.probe and ref.probe and cand.probe.get("hits") and ref.probe.get("hits"):
        def inc(p):
            h = {k.split("|", 1)[1]: int(v) for k, v in p["hits"].items() if k.split("|", 1)[0] == "include"}
            return h, {k: clip_source_group(k, cand.include_groups)[1] or "?" for k in h}

        c_, r_ = inc(cand.probe), inc(ref.probe)
        res = paired_difference(c_[0], r_[0], {**r_[1], **c_[1]}, m["top1"], opts["boot"], opts["seed"])
        diff = [k for k in ("profile", "split", "per_signer", "data_seed") if cand.probe.get(k) != ref.probe.get(k)]
        if diff:
            res["note"] = "the probes differ in " + ", ".join(diff) + " (paired on the common INCLUDE queries)"
        out.append(_check("b", "encoder probe INCLUDE val -> ISLRTC r1", "validation", res))
    else:
        out.append(_check("b", "encoder probe INCLUDE val -> ISLRTC r1", "validation",
                          _na("no current encoder_probe.json with per-query hits in both work folders (run "
                              "encoder-probe --out <work>/eval/encoder_probe.json)")))
    # (b) E2 one-shot: held-out-concept queries of the validation part
    c_, r_ = cand.hits("E2", part="val"), ref.hits("E2", part="val")
    if c_ is not None and r_ is not None:
        res = paired_difference(c_[0], r_[0], {**r_[1], **c_[1]}, m["oneshot"], opts["boot"], opts["seed"])
        out.append(_check("b", "E2 one-shot top-1, validation-part queries", "validation", res))
    else:
        out.append(_check("b", "E2 one-shot top-1, validation-part queries", "validation",
                          _na("no E2 per-clip records with parts (evaluate --all --per-clip) in both")))
    # (c) certified worst-source strict lower bound (fitted on validation / out-of-fold clips)
    out.append(_check("c", "strict worst-source lower bound (fusion.json)", "validation", lower_bound_check(cand, ref, m)))
    # (d) ISL500 validation top-1 against the baseline (same validation clips only)
    out.append(_check("d", "ISL500 validation top-1 (classifier)", "validation", isl500_val_check(cand, base, m)))
    return out


def _strict_lb(side: Side) -> tuple[float | None, str | None, dict]:
    a = ((side.fusion or {}).get("achieved") or {}).get("strict") or {}
    return a.get("precision_lower_bound"), a.get("worst_source"), a


def lower_bound_check(cand: Side, ref: Side, m: dict) -> dict:
    c_lb, c_ws, ca = _strict_lb(cand)
    r_lb, r_ws, _ = _strict_lb(ref)
    if c_lb is None or r_lb is None:
        return _na("a fusion.json without a strict precision_lower_bound (refit with the 15.25 code)")
    cal = lambda s: ((s.fusion or {}).get("certification") or {}).get("calibration_clips")  # noqa: E731
    return {"cand": c_lb, "ref": r_lb, "delta": round(c_lb - r_lb, 4), "cand_worst_source": c_ws,
            "ref_worst_source": r_ws, "cand_rule": ca.get("rule"), "margin": m["lower_bound"],
            "calibration_clips": {"cand": cal(cand), "ref": cal(ref)},
            "status": "pass" if c_lb >= r_lb - m["lower_bound"] else "fail",
            "note": "each number is already a 95 % lower bound (worst major source, design-effect adjusted)"}


def isl500_val_check(cand: Side, base: Side | None, m: dict) -> dict:
    if base is None:
        return {"status": "not_applicable", "reason": "no --isl500-baseline given"}
    c = ((cand.cls_meta or {}).get("val_by_source") or {}).get("isl500")
    b = ((base.cls_meta or {}).get("val_by_source") or {}).get("isl500")
    if not c:
        return {"status": "not_applicable", "reason": "the candidate's classifier has no ISL500 validation clips"}
    if not b:
        return _na("the baseline classifier records no ISL500 validation top-1 (val_by_source)")
    if c["n"] != b["n"]:
        return _na(f"different ISL500 validation sets (n {c['n']} vs {b['n']}; signer split v2 moves 2 signers to "
                   "validation): only the test check can compare them")
    return unpaired_difference((c["top1"], c["n"]), (b["top1"], b["n"]), m["isl500"],
                               "the classifiers were early-stopped on these clips")


def confirmation_checks(cand: Side, ref: Side, base: Side | None, m: dict, opts: dict) -> list[dict]:
    """The single confirmation on test of the selected candidate."""
    out = []
    # (a) E1b on the common test clips, strict labels
    c_, r_ = cand.hits("E1b"), ref.hits("E1b")
    ce, re_ = cand.protocol("E1b"), ref.protocol("E1b")
    if ce.get("split") and re_.get("split") and ce["split"] != re_["split"]:
        out.append(_check("a", "E1b top-1 on the common test clips", "test",
                          _na(f"different primary splits ({ce['split']} vs {re_['split']})")))
    else:
        res = _paired_or_unpaired(c_, r_, ((ce.get("metrics") or {}).get("top1"), ce.get("n")),
                                  ((re_.get("metrics") or {}).get("top1"), re_.get("n")), m["top1"], opts,
                                  "each side on its own class set")
        out.append(_check("a", "E1b top-1 on the common test clips", "test", res))
    lf = ce.get("like_for_like") or {}
    if lf:
        out.append(_check("a", "E1b like for like (evaluate --compare-to)", "test",
                          {"status": "pass", "like_for_like": lf,
                           "note": "information: scores masked to the reference's classes"}, role="info"))
    # (a) CV pooled over every fold (the primary fold included: this is the test stage)
    ccv, rcv = ce.get("cv") or {}, re_.get("cv") or {}
    folds = sorted(set(_cv_folds(cand)) & set(_cv_folds(ref)))
    ch, rh, groups, paired = {}, {}, {}, bool(c_ and r_)
    for s in folds:
        a, b = cand.hits("E1b_cv", split=s), ref.hits("E1b_cv", split=s)
        if a is None or b is None:
            paired = False
            break
        ch.update(a[0])
        rh.update(b[0])
        groups.update({k: f"{s}:{v}" for k, v in {**b[1], **a[1]}.items()})
    if paired and folds:
        ch.update(c_[0])
        rh.update(r_[0])
        groups.update({k: f"primary:{v}" for k, v in {**r_[1], **c_[1]}.items()})
        res = paired_difference(ch, rh, groups, m["top1"], opts["boot"], opts["seed"])
    else:
        res = unpaired_difference((ccv.get("pooled_top1"), ccv.get("pooled_n")),
                                  (rcv.get("pooled_top1"), rcv.get("pooled_n")), m["top1"],
                                  "each side on its own classes' clips")
    out.append(_check("a", "CV pooled top-1 (every fold)", "test", res))
    # (b) E2_seen and E2 one-shot (test-part queries)
    c_, r_ = cand.hits("E2_seen"), ref.hits("E2_seen")
    cs, rs = cand.protocol("E2_seen"), ref.protocol("E2_seen")
    res = _paired_or_unpaired(c_, r_, ((cs.get("metrics") or {}).get("top1"), cs.get("n")),
                              ((rs.get("metrics") or {}).get("top1"), rs.get("n")), m["top1"], opts)
    out.append(_check("b", "E2_seen top-1 (INCLUDE test -> ISLRTC rows)", "test", res))
    c_, r_ = cand.hits("E2", part="test"), ref.hits("E2", part="test")
    c2, r2 = cand.protocol("E2"), ref.protocol("E2")
    res = _paired_or_unpaired(c_, r_, ((c2.get("metrics") or {}).get("top1"), c2.get("n")),
                              ((r2.get("metrics") or {}).get("top1"), r2.get("n")), m["oneshot"], opts,
                              "validation and test queries pooled (no parts recorded)")
    out.append(_check("b", "E2 one-shot top-1, test-part queries", "test", res))
    # (c) the strict certificate holds on test, per major source
    out.append(_check("c", "strict certificate holds on test (every major source)", "test", certificate_check(cand, ref)))
    # (d) ISL500 test top-1 against the baseline: classifier alone (E1c) and fused (E3)
    for key, label in (("E1c", "classifier alone"), ("E3", "fused")):
        name = f"ISL500 test top-1 ({label})"
        if base is None:
            out.append(_check("d", name, "test", {"status": "not_applicable", "reason": "no --isl500-baseline given"}))
            continue
        c_, b_ = cand.hits(key, source="isl500"), base.hits(key, source="isl500")
        if c_ is not None and not c_[0]:
            out.append(_check("d", name, "test", {"status": "not_applicable",
                                                  "reason": "the candidate has no ISL500 test clips"}))
            continue
        crate = _isl500_rate(cand, key)
        brate = _isl500_rate(base, key)
        if c_ is None and crate[1] is None:
            out.append(_check("d", name, "test", {"status": "not_applicable",
                                                  "reason": "the candidate reports no ISL500 test clips"}))
            continue
        out.append(_check("d", name, "test", _paired_or_unpaired(c_, b_, crate, brate, m["isl500"], opts)))
    return out


def _isl500_rate(side: Side, key: str) -> tuple[float | None, int | None]:
    e = side.protocol(key)
    for blk in (e.get("top1_by_source"), (e.get("word_labels") or {}).get("top1_by_source")):
        r = (blk or {}).get("isl500")
        if r:
            return r.get("top1_strict", r.get("top1")), r.get("n")
    return None, None


def certificate_check(cand: Side, ref: Side) -> dict:
    """For every major source of the candidate's strict certification: the test precision of the strict threshold
    rule (E3) is at least the validation lower bound it was certified with."""
    _, _, ach = _strict_lb(cand)
    by_val = ach.get("by_source") or {}
    major = [s for s, r in by_val.items() if r.get("major")]
    by_test = (((cand.protocol("E3").get("policies") or {}).get("strict") or {}).get("by_source")) or {}
    if not major:
        return _na("fusion.json records no major source for strict")
    if not by_test:
        return _na("E3 has no per-source strict block (re-run evaluate --all with the 15.25 code)")
    rows, worst = {}, "pass"
    for s in major:
        t = by_test.get(s) or {}
        lb = by_val[s].get("lower_bound")
        k_n = t.get("threshold_n_accepted")
        prec = t.get("threshold_precision")
        if not k_n or prec is None:
            st = "not_available"
        elif k_n < MIN_CONFIRM_ACCEPTS:
            st = "inconclusive"
        else:
            st = "pass" if prec >= (lb or 0.0) else "fail"
        rows[s] = {"val_lower_bound": lb, "test_precision": prec, "test_accepts": k_n,
                   "test_interval": wilson(int(round((prec or 0) * (k_n or 0))), k_n or 0), "status": st}
        worst = st if SEVERITY[st] < SEVERITY[worst] else worst
    rs = (((ref.protocol("E3").get("policies") or {}).get("strict") or {}).get("by_source")) or {}
    return {"status": worst, "by_source": rows, "rule": ach.get("rule"),
            "ref_test_precision": {s: (rs.get(s) or {}).get("threshold_precision") for s in major},
            "note": "test precision of the strict threshold rule >= the validation lower bound it was certified "
                    f"with; fewer than {MIN_CONFIRM_ACCEPTS} test accepts: inconclusive"}


def gate_status(checks: list[dict]) -> dict[str, str]:
    out = {}
    for g in GATES:
        sts = [c["status"] for c in checks if c["gate"] == g and c["role"] == "gate"]
        out[g] = min(sts, key=SEVERITY.get) if sts else "not_available"
    return out


# --------------------------------------------------------------------------------------------- driver

def compare(reference: Path, candidates: dict[str, Path], *, baseline: Path | None = None, out: Path,
            splits_dir: Path | None = None, margins: dict | None = None, select_only: bool = False,
            boot: int = 2000, seed: int = 0) -> dict:
    """Selection on validation for every candidate, then one confirmation on test for the selected one. Writes
    ``<out>/compare.json`` and ``compare.md``; returns the document."""
    m = {**DEFAULT_MARGINS, **(margins or {})}
    opts = {"boot": boot, "seed": seed}
    groups = include_signer_groups(splits_dir)
    ref = Side("reference", reference, groups)
    base = Side("isl500-baseline", baseline, groups) if baseline else None
    sides = {n: Side(n, p, groups) for n, p in candidates.items()}
    doc: dict = {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "reference": {"work": str(ref.work), "notes": ref.notes, "components": ref.components},
                 "isl500_baseline": None if base is None else {"work": str(base.work), "notes": base.notes},
                 "margins": m, "bootstrap": opts, "gates": GATES,
                 "rule": {"selection": "validation / out-of-fold checks only; a failed selection gate rejects the "
                                       "candidate; among the others: most passed selection checks, then the CV "
                                       "difference, then the probe difference",
                          "confirmation": "the selected candidate only, once on test; ship when every test gate is "
                                          "pass (or not applicable)",
                          "check": "pass: interval lower end >= -margin; fail: upper end < -margin; else "
                                   "inconclusive. Paired: McNemar exact + clip-level interval, and a signer-group "
                                   "bootstrap with >= %d groups (the wider interval counts)" % MIN_INTERVAL_GROUPS},
                 "candidates": {}}
    ranked = []
    for name, s in sides.items():
        sel = selection_checks(s, ref, base, m, opts)
        gs = gate_status(sel)
        eligible = "fail" not in gs.values()
        entry = {"work": str(s.work), "notes": s.notes, "stale": s.stale, "components": s.components,
                 "selection": {"checks": sel, "gates": gs, "eligible": eligible}}
        doc["candidates"][name] = entry
        if eligible:
            def delta(g, ch=sel):
                v = next((c.get("delta") for c in ch if c["gate"] == g and c["role"] == "gate"
                          and c.get("delta") is not None), None)
                return v if v is not None else -1.0
            npass = sum(1 for c in sel if c["role"] == "gate" and c["status"] == "pass")
            ranked.append(((npass, delta("a"), delta("b")), name))
    ranked.sort(key=lambda t: t[0], reverse=True)
    selected = ranked[0][1] if ranked else None
    doc["selected"] = selected
    doc["ranking"] = [{"candidate": n, "key": list(k)} for k, n in ranked]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, e in doc["candidates"].items():
        if name != selected:
            e["verdict"] = ("rejected on validation: " + ", ".join(f"gate ({g}) {st}" for g, st in
                                                                     e["selection"]["gates"].items() if st == "fail")
                            if not e["selection"]["eligible"] else "eligible on validation, not selected; test not "
                                                                    "looked at")
    if selected is None:
        doc["verdict"] = ("no candidate passes selection: ship the research pack with commercial-safe's encoder and "
                          "classifier (plus research index rows for extra vocabulary), per the QA plan")
    elif select_only:
        doc["candidates"][selected]["verdict"] = "selected on validation; test not looked at (--select-only)"
        doc["verdict"] = f"{selected} selected on validation; confirm on test before shipping"
    else:
        s = sides[selected]
        led = out / LEDGER
        before = [json.loads(x) for x in led.read_text().splitlines() if x.strip()] if led.exists() else []
        tc = confirmation_checks(s, ref, base, m, opts)
        tg = gate_status(tc)
        ship = all(v in ("pass", "not_applicable") for v in tg.values()) and not s.stale and not ref.stale
        with led.open("a") as f:
            f.write(json.dumps({"time": doc["created"], "candidate": selected, "work": str(s.work),
                                "components": s.components, "gates": tg}) + "\n")
        e = doc["candidates"][selected]
        e["confirmation"] = {"checks": tc, "gates": tg, "ship": ship, "test_looks_before": len(before),
                             "warning": (f"the test set was looked at {len(before)} time(s) before from {out}: this "
                                         "is no longer a single confirmation" if before else None)}
        failed = [f"({g}) {st}" for g, st in tg.items() if st not in ("pass", "not_applicable")]
        e["verdict"] = ("selected on validation and confirmed on test: ship" if ship else
                        "selected on validation, NOT confirmed on test: do not ship ("
                        + (", ".join(failed) if failed else "stale inputs") + ")")
        doc["verdict"] = f"{selected}: {e['verdict']}"
    (out / "compare.json").write_text(json.dumps(doc, indent=1, default=str))
    (out / "compare.md").write_text(markdown(doc))
    return doc


# --------------------------------------------------------------------------------------------- markdown

def _f(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _rows(checks: list[dict]) -> list[str]:
    out = ["| gate | check | role | candidate | reference | difference | 95 % interval | McNemar p | n | status |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in checks:
        iv = c.get("interval")
        n = c.get("common") or c.get("n_cand")
        out.append(f"| ({c['gate']}) | {c['check']} | {c['role']} | {_f(c.get('cand'))} | {_f(c.get('ref'))} | "
                   f"{_f(c.get('delta'))} | {'-' if not iv else f'{iv[0]:.4f} to {iv[1]:.4f}'} | "
                   f"{_f(c.get('p_mcnemar'))} | {_f(n)} | **{c['status']}** |")
    notes = [f"* ({c['gate']}) {c['check']}: {c.get('reason') or c.get('note') or c.get('group_interval_reason')}"
             for c in checks if c.get("reason") or c.get("note") or c.get("group_interval_reason")]
    return out + ([""] + notes if notes else [])


def markdown(doc: dict) -> str:
    m = doc["margins"]
    lines = ["# ISL pack selection (compare-packs)", "",
             f"Created {doc['created']}. Reference: `{doc['reference']['work']}`. "
             + (f"ISL500 baseline: `{doc['isl500_baseline']['work']}`." if doc.get("isl500_baseline") else
                "No ISL500 baseline: gate (d) not applicable."), "",
             f"**Verdict: {doc['verdict']}**", "",
             f"Margins (non-inferiority): top-1 {m['top1']}, one-shot {m['oneshot']}, ISL500 {m['isl500']}, "
             f"lower bound {m['lower_bound']}. {doc['rule']['check']}.", "",
             "Selection uses validation and out-of-fold numbers only; only the selected candidate is looked at on "
             "test, once.", ""]
    for g, t in doc["gates"].items():
        lines.append(f"* gate ({g}): {t}")
    for n in doc["reference"].get("notes") or []:
        lines.append(f"* reference: {n}")
    for name, e in doc["candidates"].items():
        lines += ["", f"## {name}", "", f"`{e['work']}`", "", f"**{e.get('verdict')}**", ""]
        lines += [f"* {n}" for n in e.get("notes") or []]
        lines += ["", "### Selection (validation / out-of-fold)", ""] + _rows(e["selection"]["checks"])
        lines += ["", "Gates: " + ", ".join(f"({g}) {st}" for g, st in e["selection"]["gates"].items())]
        cf = e.get("confirmation")
        if cf:
            lines += ["", "### Confirmation (test, once)", ""]
            if cf.get("warning"):
                lines += [f"**Warning: {cf['warning']}**", ""]
            lines += _rows(cf["checks"])
            lines += ["", "Gates: " + ", ".join(f"({g}) {st}" for g, st in cf["gates"].items())
                      + f". Ship: **{cf['ship']}**"]
    return "\n".join(lines) + "\n"


__all__ = ["compare", "paired_difference", "unpaired_difference", "mcnemar_exact", "group_bootstrap_diff",
           "selection_checks", "confirmation_checks", "certificate_check", "gate_status", "Side", "DEFAULT_MARGINS",
           "include_signer_groups", "clip_source_group", "LEDGER"]
