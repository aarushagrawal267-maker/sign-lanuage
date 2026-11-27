"""Variant forms instead of forced merges (QA plan 1.3, critics 7 and 13, docs/ISL_DESIGN.md 15.36).

ISL500 and INCLUDE sometimes sign one word in two different ways ("hot", "short", "tight", ...). The linker gives both
the same concept id, so the encoder is pulled to put two different signs together and the classifier gets one mode
for two forms. The audit finds such words and proposes a **variant concept** for the ISL500 clips:
``isl.<word>~isl500`` with ``alias_of = isl.<word>``. The output word is unchanged (the alias group is summed in the
fused decision, docs 15.34), but the two forms are trained apart.

How a word is chosen (all of it on ISL500 **train** clips and INCLUDE **train** clips; no validation or test clip):

1. **Classifier agreement.** The commercial-safe classifier (trained on INCLUDE + ISLRTC, never on ISL500) scores
   every ISL500 train clip of a word it knows. A word with at least ``min_clips`` clips whose top-5 agreement (the
   word or a twin of its alias group in the top 5) is below ``max_agreement`` is a candidate.
2. **Dictionary cross-check (critic 13), calibrated on signer shift.** Low agreement can be signer shift (new
   signers, same sign), not another sign form. Relinking then takes away supervision INCLUDE needs. So with the
   commercial-safe encoder the ISL500 centroid of the word retrieves against the ISLRTC dictionary rows. The INCLUDE
   centroid is NOT the yardstick: that encoder was trained on those INCLUDE clips, so its rank is in-sample (almost
   always 1); it is only a sanity check that the row exists and is reachable. The yardstick is the ISL500 rank of
   the words the classifier agrees on (same form, new signers, same encoder): ``different_form`` only when the
   word's ISL500 rank is above the ``dict_calib_percentile`` (95th) percentile of that spread (and above
   ``dict_rank_floor``). The rule's false-alarm rate on the agreeing words (leave-one-out) is in the summary; above
   ``dict_max_false_alarm``, or with fewer than ``dict_calib_min_words`` agreeing words, the rule is marked not
   usable and the link step refuses the file unless told to accept it.
3. **Reverse direction (optional).** A classifier trained on ISL500 scores INCLUDE clips of the word. Agreement at or
   above ``reverse_min_agreement`` means the forms are compatible (``no_difference``: never relinked).

Decision: relink only on evidence of a different form: dictionary ``different_form`` (and the reverse direction, if
run, does not say ``no_difference``), or dictionary ``inconclusive`` with the reverse direction ``different_form``.
Every word's evidence is written next to the decision.

Scope (critic 7): the audit is a proposal written to the profile's work folder (``<work>/variant_audit.json``),
never to the shared ``data/isl/derived``. A person approves it by copying it to ``<work>/variant_links.json`` (or
names it with ``link-sources --variant-links FILE``). The link step reads only that, freezes the relinks in
``link_settings.json`` with the file's SHA-256, and every later step uses the frozen copy. The file records the
ISL500 train signers it used; the link step refuses it when one of them is no longer in train (a split change).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

VARIANT_LINKS = "variant_links.json"  # the approved file the link step reads (auto)
VARIANT_AUDIT = "variant_audit.json"  # what variant-audit writes: a proposal, never read by the link step itself
VARIANT_KIND = "isl-variant-links"
AUDIT_VERSION = "isl-variant-audit-2"  # 2: dictionary rule calibrated on agreeing words (1: in-sample INCLUDE rank)
VARIANT_SOURCE = "isl500"  # the only source audited (the INCLUDE-trained reference has never seen it)
VARIANT_SUFFIX = "~isl500"

DIFFERENT, NO_DIFFERENCE, INCONCLUSIVE = "different_form", "no_difference", "inconclusive"


def variant_concept_id(root: str, source: str = VARIANT_SOURCE) -> str:
    return f"{root}~{source}"


@dataclass
class AuditParams:
    min_clips: int = 20  # ISL500 train clips a word needs to be judged
    max_agreement: float = 0.3  # top-k agreement below this: a candidate
    top_k: int = 5
    dict_rank_floor: int = 10  # ISL500 within this dictionary rank: no difference
    dict_calib_percentile: float = 95.0  # ISL500 rank above this percentile of the agreeing words': another form
    dict_calib_min_words: int = 20  # agreeing words with a dictionary row needed to calibrate the rule
    dict_max_false_alarm: float = 0.10  # leave-one-out false-alarm rate on agreeing words above this: not usable
    dict_include_max_rank: int = 20  # sanity check: INCLUDE must reach the word's dictionary row within this rank
    reverse_min_agreement: float = 0.3  # reverse direction at or above this: the forms are compatible
    max_per_class: int = 0  # 0 = every train clip of a word; else a seeded sample (shorter runs)
    seed: int = 0


# --------------------------------------------------------------------------------------------- pure helpers

def group_of(cid: str, alias_of: Mapping[str, str] | None) -> set[str]:
    """``cid`` and every member of its alias group."""
    from app.isl.vocab.aliases import alias_root

    al = dict(alias_of or {})
    r = alias_root(cid, al)
    return {cid, r} | {a for a, x in al.items() if x == r}


def topk_agreement(P: np.ndarray, class_names: list[str], targets: list[set[str]], k: int = 5) -> tuple[np.ndarray,
                                                                                                         np.ndarray]:
    """Per clip: is a member of its target set the top-1 / in the top-k of ``P`` (n, classes)?"""
    P = np.asarray(P)
    if not len(P):
        return np.zeros(0, bool), np.zeros(0, bool)
    order = np.argsort(-P, axis=1, kind="stable")[:, :k]
    names = np.asarray(class_names, dtype=object)
    top = names[order]
    h1 = np.array([top[i, 0] in t for i, t in enumerate(targets)], bool)
    hk = np.array([bool(set(top[i]) & t) for i, t in enumerate(targets)], bool)
    return h1, hk


class Dictionary:
    """Dictionary rows (ISLRTC) embedded by the reference encoder: concept-level retrieval ranks."""

    def __init__(self, emb: np.ndarray, concepts: list[str]):
        emb = np.asarray(emb, np.float32)
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        self.concepts = sorted(set(concepts))
        pos = {c: i for i, c in enumerate(self.concepts)}
        rc = np.asarray([pos[c] for c in concepts], np.int64)
        order = np.argsort(rc, kind="stable")
        self.emb = emb[order]
        rc = rc[order]
        self._starts = np.flatnonzero(np.r_[True, rc[1:] != rc[:-1]]) if len(rc) else np.zeros(0, np.int64)
        self.pos = pos

    def __len__(self) -> int:
        return len(self.concepts)

    def concept_scores(self, q: np.ndarray) -> np.ndarray:
        s = self.emb @ np.asarray(q, np.float32)
        return np.maximum.reduceat(s, self._starts) if len(s) else s

    def rank(self, q: np.ndarray, targets: set[str]) -> tuple[int | None, list[str], float | None]:
        """(1-based rank of the best target concept among every dictionary concept, top-3 concepts, best target
        score). None when no target has a dictionary row."""
        idx = [self.pos[t] for t in targets if t in self.pos]
        cs = self.concept_scores(q)
        top3 = [self.concepts[i] for i in np.argsort(-cs, kind="stable")[:3]]
        if not idx:
            return None, top3, None
        best = float(cs[idx].max())
        others = np.ones(len(cs), bool)
        others[idx] = False
        return int((cs[others] > best).sum()) + 1, top3, best


def _centroid(E: np.ndarray | None) -> np.ndarray | None:
    if E is None or not len(E):
        return None
    c = np.asarray(E, np.float32).mean(0)
    n = float(np.linalg.norm(c))
    return c / n if n > 1e-8 else None


def dictionary_verdict(rank_isl500: int | None, rank_include: int | None, p: AuditParams,
                       threshold: float | None) -> tuple[str, str]:
    """The dictionary verdict of one word. ``threshold``: the calibrated ISL500 rank (the ``dict_calib_percentile``
    of the words the classifier agrees on, :func:`calibrate`); None = not calibrated (inconclusive)."""
    if rank_isl500 is None or rank_include is None:
        why = ("the word has no ISLRTC dictionary row" if rank_isl500 is None and rank_include is None else
               "no INCLUDE train clips of the word" if rank_include is None else "no ISL500 centroid")
        return INCONCLUSIVE, why
    if threshold is None:
        return INCONCLUSIVE, (f"the dictionary rule is not calibrated (fewer than {p.dict_calib_min_words} words the "
                              "classifier agrees on have a dictionary row)")
    bar = max(float(p.dict_rank_floor), float(threshold))
    if rank_isl500 <= bar:
        return NO_DIFFERENCE, (f"ISL500 retrieves the dictionary row at rank {rank_isl500}, within the spread of the "
                               f"words the classifier agrees on (<= {bar:g}): signer shift, not another form")
    if rank_include > p.dict_include_max_rank:
        return INCONCLUSIVE, (f"INCLUDE does not reach the dictionary row either (rank {rank_include} > "
                              f"{p.dict_include_max_rank}): the row may be another form")
    return DIFFERENT, (f"ISL500 retrieves the dictionary row only at rank {rank_isl500} (> {bar:g}, the "
                       f"{p.dict_calib_percentile:g}th percentile of the words the classifier agrees on): another form")


def calibrate(pairs: list[tuple[int, int | None]], p: AuditParams) -> dict:
    """Calibrate the dictionary rule on signer shift (review of b03eba0). ``pairs``: (ISL500 rank, INCLUDE rank) of
    the words the reference classifier agrees on (same form, new signers). The threshold is the
    ``dict_calib_percentile`` of their ISL500 ranks. The false-alarm rate is leave-one-out: how often an agreeing word
    would be called another form by the threshold of the other agreeing words. ``usable``: enough words and a rate
    at most ``dict_max_false_alarm``."""
    pairs = [(int(a), b) for a, b in pairs if a is not None]
    r = np.asarray([a for a, _ in pairs], float)
    out: dict = {"rule": "ISL500 rank above the percentile of the agreeing words' ISL500 ranks (and the floor), "
                         "INCLUDE reaching the row (sanity check)",
                 "n_words": len(pairs), "percentile": p.dict_calib_percentile, "min_words": p.dict_calib_min_words,
                 "floor": p.dict_rank_floor, "max_false_alarm": p.dict_max_false_alarm}
    if len(pairs) < max(1, p.dict_calib_min_words):
        return {**out, "threshold": None, "usable": False, "false_alarms": None,
                "reason": f"{len(pairs)} agreeing words with a dictionary row (< {p.dict_calib_min_words})"}
    thr = float(np.percentile(r, p.dict_calib_percentile))
    fa = 0
    for i, (a, b) in enumerate(pairs):
        others = np.delete(r, i)
        t = float(np.percentile(others, p.dict_calib_percentile)) if len(others) else thr
        if a > max(p.dict_rank_floor, t) and b is not None and b <= p.dict_include_max_rank:
            fa += 1
    rate = fa / len(pairs)
    q = {f"p{k}": float(np.percentile(r, k)) for k in (50, 80, 90, 95)}
    return {**out, "threshold": thr, "bar": max(float(p.dict_rank_floor), thr), "isl500_rank_quantiles": q,
            "isl500_rank_max": int(r.max()),
            "false_alarms": {"n": fa, "of": len(pairs), "rate": rate, "method": "leave-one-out"},
            "usable": rate <= p.dict_max_false_alarm,
            **({} if rate <= p.dict_max_false_alarm else
               {"reason": f"false-alarm rate {rate:.3f} on agreeing words > {p.dict_max_false_alarm}"})}


def decide(agreement: float | None, n: int, dict_v: str, rev_v: str | None, p: AuditParams) -> tuple[str, str]:
    if n < p.min_clips:
        return "keep", f"fewer than {p.min_clips} ISL500 train clips"
    if agreement is None:
        return "keep", "no clip could be scored"
    if agreement >= p.max_agreement:
        return "keep", f"the reference classifier agrees (top-{p.top_k} {agreement:.3f} >= {p.max_agreement})"
    if rev_v == NO_DIFFERENCE:
        return "keep", "reverse direction agrees: signer shift, not another form"
    if dict_v == DIFFERENT:
        return "relink", "dictionary: another form" + (" ; reverse direction disagrees too" if rev_v else "")
    if dict_v == NO_DIFFERENCE:
        return "keep", "dictionary: no difference (signer shift)"
    if rev_v == DIFFERENT:
        return "relink", "dictionary inconclusive; reverse direction: another form"
    return "keep", "no evidence of another form (dictionary inconclusive" + (
        ", no reverse check)" if rev_v is None else ")")


# --------------------------------------------------------------------------------------------- the audit

# clips -> (P (n, classes) or None, E (n, dim) unit rows or None, the clips that could be scored)
ScoreFn = Callable[[list], tuple[np.ndarray | None, np.ndarray | None, list]]


@dataclass
class AuditModels:
    class_names: list[str]  # the reference classifier's classes
    score: ScoreFn  # reference classifier + encoder
    dictionary: Dictionary | None = None
    reverse_class_names: list[str] | None = None
    reverse_score: ScoreFn | None = None  # reverse classifier (P only)
    provenance: dict | None = None
    embed: ScoreFn | None = None  # encoder only (the INCLUDE centroid needs no classifier pass); default ``score``


def _sample(clips: list, n: int, seed: str) -> list:
    if n <= 0 or len(clips) <= n:
        return list(clips)
    return sorted(random.Random(seed).sample(clips, n), key=lambda c: c.clip_id)


def run_audit(isl500_train: list, include_train: list, models: AuditModels, params: AuditParams | None = None,
              alias_of: Mapping[str, str] | None = None, reverse_clips: list | None = None,
              progress: Callable[[str], None] | None = None) -> dict:
    """The audit over ISL500 ``isl500_train`` clips (train part only; the caller selects them) against the reference
    models. ``include_train``: INCLUDE clips for the dictionary centroid (train part). ``reverse_clips``: INCLUDE
    clips the reverse classifier scores (default ``include_train``). ``alias_of``: twin groups (a twin in the top-k
    counts as agreement). Returns the variant-links document without file metadata (see :func:`write_variant_links`)."""
    p = params or AuditParams()
    say = progress or (lambda m: None)
    known = set(models.class_names)
    by: dict[str, list] = defaultdict(list)
    for c in isl500_train:
        if c.concept_id and c.source == VARIANT_SOURCE:
            by[c.concept_id].append(c)
    cand = {w: v for w, v in by.items() if group_of(w, alias_of) & known}
    not_in = sorted(set(by) - set(cand))
    inc_by: dict[str, list] = defaultdict(list)
    for c in include_train:
        if c.concept_id:
            inc_by[c.concept_id].append(c)
    rev_by: dict[str, list] = defaultdict(list)
    for c in (include_train if reverse_clips is None else reverse_clips):
        if c.concept_id:
            rev_by[c.concept_id].append(c)
    rev_known = set(models.reverse_class_names or [])
    embed = models.embed or models.score
    classes: dict[str, dict] = {}
    pending: dict[str, tuple] = {}  # word -> (agreement, n used, ISL500 rank, INCLUDE rank, reverse verdict)
    words = sorted(cand)
    for i, w in enumerate(words):
        grp = group_of(w, alias_of)
        clips = _sample(cand[w], p.max_per_class, f"{p.seed}:{w}")
        P, E, used = models.score(clips)
        ev: dict = {"n_clips": len(used), "n_clips_total": len(cand[w]),
                    "n_signers": len({c.group or c.clip_id for c in used}), "agreement_top1": None,
                    f"agreement_top{p.top_k}": None, "predicted_as": []}
        agree = None
        if used and P is not None:
            h1, hk = topk_agreement(P, models.class_names, [grp] * len(used), p.top_k)
            agree = float(hk.mean())
            ev["agreement_top1"], ev[f"agreement_top{p.top_k}"] = float(h1.mean()), agree
            top1 = Counter(models.class_names[j] for j in np.asarray(P).argmax(1))
            ev["predicted_as"] = [[k, round(v / len(used), 3)] for k, v in top1.most_common(3)]
        # dictionary ranks for every word: the words the classifier agrees on calibrate the rule (critic 13)
        r5 = ri = None
        if models.dictionary is not None:
            inc = _sample([x for g in sorted(grp) for x in inc_by.get(g, [])], p.max_per_class, f"{p.seed}:inc:{w}")
            _, Ei, used_i = embed(inc) if inc else (None, None, [])
            c5, ci = _centroid(E), _centroid(Ei)
            r5, top5_, s5 = models.dictionary.rank(c5, grp) if c5 is not None else (None, [], None)
            ri, topi, si = models.dictionary.rank(ci, grp) if ci is not None else (None, [], None)
            ev["dictionary"] = {"rows": sorted(grp & set(models.dictionary.pos)), "rank_isl500": r5,
                                "rank_include": ri, "rank_include_note": "in-sample (the encoder trained on these "
                                "INCLUDE clips): a sanity check only", "n_concepts": len(models.dictionary),
                                "n_include_clips": len(used_i), "isl500_top3": top5_, "include_top3": topi,
                                "centroid_cosine": float(c5 @ ci) if c5 is not None and ci is not None else None}
        rev_v = None
        if models.reverse_score is not None:
            rc = [x for g in sorted(grp) for x in rev_by.get(g, [])]
            rc = _sample(rc, p.max_per_class, f"{p.seed}:rev:{w}")
            if not (grp & rev_known):
                ev["reverse"] = {"run": False, "reason": "the reverse classifier has no class of this word"}
            elif not rc:
                ev["reverse"] = {"run": False, "reason": "no INCLUDE clips of this word"}
            else:
                Pr, _, used_r = models.reverse_score(rc)
                if used_r and Pr is not None:
                    _, hr = topk_agreement(Pr, models.reverse_class_names, [grp] * len(used_r), p.top_k)
                    ra = float(hr.mean())
                    rev_v = NO_DIFFERENCE if ra >= p.reverse_min_agreement else DIFFERENT
                    ev["reverse"] = {"run": True, "n_clips": len(used_r), f"agreement_top{p.top_k}": ra,
                                     "verdict": rev_v}
                else:
                    ev["reverse"] = {"run": False, "reason": "no INCLUDE clip could be scored"}
        classes[w] = ev
        pending[w] = (agree, len(used), r5, ri, rev_v)
        if (i + 1) % 20 == 0 or i + 1 == len(words):
            say(f"variant-audit {i + 1}/{len(words)} words scored")
    # calibrate the dictionary rule on the words the classifier agrees on (same form, signer shift only)
    agreeing_w = [w for w, (a, n, *_r) in pending.items() if n >= p.min_clips and a is not None
                  and a >= p.max_agreement]
    cal = calibrate([(pending[w][2], pending[w][3]) for w in agreeing_w if pending[w][2] is not None], p) \
        if models.dictionary is not None else None
    thr = cal["threshold"] if cal is not None else None
    relinks: dict[str, str] = {}
    for w in words:
        agree, n, r5, ri, rev_v = pending[w]
        ev = classes[w]
        dict_v, dict_why = INCONCLUSIVE, "no dictionary (reference index) given"
        if models.dictionary is not None:
            dict_v, dict_why = dictionary_verdict(r5, ri, p, thr)
            ev["dictionary"].update(verdict=dict_v, reason=dict_why)
        decision, why = decide(agree, n, dict_v, rev_v, p)
        ev["decision"], ev["reason"] = decision, why
        if decision == "relink":
            ev["variant_id"] = variant_concept_id(w)
            relinks[w] = ev["variant_id"]
    say(f"variant-audit: {len(relinks)} relinks")
    judged = [e for e in classes.values() if e["n_clips"] >= p.min_clips]
    low = [e for e in judged if e[f"agreement_top{p.top_k}"] is not None
           and e[f"agreement_top{p.top_k}"] < p.max_agreement]
    summary = {"words_isl500_train": len(by), "words_in_reference": len(cand), "words_not_in_reference": len(not_in),
               "words_judged": len(judged), "low_agreement": len(low), "relinked": len(relinks),
               "kept_low_agreement": len(low) - len(relinks),
               "kept_by_reason": dict(Counter(e["reason"].split(" (")[0] for e in low if e["decision"] == "keep")),
               "clips_scored": sum(e["n_clips"] for e in classes.values()),
               # the dictionary rule, calibrated on the words the classifier agrees on: its threshold, and how often
               # it says "another form" on those words (leave-one-out). Not usable: the link step refuses the file
               "dictionary_calibration": cal,
               "dictionary_rule_usable": cal["usable"] if cal is not None else None,
               "dictionary_false_alarms_on_agreeing_words": (cal or {}).get("false_alarms")}
    return {"kind": VARIANT_KIND, "audit_version": AUDIT_VERSION, "source": VARIANT_SOURCE, "part": "train",
            "params": asdict(p), "alias_of_used": dict(sorted(dict(alias_of or {}).items())),
            "reference": dict(models.provenance or {}), "summary": summary,
            "relinks": {VARIANT_SOURCE: dict(sorted(relinks.items()))}, "classes": classes,
            "not_in_reference": not_in}


# --------------------------------------------------------------------------------------------- files

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_variant_links(path: Path, doc: dict, *, profile: str, work: Path, extra: dict | None = None) -> dict:
    """Write the audit document (atomic) and return ``{"file", "sha256"}``."""
    d = {**doc, "profile": profile, "work": str(work), "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
         **(extra or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(path)
    return {"file": str(path), "sha256": file_sha256(path)}


class VariantLinksError(ValueError):
    pass


def split_record(seed: int, params: Mapping, train_groups) -> dict:
    """What the audit records of the ISL500 signer split (review of b03eba0): the seed, the split parameters and the
    sorted train signer groups (and their SHA-256). The link step refuses the file when one of these groups is no
    longer in train (``check_split``)."""
    g = sorted({str(x) for x in train_groups if x})
    return {"source": VARIANT_SOURCE, "seed": int(seed), "params": dict(params), "train_groups": g,
            "train_groups_sha256": hashlib.sha256("\n".join(g).encode()).hexdigest()}


def check_split(audited: Mapping | None, current_train_groups) -> dict:
    """``{"ok", "moved_groups", ...}``: are the ISL500 signer groups the audit used as train still train now? A group
    that moved to val / test would let held-out signers decide training labels (a leak into data selection)."""
    if not audited or not audited.get("train_groups"):
        return {"ok": False, "reason": "the file records no ISL500 train groups: the split cannot be checked",
                "moved_groups": None}
    cur = {str(x) for x in current_train_groups if x}
    moved = sorted(set(audited["train_groups"]) - cur)
    return {"ok": not moved, "moved_groups": moved, "audited_groups": len(audited["train_groups"]),
            "audited_sha256": audited.get("train_groups_sha256"), "audited_params": audited.get("params"),
            "audited_seed": audited.get("seed"),
            **({"reason": f"{len(moved)} ISL500 signer group(s) the audit used as train are not train now "
                          f"(e.g. {moved[:3]}): re-run the audit on the current split"} if moved else {})}


def load_variant_links(path: Path, profile: str, *, accept_noisy: bool = False) -> tuple[dict[str, dict[str, str]],
                                                                                          dict]:
    """``(relinks, meta)`` of an audit file: ``relinks`` = source -> {concept: variant concept}; ``meta`` = file,
    sha256, audit version, profile, counts, the evidence scope (``parts_used``) and the dictionary rule's
    calibration (recorded in ``link_settings.json``). Refuses (``VariantLinksError``) a file of another kind, audit
    version or profile (critic 7: one profile's audit never relinks another's data), a relink that is not
    ``<concept>~<source>``, and a file whose dictionary rule is marked not usable unless ``accept_noisy``. The split
    check needs the current data: :func:`check_split` (the link step)."""
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise VariantLinksError(f"{path}: not a readable variant-links file ({e})") from e
    if not isinstance(d, dict) or d.get("kind") != VARIANT_KIND:
        raise VariantLinksError(f"{path}: not a variant-links file (kind {d.get('kind') if isinstance(d, dict) else None!r})")
    if d.get("audit_version") != AUDIT_VERSION:
        raise VariantLinksError(f"{path}: audit version {d.get('audit_version')!r}, not {AUDIT_VERSION!r} (an older "
                                "audit used an uncalibrated dictionary rule): run variant-audit again")
    if d.get("profile") != profile:
        raise VariantLinksError(f"{path} was audited for profile {d.get('profile')!r}, not {profile!r}")
    rel: dict[str, dict[str, str]] = {}
    for src, m in (d.get("relinks") or {}).items():
        for w, v in (m or {}).items():
            if v != variant_concept_id(w, src):
                raise VariantLinksError(f"{path}: relink {w} -> {v} is not {variant_concept_id(w, src)}")
            rel.setdefault(str(src), {})[str(w)] = str(v)
    summ = d.get("summary") or {}
    usable = summ.get("dictionary_rule_usable")
    n_rel = sum(len(m) for m in rel.values())
    if usable is False and n_rel and not accept_noisy:
        cal = summ.get("dictionary_calibration") or {}
        raise VariantLinksError(f"{path}: the audit marks its dictionary rule not usable ({cal.get('reason') or '-'}): "
                                f"its {n_rel} relinks may be signer shift. Read the evidence; --accept-noisy links "
                                "with it anyway (recorded)")
    fa = summ.get("dictionary_false_alarms_on_agreeing_words") or {}
    meta = {"file": str(path), "sha256": file_sha256(path), "audit_version": d.get("audit_version"),
            "profile": d.get("profile"), "work": d.get("work"), "created": d.get("created"), "n_relinks": n_rel,
            "parts_used": d.get("parts_used") or {"isl500": d.get("part", "train")},
            "dictionary_rule_usable": usable, "dictionary_false_alarm_rate": fa.get("rate"),
            "dictionary_threshold": (summ.get("dictionary_calibration") or {}).get("threshold"),
            "top_k": (d.get("params") or {}).get("top_k")}
    if usable is False and n_rel:
        meta["accepted_noisy"] = True
    return rel, meta


def variant_links_doc_split(path: Path) -> dict | None:
    """The ``isl500_split`` record of an audit file (None when missing or unreadable)."""
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return d.get("isl500_split") if isinstance(d, dict) else None


# --------------------------------------------------------------------------------------------- real models

def model_scorer(classifier=None, encoder=None, fcfg=None, chunk: int = 32) -> ScoreFn:
    """Score clips with a ``Predictor`` (mean softmax over the engine's three crops) and / or an ``EncoderPredictor``
    (mean crop embedding, renormalised). Clips whose features fail are left out."""
    from app.features.build import FeatureError
    from app.isl.evaluation.runners import _features

    fc = fcfg or (classifier.feature_config if classifier is not None else encoder.feature_config)

    def score(clips):
        P, E, used = [], [], []
        for a in range(0, len(clips), chunk):
            feats, ok = [], []
            for c in clips[a:a + chunk]:
                try:
                    feats.append(_features(c, fc))
                except (FeatureError, OSError, ValueError, KeyError):
                    continue
                ok.append(c)
            if not ok:
                continue
            n = [len(f) for f in feats]
            X = np.concatenate(feats)
            cut = np.cumsum(n)[:-1]
            if classifier is not None:
                pr, _ = classifier.predict(X)
                P += [x.mean(0) for x in np.split(np.atleast_2d(pr), cut)]
            if encoder is not None:
                z, _ = encoder.embed(X)
                E += [x.mean(0) for x in np.split(z, cut)]
            used += ok
        Pm = np.stack(P).astype(np.float32) if P else (np.zeros((0, 0), np.float32) if classifier is not None else None)
        Em = None
        if encoder is not None:
            Em = np.stack(E).astype(np.float32) if E else np.zeros((0, 0), np.float32)
            if len(Em):
                Em = Em / np.maximum(np.linalg.norm(Em, axis=1, keepdims=True), 1e-8)
        return Pm, Em, used

    return score


def dictionary_from_index(index, sources: tuple[str, ...] = ("islrtc",)) -> Dictionary | None:
    """The dictionary rows of an ``EmbeddingIndex`` (default: ISLRTC rows only)."""
    keep = [i for i, r in enumerate(index.rows) if r.get("source") in sources]
    if not keep:
        return None
    return Dictionary(index.emb[keep], [index.rows[i]["concept_id"] for i in keep])
