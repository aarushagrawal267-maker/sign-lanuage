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
2. **Dictionary cross-check (critic 13).** Low agreement can be signer shift (new signers, same sign), not another
   sign form. Relinking then takes away supervision INCLUDE needs. So with the commercial-safe encoder the ISL500
   centroid and the INCLUDE centroid of the word both retrieve against the ISLRTC dictionary rows. The verdict is
   ``different_form`` only when INCLUDE retrieves the word's dictionary row well (rank <= ``dict_include_max_rank``)
   and ISL500 retrieves it clearly worse (rank > max(``dict_rank_floor``, ``dict_rank_ratio`` x INCLUDE's rank)).
   ISL500 retrieving it about as well is ``no_difference`` (signer shift); missing rows or clips, or INCLUDE not
   retrieving the row either, is ``inconclusive``.
3. **Reverse direction (optional).** A classifier trained on ISL500 scores INCLUDE clips of the word. Agreement at or
   above ``reverse_min_agreement`` means the forms are compatible (``no_difference``: never relinked).

Decision: relink only on evidence of a different form: dictionary ``different_form`` (and the reverse direction, if
run, does not say ``no_difference``), or dictionary ``inconclusive`` with the reverse direction ``different_form``.
Every word's evidence is written next to the decision.

Scope (critic 7): the audit file is written to the profile's work folder (``<work>/variant_links.json``), never to the
shared ``data/isl/derived``. The link step of that work folder reads it (or a file given explicitly), freezes the
relinks in ``link_settings.json`` with the file's SHA-256, and every later step uses the frozen copy.
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

VARIANT_LINKS = "variant_links.json"
VARIANT_KIND = "isl-variant-links"
AUDIT_VERSION = "isl-variant-audit-1"
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
    dict_rank_ratio: float = 3.0  # ISL500 rank above ratio x INCLUDE rank (and the floor): a different form
    dict_include_max_rank: int = 20  # INCLUDE itself must retrieve the word's dictionary row within this rank
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


def dictionary_verdict(rank_isl500: int | None, rank_include: int | None, p: AuditParams) -> tuple[str, str]:
    if rank_isl500 is None or rank_include is None:
        why = ("the word has no ISLRTC dictionary row" if rank_isl500 is None and rank_include is None else
               "no INCLUDE train clips of the word" if rank_include is None else "no ISL500 centroid")
        return INCONCLUSIVE, why
    bar = max(p.dict_rank_floor, p.dict_rank_ratio * rank_include)
    if rank_isl500 <= bar:
        return NO_DIFFERENCE, (f"ISL500 retrieves the dictionary row at rank {rank_isl500}, INCLUDE at {rank_include}: "
                               "about as well (signer shift, not another form)")
    if rank_include > p.dict_include_max_rank:
        return INCONCLUSIVE, (f"INCLUDE does not retrieve the dictionary row either (rank {rank_include} > "
                              f"{p.dict_include_max_rank})")
    return DIFFERENT, (f"INCLUDE retrieves the dictionary row at rank {rank_include}, ISL500 only at {rank_isl500} "
                       f"(> {bar:g}): ISL500 signs another form")


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
    classes: dict[str, dict] = {}
    relinks: dict[str, str] = {}
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
        # dictionary cross-check (critic 13): for every word, so the report also shows how often the rule fires on
        # words the classifier agrees on (its false-alarm rate)
        dict_v, dict_why = INCONCLUSIVE, "no dictionary (reference index) given"
        if models.dictionary is not None:
            inc = _sample([x for g in sorted(grp) for x in inc_by.get(g, [])], p.max_per_class, f"{p.seed}:inc:{w}")
            _, Ei, used_i = models.score(inc) if inc else (None, None, [])
            c5, ci = _centroid(E), _centroid(Ei)
            r5, top5_, s5 = models.dictionary.rank(c5, grp) if c5 is not None else (None, [], None)
            ri, topi, si = models.dictionary.rank(ci, grp) if ci is not None else (None, [], None)
            dict_v, dict_why = dictionary_verdict(r5, ri, p)
            ev["dictionary"] = {"rows": sorted(grp & set(models.dictionary.pos)), "rank_isl500": r5, "rank_include": ri,
                                "n_concepts": len(models.dictionary), "n_include_clips": len(used_i),
                                "isl500_top3": top5_, "include_top3": topi,
                                "centroid_cosine": float(c5 @ ci) if c5 is not None and ci is not None else None,
                                "verdict": dict_v, "reason": dict_why}
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
        decision, why = decide(agree, len(used), dict_v, rev_v, p)
        ev["decision"], ev["reason"] = decision, why
        if decision == "relink":
            ev["variant_id"] = variant_concept_id(w)
            relinks[w] = ev["variant_id"]
        classes[w] = ev
        if (i + 1) % 20 == 0 or i + 1 == len(words):
            say(f"variant-audit {i + 1}/{len(words)} words, {len(relinks)} relinks")
    judged = [e for e in classes.values() if e["n_clips"] >= p.min_clips]
    low = [e for e in judged if e[f"agreement_top{p.top_k}"] is not None
           and e[f"agreement_top{p.top_k}"] < p.max_agreement]
    agreeing = [e for e in judged if e not in low]
    fa = [e for e in agreeing if (e.get("dictionary") or {}).get("verdict") == DIFFERENT]
    summary = {"words_isl500_train": len(by), "words_in_reference": len(cand), "words_not_in_reference": len(not_in),
               "words_judged": len(judged), "low_agreement": len(low), "relinked": len(relinks),
               "kept_low_agreement": len(low) - len(relinks),
               "kept_by_reason": dict(Counter(e["reason"].split(" (")[0] for e in low if e["decision"] == "keep")),
               "clips_scored": sum(e["n_clips"] for e in classes.values()),
               # how often the dictionary rule says "another form" on words the classifier agrees on: a high
               # rate means the cross-check cannot tell forms apart with this encoder
               "dictionary_false_alarms_on_agreeing_words": {"n": len(fa), "of": len(agreeing),
                                                             "rate": (len(fa) / len(agreeing)) if agreeing else None}}
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


def load_variant_links(path: Path, profile: str) -> tuple[dict[str, dict[str, str]], dict]:
    """``(relinks, meta)`` of an audit file: ``relinks`` = source -> {concept: variant concept}; ``meta`` = file,
    sha256, audit version, profile and counts (recorded in ``link_settings.json``). Refuses (``VariantLinksError``)
    a file of another kind or another profile (critic 7: one profile's audit never relinks another's data), and a
    relink that is not ``<concept>~<source>``."""
    path = Path(path)
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise VariantLinksError(f"{path}: not a readable variant-links file ({e})") from e
    if not isinstance(d, dict) or d.get("kind") != VARIANT_KIND:
        raise VariantLinksError(f"{path}: not a variant-links file (kind {d.get('kind') if isinstance(d, dict) else None!r})")
    if d.get("profile") != profile:
        raise VariantLinksError(f"{path} was audited for profile {d.get('profile')!r}, not {profile!r}")
    rel: dict[str, dict[str, str]] = {}
    for src, m in (d.get("relinks") or {}).items():
        for w, v in (m or {}).items():
            if v != variant_concept_id(w, src):
                raise VariantLinksError(f"{path}: relink {w} -> {v} is not {variant_concept_id(w, src)}")
            rel.setdefault(str(src), {})[str(w)] = str(v)
    meta = {"file": str(path), "sha256": file_sha256(path), "audit_version": d.get("audit_version"),
            "profile": d.get("profile"), "created": d.get("created"),
            "n_relinks": sum(len(m) for m in rel.values())}
    return rel, meta


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
