"""Language packs: list, vocabulary, Hindi lexicon (summary, entries, corrections), approximate sentences,
clip analysis, results and "My signs" (docs/ISL_DESIGN.md sections 4.4, 7.4 and 8). The legacy ASL routes in
``recognition.py`` are unchanged."""

from __future__ import annotations

import re
import threading

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.features.build import sequence_quality
from app.landmarks.raw import RawSequence
from app.packs.registry import UnknownLanguage, get_registry

router = APIRouter(tags=["languages"])


def _pack(lang: str):
    try:
        return get_registry().get(lang)
    except UnknownLanguage as e:
        raise HTTPException(404, str(e)) from e


def _isl_pack(lang: str):
    pack = _pack(lang)
    if pack.info.kind != "isl-hybrid":
        raise HTTPException(400, f"'{lang}' does not support this (only file-based sign packs such as ISL do)")
    return pack


def _recognizer(pack):
    rec, info, problem = pack.recognizer()
    if rec is None:
        raise HTTPException(409, problem or "this language has no recogniser")
    return rec, info, problem


# ---------------------------------------------------------------------------------------------- packs

@router.get("/api/languages", summary="Available sign languages (language packs)")
def list_languages():
    reg = get_registry()
    return {"default": reg.default_id(), "languages": [i.to_json() for i in reg.list()]}


@router.get("/api/languages/{lang}", summary="One language pack with model info and a results summary")
def get_language(lang: str):
    pack = _pack(lang)
    info = pack.info
    model = None
    if info.kind == "experiment-registry":
        _, model, _ = pack.predictor()
    elif info.status != "error":
        _, model, _ = pack.recognizer()
    res = pack.results() if hasattr(pack, "results") else None
    summary = None
    if res:
        summary = {k: res.get(k) for k in ("profile", "created", "label", "smoke", "protocols") if k in res}
    return {"pack": info.to_json(), "model": model, "results": summary}


@router.get("/api/languages/{lang}/results", summary="Evaluation results of a language pack (with protocol labels)")
def language_results(lang: str):
    pack = _pack(lang)
    res = pack.results() if hasattr(pack, "results") else None
    if res is None:
        raise HTTPException(404, "no evaluation results for this language yet")
    return res


# ---------------------------------------------------------------------------------------------- vocabulary

def _hi_of(pack):
    lex = pack.lexicon() if hasattr(pack, "lexicon") else None
    return (lambda cid: lex.get(cid)) if lex is not None else (lambda cid: None)


def _vocab_row(i: int, r: dict, hi_get) -> dict:
    hi = hi_get(r["concept_id"])
    variants = r.get("variants", []) or []
    return {"class_index": i, "concept_id": r["concept_id"], "headword": r.get("headword"),
            "text": {"en": (r.get("display") or {}).get("en") or r.get("headword"), "hi": hi["hi"] if hi else None},
            "hi_review": hi.get("review") if hi else None, "hi_alt": (hi.get("hi_alt") or []) if hi else [],
            "hi_source": hi.get("source") if hi else None, "hi_reviewed": bool(hi.get("reviewed")) if hi else False,
            "kind": r.get("kind", "word"), "category": r.get("category"),
            "in_classifier": bool(r.get("in_classifier")), "indexable": r.get("indexable", True),
            "n_variants": len(variants), "sources": sorted({v.get("source") for v in variants if v.get("source")}),
            "regions": sorted({v["region"] for v in variants if v.get("region")}),
            "review": (r.get("review") or {}).get("status", "auto")}


@router.get("/api/languages/{lang}/vocabulary", summary="Paged, searchable vocabulary of a language pack")
def language_vocabulary(lang: str, q: str | None = None, category: str | None = None, source: str | None = None,
                        kind: str | None = None, has_hi: bool | None = None, in_classifier: bool | None = None,
                        review: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    pack = _pack(lang)
    vocab = pack.vocabulary()
    hi_get = _hi_of(pack)
    total, rows = vocab.search(q, category=category, source=source, kind=kind, in_classifier=in_classifier,
                               has_hi=has_hi, review=review, offset=offset, limit=limit, hi_lookup=hi_get)
    return {"total": total, "offset": offset, "limit": limit, "items": [_vocab_row(i, r, hi_get) for i, r in rows]}


_preview_cache: dict = {}
_preview_lock = threading.Lock()


def _islrtc_paths() -> tuple[dict, dict]:
    """(clip id -> landmark path, clip id -> trim) from the local ISL data (cached per file mtimes)."""
    from app.isl.data.manifest import read_manifest
    from app.isl.paths import isl_paths

    p = isl_paths()
    man, trims = p.manifest("islrtc"), p.trim_sidecar
    key = tuple(f.stat().st_mtime_ns if f.exists() else 0 for f in (man, trims)) + (str(p.root),)
    with _preview_lock:
        if _preview_cache.get("key") != key:
            recs = {r["id"]: str(p.resolve(r["path"])) for r in read_manifest(man) if r.get("n_frames")}
            tr = {r["id"]: r for r in read_manifest(trims, only_ok=False)}
            _preview_cache.update(key=key, recs=recs, trims=tr)
        return _preview_cache["recs"], _preview_cache["trims"]


def _preview(entry: dict, max_frames: int = 48) -> dict | None:
    from pathlib import Path

    try:
        recs, trims = _islrtc_paths()
    except OSError:
        return None
    for v in entry.get("variants", []):
        for cid in v.get("clips", []):
            if cid in recs and Path(recs[cid]).exists():
                raw = RawSequence.load(Path(recs[cid]))
                tr = trims.get(cid)
                if tr and tr.get("method") != "empty":
                    raw = raw.slice(int(tr["start"]), int(tr["end"]))
                step = max(1, len(raw) // max_frames)
                return {"clip": cid, "variant_id": v.get("variant_id"), "width": raw.width, "height": raw.height,
                        "frames": [raw.frame_to_json(i) for i in range(0, len(raw), step)]}
    return None


@router.get("/api/languages/{lang}/vocabulary/{concept_id}", summary="One vocabulary entry with variants and preview")
def language_vocabulary_entry(lang: str, concept_id: str, preview: bool = True):
    pack = _pack(lang)
    vocab = pack.vocabulary()
    r = vocab.get(concept_id)
    if r is None:
        raise HTTPException(404, f"unknown concept {concept_id!r}")
    hi = _hi_of(pack)(concept_id)
    out = {**r, "class_index": vocab.index_of(concept_id), "hindi": hi}
    res = pack.results() if hasattr(pack, "results") else None
    if res:
        per = ((res.get("per_class") or {}).get(concept_id))
        out["metrics"] = per
    if preview and pack.info.kind == "isl-hybrid":
        out["preview"] = _preview(r)
    return out


@router.get("/api/languages/{lang}/lexicon", summary="Hindi lexicon summary (coverage, sources, review states)")
def lexicon_summary(lang: str, local: bool = False):
    from app.isl.text.lexicon import EN_MISSING, HI_MISSING, REVIEW_LABELS

    pack = _pack(lang)
    lex = pack.lexicon() if hasattr(pack, "lexicon") else None
    n_vocab = len(pack.vocabulary())
    if lex is None:
        return {"language": lang, "available": False, "concepts": n_vocab, "entries": 0,
                "missing_text": {"en": EN_MISSING, "hi": HI_MISSING}}
    st = lex.stats()
    out = {"language": lang, "available": True, "concepts": n_vocab, **st,
           "coverage": round(st["entries"] / n_vocab, 4) if n_vocab else 0.0,
           "review_labels": REVIEW_LABELS, "missing_text": {"en": EN_MISSING, "hi": HI_MISSING}}
    if local:
        out["local_rows"] = lex.local_rows()
    return out


@router.get("/api/languages/{lang}/lexicon/{concept_id}", summary="Hindi entry of one concept")
def lexicon_entry(lang: str, concept_id: str):
    from app.isl.text.lexicon import REVIEW_LABELS

    pack = _pack(lang)
    vocab = pack.vocabulary()
    if vocab.get(concept_id) is None:
        raise HTTPException(404, f"unknown concept {concept_id!r}")
    hi = _hi_of(pack)(concept_id)
    return {"concept_id": concept_id, "english": vocab.english(concept_id), "hindi": hi,
            "available": hi is not None, "label": REVIEW_LABELS.get(hi.get("review")) if hi else None}


class SentenceWord(BaseModel):
    concept_id: str | None = Field(None, max_length=160)
    en: str | None = Field(None, max_length=120)
    hi: str | None = Field(None, max_length=120)


class SentenceIn(BaseModel):
    concept_ids: list[str] = Field(default_factory=list, max_length=40)
    words: list[SentenceWord] = Field(default_factory=list, max_length=40)


@router.post("/api/languages/{lang}/sentence",
             summary="Approximate English + Hindi sentence from a sequence of recognised signs (rule-based)")
def build_sentence_route(lang: str, body: SentenceIn):
    """``concept_ids`` (recognised concepts, in signing order) or ``words`` (for My signs / spelled words:
    ``{"concept_id"?, "en", "hi"?}``). The result is always labelled approximate."""
    from app.isl.text.sentence import build_both, make_token, tokens_from_concepts

    pack = _pack(lang)
    vocab = pack.vocabulary()
    lex = pack.lexicon() if hasattr(pack, "lexicon") else None
    if body.concept_ids and body.words:
        raise HTTPException(422, "send concept_ids or words, not both")
    if body.concept_ids:
        unknown = [c for c in body.concept_ids if vocab.get(c) is None]
        if unknown:
            raise HTTPException(404, f"unknown concept(s): {', '.join(unknown[:5])}")
        tokens = tokens_from_concepts(body.concept_ids, vocab, lex)
    else:
        tokens = []
        for w in body.words:
            if w.concept_id and vocab.get(w.concept_id) is not None:
                t = tokens_from_concepts([w.concept_id], vocab, lex)[0]
                if w.hi:
                    t.hi = w.hi
                tokens.append(t)
            elif w.en:
                tokens.append(make_token(w.concept_id or f"word:{w.en}", w.en.strip(), (w.hi or "").strip() or None))
            else:
                raise HTTPException(422, "each word needs a known concept_id or an English text")
    if not tokens:
        raise HTTPException(422, "no words given")
    out = build_both(tokens)
    out["language"] = lang
    if pack.info.sign_language != "ISL":
        out["note"] = "The word-order rules are written for ISL; for other sign languages the result is rougher."
    return out


class LexiconPatch(BaseModel):
    hi: str = Field(min_length=1, max_length=120)
    hi_alt: list[str] = Field(default_factory=list, max_length=10)
    review: str = "reviewed"
    reviewer: str | None = Field(None, max_length=80)


@router.patch("/api/languages/{lang}/lexicon/{concept_id}", summary="Local Hindi correction / review of one concept")
def patch_lexicon(lang: str, concept_id: str, body: LexiconPatch):
    pack = _isl_pack(lang)
    if pack.vocabulary().get(concept_id) is None:
        raise HTTPException(404, f"unknown concept {concept_id!r}")
    try:
        return pack.lexicon().set_local(concept_id, body.hi, body.hi_alt, body.review, body.reviewer,
                                        english=pack.vocabulary().english(concept_id))
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# ---------------------------------------------------------------------------------------------- analysis

class ClipIn(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[dict]
    policy: str = "balanced"
    top_k: int = Field(5, ge=1, le=20)
    spell: bool = False


def analyze_clip_for_language(lang: str, clip) -> dict:
    pack = _pack(lang)
    if pack.info.kind == "experiment-registry":
        from app.api.recognition import LandmarkClip, analyze_landmarks

        return analyze_landmarks(LandmarkClip(width=clip.width, height=clip.height, frames=clip.frames))
    rec, info, problem = _recognizer(pack)
    try:
        seq = RawSequence.from_frames(clip.frames, clip.width, clip.height)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    if len(seq) == 0:
        raise HTTPException(422, "the clip has no frames")
    policy = getattr(clip, "policy", "balanced")
    if policy not in ("strict", "balanced", "lenient"):
        raise HTTPException(422, "policy must be strict, balanced or lenient")
    if getattr(clip, "spell", False):
        if getattr(rec, "speller", None) is None:
            raise HTTPException(409, "spelling is not available for this language yet")
        r = rec.spell_recognition(seq, top_k=getattr(clip, "top_k", 5))
    else:
        r = rec.recognize([seq], top_k=getattr(clip, "top_k", 5), policy=policy)
    out = r.to_json()
    out["top"] = out["candidates"]
    out["quality"] = sequence_quality(seq)
    out["model"] = info
    out["message"] = problem
    return out


@router.post("/api/languages/{lang}/analyze/landmarks", summary="Recognise a landmark clip (top-5 with decision)")
def analyze_landmarks_lang(lang: str, clip: ClipIn):
    return analyze_clip_for_language(lang, clip)


# ---------------------------------------------------------------------------------------------- my signs

class TakeIn(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: list[dict] = Field(max_length=300)


class MySignIn(BaseModel):
    text_en: str = Field(min_length=1, max_length=80)
    text_hi: str | None = Field(None, max_length=80)
    concept_id: str | None = Field(None, max_length=120)
    region: str | None = Field(None, max_length=80)
    note: str | None = Field(None, max_length=300)
    author: str | None = Field(None, max_length=80)
    dry_run: bool = False
    takes: list[TakeIn] = Field(min_length=1, max_length=5)


class MySignPatch(BaseModel):
    text_en: str | None = Field(None, max_length=80)
    text_hi: str | None = Field(None, max_length=80)
    region: str | None = Field(None, max_length=80)
    note: str | None = Field(None, max_length=300)
    enabled: bool | None = None
    concept_id: str | None = Field(None, max_length=120)


@router.get("/api/languages/{lang}/my-signs", summary="Your own (personal / regional) signs")
def list_my_signs(lang: str):
    pack = _isl_pack(lang)
    return {"signs": [s.to_json() for s in pack.my_signs().list()]}


@router.post("/api/languages/{lang}/my-signs", summary="Add your own sign from 3-5 recordings (landmarks only)")
def add_my_sign(lang: str, body: MySignIn):
    pack = _isl_pack(lang)
    rec, _, _ = _recognizer(pack)
    vocab = pack.vocabulary()
    if body.concept_id and vocab.get(body.concept_id) is None:
        raise HTTPException(404, f"unknown concept {body.concept_id!r}")
    try:
        takes = [RawSequence.from_frames(t.frames, t.width, t.height) for t in body.takes]
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    text_hi = body.text_hi
    if body.concept_id and not text_hi:
        hi = _hi_of(pack)(body.concept_id)
        text_hi = hi["hi"] if hi else None
    enrol = rec.fusion.enrol
    res = pack.my_signs().add(
        body.text_en, takes, embedder=rec.user_embedder, text_hi=text_hi, concept_id=body.concept_id,
        region=body.region, note=body.note, author=body.author, dry_run=body.dry_run,
        global_search=rec.global_search if rec.index is not None else None, s_min=rec.fusion.s_min,
        c_min=float(enrol.get("c_min", 0.5)), min_takes=int(enrol.get("min_takes", 3)),
        max_takes=int(enrol.get("max_takes", 5)), describe_concept=lambda c: f"{vocab.english(c)} (ISLRTC dictionary)")
    return res.to_json()


class LetterIn(BaseModel):
    letter: str = Field(min_length=1, max_length=1)
    take: TakeIn


@router.get("/api/languages/{lang}/my-signs/alphabet", summary="Your alphabet calibration for spelling mode")
def get_alphabet(lang: str):
    from app.isl.fingerspell.prototypes import MIN_LETTERS

    pack = _isl_pack(lang)
    counts = pack.alphabet().counts()
    return {"letters": counts, "ready": len(counts) >= MIN_LETTERS, "min_letters": MIN_LETTERS,
            "trained_model": pack.present().get("letters", False)}


@router.post("/api/languages/{lang}/my-signs/alphabet", summary="Record one letter of the manual alphabet")
def add_alphabet_letter(lang: str, body: LetterIn):
    from app.isl.fingerspell.prototypes import MIN_LETTERS, hold_features

    pack = _isl_pack(lang)
    try:
        raw = RawSequence.from_frames(body.take.frames, body.take.width, body.take.height)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e)) from e
    hf = hold_features(raw) if len(raw) >= 2 else None
    if hf is None:
        raise HTTPException(422, "No hands or body found in this recording. Please hold the letter still in view.")
    feats, hold_ms = hf
    try:
        counts = pack.alphabet().add(body.letter, feats)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return {"letter": body.letter.upper(), "hold_ms": hold_ms, "letters": counts, "ready": len(counts) >= MIN_LETTERS,
            "message": "Saved." if hold_ms > 0 else "Saved, but no clear still moment was found; hold the letter still."}


@router.delete("/api/languages/{lang}/my-signs/alphabet", summary="Forget the alphabet calibration")
def reset_alphabet(lang: str):
    _isl_pack(lang).alphabet().reset()
    return {"deleted": True}


@router.patch("/api/languages/{lang}/my-signs/{sign_id}", summary="Rename, translate or enable/disable a sign")
def patch_my_sign(lang: str, sign_id: str, body: MySignPatch):
    pack = _isl_pack(lang)
    try:
        return pack.my_signs().update(sign_id, **body.model_dump()).to_json()
    except KeyError:
        raise HTTPException(404, "unknown sign") from None
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.delete("/api/languages/{lang}/my-signs/{sign_id}", summary="Delete one of your signs (and its recordings)")
def delete_my_sign(lang: str, sign_id: str):
    pack = _isl_pack(lang)
    try:
        pack.my_signs().delete(sign_id)
    except KeyError:
        raise HTTPException(404, "unknown sign") from None
    return {"deleted": True}


@router.get("/api/languages/{lang}/my-signs/export", summary="Export signs as a .signpack file (needs consent)")
def export_my_signs(lang: str, ids: str | None = None, title: str = "My signs", region: str | None = None,
                    author: str | None = None, licence: str = "CC-BY-4.0", include_face: bool = False):
    from app.isl.mysigns.packfile import PackError, export_pack

    pack = _isl_pack(lang)
    rec, _, _ = pack.recognizer()
    try:
        data = export_pack(pack.my_signs(), [i for i in ids.split(",") if i] if ids else None, title=title,
                           region=region, author=author, licence=licence, include_face=include_face,
                           encoder_version=rec.user_embedder.version if rec is not None else None,
                           app_version=pack.manifest.version or "")
    except PackError as e:
        raise HTTPException(422, str(e)) from e
    fname = re.sub(r"[^A-Za-z0-9_. -]", "_", title)[:60].strip() or "signs"
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fname}.signpack"'})


@router.post("/api/languages/{lang}/my-signs/import", summary="Import a .signpack file (merge or replace)")
async def import_my_signs(lang: str, file: UploadFile = File(...), mode: str = Form("merge")):
    from app.isl.mysigns.packfile import MAX_BYTES, PackError, import_pack

    pack = _isl_pack(lang)
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "sign pack larger than 50 MB")
    rec, _, _ = pack.recognizer()
    try:
        out = import_pack(pack.my_signs(), data, mode, encoder_version=rec.user_embedder.version if rec else None)
    except PackError as e:
        raise HTTPException(422, str(e)) from e
    out["conflicts"] = []  # conflicts are shown per sign on the My signs page (same check as enrolment)
    return out


def demo_allowed(path: str) -> bool:
    """Write routes that stay enabled in the public read-only demo (pure analysis, nothing stored)."""
    return path.startswith("/api/languages/") and (path.endswith("/analyze/landmarks") or path.endswith("/sentence"))

