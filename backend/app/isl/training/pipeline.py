"""ISL build pipeline glue used by ``python -m app.isl.cli``: work folders, component loading, evaluation and
``build-pack`` (docs/ISL_DESIGN.md section 11).

Work folder per profile: ``<artifacts>/isl/<profile>/`` (classifier.pt, encoder.pt, index/, fusion.json,
results.json, letters.pt). ``build-pack`` copies them with the vocabulary, the Hindi lexicon (when present) and
a spelling lexicon into ``<artifacts>/packs/<pack id>/`` - ``isl`` for commercial-safe, ``isl-research`` for
research - which the app loads in place of the tracked ``models/packs/isl`` (vocabulary only).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

from app.core.config import get_settings
from app.features.spec import FeatureConfig
from app.isl.models.fusion import FusionParams
from app.isl.paths import PACK_DIR
from app.isl.profiles import get_profile

PACK_IDS = {"commercial-safe": "isl", "research": "isl-research"}
PACK_VERSION = "2026.11.0"
# ISLRTC dictionary: GODL-India, but the publisher forbids resale and profiteering (docs/DATASETS_ISL.md).
ISLRTC_TERMS = ("free, non-profit use with acknowledgement only (ISLRTC FAQ Q.23); "
                "paid or ad-funded use needs ISLRTC permission")


def workdir(profile: str) -> Path:
    """``<artifacts>/isl/<profile>``; ``SLR_ISL_WORKDIR`` overrides the parent folder (smoke runs of the phase-2
    driver use it so they never overwrite a real build)."""
    import os

    base = Path(os.environ["SLR_ISL_WORKDIR"]) if os.environ.get("SLR_ISL_WORKDIR") else get_settings().artifacts_dir / "isl"
    d = base / get_profile(profile).name
    d.mkdir(parents=True, exist_ok=True)
    return d


def pack_dir(profile: str) -> Path:
    return get_settings().artifacts_dir / "packs" / PACK_IDS[get_profile(profile).name]


def load_recognizer(work: Path, fcfg: FeatureConfig, vocab_path: Path | None = None, profile: str = "commercial-safe",
                    fusion: FusionParams | None = None, with_index: bool = True):
    """An ``IslRecognizer`` over the components present in a work folder (no My signs)."""
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.models.index import EmbeddingIndex
    from app.isl.recognizer import IslRecognizer
    from app.models.checkpoint import Predictor
    from app.packs.vocabulary import Vocabulary

    vocab = Vocabulary.load(vocab_path or PACK_DIR / "vocabulary.jsonl")
    clf = Predictor(work / "classifier.pt") if (work / "classifier.pt").exists() else None
    enc = EncoderPredictor(work / "encoder.pt") if (work / "encoder.pt").exists() else None
    idx = None
    if with_index and enc is not None and (work / "index" / "meta.json").exists():
        idx = EmbeddingIndex.load(work / "index", expect_encoder_version=enc.version)
    if fusion is None:
        fusion = FusionParams.load(work / "fusion.json") if (work / "fusion.json").exists() else FusionParams(profile=profile)
    return IslRecognizer("isl", fcfg, vocab, classifier=clf, encoder=enc, index=idx, fusion=fusion, profile=profile)


def evaluate(work: Path, data, fcfg: FeatureConfig, smoke: bool, part: str = "test", vocab_path: Path | None = None) -> dict:
    from app.isl.evaluation.protocols import e1_classifier, e2_retrieval, e3_decisions, e6_latency
    from app.isl.evaluation.report import markdown_table, write_results
    from app.isl.training.fit_fusion import collect_evidence

    rec = load_recognizer(work, fcfg, vocab_path, profile=data.profile)
    clips = data.include_part(part)
    inputs, truths, used = collect_evidence(rec, clips)
    si = data.split.get("signer_independent")
    lab = data.split.get("label") or data.split.get("name")
    protocols = {}
    e1 = e1_classifier(inputs, truths, used, f"E1 core words, closed set, INCLUDE {part} fold of {data.split.get('name')} "
                                              f"({lab}; signer-independent: {si})")
    if e1:
        protocols["E1"] = e1
    e2 = e2_retrieval(rec, used, "E2 INCLUDE clips vs ISLRTC dictionary rows only (another signer). "
                                 "Seen-class: the encoder trained on these concepts' INCLUDE train clips; "
                                 "the held-out-concept one-shot protocol is phase 2")
    if e2:
        protocols["E2"] = e2
    protocols["E3"] = e3_decisions(inputs, truths, used, rec.fusion, "E3 fused system with accept / suggest / abstain "
                                   "(fusion fitted on the validation fold)")
    protocols["E6"] = e6_latency(rec, used)
    per_class = {}
    for inp, t in zip(inputs, truths):
        from app.isl.models.fusion import fuse

        r = fuse(inp, rec.fusion)
        d = per_class.setdefault(t, {"n": 0, "top1": 0, "top5": 0})
        d["n"] += 1
        d["top1"] += int(bool(r.keys) and r.keys[0] == t)
        d["top5"] += int(t in r.keys[:5])
    per_class = {k: {"n": v["n"], "top1": v["top1"] / v["n"], "top5": v["top5"] / v["n"]} for k, v in per_class.items()}
    from app.isl.training.readiness import partial_data_flag

    doc = write_results(work / "results.json", protocols, profile=data.profile, smoke=smoke, data=data.summary(),
                        extra={"per_class": per_class, "partition": part}, partial_data=partial_data_flag(work))
    (work / "results.md").write_text(markdown_table(doc) + "\n")
    return doc


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


CORE_LICENCES = {"include": {"licence": "CC-BY-4.0", "commercial": True},
                 "islrtc": {"licence": "GODL-India", "commercial": True, "terms": ISLRTC_TERMS}}


def pack_licences(work: Path, profile: str | None = None) -> list[dict]:
    """Licence entries of a pack: only the sources that really contributed clips to this build (``sources.json``
    ``used`` counts > 0, written by the phase-2 commands), with their roles and clip counts. INCLUDE and ISLRTC are
    always named because the pack's vocabulary is built from their word lists (role ``vocabulary`` when no clip of
    theirs was used). Without ``sources.json`` (phase-1 commands) the build used exactly INCLUDE + ISLRTC.
    With ``profile``, rows of sources outside that profile (left in a reused work folder by an older build, e.g.
    WLASL before it was removed) are ignored: no current command of the profile can have read them."""
    allowed = None
    if profile is not None:
        from app.isl.profiles import get_profile

        allowed = set(get_profile(profile).all_sources()) | set(CORE_LICENCES)
    used = work / "sources.json"
    rows: dict[str, dict] = {}
    if used.exists():
        try:
            doc = json.loads(used.read_text())
        except ValueError:
            doc = {}
        for name, d in sorted(doc.items()):
            if not isinstance(d, dict) or (allowed is not None and name not in allowed):
                continue
            per_role = d.get("used")
            if per_role is None:  # sources.json of an older build: no per-role counts (unknown count = assume used)
                n = (int(d["clips"] or 0) if "clips" in d else 1) if d.get("available") else 0
                per_role = {r: n for r in (d.get("roles") or ["words"])}
            roles = sorted(r for r, k in per_role.items() if int(k or 0) > 0)
            if not roles:
                continue
            rows[name] = {"source": name, "licence": d.get("licence", ""), "commercial": bool(d.get("commercial", False)),
                          "role": ",".join(roles), "clips": {r: int(per_role[r]) for r in roles}}
    else:
        rows = {k: {"source": k, **v, "role": "words"} for k, v in CORE_LICENCES.items()}
    for k, v in CORE_LICENCES.items():
        rows[k] = {**rows.get(k, {"source": k, "role": "vocabulary"}), **v}
    order = list(CORE_LICENCES) + sorted(x for x in rows if x not in CORE_LICENCES)
    return [rows[k] for k in order]


def check_letters_profile(path: Path, profile: str) -> None:
    """Refuse a letter model trained for another data profile, or (commercial-safe) one whose recorded letter
    sources include a source the profile may not read (ADR-ISL-6; legacy checkpoints without a profile pass)."""
    import torch

    from app.isl.profiles import LETTER_SOURCES

    prof = get_profile(profile)
    meta = torch.load(path, map_location="cpu", weights_only=False).get("meta") or {}
    if meta.get("profile") and meta["profile"] != prof.name:
        raise ValueError(f"{path} was trained for profile {meta['profile']!r}, not {prof.name!r}")
    known = {s for v in LETTER_SOURCES.values() for s in v}
    used = set(str(meta.get("source") or "").split(",")) | set(((meta.get("data") or {}).get("per_source") or {}))
    bad = sorted((used & known) - set(LETTER_SOURCES[prof.name]))
    if bad:
        raise ValueError(f"{path} was trained on letter sources outside profile {prof.name!r}: {bad}")


def build_pack(work: Path, profile: str, fcfg: FeatureConfig, out: Path | None = None, smoke: bool = False,
               vocab_path: Path | None = None) -> dict:
    from app.isl.fingerspell.lexicon import build_spelling_lexicon, write_spelling_lexicon
    from app.packs.vocabulary import Vocabulary

    prof = get_profile(profile)
    if (work / "letters.pt").exists():
        check_letters_profile(work / "letters.pt", prof.name)
    out = out or pack_dir(prof.name)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files = {}
    for name in ("classifier.pt", "encoder.pt", "letters.pt", "fusion.json"):
        if (work / name).exists():
            shutil.copy2(work / name, tmp / name)
            files[name] = _sha(tmp / name)
    if (work / "index" / "meta.json").exists():
        shutil.copytree(work / "index", tmp / "index")
        files["index/embeddings.f16.npy"] = _sha(tmp / "index" / "embeddings.f16.npy")
    # the results the app serves (/api/languages/<id>/results): the phase-2 ``evaluate --all`` report when it exists
    # (same top-level fields: label, smoke, partial_data, protocols), never a phase-1 ``results.json`` left in a
    # reused work folder by an older run; the phase-1 ``evaluate`` results only when there is no phase-2 report
    if (work / "eval" / "report.json").exists():  # evaluate --all report travels with the pack
        shutil.copytree(work / "eval", tmp / "eval")
        shutil.copy2(work / "eval" / "report.json", tmp / "results.json")
        files["results.json"] = _sha(tmp / "results.json")
    elif (work / "results.json").exists():
        shutil.copy2(work / "results.json", tmp / "results.json")
        files["results.json"] = _sha(tmp / "results.json")
    shutil.copy2(vocab_path or PACK_DIR / "vocabulary.jsonl", tmp / "vocabulary.jsonl")
    extra = work / "extra_concepts.jsonl"  # concepts only other sources (WSLP, ISL500, ...) know: appended
    if extra.exists() and extra.stat().st_size:
        have = {r["concept_id"] for r in Vocabulary.load(tmp / "vocabulary.jsonl").rows}
        with extra.open(encoding="utf-8") as src, (tmp / "vocabulary.jsonl").open("a", encoding="utf-8") as dst:
            for ln in src:
                try:
                    if ln.strip() and json.loads(ln)["concept_id"] not in have:
                        dst.write(ln if ln.endswith("\n") else ln + "\n")
                except (json.JSONDecodeError, KeyError):
                    continue
    if (PACK_DIR / "lexicon_hi.jsonl").exists():  # only the rows of this pack's vocabulary
        ids = {r["concept_id"] for r in Vocabulary.load(tmp / "vocabulary.jsonl").rows}
        with (PACK_DIR / "lexicon_hi.jsonl").open(encoding="utf-8") as src, \
                (tmp / "lexicon_hi.jsonl").open("w", encoding="utf-8") as dst:
            for ln in src:
                try:
                    if json.loads(ln).get("concept_id") in ids:
                        dst.write(ln)
                except json.JSONDecodeError:
                    continue
    write_spelling_lexicon(build_spelling_lexicon(Vocabulary.load(tmp / "vocabulary.jsonl").rows),
                           tmp / "spelling_lexicon.txt")
    enc_version = None
    if (work / "encoder.pt").exists():
        import torch

        enc_version = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)["meta"]["encoder_version"]
    from app.isl.training.readiness import partial_data_flag

    base = json.loads((PACK_DIR / "pack.json").read_text())
    licences = pack_licences(work, prof.name)
    partial = partial_data_flag(work)
    # the commercial check reads every contributing row, also those outside the profile (fail closed)
    nc = [x["source"] for x in pack_licences(work) if not x.get("commercial", False)]
    if prof.commercial and nc:
        raise ValueError(f"refusing to build a commercial-safe pack with non-commercial sources: {nc}")
    man = {**base, "id": PACK_IDS[prof.name], "version": PACK_VERSION + ("-smoke" if smoke else ""),
           "profile": prof.name, "feature_config": fcfg.to_dict(), "licences": licences,
           "display_name": base["display_name"] if prof.commercial else
           {"en": "Indian Sign Language (ISL, research models)", "hi": "भारतीय सांकेतिक भाषा (शोध)"},
           "partial_data": bool(partial or smoke),
           "note": ("SMOKE BUILD on partial data: proves the pipeline, not usable accuracy. " if smoke else
                    "PARTIAL DATA BUILD: trained while ingestion was incomplete (--allow-partial); not the app's "
                    "accuracy. " if partial else "")
           + f"Built {time.strftime('%Y-%m-%d %H:%M')} by python -m app.isl.cli build-pack.",
           "checksums": files}
    man["components"] = dict(base["components"])
    if enc_version:
        man["components"]["encoder"] = {**man["components"]["encoder"], "version": enc_version}
        man["components"]["index"] = {**man["components"]["index"], "encoder_version": enc_version}
    (tmp / "pack.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    from app.packs.registry import reset_registries

    reset_registries()
    return {"pack": str(out), "id": man["id"], "files": sorted(files), "size_mb": round(size / 1e6, 1)}
