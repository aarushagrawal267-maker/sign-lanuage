"""ISL build pipeline glue used by ``python -m app.isl.cli``: work folders, component loading, evaluation and
``build-pack`` (docs/ISL_DESIGN.md section 11).

Work folder per profile: ``<artifacts>/isl/<profile>/`` (classifier.pt, encoder.pt, index/ (training parts only:
fit-fusion and evaluation), index_final/ (every reference: the shipped pack's index), fusion.json, results.json,
letters.pt). ``build-pack`` copies them with the vocabulary, the Hindi lexicon (when present) and
a spelling lexicon into ``<artifacts>/packs/<pack id>/`` - ``isl`` for commercial-safe, ``isl-research`` for
research - which the app loads in place of the tracked ``models/packs/isl`` (vocabulary only).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
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


def shipped_index(work: Path) -> tuple[Path | None, str | None]:
    """The index a work folder ships to the app: ``index_final`` (every reference) when it exists and is not older
    than the training-only ``index``, else ``index`` (None when neither exists). An ``index_final`` older than
    ``index`` was built from older data or row logic (``index`` was rebuilt without ``index_final``): it is not
    shipped and the second value says why (docs/ISL_DESIGN.md 15.22)."""
    fin, tr = work / "index_final" / "meta.json", work / "index" / "meta.json"
    if not fin.exists():
        return (tr.parent if tr.exists() else None), None
    if tr.exists():
        cf = json.loads(fin.read_text()).get("created") or ""
        ct = json.loads(tr.read_text()).get("created") or ""
        if cf < ct:  # ISO UTC strings ("" = unknown: an older index_final, also not shipped over a newer index)
            return tr.parent, (f"index_final ({cf or 'no time'}) is older than index ({ct}): index was rebuilt "
                               "without index_final; shipping the training-only index (run the index_final step)")
    return fin.parent, None


def extra_concept_rows(work: Path, have: set[str]) -> list[dict]:
    """Rows of ``<work>/extra_concepts.jsonl`` (concepts only other sources know, written by the link step) whose
    concept id is not in ``have``, in file order: exactly the rows ``build_pack`` appends to the pack's vocabulary
    (so the vocabulary indices match). Bad lines are skipped; no file = no rows."""
    f = Path(work) / "extra_concepts.jsonl"
    out: list[dict] = []
    if not f.exists() or not f.stat().st_size:
        return out
    seen = set(have)
    for ln in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln) if ln.strip() else None
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and r.get("concept_id") and r["concept_id"] not in seen:
            seen.add(r["concept_id"])
            out.append(r)
    return out


def load_recognizer(work: Path, fcfg: FeatureConfig, vocab_path: Path | None = None, profile: str = "commercial-safe",
                    fusion: FusionParams | None = None, with_index: bool = True, for_app: bool = False):
    """An ``IslRecognizer`` over the components present in a work folder (no My signs).

    Default (fit-fusion, evaluation, every measuring caller): the index is ``<work>/index`` and must pass the leak
    guard (``EmbeddingIndex.check_for_evaluation``): built from training parts only (``contains_eval`` False, not
    unknown) with this very ``encoder.pt`` (SHA-256), else ``LeakageError``. ``for_app=True`` (a recogniser that is
    used, not measured): ``<work>/index_final`` when it exists and is not older than ``index`` (every reference,
    validation and test clips too), else ``<work>/index``; only the encoder SHA-256 is checked
    (``IndexEncoderMismatch``). docs 15.21. In the default mode a missing ``<work>/index`` next to an
    ``index_final`` / ``index.old`` / ``index.tmp`` raises ``IndexMissingError`` (15.22)."""
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.models.index import EmbeddingIndex
    from app.isl.recognizer import IslRecognizer
    from app.models.checkpoint import Predictor
    from app.packs.vocabulary import Vocabulary

    vocab = Vocabulary.load(vocab_path or PACK_DIR / "vocabulary.jsonl")
    extra = extra_concept_rows(work, {r["concept_id"] for r in vocab.rows})
    if extra:
        # the concepts only other sources know (link step's extra_concepts.jsonl) are in the shipped pack's
        # vocabulary (build_pack appends them): fit-fusion and evaluation must see the same vocabulary, or those
        # classes get prior 0 and vocabulary index -1 while measuring and prior 0.5 in the app (docs 15.27)
        vocab = Vocabulary(vocab.rows + extra)
    clf = Predictor(work / "classifier.pt") if (work / "classifier.pt").exists() else None
    enc = EncoderPredictor(work / "encoder.pt") if (work / "encoder.pt").exists() else None
    idx = None
    folder = work / "index"
    if for_app:
        folder = shipped_index(work)[0] or folder
    if with_index and enc is not None and not for_app and not (folder / "meta.json").exists():
        # measuring without retrieval because work/index is missing, while another index is there, would fit or
        # report classifier-only numbers without saying so (docs 15.22)
        other = [work / n for n in ("index_final", "index.old", "index.tmp") if (work / n).exists()]
        if other:
            from app.isl.models.index import IndexMissingError

            raise IndexMissingError(
                f"{folder} is missing but {', '.join(str(o) for o in other)} exists: fit-fusion and evaluation need "
                "the training-only index; run the 'index' step (scripts/phase2_train.sh --force index,... or "
                "`python -m app.isl.cli build-index` without --all-include). index.old / index.tmp are left by an "
                "interrupted index save")
    if with_index and enc is not None and (folder / "meta.json").exists():
        if for_app:
            idx = EmbeddingIndex.load(folder, expect_encoder_version=enc.version, encoder_path=enc.path)
        else:  # the leak guard checks the encoder SHA-256 itself (LeakageError, also for an index that records none)
            idx = EmbeddingIndex.load(folder, expect_encoder_version=enc.version)
            idx.check_for_evaluation(enc.path, where=str(folder))
    if fusion is None:
        # no fusion.json: uncalibrated defaults, uncertified (every policy confirm-only, docs 15.25)
        fusion = (FusionParams.load(work / "fusion.json") if (work / "fusion.json").exists()
                  else FusionParams(profile=profile, certification={}))
    return IslRecognizer("isl", fcfg, vocab, classifier=clf, encoder=enc, index=idx, fusion=fusion, profile=profile)


def evaluate(work: Path, data, fcfg: FeatureConfig, smoke: bool, part: str = "test", vocab_path: Path | None = None) -> dict:
    from app.isl.evaluation.protocols import e1_classifier, e2_retrieval, e3_decisions, e6_latency
    from app.isl.evaluation.report import markdown_table, write_results
    from app.isl.training.fit_fusion import collect_evidence

    from app.isl.training.fit_fusion import check_index_for_fitting

    rec = load_recognizer(work, fcfg, vocab_path, profile=data.profile)
    clips = data.include_part(part)
    check_index_for_fitting(rec, clips, data)  # no index row of a clip being measured; this split's clips (15.22)
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
               vocab_path: Path | None = None, allow_profile_mix: bool = False) -> dict:
    """Assemble the pack folder from a work folder. ``allow_profile_mix`` (opt-in, docs/ISL_DESIGN.md 15.44): a
    component built under another data profile that is licence-compatible with this one (a commercial-safe encoder
    or classifier in a research pack, never the reverse) is recorded in ``pack.json`` ``mixed_profiles``, which is
    what lets the app load it; an incompatible one is refused. Without it nothing changes: such a pack builds as
    before and the app refuses it (``ProfileMismatch``)."""
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
    # the shipped index: <work>/index_final (every reference, validation and test clips too: fine for the app, which
    # is not measured with it), else the training-only <work>/index. Never one embedded with another encoder file.
    # (an index_final older than index is not shipped: see shipped_index)
    index_info = None
    src_index, index_note = shipped_index(work)
    if index_note:
        print(f"build-pack: {index_note}", file=sys.stderr)
    if src_index is not None:
        from app.isl.models.index import EmbeddingIndex, IndexMeta

        imeta = IndexMeta.from_json(json.loads((src_index / "meta.json").read_text()))
        if imeta.encoder_sha256 and (work / "encoder.pt").exists() and imeta.encoder_sha256 != _sha(work / "encoder.pt"):
            # same message and type as the app's loader would raise
            EmbeddingIndex.load(src_index, encoder_path=work / "encoder.pt")
        shutil.copytree(src_index, tmp / "index")
        files["index/embeddings.f16.npy"] = _sha(tmp / "index" / "embeddings.f16.npy")
        index_info = {"built_from": src_index.name, "contains_eval": imeta.contains_eval,
                      "encoder_sha256": imeta.encoder_sha256 or None}
        if imeta.row_filters:  # e.g. WSLP twins of ISLRTC rows left out (docs 15.32)
            index_info["row_filters"] = imeta.row_filters
        if index_note:
            index_info["note"] = index_note
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
            # a variant concept (``alias_of`` its word, docs 15.36) shows its word's Hindi
            var_of: dict[str, list[str]] = {}
            for r in Vocabulary.load(tmp / "vocabulary.jsonl").rows:
                if r.get("alias_of") and r.get("concept_id"):
                    var_of.setdefault(r["alias_of"], []).append(r["concept_id"])
            have_hi = set()
            for ln in src:
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get("concept_id") in ids:
                    dst.write(ln if ln.endswith("\n") else ln + "\n")
                    have_hi.add(row["concept_id"])
                for v in var_of.get(row.get("concept_id"), ()):
                    if v not in have_hi:
                        dst.write(json.dumps({**row, "concept_id": v, "alias_of": row["concept_id"]},
                                             ensure_ascii=False) + "\n")
                        have_hi.add(v)
    write_spelling_lexicon(build_spelling_lexicon(Vocabulary.load(tmp / "vocabulary.jsonl").rows),
                           tmp / "spelling_lexicon.txt")
    enc_version = None
    class_spaces = {}  # the components' class spaces when their meta records one (docs 15.32; older: none)
    if (work / "encoder.pt").exists():
        import torch

        enc_meta = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)["meta"]
        enc_version = enc_meta["encoder_version"]
        if enc_meta.get("class_space"):
            class_spaces["encoder"] = enc_meta["class_space"]
    if (work / "classifier.pt").exists():
        import torch

        cls_meta = torch.load(work / "classifier.pt", map_location="cpu", weights_only=False).get("meta") or {}
        if cls_meta.get("class_space"):
            class_spaces["classifier"] = cls_meta["class_space"]
    if (work / "fusion.json").exists():
        # a fusion.json fitted on another classifier / encoder class space is never shipped with this one (docs
        # 15.33); one fitted before 15.33 records nothing and passes as before
        from app.isl.training.common import fusion_fit_space, fusion_space_mismatch

        fitted_on = FusionParams.load(work / "fusion.json").fitted_on
        bad = fusion_space_mismatch(fitted_on, fusion_fit_space(
            cls_meta if (work / "classifier.pt").exists() else None,
            enc_meta if (work / "encoder.pt").exists() else None,
            imeta.row_filters if src_index is not None else None)) if fitted_on else []
        if bad:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"{work / 'fusion.json'} was fitted on another class space ({', '.join(bad)}) than the "
                             "components of this pack: run fit-fusion (and evaluate) again before build-pack")
    # components built under another data profile (docs 15.44): recorded only on request and only when compatible
    comp_prof = {"classifier": cls_meta.get("profile") if (work / "classifier.pt").exists() else None,
                 "encoder": enc_meta.get("profile") if (work / "encoder.pt").exists() else None,
                 "index": imeta.profile if src_index is not None else None,
                 "fusion": FusionParams.load(work / "fusion.json").profile if (work / "fusion.json").exists() else None}
    other = {k: v for k, v in comp_prof.items() if v and v != prof.name}
    # an encoder warm-started from another work folder (docs 15.46) is a component of THIS profile (trained here: its
    # meta.profile is this one), so it is not a mixed-profile component; its warm start must still be licence-
    # compatible with the pack (commercial-safe -> research, never the reverse), and its lineage is recorded
    enc_ws = enc_meta.get("warm_start") if (work / "encoder.pt").exists() else None
    if isinstance(enc_ws, dict):
        from app.isl.profiles import component_profile_allowed

        if not component_profile_allowed(enc_ws.get("profile"), prof.name):
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"refusing to build a {prof.name} pack: its encoder was warm-started from "
                             f"{enc_ws.get('path')} of profile {enc_ws.get('profile')!r}, which is not licence-"
                             "compatible with it (docs 15.44, 15.46)")
    mixed = {}
    if other and allow_profile_mix:
        from app.isl.profiles import component_profile_allowed

        bad = {k: v for k, v in other.items() if not component_profile_allowed(v, prof.name)}
        if bad:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"refusing to build a {prof.name} pack with components of another data profile: {bad} "
                             "(only a commercial-safe component in a research pack is licence-compatible)")
        fname = {"classifier": "classifier.pt", "encoder": "encoder.pt", "fusion": "fusion.json",
                 "index": "index/embeddings.f16.npy"}  # the app checks each recorded SHA-256 (docs 15.45)
        for k, v in other.items():
            mixed[k] = {"profile": v, **({"sha256": files[fname[k]]} if fname.get(k) in files else {}),
                        "reason": f"licence-compatible: every source the {v} profile reads is allowed in {prof.name}; "
                                  "recorded by build-pack --allow-profile-mix (docs/ISL_DESIGN.md 15.44)"}
    elif other:
        print(f"build-pack: components of another data profile {other}: the app will refuse this pack "
              "(ProfileMismatch) unless it is built with --allow-profile-mix (docs 15.44)", file=sys.stderr)
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
    if index_info is not None:
        man["components"]["index"] = {**man["components"]["index"], **index_info}
    for comp, cs in class_spaces.items():
        man["components"][comp] = {**(man["components"].get(comp) or {}), "class_space": cs}
    if isinstance(enc_ws, dict):  # docs 15.46: the encoder's warm-start lineage travels with the pack
        ss = enc_ws.get("self_supervised") or {}
        man["components"]["encoder"] = {**(man["components"].get("encoder") or {}), "warm_start": {
            "path": enc_ws.get("path"), "sha256": enc_ws.get("sha256"), "profile": enc_ws.get("profile"),
            "encoder_version": enc_ws.get("encoder_version"),
            "pretrain_list": ss.get("list_file"), "pretrain_list_sha256": ss.get("list_sha256_recorded")
            or ss.get("list_sha256"),
            "note": (f"a {prof.name} encoder (trained on this profile's data) warm-started from a "
                     f"{enc_ws.get('profile')} encoder: not a mixed-profile component; the warm start is licence-"
                     "compatible with this profile (docs 15.44, 15.46)")}}
    # the variant-audit file whose relinks trained this pack (docs 15.36-37): shipped numbers trace back to it
    from app.isl.evaluation.runners import variant_links_summary
    from app.isl.training.multisource import read_link_settings

    vl = variant_links_summary(read_link_settings(work))
    if vl:
        man["variant_links"] = vl
    if mixed:
        man["mixed_profiles"] = mixed
    (tmp / "pack.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    from app.packs.registry import reset_registries

    reset_registries()
    return {"pack": str(out), "id": man["id"], "files": sorted(files), "size_mb": round(size / 1e6, 1)}
