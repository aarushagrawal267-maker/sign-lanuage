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


def workdir(profile: str) -> Path:
    d = get_settings().artifacts_dir / "isl" / get_profile(profile).name
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
    doc = write_results(work / "results.json", protocols, profile=data.profile, smoke=smoke, data=data.summary(),
                        extra={"per_class": per_class, "partition": part})
    (work / "results.md").write_text(markdown_table(doc) + "\n")
    return doc


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_pack(work: Path, profile: str, fcfg: FeatureConfig, out: Path | None = None, smoke: bool = False,
               vocab_path: Path | None = None) -> dict:
    from app.isl.fingerspell.lexicon import build_spelling_lexicon, write_spelling_lexicon
    from app.packs.vocabulary import Vocabulary

    prof = get_profile(profile)
    out = out or pack_dir(prof.name)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files = {}
    for name in ("classifier.pt", "encoder.pt", "letters.pt", "fusion.json", "results.json"):
        if (work / name).exists():
            shutil.copy2(work / name, tmp / name)
            files[name] = _sha(tmp / name)
    if (work / "index" / "meta.json").exists():
        shutil.copytree(work / "index", tmp / "index")
        files["index/embeddings.f16.npy"] = _sha(tmp / "index" / "embeddings.f16.npy")
    shutil.copy2(vocab_path or PACK_DIR / "vocabulary.jsonl", tmp / "vocabulary.jsonl")
    if (PACK_DIR / "lexicon_hi.jsonl").exists():
        shutil.copy2(PACK_DIR / "lexicon_hi.jsonl", tmp / "lexicon_hi.jsonl")
    write_spelling_lexicon(build_spelling_lexicon(Vocabulary.load(tmp / "vocabulary.jsonl").rows),
                           tmp / "spelling_lexicon.txt")
    enc_version = None
    if (work / "encoder.pt").exists():
        import torch

        enc_version = torch.load(work / "encoder.pt", map_location="cpu", weights_only=False)["meta"]["encoder_version"]
    base = json.loads((PACK_DIR / "pack.json").read_text())
    licences = [{"source": "include", "licence": "CC-BY-4.0", "commercial": True},
                {"source": "islrtc", "licence": "GODL-India", "commercial": True}]
    if not prof.commercial:
        licences += [{"source": "cislr", "licence": "AFL-3.0 (pending review)", "commercial": False},
                     {"source": "isign", "licence": "CC-BY-NC-SA-4.0", "commercial": False}]
    man = {**base, "id": PACK_IDS[prof.name], "version": PACK_VERSION + ("-smoke" if smoke else ""),
           "profile": prof.name, "feature_config": fcfg.to_dict(), "licences": licences,
           "display_name": base["display_name"] if prof.commercial else
           {"en": "Indian Sign Language (ISL, research models)", "hi": "भारतीय सांकेतिक भाषा (शोध)"},
           "note": ("SMOKE BUILD on partial data: proves the pipeline, not usable accuracy. " if smoke else "")
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
