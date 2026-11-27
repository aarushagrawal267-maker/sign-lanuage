"""File-based ISL pack (``kind: "isl-hybrid"``): builds an ``IslRecognizer`` lazily from the pack folder.

Missing components degrade gracefully (docs/ISL_DESIGN.md 4.1): no classifier -> retrieval only; no index ->
classifier only; neither -> status ``missing_models`` (only the user's own signs can be recognised). Components
built under different data profiles are refused (ADR-ISL-6).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.core.config import REPO_ROOT, get_settings
from app.features.spec import FeatureConfig
from app.packs.registry import PackInfo
from app.packs.schema import PackManifest
from app.packs.vocabulary import Vocabulary

log = logging.getLogger(__name__)

BUILD_HINT = ("Build the ISL models with 'python -m app.isl.cli smoke' (quick test) or the phase-2 commands in "
              "docs/ISL_DESIGN.md section 11, then restart the backend.")


class ProfileMismatch(ValueError):
    pass


MIX_HINT = ("a commercial-safe component may serve in a research pack only when the pack records it: rebuild with "
            "'python -m app.isl.cli build-pack --profile research --allow-profile-mix' (docs/ISL_DESIGN.md 15.44)")


def _file_sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_component_profiles(manifest: PackManifest, profiles: dict[str, str | None],
                             files: dict[str, Path | None] | None = None) -> dict[str, dict]:
    """ADR-ISL-6, never mix data profiles, with one explicit exception (docs/ISL_DESIGN.md 15.44): a component built
    under another profile is accepted only when (1) it is licence-compatible with the pack's profile
    (``profiles.component_profile_allowed``: a commercial-safe component in a research pack, never the reverse) and
    (2) the pack manifest records it in ``mixed_profiles.<component>`` with that profile (and, when recorded, the
    file's SHA-256). Returns the accepted records; raises ``ProfileMismatch`` otherwise."""
    from app.isl.profiles import component_profile_allowed

    prof = manifest.profile
    declared = getattr(manifest, "mixed_profiles", None) or {}
    accepted, refused = {}, {}
    for k, v in profiles.items():
        if not v or v == prof or k == "pack":
            continue
        rec = declared.get(k) if isinstance(declared, dict) else None
        if not component_profile_allowed(v, prof):
            refused[k] = f"{v} (never allowed in a {prof} pack)"
        elif not isinstance(rec, dict) or rec.get("profile") != v:
            refused[k] = f"{v} (licence-compatible, but not recorded in the pack manifest)"
        elif rec.get("sha256") and (files or {}).get(k) is not None and Path(files[k]).is_file() \
                and _file_sha256(files[k]) != rec["sha256"]:
            refused[k] = f"{v} (the manifest's allowance was recorded for another {k} file)"
        else:
            accepted[k] = {"profile": v, **{x: rec[x] for x in ("sha256", "reason") if x in rec}}
    if refused:
        hint = f"; {MIX_HINT}" if any("not recorded" in r for r in refused.values()) else ""
        raise ProfileMismatch(f"components built with another data profile than the pack ({prof}): {refused}{hint}")
    return accepted


class IslPack:
    def __init__(self, manifest: PackManifest, folder: Path):
        self.manifest = manifest
        self.folder = Path(folder)
        self._lock = threading.Lock()
        self._cache: dict = {}
        self._vocab: tuple | None = None
        self._lexicon: tuple | None = None
        self._store = None

    # ----------------------------------------------------------------- files
    def _path(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = (self.folder / rel).resolve()
        return p if self.folder.resolve() in p.parents or p == self.folder.resolve() else None

    def component_paths(self) -> dict[str, Path | None]:
        c = self.manifest.components or {}
        fs = c.get("fingerspelling") or {}
        return {
            "classifier": self._path((c.get("classifier") or {}).get("checkpoint")),
            "encoder": self._path((c.get("encoder") or {}).get("checkpoint")),
            "index": self._path((c.get("index") or {}).get("dir")),
            "letters": self._path(fs.get("checkpoint")),
            "spelling_lexicon": self._path(fs.get("lexicon")),
            "fusion": self._path(c.get("fusion") if isinstance(c.get("fusion"), str) else None),
            "vocabulary": self._path(self.manifest.vocabulary),
            "lexicon_hi": self._lexicon_path(),
            "results": self._path(self.manifest.results),
        }

    def _lexicon_path(self) -> Path | None:
        """The pack's Hindi lexicon. A local build (artifacts/packs/<id>) made before the lexicon existed falls back
        to the tracked copy in models/packs/<id>, which is keyed by the same concept ids; rows of concepts that
        are not in this pack's vocabulary are simply never looked up."""
        rel = self.manifest.lexicons.get("hi")
        p = self._path(rel)
        if p is None or p.exists():
            return p
        tracked = REPO_ROOT / "models" / "packs" / self.manifest.id
        if tracked.resolve() == self.folder.resolve():
            return p
        alt = (tracked / rel).resolve()
        return alt if tracked.resolve() in alt.parents and alt.exists() else p

    def present(self) -> dict[str, bool]:
        out = {}
        for k, p in self.component_paths().items():
            if k == "index":
                out[k] = p is not None and (p / "meta.json").exists()
            else:
                out[k] = p is not None and p.exists()
        return out

    @property
    def feature_config(self) -> FeatureConfig:
        return FeatureConfig.from_dict(self.manifest.feature_config or {"canonical_hand": True})

    def status(self) -> tuple[str, str | None]:
        pr = self.present()
        if not pr["vocabulary"]:
            return "error", f"the ISL pack in {self.folder} has no vocabulary file"
        if pr["classifier"] or (pr["encoder"] and pr["index"]):
            if "error" in self._cache:
                return "error", self._cache["error"]
            return "ready", None
        return "missing_models", ("The ISL models are not built yet, so only your own signs (My signs) can be "
                                  "recognised. " + BUILD_HINT)

    @property
    def info(self) -> PackInfo:
        m = self.manifest
        st, msg = self.status()
        caps = set(m.capabilities)
        pr = self.present()
        if not pr["letters"] and self.alphabet().model() is None:
            caps.discard("spelling")
        if not (pr["encoder"] and pr["index"]):
            caps.discard("retrieval")
        return PackInfo(m.id, m.kind, m.sign_language, m.standard, m.display_name, m.output_languages,
                        frozenset(caps), m.profile, m.version, [x.model_dump() for x in m.licences], st, msg,
                        str(self.folder), {"components": pr})

    # ----------------------------------------------------------- vocabulary
    def vocabulary(self) -> Vocabulary:
        p = self.component_paths()["vocabulary"]
        if p is None or not p.exists():
            return Vocabulary([])
        key = p.stat().st_mtime_ns
        if self._vocab is None or self._vocab[0] != key:
            self._vocab = (key, Vocabulary.load(p))
        return self._vocab[1]

    def lexicon(self):
        from app.isl.text.lexicon import HindiLexicon

        base = self.component_paths()["lexicon_hi"]
        local = get_settings().data_dir / "isl" / "lexicon_hi.local.jsonl"
        key = tuple(x.stat().st_mtime_ns if x is not None and x.exists() else 0 for x in (base, local))
        if self._lexicon is None or self._lexicon[0] != key:
            self._lexicon = (key, HindiLexicon(base, local))
        return self._lexicon[1]

    def my_signs(self):
        from app.isl.mysigns.store import MySignsStore

        root = get_settings().my_signs_dir
        if self._store is None or self._store.dir.parent != root:
            self._store = MySignsStore(root, self.manifest.id)
        return self._store

    def results(self) -> dict | None:
        p = self.component_paths()["results"]
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except ValueError:
            return None

    def fusion_params(self):
        from app.isl.models.fusion import FusionParams

        p = self.component_paths()["fusion"]
        if p is not None and p.exists():
            return FusionParams.load(p)
        # no fusion.json: uncalibrated defaults, uncertified (every policy confirm-only, docs/ISL_DESIGN.md 15.25)
        return FusionParams(profile=self.manifest.profile, certification={})

    # ----------------------------------------------------------- recogniser
    def _key(self) -> tuple:
        out = []
        for _k, p in sorted(self.component_paths().items()):
            if p is None or not p.exists():
                out.append(0)
            elif p.is_dir():
                out.append((p / "meta.json").stat().st_mtime_ns if (p / "meta.json").exists() else 0)
            else:
                out.append(p.stat().st_mtime_ns)
        alpha = self.alphabet().path
        return (self.manifest.version, *out, str(get_settings().my_signs_dir),
                alpha.stat().st_mtime_ns if alpha.exists() else 0)

    def _load(self):
        from app.isl.fingerspell.speller import Speller
        from app.isl.models.encoder import EncoderPredictor
        from app.isl.models.index import EmbeddingIndex
        from app.isl.recognizer import IslRecognizer
        from app.models.checkpoint import Predictor

        paths, pr = self.component_paths(), self.present()
        prof = self.manifest.profile
        vocab = self.vocabulary()
        fusion = self.fusion_params()
        classifier = Predictor(paths["classifier"], "cpu") if pr["classifier"] else None
        encoder = EncoderPredictor(paths["encoder"], "cpu") if pr["encoder"] else None
        index = None
        if encoder is not None and pr["index"]:
            want = (self.manifest.components.get("index") or {}).get("encoder_version") or encoder.version
            # the index of a pack may contain evaluation clips (index_final): fine for the app; but it must have been
            # embedded with this very encoder file when it records one (docs 15.21)
            index = EmbeddingIndex.load(paths["index"], expect_encoder_version=want, encoder_path=paths["encoder"])
            if index.meta.encoder_version != encoder.version:
                raise ProfileMismatch(f"index encoder {index.meta.encoder_version!r} != encoder {encoder.version!r}")
        speller = self._speller(vocab, paths, pr, Speller)
        # ADR-ISL-6: never mix data profiles
        profiles = {"pack": prof}
        if classifier is not None:
            profiles["classifier"] = classifier.meta.get("profile")
        if encoder is not None:
            profiles["encoder"] = encoder.meta.get("profile")
        if index is not None:
            profiles["index"] = index.meta.profile
        if fusion.profile:
            profiles["fusion"] = fusion.profile
        if speller is not None and pr["letters"]:  # the trained letter model (not the user's own alphabet)
            profiles["letters"] = (getattr(speller.model, "meta", None) or {}).get("profile")
        # a recorded, licence-compatible exception is accepted (docs 15.44); anything else raises as before
        # every file an allowance can be recorded for is checked against its SHA-256 (docs 15.45)
        mixed = check_component_profiles(self.manifest, profiles, {
            "classifier": paths["classifier"], "encoder": paths["encoder"], "letters": paths["letters"],
            "fusion": paths["fusion"],
            "index": None if paths["index"] is None else paths["index"] / "embeddings.f16.npy"})
        rec = IslRecognizer(self.manifest.id, self.feature_config, vocab, lexicon=self.lexicon(), classifier=classifier,
                            encoder=encoder, index=index, speller=speller, fusion=fusion, my_signs=self.my_signs(),
                            profile=prof)
        info = {"language": self.manifest.id, "name": f"{self.manifest.display_name.get('en', self.manifest.id)}",
                "version": self.manifest.version, "profile": prof, "num_classes": len(vocab),
                "threshold": rec.recommended_threshold, "seq_len": self.feature_config.seq_len,
                "feature_config": self.feature_config.to_dict(), "experiment_id": None,
                "components": {"classifier": None if classifier is None else {
                                   "classes": len(classifier.class_names), "split": classifier.meta.get("split"),
                                   "val_top1": classifier.meta.get("val_top1"), "smoke": classifier.meta.get("smoke", False)},
                               "encoder": None if encoder is None else {"version": encoder.version, "dim": encoder.dim,
                                                                         "smoke": encoder.meta.get("smoke", False)},
                               "index": None if index is None else {"rows": len(index), "concepts": len(index.concepts)},
                               "spelling": None if speller is None else {"experimental": speller.experimental,
                                                                         "lexicon": len(speller.decoder.trie)},
                               "fusion": {"fitted": fusion.fitted, "version": fusion.version,
                                          # per policy: rule, certified worst-source precision, floor (docs 15.25)
                                          "certification": {pol: fusion.certification_info(pol)
                                                            for pol in ("strict", "balanced", "lenient")}}},
                "is_synthetic": False}
        if mixed:
            info["mixed_profiles"] = mixed
        return rec, info

    def alphabet(self):
        from app.isl.fingerspell.prototypes import AlphabetStore

        return AlphabetStore(get_settings().my_signs_dir / self.manifest.id)

    def _speller(self, vocab, paths, pr, Speller):
        """Trained letter model when the pack has one, else the user's alphabet calibration (prototypes)."""
        from app.isl.fingerspell.decoder import LexiconTrie, SpellDecoder
        from app.isl.fingerspell.lexicon import build_spelling_lexicon
        from app.isl.fingerspell.model import LetterPredictor

        if pr["letters"]:
            letters = LetterPredictor(paths["letters"])
        else:
            letters = self.alphabet().model()
            if letters is None:
                return None
        cw = {}
        for r in vocab.rows:
            if r.get("kind", "word") in ("word", "number", "multiword") and r.get("headword"):
                cw.setdefault(r["headword"], r["concept_id"])
        lex = paths["spelling_lexicon"]
        if lex is not None and lex.exists():
            trie = LexiconTrie.from_lines(lex.read_text(encoding="utf-8").splitlines())
        else:
            trie = LexiconTrie(build_spelling_lexicon(vocab.rows))
        return Speller(letters, SpellDecoder(trie), cw)

    def recognizer(self):
        st, msg = self.status()
        if st == "error" and "error" not in self._cache:
            return None, None, msg
        key = self._key()
        with self._lock:
            if self._cache.get("key") != key:
                self._cache = {"key": key}
                try:
                    self._cache["rec"], self._cache["info"] = self._load()
                except Exception as e:  # noqa: BLE001 - a broken component gives a clear message, never a 500
                    m = f"The ISL pack could not be loaded ({type(e).__name__}: {str(e)[:300]}). {BUILD_HINT}"
                    log.error("%s [%s]", m, self.folder)
                    self._cache["error"] = m
            if "error" in self._cache:
                return None, None, self._cache["error"]
            rec, info = self._cache["rec"], self._cache["info"]
        problem = None if rec.has_global_model else msg
        return rec, info, problem
