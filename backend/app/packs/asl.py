"""ASL as a language pack: wraps the existing active-experiment registry, unchanged."""

from __future__ import annotations

from pathlib import Path

from app.inference import registry as experiment_registry
from app.inference.recognizer import ClassifierRecognizer
from app.packs.registry import PackInfo
from app.packs.schema import PackManifest
from app.packs.vocabulary import Vocabulary


class AslExperimentPack:
    def __init__(self, manifest: PackManifest, folder: Path):
        self.manifest = manifest
        self.folder = folder

    def predictor(self):
        """(predictor, info, problem) of the active experiment - exactly ``registry.load_active()``."""
        return experiment_registry.load_active()

    @property
    def info(self) -> PackInfo:
        m = self.manifest
        try:
            p, _, problem = experiment_registry.load_active()
        except Exception as e:  # noqa: BLE001 - e.g. unreadable database
            p, problem = None, f"{type(e).__name__}: {e}"
        return PackInfo(m.id, m.kind, m.sign_language, m.standard, m.display_name, m.output_languages,
                        frozenset(m.capabilities), m.profile, m.version, [x.model_dump() for x in m.licences],
                        "ready" if p is not None else "missing_models", problem, str(self.folder))

    def recognizer(self):
        p, info, problem = experiment_registry.load_active()
        if p is None:
            return None, None, problem
        return ClassifierRecognizer(p, self.manifest.id, lexicon=self.lexicon()), info, None

    def lexicon(self):
        """Hindi for the WLASL100 glosses (``lexicon_hi.jsonl`` next to pack.json; concept id = gloss), or None."""
        from app.isl.text.lexicon import HindiLexicon

        rel = self.manifest.lexicons.get("hi")
        path = self.folder / rel if rel else None
        if path is None or not path.exists():
            return None
        key = path.stat().st_mtime_ns
        cached = getattr(self, "_lexicon", None)
        if cached is None or cached[0] != key:
            self._lexicon = (key, HindiLexicon(path, None))
        return self._lexicon[1]

    def vocabulary(self) -> Vocabulary:
        p, _, _ = experiment_registry.load_active()
        return Vocabulary.from_class_names(list(p.class_names) if p is not None else [])

    def results(self) -> dict | None:
        return None
