"""A tiny synthetic ISL "world" for model tests: INCLUDE-like multi-signer clips, ISLRTC-like single clips,
a vocabulary and a split, written in the real on-disk formats (no real data, no network)."""

from __future__ import annotations

import json
from pathlib import Path

from app.dataset.synthetic import make_sequence
from app.isl.paths import isl_paths
from tests.isl.helpers import write_manifest

CORE = ["apple", "book", "cat", "dog"]  # INCLUDE + ISLRTC
DICT_ONLY = ["egg", "fish"]  # ISLRTC only


def concept(w: str) -> str:
    return f"isl.{w}"


def make_world(root: Path, signers: int = 6, seed: int = 0) -> dict:
    P = isl_paths(root / "isl")
    (P.landmarks / "include").mkdir(parents=True, exist_ok=True)
    (P.landmarks / "islrtc").mkdir(parents=True, exist_ok=True)
    inc_rows, isl_rows, vocab, assign = [], [], [], {}
    words = CORE + DICT_ONLY
    for ci, w in enumerate(words):
        variants = []
        name = w.capitalize()
        raw = make_sequence(ci, signer=50, n_frames=45, fps=12.0, n_classes=len(words))
        fn = f"{name[0]}__{name}.npz"
        raw.save(P.landmarks / "islrtc" / fn)
        isl_rows.append({"id": f"islrtc/{name[0]}/{name}.mp4", "path": f"data/isl/landmarks/islrtc/{fn}",
                         "n_frames": len(raw), "hands_frac": 0.9, "fps": 12.0, "stop": "eof", "ok": True})
        variants.append({"variant_id": f"islrtc:{name[0]}/{name}", "source": "islrtc",
                         "clips": [f"islrtc/{name[0]}/{name}.mp4"]})
        if w in CORE:
            word_dir = f"Things/{ci + 1}. {name}"
            for s in range(signers):
                r = make_sequence(ci, signer=s, n_frames=40 + 2 * s, fps=12.0, n_classes=len(words))
                f = f"Things__{ci}__{s}.npz"
                r.save(P.landmarks / "include" / f)
                stem = f"MVI_{1000 + 10 * ci + s}"
                inc_rows.append({"id": f"include/Things/{ci + 1}._{name}/{stem}", "member": f"{word_dir}/{stem}.MOV",
                                 "path": f"data/isl/landmarks/include/{f}", "n_frames": len(r), "hands_frac": 0.9,
                                 "fps": 12.0, "ok": True})
                part = "train" if s < signers - 2 else ("val" if s == signers - 2 else "test")
                assign[f"include/Things/{ci + 1}._{name}/{stem}"] = part
            variants.append({"variant_id": "include:" + word_dir.replace(" ", "_"), "source": "include",
                             "clips_count": signers, "link": "exact"})
        vocab.append({"concept_id": concept(w), "headword": w, "display": {"en": w}, "kind": "word",
                      "category": "things", "variants": variants, "in_classifier": w in CORE, "indexable": True})
    write_manifest(P.manifest("include"), inc_rows)
    write_manifest(P.manifest("islrtc"), isl_rows)
    vpath = root / "vocabulary.jsonl"
    with vpath.open("w") as fh:
        for r in sorted(vocab, key=lambda r: r["concept_id"]):
            fh.write(json.dumps(r) + "\n")
    P.splits.mkdir(parents=True, exist_ok=True)
    (P.splits / "test-split.json").write_text(json.dumps({"name": "test-split", "kind": "session", "label": "synthetic",
                                                          "signer_independent": "yes", "assign": assign}))
    return {"paths": P, "vocab": vpath, "split": "test-split"}
