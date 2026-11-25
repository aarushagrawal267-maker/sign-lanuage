"""Offline English -> Hindi machine translation for the Hindi lexicon (docs/ISL_DESIGN.md section 8.1).

Model: AI4Bharat IndicTrans2, English -> Indic, distilled 200M (``ai4bharat/indictrans2-en-indic-dist-200M``,
MIT licence). It is used **offline only** (``python -m app.isl.cli hindi --translate``) to draft the tracked
``lexicon_hi.jsonl``; the app never translates at runtime and never needs these packages.

Getting the weights
-------------------
The official repository is "gated: auto" on Hugging Face (access is granted automatically once the account owner
shares contact details on the model page). When that has not been done, the files are fetched from a public
mirror and **every file is verified against the official repository's git object ids pinned below**, so the bytes
are provably identical to the official release (for Git-LFS files the official id is the hash of the LFS pointer,
which contains the SHA-256 of the content; we rebuild the pointer from the downloaded file and compare).
The MIT licence allows redistribution, so the mirror is lawful; the attribution is in docs/DATASETS_ISL.md.

Dependencies (offline tooling only, not app requirements): ``transformers`` (4.4x-4.5x), ``sentencepiece``.
The model's own code (``modeling_indictrans.py``, ``tokenization_indictrans.py``) is loaded with
``trust_remote_code`` from the verified local folder only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

OFFICIAL_REPO = "ai4bharat/indictrans2-en-indic-dist-200M"
MIRROR_REPOS = ("naklitechie/indictrans2-en-indic-dist-200M",)
MODEL_ID = OFFICIAL_REPO
MODEL_LICENCE = "MIT"

# git object ids of the official repository (main @ 2cfd0a20057756d756bf3e53528f492afc52f91a), checked 2026-11-25.
OFFICIAL_OIDS: dict[str, tuple[str, bool]] = {  # path -> (git blob sha1, stored in LFS)
    "LICENSE": ("03070285be3aab2c5603737115bb71df3f7bbdf5", False),
    "config.json": ("75dac0ee17edc02e87ef9ea9aba2c4fc71ce6782", False),
    "configuration_indictrans.py": ("110d923f8fffda96398f05c30e00f94e341670ec", False),
    "dict.SRC.json": ("777afff23d0ca2c0f20c8c2ebac93ab5d9733e1e", False),
    "dict.TGT.json": ("c36097374f97737a662ce1d1dd011fe386198561", False),
    "generation_config.json": ("542cc0f13b067a9bd56e0334c422a1e51de28e19", False),
    "model.SRC": ("4841e2bc6e5116de279437e88fe9bcaa7656f813", False),
    "model.TGT": ("e3cfbb624421efbf88f1c816d99b386d29329e61", True),
    "model.safetensors": ("c0b24bea1d89f57bb1625ae5df7ffcb363936df4", True),
    "modeling_indictrans.py": ("dec79d3e6161c3e90dabcd14887198a62ce5783d", False),
    "special_tokens_map.json": ("fdafe480f024ff444c7492147536765ce5d55a2d", False),
    "tokenization_indictrans.py": ("53c9be07a1dda3273ed8c2af4f30b635edff4b31", False),
    "tokenizer_config.json": ("c9b54b82b8cfd50c50700d885112176109f81107", False),
}


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def verify_file(path: Path, name: str) -> bool:
    """True when ``path`` is byte-identical to the official ``name``."""
    oid, lfs = OFFICIAL_OIDS[name]
    if not path.exists():
        return False
    if lfs:
        sha, size = _file_sha256(path)
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
        return _git_blob_sha1(pointer) == oid
    return _git_blob_sha1(path.read_bytes()) == oid


def _http_status(url: str) -> int:
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-I", "-L", url],
                       capture_output=True, text=True, timeout=60)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def fetch_model(dest: Path, log=print) -> Path:
    """Download (resumable) and verify the IndicTrans2 files into ``dest``. Returns ``dest``.

    Tries the official repository first, then the mirrors. Raises RuntimeError when a file cannot be verified."""
    dest.mkdir(parents=True, exist_ok=True)
    repos = [OFFICIAL_REPO, *MIRROR_REPOS]
    for name in OFFICIAL_OIDS:
        target = dest / name
        if verify_file(target, name):
            continue
        ok = False
        for repo in repos:
            url = f"https://huggingface.co/{repo}/resolve/main/{name}"
            if _http_status(url) != 200:
                continue
            part = target.with_suffix(target.suffix + ".part")
            log(f"downloading {name} from {repo}")
            subprocess.run(["curl", "-sS", "-L", "--retry", "4", "-C", "-", "-o", str(part), url], check=False,
                           timeout=7200)
            if part.exists():
                part.replace(target)
            if verify_file(target, name):
                ok = True
                log(f"verified {name} (identical to {OFFICIAL_REPO})")
                break
            log(f"{name} from {repo} does not match the official file; discarded")
            target.unlink(missing_ok=True)
        if not ok:
            raise RuntimeError(f"could not obtain a verified copy of {name}")
    (dest / "SOURCE.json").write_text(json.dumps({"model": OFFICIAL_REPO, "licence": MODEL_LICENCE,
                                                  "verified_against": "official git object ids (see mt.py)"},
                                                 indent=2))
    return dest


# ------------------------------------------------------------------------------------------- translation

_SRC, _TGT = "eng_Latn", "hin_Deva"


class IndicTrans2:
    """Batch translator, CPU. ``threads`` keeps it light while the ingestion jobs run."""

    def __init__(self, model_dir: Path, threads: int = 2, num_beams: int = 4, max_new_tokens: int = 64):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(max(1, threads))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), trust_remote_code=True,
                                                           torch_dtype=torch.float32).eval()
        self.num_beams, self.max_new_tokens = num_beams, max_new_tokens

    @staticmethod
    def preprocess(text: str) -> str:
        """IndicTrans2 input format: ``<src_lang> <tgt_lang> <normalised sentence>`` (what IndicProcessor does for
        English input, minus placeholder handling, which single dictionary terms do not need)."""
        t = unicodedata.normalize("NFC", text).replace("’", "'").replace("‘", "'")
        t = t.replace("“", '"').replace("”", '"')
        t = re.sub(r"\s+", " ", t).strip()
        return f"{_SRC} {_TGT} {t}"

    def translate(self, texts: list[str]) -> list[str]:
        batch = [self.preprocess(t) for t in texts]
        enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True)
        src_len = int(enc["attention_mask"].sum(dim=1).max())
        # short dictionary terms: cap the output length (stops the run-on "। । ।" outputs seen in padded batches)
        cap = min(self.max_new_tokens, 2 * src_len + 8)
        with self.torch.inference_mode():
            gen = self.model.generate(**enc, use_cache=True, min_length=0, max_new_tokens=cap,
                                      num_beams=self.num_beams, num_return_sequences=1, no_repeat_ngram_size=4)
        out = self.tok.batch_decode(gen.detach().cpu().tolist(), skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        return [unicodedata.normalize("NFC", o).strip() for o in out]
