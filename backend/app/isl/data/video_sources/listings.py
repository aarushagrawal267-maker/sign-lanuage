"""Turn each source's metadata (CSV, zip central directory, Mendeley file list, HF rows) into work items.

Everything here is pure (no network) so it is unit-tested; ``scripts/ingest_isl_video.py`` fetches the metadata and
the engine (``engine.py``) processes the items. Item ids are stable across runs (resume) and never depend on the
order of a listing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Item:
    id: str  # "<prefix>/<stable key>"
    gloss: str | None  # None = unlabelled
    fetch: dict  # {"kind": "zip", "archive": url, "member": name} | {"kind": "url", "url": u} | {"kind": "hfrow", ...}
    suffix: str  # temp file suffix (.mp4, .jpg ...)
    meta: dict = field(default_factory=dict)  # extra manifest fields
    order: int = 0  # processing order (zip header offset for sequential range reads)
    out_name: str | None = None  # landmark file name (default: from the id)


def safe_name(s: str, n: int = 150) -> str:
    """File-name-safe, readable and unique: ``/`` -> ``__``, spaces -> ``_``; any other character outside
    ``[A-Za-z0-9._-]`` (brackets, Hindi letters) or an over-long name adds a short hash of the original."""
    base = s.replace("/", "__").replace(" ", "_")
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
    if len(out) > n or out != base:
        out = out[: n - 9] + "-" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
    return out.strip("-") or hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _stem(name: str) -> str:
    return PurePosixPath(name).stem


def _suffix(name: str) -> str:
    return PurePosixPath(name).suffix.lower()


def read_csv_text(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))


# ----------------------------------------------------------------------------------------------- WSLP

def wslp_items(csv_text: str, members: Iterable[tuple[str, int]], archive: str, split: str = "train",
               labelled: bool = True) -> list[Item]:
    """WR clips: ``<split>/<uid>.mp4`` + CSV ``,uid,word`` (train). Val/test CSVs carry only ``uid``."""
    words = {}
    for r in read_csv_text(csv_text):
        uid = (r.get("uid") or "").strip()
        if uid:
            words[uid] = (r.get("word") or "").strip() or None
    out = []
    for name, off in members:
        if _suffix(name) not in VIDEO_EXT:
            continue
        uid = _stem(name)
        gloss = words.get(uid) if labelled else None
        out.append(Item(
            id=f"wslp/{uid}", gloss=gloss, suffix=_suffix(name), order=off,
            fetch={"kind": "zip", "archive": archive, "member": name},
            meta={"uid": uid, "split": split, "labelled": gloss is not None, "member": name,
                  "in_csv": uid in words},
        ))
    return out


# ----------------------------------------------------------------------------------------------- CISLR

def cislr_items(rows: dict[str, dict], split: dict[str, str], members: Iterable[tuple[str, int]],
                archive: str) -> list[Item]:
    """Same ids / output names as ``app.isl.data.cislr.ingest_cislr`` so both writers share one manifest."""
    from app.isl.data.cislr import _safe

    out = []
    for name, off in members:
        if _suffix(name) not in VIDEO_EXT or "__MACOSX" in name:
            continue
        uid = _stem(name)
        row = rows.get(uid, {})
        gloss = row.get("word") or PurePosixPath(name).parent.name or None
        out.append(Item(
            id=f"cislr/{uid}", gloss=gloss, suffix=_suffix(name), order=off, out_name=_safe(uid) + ".npz",
            fetch={"kind": "zip", "archive": archive, "member": name},
            meta={"uid": uid, "category": row.get("category"), "member": name, "split": split.get(uid),
                  "labelled": bool(gloss)},
        ))
    return out


# ----------------------------------------------------------------------------------------------- ISL-Fingerspelling

def fingerspelling_items(letter_csv: str, timestamps_csv: str, files: Iterable[str], url_of) -> list[Item]:
    """One item per ``videos/<uid>.mp4``; the label is the letter sequence (authoritative per the dataset card) and
    the per-letter frame boundaries are kept in the manifest row (``letters``: letter, start_frame, end_frame in
    *segment* frames at the source fps; with ``step_fps=30`` our ``t_ms`` = frame * 1000 / fps)."""
    letters: dict[str, list[dict]] = defaultdict(list)
    for r in read_csv_text(letter_csv):
        letters[r["uid"]].append({"i": int(r["char_index"]), "letter": r["letter"],
                                  "start_frame": int(r["start_frame"]), "end_frame": int(r["end_frame"])})
    info: dict[str, dict] = {}
    for r in read_csv_text(timestamps_csv):
        info.setdefault(r["uid"], {"signer": r.get("signer"), "signer_split": r.get("signer_split"),
                                   "standard_split": r.get("standard_split"), "src_fps": float(r.get("fps") or 0) or None,
                                   "transcript": r.get("transcript"), "video_id": r.get("video_id")})
    out = []
    for path in sorted(files):
        if _suffix(path) not in VIDEO_EXT:
            continue
        uid = _stem(path)
        ls = sorted(letters.get(uid, []), key=lambda x: x["i"])
        text = "".join(x["letter"] for x in ls) if ls else (info.get(uid, {}).get("transcript") or None)
        meta = {"uid": uid, "labelled": bool(text), "category": "fingerspelling",
                "letters": [{k: x[k] for k in ("letter", "start_frame", "end_frame")} for x in ls]}
        meta.update(info.get(uid, {}))
        meta["group"] = meta.get("signer")
        out.append(Item(id=f"islfs/{uid}", gloss=text, suffix=_suffix(path),
                        fetch={"kind": "url", "url": url_of(path)}, meta=meta))
    return out


# ----------------------------------------------------------------------------------------------- ISL-CSLTR

def csltr_glosses(csv_text: str) -> dict[str, list[str]]:
    """``Sentence,SIGN GLOSSES`` -> {normalised sentence: [GLOSS, ...]}."""
    out = {}
    for r in read_csv_text(csv_text):
        vals = list(r.values())
        if len(vals) >= 2 and vals[0]:
            out[norm_sentence(vals[0])] = [g for g in str(vals[1]).split() if g]
    return out


def norm_sentence(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split())


def csltr_sentence_items(members: Iterable[tuple[str, int]], archive: str, glosses: dict[str, list[str]],
                         folder: str = "Videos_Sentence_Level/") -> list[Item]:
    out = []
    for name, off in members:
        if folder not in name or _suffix(name) not in VIDEO_EXT:
            continue
        parts = PurePosixPath(name).parts
        i = next(k for k, p in enumerate(parts) if p == folder.rstrip("/"))
        if len(parts) < i + 3:
            continue
        sentence = parts[i + 1]
        stem = _stem(name)
        out.append(Item(
            id=f"csltr/{sentence}/{stem}".replace(" ", "_"), gloss=sentence.strip(), suffix=_suffix(name), order=off,
            fetch={"kind": "zip", "archive": archive, "member": name},
            meta={"member": name, "category": "sentence", "labelled": True,
                  "glosses": glosses.get(norm_sentence(sentence)), "file_hint": stem},
        ))
    return out


def csltr_word_items(members: Iterable[tuple[str, int]], archive: str, folder: str = "Frames_Word_Level/") -> list[Item]:
    out = []
    for name, off in members:
        if folder not in name or _suffix(name) not in IMAGE_EXT:
            continue
        parts = PurePosixPath(name).parts
        i = next(k for k, p in enumerate(parts) if p == folder.rstrip("/"))
        if len(parts) < i + 3:
            continue
        word = parts[i + 1].strip()
        out.append(Item(
            id=f"csltr_word/{word}/{_stem(name)}".replace(" ", "_"), gloss=word.lower(), suffix=_suffix(name),
            order=off, fetch={"kind": "zip", "archive": archive, "member": name},
            meta={"member": name, "category": "word_frame", "labelled": True},
        ))
    return out


# ----------------------------------------------------------------------------------------------- Mendeley

def mendeley_folder_paths(folders: list[dict]) -> dict[str, str]:
    """folder id -> "A/B/C" from the public API's flat folder list (``id``, ``name``, ``parent_id``)."""
    by = {f["id"]: f for f in folders}
    out = {}
    for fid in by:
        parts, cur, seen = [], fid, set()
        while cur in by and cur not in seen:
            seen.add(cur)
            parts.append(by[cur]["name"])
            cur = by[cur].get("parent_id")
        out[fid] = "/".join(reversed(parts))
    return out


def _dl(f: dict) -> str:
    return (f.get("content_details") or {}).get("download_url", "")


def _sha(f: dict) -> str | None:
    return (f.get("content_details") or {}).get("sha256_hash")


def emergency_archives(files: list[dict], folder_paths: dict[str, str], folder: str = "Raw_Data") -> list[dict]:
    """The per-word zips of one folder: [{"word", "url", "size", "sha256", "filename"}]."""
    out = []
    for f in files:
        if folder_paths.get(f.get("folder_id"), "").split("/")[-1] != folder or not f["filename"].lower().endswith(".zip"):
            continue
        word = f["filename"].rsplit("_", 1)[0].lower()
        out.append({"word": word, "url": _dl(f), "size": int(f["size"]), "sha256": _sha(f), "filename": f["filename"]})
    return sorted(out, key=lambda a: a["word"])


def emergency_items(word: str, archive_url: str, members: Iterable[tuple[str, int]]) -> list[Item]:
    """Members ``<word>_Raw/<word>_<participant>_<take>.AVI``."""
    out = []
    for name, off in members:
        if _suffix(name) not in VIDEO_EXT:
            continue
        stem = _stem(name)
        m = re.search(r"_(\d{3})_(\d{2})$", stem)
        out.append(Item(
            id=f"emergency/{word}/{stem}", gloss=word, suffix=_suffix(name), order=off,
            fetch={"kind": "zip", "archive": archive_url, "member": name},
            meta={"member": name, "category": "emergency", "labelled": True,
                  "group": f"emergency-p{m.group(1)}" if m else None, "take": m.group(2) if m else None},
        ))
    return out


AGRI_FIX = {"cococnut": "coconut"}


def agri_items(files: list[dict], folder_paths: dict[str, str], folder: str = "Original") -> list[Item]:
    out = []
    for f in files:
        path = folder_paths.get(f.get("folder_id"), "")
        if path.split("/")[-1] != folder or _suffix(f["filename"]) not in VIDEO_EXT:
            continue
        subject = next((p for p in path.split("/") if p.lower().startswith("subject")), "subject")
        word = f["filename"].split("_", 1)[0].lower()
        word = AGRI_FIX.get(word, word)
        sid = subject.replace(" ", "").lower()
        out.append(Item(
            id=f"agri/{sid}/{_stem(f['filename'])}", gloss=word, suffix=_suffix(f["filename"]),
            fetch={"kind": "url", "url": _dl(f), "sha256": _sha(f)},
            meta={"file": f["filename"], "folder": path, "category": "agriculture", "labelled": True,
                  "group": f"agri-{sid}", "expected_sha256": _sha(f)},
        ))
    return sorted(out, key=lambda i: i.id)


def static_label(folder: str, parent: str) -> tuple[str, str, str]:
    """Static-image folder -> (gloss, variant, category). E1/E2 -> E, 9a/9b -> 9 (two signs in use for each)."""
    f = folder.strip()
    p = parent.lower()
    cat = "hindi_vowel" if "hindi" in p else "digit" if "numer" in p or f[:1].isdigit() else "letter"
    if cat == "letter":
        m = re.fullmatch(r"([A-Za-z])(\d?)", f)
        return (m.group(1).upper(), f.upper(), cat) if m else (f, f, cat)
    if cat == "digit":
        m = re.fullmatch(r"(\d)([a-z]?)", f)
        return (m.group(1), f, cat) if m else (f, f, cat)
    return f, f, cat


def _stable_key(seed: int, s: str) -> str:
    return hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()


def balanced_sample(by_class: dict[str, dict[str, list[str]]], per_class: int, seed: int = 13) -> dict[str, list[str]]:
    """{class: {group: [key...]}} -> {class: [key...]}: up to ``per_class`` keys per class, taken round-robin over
    the groups (so kids / teens / adults and sleeve types are all represented), deterministic for a seed."""
    out = {}
    for cls, groups in by_class.items():
        queues = {g: sorted(keys, key=lambda k: _stable_key(seed, k)) for g, keys in sorted(groups.items())}
        picked: list[str] = []
        while len(picked) < per_class and any(queues.values()):
            for g in list(queues):
                if queues[g] and len(picked) < per_class:
                    picked.append(queues[g].pop(0))
        out[cls] = picked
    return out


def static_items(files: list[dict], folder_paths: dict[str, str], per_class: int | None = None,
                 seed: int = 13) -> list[Item]:
    """Mendeley 7tsw22y96w: ``ISL Images/<age>/<age sleeves>/<English Alphabet|Hindi Vowels|Numerals>/<label>/x.jpg``.

    ``per_class`` None (or <= 0, the catalogue default since 25 Nov 2026): **every** photo; N: up to N per folder.
    Either way the order is deterministic for a seed: folder by folder (sorted), and inside a folder round-robin over
    the kids / teens / adults x full / half sleeves groups (each group in a seeded hash order). So the first N items
    of a folder are exactly the old ``per_class=N`` sample, and the ids (``isl_static/<Mendeley file id>``) never
    depend on the cap: rows ingested from the 300-per-folder list stay valid."""
    by_class: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    info = {}
    for f in files:
        path = folder_paths.get(f.get("folder_id"), "")
        parts = path.split("/")
        if len(parts) < 3 or _suffix(f["filename"]) not in IMAGE_EXT:
            continue
        label, parent, group_dir, age_dir = parts[-1], parts[-2], parts[-3], parts[-4] if len(parts) >= 4 else ""
        cls = f"{parent}/{label}"
        by_class[cls][group_dir].append(f["id"])
        info[f["id"]] = (f, label, parent, group_dir, age_dir)
    cap = per_class if per_class is not None and per_class > 0 else max((sum(map(len, g.values()))
                                                                          for g in by_class.values()), default=0)
    picked = balanced_sample(by_class, cap, seed)
    out = []
    for cls in sorted(picked):
        for fid in picked[cls]:
            f, label, parent, group_dir, age_dir = info[fid]
            gloss, variant, cat = static_label(label, parent)
            age = re.sub(r"^\d+\.\s*", "", age_dir).split(" ")[0].lower() if age_dir else None
            sleeves = "full" if "full" in group_dir.lower() else "half" if "half" in group_dir.lower() else None
            out.append(Item(
                id=f"isl_static/{fid}", gloss=gloss, suffix=_suffix(f["filename"]),
                fetch={"kind": "url", "url": _dl(f), "sha256": _sha(f)},
                meta={"file": f["filename"], "folder": folder_paths.get(f.get("folder_id")), "variant": variant,
                      "category": cat, "labelled": True, "age_group": age, "sleeves": sleeves,
                      "group": f"static-{age}-{sleeves}", "expected_sha256": _sha(f)},
            ))
    return out


# ----------------------------------------------------------------------------------------------- HF image sets

def hf_label(name: str) -> tuple[str, str]:
    n = str(name).strip()
    return (n.upper(), "letter") if n.isalpha() and len(n) == 1 else (n, "digit" if n.isdigit() else "other")


def hf_rows_items(prefix: str, dataset: str, split: str, labels: list[tuple[int, int]], names: list[str],
                  per_class: int | None = 300, seed: int = 13) -> list[Item]:
    """``labels`` = [(row_idx, label_idx)] from the datasets-server rows API. ``per_class`` N: a deterministic
    sample of up to N rows per class; ``None`` (or <= 0): every row. Either way the items are ordered by class,
    then row (the engine fetches them in row order, so each 100-row page of the rows API is read once)."""
    by_class: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row, lab in labels:
        by_class[str(lab)]["all"].append(str(row))
    if per_class is None or per_class <= 0:
        picked = {cls: list(g["all"]) for cls, g in by_class.items()}
    else:
        picked = balanced_sample(by_class, per_class, seed)
    out = []
    for lab in sorted(picked, key=int):
        name = names[int(lab)] if int(lab) < len(names) else str(lab)
        gloss, cat = hf_label(name)
        for row in sorted(picked[lab], key=int):
            out.append(Item(
                id=f"{prefix}/{split}/{row}", gloss=gloss, suffix=".jpg", order=int(row),
                fetch={"kind": "hfrow", "dataset": dataset, "split": split, "row": int(row)},
                meta={"split": split, "row": int(row), "label_name": name, "category": cat, "labelled": True},
            ))
    return out
