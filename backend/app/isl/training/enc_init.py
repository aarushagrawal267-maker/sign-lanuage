"""Encoder warm start from another work folder (docs/ISL_DESIGN.md 15.46).

``train-encoder --init <dir>/encoder.pt`` and ``train-oneshot --init-dir <dir>`` (fold k from
``<dir>/oneshot/fold<k>/encoder.pt``) start the encoders from another run's fine-tuned encoders (e.g. the research run
from the commercial-safe ones). An explicit warm start is checked before anything is trained (:func:`check_init`):
the file exists and is an ISL encoder, its data profile may serve in this run's profile (a commercial-safe encoder in
a research run, never the reverse: :func:`app.isl.profiles.component_profile_allowed`), the same feature config and
model shape, the same primary split, and a supervised encoder must have been trained without every concept the new
model must not see (``excluded_concepts`` a superset of the fold's held-out set). :func:`init_lineage` is what the new
checkpoint records in ``meta.warm_start`` and what the evaluation report and the pack read: path, SHA-256, profile
and, honestly, the self-supervised pre-training the warm start descends from (its ``pretrain_clips.txt``).

``python -m app.isl.training.enc_init --profile P --hidden H --layers L --emb-dim E FILE...`` is the driver's
start-time check (exit 2 on a refusal) and prints each file's SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PRETRAIN_LIST = "pretrain_clips.txt"  # = multisource.PRETRAIN_LIST (kept torch- / data-free here)
SHAPE_KEYS = ("hidden", "layers", "emb_dim")


def sha256_file(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_meta(path) -> dict:
    import torch

    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    return dict(ck.get("meta") or {}) if isinstance(ck, dict) else {}


def is_pretrained_only(meta: dict) -> bool:
    """A self-supervised checkpoint (``pretrain``, any version ``isl-enc-pre-N``): it never saw a label, so no
    concept exclusion applies. A fine-tuned encoder always records ``excluded_concepts``."""
    return str(meta.get("encoder_version") or "").startswith("isl-enc-pre-") and "excluded_concepts" not in meta


def init_work_folder(path) -> Path:
    """The work folder an encoder file belongs to: ``<work>/encoder.pt``, ``<work>/oneshot/fold<k>/encoder.pt``."""
    p = Path(path).resolve()
    if p.parent.name.startswith("fold") and p.parent.parent.name == "oneshot":
        return p.parents[2]
    return p.parent


def self_supervised_lineage(path, meta: dict) -> dict:
    """The self-supervised pre-training an encoder descends from: its clip list (the one that saw unlabelled clips),
    found next to the pre-trained checkpoint it was initialised from (or in its own work folder) and verified against
    the SHA-256 recorded at pre-training time. A warm start of a warm start passes its recorded lineage on."""
    from app.isl.training.pose_dedup import DEDUP_FILE

    ws = meta.get("warm_start")
    if isinstance(ws, dict) and ws.get("self_supervised") is not None:
        return dict(ws["self_supervised"])
    pt = meta.get("pretraining")
    if not pt:
        return {"pretrained": False, "reason": f"{path} had no self-supervised pre-training"}
    ck = Path(path) if is_pretrained_only(meta) else Path(pt.get("init_from") or path)
    recorded = (pt.get("clip_list") or {}).get("sha256")
    folders = []
    for f in (ck.parent, init_work_folder(path)):
        if f not in folders:
            folders.append(f)
    found, actual = None, None
    for f in folders:
        s = sha256_file(f / PRETRAIN_LIST)
        if s is None:
            continue
        if found is None or (recorded and s == recorded):
            found, actual = f / PRETRAIN_LIST, s
        if recorded and s == recorded:
            break
    out = {"pretrained": True, "checkpoint": str(ck), "list_file": str(found) if found else None,
           "list_sha256": actual, "list_sha256_recorded": recorded,
           "list_verified": bool(found and recorded and actual == recorded),
           "clips": (pt.get("clip_list") or {}).get("clips", pt.get("clips")), "sources": pt.get("sources"),
           "profile": meta.get("profile")}
    if found is not None and (found.parent / DEDUP_FILE).exists():
        out["sign_library_dedup"] = str(found.parent / DEDUP_FILE)
    return out


def init_lineage(path, meta: dict | None = None, fold: int | None = None) -> dict:
    """What a warm-started checkpoint records in ``meta.warm_start`` (docs 15.46)."""
    from app.isl.training.loop import digest

    meta = read_meta(path) if meta is None else meta
    excl = sorted(meta.get("excluded_concepts") or [])
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "work": str(init_work_folder(path)),
            "profile": meta.get("profile"), "encoder_version": meta.get("encoder_version"),
            "pretrained_only": is_pretrained_only(meta), "excluded_concepts": len(excl),
            "excluded_concepts_sha256": digest(excl), "split": (meta.get("data") or {}).get("split"),
            "signer_split": (meta.get("data") or {}).get("signer_split") or "v1",
            "train_sources": (meta.get("data") or {}).get("train_sources"), "model_kwargs": meta.get("model_kwargs"),
            **({"fold": fold} if fold is not None else {}),
            "self_supervised": self_supervised_lineage(path, meta)}


def no_isl500_proof(path, meta: dict | None = None) -> dict:
    """Signer split v2 only moves ISL500 signers (docs 15.42): an encoder that never read an ISL500 clip, labelled or
    not, cannot have seen a v2 validation / test clip. Proven from the checkpoint, never assumed: its data profile
    reads no ISL500 source, its recorded training sources and pre-training sources have no ISL500 clip, and the same
    for a warm start it descends from. ``{"ok": bool, "reasons": [...]}`` (reasons = why not, or the evidence)."""
    from app.isl.profiles import get_profile

    try:
        meta = read_meta(path) if meta is None else meta
    except Exception as e:  # noqa: BLE001 - unreadable = not proven
        return {"ok": False, "reasons": [f"cannot read {path}: {e}"]}
    why, ev = [], []
    prof = meta.get("profile")
    try:
        if not prof:  # get_profile(None) would mean the default profile: never assumed
            raise ValueError
        srcs = set(get_profile(prof).all_sources())
    except ValueError:
        return {"ok": False, "reasons": [f"unknown data profile {prof!r}"]}
    if "isl500" in srcs:
        why.append(f"its profile {prof} reads ISL500")
    else:
        ev.append(f"profile {prof} has no ISL500 source")
    ts = (meta.get("data") or {}).get("train_sources")
    if ts is not None:
        (why if "isl500" in ts else ev).append(f"training sources {sorted(ts)}")
    elif not is_pretrained_only(meta):
        why.append("no training sources recorded")
    ps = (meta.get("pretraining") or {}).get("sources")
    if ps is not None:
        (why if "isl500" in ps else ev).append(f"pre-training sources {sorted(ps)}")
    elif meta.get("pretraining"):
        why.append("no pre-training sources recorded")
    ws = meta.get("warm_start")
    if isinstance(ws, dict):
        try:
            if not ws.get("profile"):
                raise ValueError
            wsrc = set(get_profile(ws["profile"]).all_sources())
        except ValueError:
            wsrc = {"isl500"}
        if "isl500" in wsrc or "isl500" in (ws.get("train_sources") or {}) \
                or "isl500" in ((ws.get("self_supervised") or {}).get("sources") or {}):
            why.append(f"its warm start {ws.get('path')} may have read ISL500")
        else:
            ev.append(f"warm start {ws.get('path')} of profile {ws.get('profile')}")
    return {"ok": not why, "reasons": why or ev}


def check_init(path, *, profile: str, fcfg=None, model_kwargs: dict | None = None, must_exclude=(),
               split: str | None = None, signer_split: str | None = None, meta: dict | None = None) -> list[str]:
    """Reasons to refuse ``path`` as an explicit warm start of a ``profile`` run ([] = accepted; docs 15.46)."""
    from app.isl.profiles import component_profile_allowed

    p = Path(path)
    if not p.is_file():
        return [f"warm start {p} is missing"]
    try:
        meta = read_meta(p) if meta is None else meta
    except Exception as e:  # noqa: BLE001
        return [f"warm start {p} cannot be read: {e}"]
    bad = []
    if meta.get("kind") != "isl-encoder":
        bad.append(f"{p} is not an ISL encoder checkpoint (kind={meta.get('kind')!r})")
    ip = meta.get("profile")
    if not component_profile_allowed(ip, profile):
        bad.append(f"{p} was trained under data profile {ip!r}, which may not warm-start a {profile} run (licence "
                   "direction: a commercial-safe encoder may start a research run, never the reverse; docs 15.44)")
    if fcfg is not None:
        from app.isl.training.train_encoder import feature_config_diff

        if meta.get("feature_config") is None:
            bad.append(f"{p} records no feature config")
        else:
            diff = feature_config_diff(meta.get("feature_config"), fcfg)
            if diff:
                bad.append(f"{p} has another feature config (field: [init, this run]) {diff}")
    if model_kwargs:
        mk = meta.get("model_kwargs") or {}
        diff = {k: [mk.get(k), model_kwargs[k]] for k in SHAPE_KEYS
                if k in model_kwargs and mk.get(k) != model_kwargs[k]}
        if diff:
            bad.append(f"{p} has another model shape (field: [init, this run]) {diff}")
    if not is_pretrained_only(meta):
        excl = set(meta.get("excluded_concepts") or [])
        missing = sorted(set(must_exclude) - excl)
        if missing:
            bad.append(f"{p} was trained WITH {len(missing)} concepts this model must never see (its excluded_concepts "
                       f"is not a superset of them; e.g. {missing[:5]})")
        d = meta.get("data") or {}
        if split and d.get("split") and d["split"] != split:
            bad.append(f"{p} was trained on split {d['split']!r}, this run uses {split!r} (its training clips may be "
                       "this run's evaluation clips)")
        if signer_split and "isl500" in (d.get("train_sources") or {}) \
                and (d.get("signer_split") or "v1") != signer_split:
            bad.append(f"{p} was trained on ISL500 with signer split {d.get('signer_split') or 'v1'}, this run uses "
                       f"{signer_split}")
    return bad


def fold_of(path) -> int | None:
    """k for ``.../oneshot/fold<k>/encoder.pt``, else None."""
    p = Path(path)
    if p.parent.parent.name == "oneshot" and p.parent.name.startswith("fold"):
        try:
            return int(p.parent.name.removeprefix("fold"))
        except ValueError:
            return None
    return None


def fold_heldout(path) -> dict[int, list[str]]:
    """``{fold: raw held-out concept ids}`` of a ``splits/oneshot_concepts.json`` (no linker expansion)."""
    import json

    doc = json.loads(Path(path).read_text())
    folds = (doc.get("protocol") or {}).get("folds", [])
    return {int(f["fold"]): list(f.get("held_out_concepts") or []) for f in folds}


def init_linker(init_meta: dict, init_path, fallback_vocab):
    """The linker the warm start's own run labelled its clips with: its work folder's ``link_settings.json`` and
    ``vocabulary.jsonl`` (read only; this run's vocabulary when the folder has none). Returns (linker, info)."""
    from app.isl.training.common import load_vocab
    from app.isl.training.multisource import GlossLinker, linker_options, read_link_settings

    wf = init_work_folder(init_path)
    vf = wf / "vocabulary.jsonl"
    vocab = load_vocab(vf) if vf.is_file() else fallback_vocab
    link = read_link_settings(wf)
    info = {"work": str(wf), "vocab": str(vf) if vf.is_file() else "this run's", "link_settings": link or {}}
    return GlossLinker.from_vocab(vocab, **linker_options(link)), info


def relabel(linker, clip) -> str | None:
    """The concept ``linker`` gives ``clip``: a clip linked by gloss is re-linked from its raw gloss; an adapter-given
    concept (before this run's variant relabel) goes through ``linker``'s variant links. None = not linked."""
    q = clip.quality or {}
    if "link" in q:
        cid, _ = linker.link(clip.raw_gloss, clip.source)
    else:
        cid = q.get("variant_of") or clip.concept_id
    if cid and linker.variants:
        cid = linker.variant_of(cid, clip.source)
    return cid


def clip_level_exclusion(data, init_meta: dict, init_path, held: set[str], fallback_vocab) -> dict:
    """Clip-level form of the superset check (docs 15.46): the init's excluded_concepts are concept ids of ITS linker,
    the held-out set is this run's. A clip this run labels with a held-out concept that the init's linker labels
    with a concept the init did not exclude was (possibly) in the init's supervised training: refused. The clips
    checked are this run's supervised-candidate clips of the init's training sources (INCLUDE train part, ISLRTC,
    other sources' train part); ``{"violations": [...], "checked": n, ...}``."""
    srcs = set(((init_meta.get("data") or {}).get("train_sources") or {}).keys())
    excl = set(init_meta.get("excluded_concepts") or [])
    cands = list(data.include_part("train")) + list(data.islrtc) + list(data.extra_part("train"))
    cands = [c for c in cands if c.concept_id in held and (not srcs or c.source in srcs)]
    if not cands:
        return {"violations": [], "checked": 0}
    linker, info = init_linker(init_meta, init_path, fallback_vocab)
    bad = []
    for c in cands:
        y = relabel(linker, c)
        if y is not None and y not in excl:
            bad.append({"clip": c.clip_id, "source": c.source, "here": c.concept_id, "init": y})
    return {"violations": bad, "checked": len(cands), "init_linker": info}


def unreachable_heldout(data, init_meta: dict, init_path, held: set[str], fallback_vocab) -> dict:
    """Held-out concepts the init cannot have trained on (docs 15.46, 28 Nov fix): the concept-id superset check
    compares this run's held-out ids with the init's ``excluded_concepts``, so an id that exists only here (e.g. a
    variant concept such as ``isl.mean~isl500`` that variant links make for ISL500 clips only) is refused although the
    init never read a clip of it. A held-out concept is excused when NO clip of any source the init could have read
    (its data profile's sources and its recorded training sources; every clip of every part, a superset of its
    training) carries it, neither in this run's labels nor in the init's own linker's labels. Unprovable (unknown
    profile, no training sources) = nothing excused. ``{"concepts": [...], "init_sources": [...], "checked": n}``."""
    from app.isl.profiles import get_profile

    excl = set(init_meta.get("excluded_concepts") or [])
    todo = set(held) - excl
    if not todo:
        return {"concepts": [], "checked": 0}
    prof = init_meta.get("profile")
    ts = (init_meta.get("data") or {}).get("train_sources")
    try:
        if not prof or not ts:
            raise ValueError
        srcs = set(get_profile(prof).all_sources()) | set(ts)
    except ValueError:
        return {"concepts": [], "checked": 0, "reason": "the init records no data profile or training sources"}
    clips = [c for c in list(data.include) + list(data.islrtc) + list(getattr(data, "extra", None) or [])
             if c.source in srcs]
    seen = {c.concept_id for c in clips if c.concept_id}
    if todo - seen:
        linker, _ = init_linker(init_meta, init_path, fallback_vocab)
        seen |= {y for y in (relabel(linker, c) for c in clips) if y}
    return {"concepts": sorted(todo - seen), "init_sources": sorted(srcs), "checked": len(clips)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.isl.training.enc_init",
                                 description="check warm-start encoders for a run (docs 15.46); exit 2 = refused")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--emb-dim", type=int)
    ap.add_argument("--oneshot-concepts", default=None,
                    help="splits/oneshot_concepts.json: a fold file (<dir>/oneshot/fold<k>/encoder.pt) must exclude "
                         "fold k's raw held-out concepts (the linked expansion is checked by train-oneshot)")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    from app.isl.training.common import ISL_FEATURES

    mk = {k: v for k, v in (("hidden", a.hidden), ("layers", a.layers), ("emb_dim", a.emb_dim)) if v is not None}
    held = fold_heldout(a.oneshot_concepts) if a.oneshot_concepts else {}
    rc = 0
    for f in a.files:
        k = fold_of(f)
        if a.oneshot_concepts and k is not None and not held.get(k):
            print(f"ENC_INIT: {f}: fold {k} has no held-out concepts (not trained): not needed")
            continue
        bad = check_init(f, profile=a.profile, fcfg=ISL_FEATURES, model_kwargs=mk,
                         must_exclude=held.get(k, ()) if k is not None else ())
        for b in bad:
            print(f"ENC_INIT refused: {b}", file=sys.stderr)
        rc = 2 if bad else rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
