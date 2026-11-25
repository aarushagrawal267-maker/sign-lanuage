"""ISL command-line interface: data commands (Data Engineer) and model commands (AI Engineer).

Run from the backend folder:  python -m app.isl.cli <command> [options]
(heavy commands: prefix with ``nice -n 10`` while the ingestion jobs run)

  status                 counts per source, quality filter, trims, vocabulary, splits, gated access
  wordnet                download WordNet (synonym lists, POS) into data/isl/nltk_data
  vocab                  build models/packs/isl/vocabulary.jsonl (+ data/isl/derived/vocab_report.json)
  trim [--redo] [--limit N] [--learned]
                         ISLRTC first-sign trims -> data/isl/derived/islrtc_trim.jsonl (stage 1; --learned
                         trains/applies the stage-2 boundary model)
  trim-review [--n 60]   review sheet PNGs (20 random clips per method) -> artifacts/reports/trim_review/
  trim-eval              synthetic end-error check + INCLUDE active-frame check -> artifacts/reports/trim_eval.json
  splits [--fetch] [--k 5] [--seed 0]
                         INCLUDE official / session / pseudo-signer splits with leakage checks, and the
                         cross-dataset one-shot concept folds -> data/isl/splits/
  dedup [--sources include islrtc cislr]
                         near-duplicate clips across/within sources -> data/isl/derived/duplicates.jsonl
  access                 Hugging Face access check for CISLR and iSign (exit 0 = all granted, 3 = pending)
  profiles               print the data profiles (commercial-safe, research)

Model commands (heavy: run with ``nice -n 10``; ``--threads`` defaults to 2 while ingestion runs). Every command
takes ``--profile commercial-safe|research`` (default: $SLR_ISL_PROFILE or commercial-safe); outputs go to
artifacts/isl/<profile>/ and build-pack copies them to artifacts/packs/<isl|isl-research>/ (git-ignored).

  train-classifier       closed-set core-word classifier (INCLUDE + ISLRTC clips of the same concepts)
  train-encoder          metric-learning encoder (SupCon + sub-centre ArcFace) over every concept with clips
  build-index            nearest-neighbour index over ISLRTC first-sign trims (+ alt ends) and INCLUDE centroids
  train-letters --clips A.npz [B.npz ...] [--letters ABC...]
                         letter model from in-order alphabet recordings (none exist in the open data yet)
  fit-fusion             fusion weights, temperature, accept/suggest thresholds, out-of-vocabulary guard (val fold)
  evaluate               E1/E2/E3/E6 on the test fold -> results.json (+ results.md)
  build-pack             assemble the language pack the app loads
  smoke                  all of the above with tiny settings (a few CPU minutes) on whatever data is on disk
  hindi [--translate]    Hindi lexicon models/packs/isl/lexicon_hi.jsonl (+ models/packs/asl-wlasl100/): curated
                         core list, number rules and offline IndicTrans2 drafts (--translate fills the cache;
                         about 40 min on 2 CPU threads); --merge-local folds reviewed local corrections back in

Phase 2 (all sources; driver: scripts/phase2_train.sh). Model commands accept --all-sources (every labelled word
source of the profile, linked to concepts, signer-held-out splits, duplicates dropped), --vocab, and the
training ones --resume / --max-minutes (exit code 75 = stopped with state saved, run again with --resume):

  link-sources           read + link every labelled source -> <work>/multisource.json, sources.json, extra_concepts.jsonl
  pretrain               self-supervised encoder pre-training (temporal contrastive + masked frames) on every
                         unlabelled / sentence source of the profile -> <work>/pretrained_encoder.pt
  train-encoder --init auto
                         fine-tune from the pre-trained encoder
  train-oneshot --folds 0 1
                         E2 encoders with each fold's concepts held out of all supervised training
  train-fingerspell      letter model from every letter source (+ leave-one-signer-out E5)
  evaluate --all         E1a/E1b/E1c/E2/E3/E4/E5/E6/E7/E8 -> <work>/eval and <pack>/eval report.json + report.md
  bench                  CPU seconds per training step for the model sizes (for the time budget)
  readiness [--allow-partial --work DIR]
                         is the data complete? (no ingestion job running, every manifest of the profile complete);
                         exit 4 = not ready. --allow-partial marks DIR/partial_data.json (reports say PARTIAL DATA)
  pretrain-dedup         OpenHands Sign Library chunks that repeat another chunk or contain an evaluation clip
                         (pose-content matcher, calibrated first) -> <work>/pretrain_dedup.json, read by pretrain
  trim [--learned-model auto|PATH|none]
                         uses artifacts/isl/trim_boundary.pt (learned end) when present: trim_version trim-2+trim-learned-1

Gated-dataset ingestion lives in ``scripts/ingest_isl_gated.py`` (CISLR, iSign).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from app.isl.paths import PACK_DIR, IslPaths, isl_paths


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------------------------- commands

def cmd_status(paths: IslPaths, args) -> int:
    from app.isl.data.manifest import read_manifest
    from app.isl.data.sources import ADAPTERS, passes_quality

    out: dict = {"root": str(paths.root)}
    for name in ADAPTERS:
        man = paths.manifest(name)
        if not man.exists():
            out[name] = {"manifest": False}
            continue
        allr = read_manifest(man, only_ok=False)
        ok = [r for r in allr if r.get("ok")]
        d = {"records": len(allr), "ok": len(ok), "failed": len(allr) - len(ok),
             "zero_frames": sum(1 for r in ok if not r.get("n_frames")),
             "quality_pass": sum(1 for r in ok if passes_quality(r))}
        if name == "islrtc":
            d["stop"] = dict(Counter(r.get("stop") for r in ok))
            if paths.islrtc_index.exists():
                d["index_files"] = len(json.loads(paths.islrtc_index.read_text()))
        if name == "include":
            d["words"] = len({(r.get("member") or "").rsplit("/", 1)[0] for r in ok})
        out[name] = d
    trims = read_manifest(paths.trim_sidecar, only_ok=False)
    out["trims"] = {"clips": len(trims), "methods": dict(Counter(r.get("method") for r in trims))}
    vocab = PACK_DIR / "vocabulary.jsonl"
    out["vocabulary"] = {"path": str(vocab), "concepts": sum(1 for _ in vocab.open()) if vocab.exists() else 0}
    out["splits"] = sorted(p.name for p in paths.splits.glob("*.json")) if paths.splits.exists() else []
    if not args.offline:
        from app.isl.data.remote import PROBE_FILES, hf_access

        out["gated_access"] = {r: hf_access(r) for r in PROBE_FILES}
    _print(out)
    return 0


def cmd_wordnet(paths: IslPaths, args) -> int:
    from app.isl.vocab.wordnet import download_wordnet, get_wordnet

    p = download_wordnet(paths.nltk_data)
    print(f"WordNet at {p}; loaded: {get_wordnet() is not None}")
    return 0


def cmd_vocab(paths: IslPaths, args) -> int:
    from app.isl.vocab import wordnet as wnx
    from app.isl.vocab.build import (
        BuildInputs,
        build_vocabulary,
        load_cislr_rows,
        load_include_words,
        load_islrtc_paths,
        load_trims,
        write_vocabulary,
    )
    from app.isl.vocab.normalize import load_overrides

    if args.no_wordnet:
        wnx.disable(True)
    inp = BuildInputs(
        islrtc_paths=load_islrtc_paths(paths), include_words=load_include_words(paths),
        cislr_rows=load_cislr_rows(paths), trims={} if args.no_trims else load_trims(paths),
        overrides=load_overrides(),
    )
    if not inp.islrtc_paths:
        print("no ISLRTC names found (data/isl/manifests/islrtc_index.json or islrtc.jsonl)", file=sys.stderr)
        return 2
    res = build_vocabulary(inp)
    out = Path(args.out) if args.out else PACK_DIR / "vocabulary.jsonl"
    write_vocabulary(res.entries, out)
    rep_path = paths.derived / "vocab_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(res.report, indent=1, ensure_ascii=False))
    summary = {k: v for k, v in res.report.items() if k not in ("include",)}
    summary["include"] = {k: v for k, v in res.report["include"].items() if k in ("words", "links")}
    summary["include"]["new"] = len(res.report["include"]["new"])
    summary["include"]["needs_review"] = len(res.report["include"]["needs_review"])
    summary["out"] = str(out)
    summary["size_mb"] = round(out.stat().st_size / 1e6, 2)
    _print(summary)
    return 0


def _islrtc_records(paths: IslPaths) -> list[dict]:
    """ISLRTC manifest rows with frames; a re-extracted row (``islrtc_recut.jsonl``, card-v2 detector) replaces the
    original row of the same id (``training.common.islrtc_overrides``)."""
    from app.isl.data.manifest import read_manifest
    from app.isl.training.common import islrtc_overrides

    over = islrtc_overrides(paths)
    return [over.get(r["id"], r) for r in read_manifest(paths.manifest("islrtc")) if r.get("n_frames")
            or over.get(r["id"], {}).get("n_frames")]


def _trim_model(args):
    """(model, version suffix, path) of the learned end model (``trim_learned``), or (None, "", None)."""
    from app.isl.data.trim_learned import MODEL_VERSION, default_model_path, load

    choice = getattr(args, "learned_model", "auto") or "auto"
    if choice == "none":
        return None, "", None
    path = default_model_path() if choice == "auto" else Path(choice)
    if not path.exists():
        if choice != "auto":
            raise SystemExit(f"learned trim model {path} not found")
        return None, "", None
    model, _ = load(path)
    return model, f"+{MODEL_VERSION}", path


def cmd_trim(paths: IslPaths, args) -> int:
    """ISLRTC first-sign trims. Uses the learned end model (``artifacts/isl/trim_boundary.pt``, the same call as
    ``trim_bench run --learned``) when it exists, rules otherwise; ``trim_version`` says which (``trim-2`` or
    ``trim-2+trim-learned-1``), so clips trimmed by rules only are redone once a model exists. Clips whose
    landmark file changed (re-extraction) are re-trimmed too; the record keeps the file it was computed on."""
    import time

    from app.isl.data.manifest import read_manifest, write_jsonl
    from app.isl.data.trim import TrimConfig, find_first_sign, trim_record_for, words_for
    from app.isl.vocab.normalize import parse_islrtc_name
    from app.landmarks.raw import RawSequence

    cfg = TrimConfig()
    model, suffix, model_path = _trim_model(args)
    version = cfg.version + suffix
    existing = {r["id"]: r for r in read_manifest(paths.trim_sidecar, only_ok=False)}
    recs = _islrtc_records(paths)
    todo = [r for r in recs if args.redo or r["id"] not in existing
            or existing[r["id"]].get("trim_version") != version
            or existing[r["id"]].get("landmarks", None if r.get("recut_of") else r["path"]) != r["path"]]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    n_err = n_learned = 0
    for i, r in enumerate(todo):
        pn = parse_islrtc_name(r["id"].removeprefix("islrtc/"))
        try:
            raw = RawSequence.load(paths.resolve(r["path"]))
            nw = words_for(pn.kind, pn.text)
            if model is not None and pn.kind != "phrase":
                from app.isl.data.trim_learned import refine_cap

                res = refine_cap(raw, find_first_sign(raw, r.get("stop"), cfg, n_words=nw), model, nw, cfg)
                n_learned += any("trim-learned" in x for x in res.notes)
                rec = res.to_record(r["id"], len(raw), version)
            else:
                rec = trim_record_for(r["id"], raw, r.get("stop"), pn.kind, cfg, n_words=nw)
                rec["trim_version"] = version
        except Exception as e:  # noqa: BLE001
            n_err += 1
            rec = {"id": r["id"], "start": 0, "end": int(r.get("n_frames") or 0), "method": "empty", "confidence": 0.0,
                   "alternatives": [], "notes": [f"{type(e).__name__}: {e}"[:200]], "trim_version": version}
        rec["stop"] = r.get("stop")
        rec["kind"] = pn.kind
        rec["landmarks"] = r["path"]
        if r.get("recut_of"):
            rec["recut"] = True
        existing[r["id"]] = rec
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(todo)} trimmed ({time.time() - t0:.0f}s)", flush=True)
    if args.learned:
        existing = _apply_learned(paths, existing, args)
    write_jsonl(paths.trim_sidecar, [existing[k] for k in sorted(existing)])
    methods = Counter(r.get("method") for r in existing.values())
    _print({"trimmed_now": len(todo), "errors": n_err, "total": len(existing), "methods": dict(methods),
            "trim_version": version, "learned_model": str(model_path) if model_path else None,
            "learned_ends_now": n_learned,
            "versions": dict(Counter(r.get("trim_version") for r in existing.values())),
            "truncated": methods.get("truncated", 0), "recut": sum(1 for r in existing.values() if r.get("recut")),
            "secs": round(time.time() - t0, 1), "sidecar": str(paths.trim_sidecar)})
    return 0


def _apply_learned(paths: IslPaths, trims: dict[str, dict], args) -> dict[str, dict]:
    from app.core.config import get_settings
    from app.isl.data.boundary import frame_features, load_model, refine, save_model, train_boundary
    from app.isl.data.sources import IncludeAdapter
    from app.isl.data.trim import TrimConfig, TrimResult
    from app.landmarks.raw import RawSequence

    cfg = TrimConfig()
    model_path = get_settings().artifacts_dir / "isl" / "boundary.pt"
    recs = {r["id"]: r for r in _islrtc_records(paths)}
    if not model_path.exists() or args.retrain:
        rng = random.Random(0)
        clean, tails = [], []
        for cid, t in trims.items():
            r = recs.get(cid)
            if r is None:
                continue
            if t.get("method") in ("source_end", "rest") and t.get("confidence", 0) >= 0.8 and len(clean) < 1500:
                raw = RawSequence.load(paths.resolve(r["path"])).slice(t["start"], t["end"])
                if len(raw) >= 6:
                    clean.append(frame_features(raw, cfg))
            elif (r.get("stop") or "") == "max_s" and t.get("method") in ("dip", "rest") and len(tails) < 1500:
                full = RawSequence.load(paths.resolve(r["path"]))
                e = int(t["end"])
                L = int(rng.uniform(2.0, 6.0) * (r.get("fps") or 12.0))
                if len(full) - e >= 12:
                    tails.append(frame_features(full.slice(e, min(len(full), e + L)), cfg))
        for c in IncludeAdapter(paths).clips():
            if len(clean) >= 2500:
                break
            try:
                clean.append(frame_features(c.load(), cfg))
            except Exception:  # noqa: BLE001
                continue
        print(f"  boundary model: {len(clean)} clean signs, {len(tails)} tails", flush=True)
        model, res = train_boundary(clean, tails, n_examples=args.examples, epochs=args.epochs, threads=args.threads)
        save_model(model, model_path, {"trim_version": cfg.version, "result": res.__dict__})
        print(f"  boundary model trained: {res}", flush=True)
    model, _ = load_model(model_path)
    changed = 0
    for cid, t in trims.items():
        r = recs.get(cid)
        if r is None or t.get("method") in ("empty", "full", "source_end", "learned"):
            continue
        raw = RawSequence.load(paths.resolve(r["path"]))
        st1 = TrimResult(t["start"], t["end"], t.get("t_start_ms", 0.0), t.get("t_end_ms", 0.0), t["method"],
                         t.get("confidence", 0.0), list(t.get("alternatives", [])), list(t.get("notes", [])))
        new = refine(raw, st1, model, cfg)
        if new.method == "learned":
            rec = new.to_record(cid, len(raw), cfg.version)
            rec.update({"stop": t.get("stop"), "kind": t.get("kind")})
            trims[cid] = rec
            changed += 1
    print(f"  learned refinement changed {changed} trims", flush=True)
    return trims


def cmd_trim_review(paths: IslPaths, args) -> int:
    from app.core.config import get_settings
    from app.isl.data.manifest import read_manifest
    from app.isl.data.trim import plot_trim
    from app.landmarks.raw import RawSequence

    out = Path(args.out) if args.out else get_settings().reports_dir / "trim_review"
    out.mkdir(parents=True, exist_ok=True)
    recs = {r["id"]: r for r in _islrtc_records(paths)}
    trims = [t for t in read_manifest(paths.trim_sidecar, only_ok=False) if t["id"] in recs]
    rng = random.Random(args.seed)
    by_m: dict[str, list] = {}
    for t in trims:
        by_m.setdefault(t["method"], []).append(t)
    per = max(1, args.n // 3)
    chosen = []
    for m in sorted(by_m):
        items = by_m[m][:]
        rng.shuffle(items)
        chosen += items[:per]
    lines = ["# ISLRTC trim review sheet", "",
             "Yellow = kept frames; purple dashed = alternative ends; top = wrist height (up = higher), "
             "bottom = motion.", ""]
    for i, t in enumerate(chosen):
        r = recs[t["id"]]
        name = f"{i:03d}_{t['method']}.png"
        plot_trim(RawSequence.load(paths.resolve(r["path"])), t, out / name, title=f"{t['id']} stop={r.get('stop')}")
        lines.append(f"- `{t['id']}` {t['method']} conf {t.get('confidence')}: ![{name}]({name})")
    (out / "index.md").write_text("\n".join(lines) + "\n")
    _print({"plots": len(chosen), "dir": str(out), "methods": {m: min(per, len(v)) for m, v in by_m.items()}})
    return 0


def align_concat(a, b, gap_frames: int = 0):
    """A followed by B, with B moved/scaled so its median shoulder centre and width match A's (synthetic
    'sign + explanation' sequences for the trim checks)."""
    import numpy as np

    from app.landmarks.raw import RawSequence

    def frame(raw):
        asp = raw.aspect
        ls, rs = raw.pose[:, 11, :2].astype(np.float64).copy(), raw.pose[:, 12, :2].astype(np.float64).copy()
        ls[:, 0] *= asp
        rs[:, 0] *= asp
        ok = ~np.isnan(ls[:, 0]) & ~np.isnan(rs[:, 0])
        c = np.median((ls[ok] + rs[ok]) / 2, axis=0)
        s = float(np.median(np.linalg.norm(ls[ok] - rs[ok], axis=1)))
        return c, s

    ca, sa = frame(a)
    cb, sb = frame(b)
    k = sa / sb if sb > 0 else 1.0

    def mv(arr, src_asp, dst_asp):
        out = arr.astype(np.float64).copy()
        x = out[..., 0] * src_asp
        y = out[..., 1]
        out[..., 0] = ((x - cb[0]) * k + ca[0]) / dst_asp
        out[..., 1] = (y - cb[1]) * k + ca[1]
        if out.shape[-1] >= 3:
            out[..., 2] = out[..., 2] * k * src_asp / dst_asp
        return out.astype(np.float32)

    dt = float(np.median(np.diff(a.t_ms))) if len(a) > 1 else 80.0
    nanf = lambda shape: np.full(shape, np.nan, np.float32)  # noqa: E731
    gp = nanf((gap_frames,) + a.pose.shape[1:])
    pose_b = mv(b.pose, b.aspect, a.aspect)
    pose_b[..., 3] = b.pose[..., 3]
    t_b = a.t_ms[-1] + dt * (gap_frames + 1) + (b.t_ms - b.t_ms[0])
    return RawSequence(
        np.concatenate([a.pose, gp, pose_b]), np.concatenate([a.left, nanf((gap_frames, 21, 3)), mv(b.left, b.aspect, a.aspect)]),
        np.concatenate([a.right, nanf((gap_frames, 21, 3)), mv(b.right, b.aspect, a.aspect)]),
        np.concatenate([a.face, nanf((gap_frames,) + a.face.shape[1:]), mv(b.face, b.aspect, a.aspect)]),
        np.concatenate([a.t_ms, a.t_ms[-1] + dt * np.arange(1, gap_frames + 1), t_b]), a.width, a.height,
        a.extractor_version)


def cmd_trim_eval(paths: IslPaths, args) -> int:
    import numpy as np

    from app.core.config import get_settings
    from app.isl.data.manifest import read_manifest
    from app.isl.data.sources import IncludeAdapter
    from app.isl.data.trim import TrimConfig, activity_signal, find_first_sign, words_for
    from app.isl.vocab.normalize import parse_islrtc_name
    from app.landmarks.raw import RawSequence

    cfg = TrimConfig()
    rng = random.Random(args.seed)
    recs = {r["id"]: r for r in _islrtc_records(paths)}
    trims = {t["id"]: t for t in read_manifest(paths.trim_sidecar, only_ok=False) if t["id"] in recs}
    model = None
    if args.learned:
        from app.core.config import get_settings as _gs
        from app.isl.data.boundary import load_model, refine

        mp = _gs().artifacts_dir / "isl" / "boundary.pt"
        if not mp.exists():
            print("no boundary model; run `trim --learned` first", file=sys.stderr)
            return 2
        model, _meta = load_model(mp)
    clean = [t for t in trims.values() if t["method"] == "source_end"]
    tails = [t for t in trims.values() if (recs[t["id"]].get("stop") == "max_s") and t["method"] in ("dip", "rest", "cap")]
    rng.shuffle(clean)
    errs, methods, within = [], Counter(), 0
    n = min(args.n, len(clean))
    for t in clean[:n]:
        a_full = RawSequence.load(paths.resolve(recs[t["id"]]["path"]))
        a = a_full.slice(0, t["end"])
        bt = rng.choice(tails) if tails else None
        if bt is None:
            break
        b_full = RawSequence.load(paths.resolve(recs[bt["id"]]["path"]))
        L = int(rng.uniform(2.0, 6.0) * 12)
        b = b_full.slice(bt["end"], min(len(b_full), bt["end"] + L))
        if len(b) < 6:
            continue
        x = align_concat(a, b, gap_frames=rng.randint(0, 3))
        pn = parse_islrtc_name(t["id"].removeprefix("islrtc/"))
        res = find_first_sign(x, "max_s", cfg, n_words=words_for(pn.kind, pn.text))
        if model is not None:
            res = refine(x, res, model, cfg)
        truth_ms = float(a.t_ms[-1])
        err = float(x.t_ms[min(len(x), res.end) - 1] - truth_ms)
        errs.append(err)
        methods[res.method] += 1
        within += abs(err) <= 300
    out: dict = {"synthetic": {
        "description": "clean ISLRTC clips (bounded source_end trims) + 2-6 s explanation-like tail of another "
                       "clip (after its stage-1 end), aligned to the same body frame; trimmer run as if stop=max_s",
        "n": len(errs), "within_300ms": round(within / max(1, len(errs)), 3),
        "median_abs_err_ms": round(float(np.median(np.abs(errs))), 1) if errs else None,
        "late_frac": round(float(np.mean(np.array(errs) > 300)), 3) if errs else None,
        "early_frac": round(float(np.mean(np.array(errs) < -300)), 3) if errs else None,
        "methods": dict(methods)}}
    # INCLUDE: one sign per clip; the trimmer (bounded) should keep >= 90% of the active frames
    keep = []
    for c in list(IncludeAdapter(paths).clips())[: args.n]:
        raw = c.load()
        try:
            sig = activity_signal(raw, cfg)
        except Exception:  # noqa: BLE001
            continue
        act = ~sig["dip"]
        if act.sum() == 0:
            continue
        res = find_first_sign(raw, "eof", cfg)
        kept = act[res.start: res.end].sum() / act.sum()
        keep.append(float(kept))
    out["include_active_kept"] = {"n": len(keep), "mean": round(float(np.mean(keep)), 3) if keep else None,
                                  "frac_ge_90pct": round(float(np.mean(np.array(keep) >= 0.9)), 3) if keep else None}
    out["methods_all"] = dict(Counter(t["method"] for t in trims.values()))
    out["learned"] = model is not None
    rp = get_settings().reports_dir / ("trim_eval_learned.json" if model is not None else "trim_eval.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(out, indent=1))
    _print(out)
    return 0


def cmd_splits(paths: IslPaths, args) -> int:
    from app.isl.data import splits as sp
    from app.isl.data.manifest import iter_jsonl
    from app.isl.data.sources import IncludeAdapter, include_match_key
    from app.isl.vocab.build import VocabIndex

    if args.fetch or (not args.no_fetch and not (paths.official_splits / "include_train.txt").exists()):
        _print({"fetched": sp.fetch_official(paths)})
    vocab_path = Path(args.vocab) if args.vocab else PACK_DIR / "vocabulary.jsonl"
    vocab = VocabIndex.load(vocab_path) if vocab_path.exists() else None
    adapter = IncludeAdapter(paths, vocab=vocab, groups={})
    clips = list(adapter.clips())
    manifest = paths.manifest("include")
    dups = [d["clips"] for d in iter_jsonl(paths.derived / "duplicates.jsonl")] if (paths.derived / "duplicates.jsonl").exists() else []
    class_of = {c.clip_id: c.concept_id or c.quality.get("word_dir") for c in clips}
    report: dict = {"include_clips": len(clips)}

    # official
    assign, rep = sp.include_official(clips, sp.official_paths(paths))
    sessions = sp.include_sessions(clips)
    lk = sp.leakage_report(assign, sessions, class_of, dups)
    sp.write_split(paths, sp.split_doc("include-official", "official", assign, lk, None, manifest, "no",
                                       {"unmatched": rep}))
    report["include-official"] = {"matched": rep["matched"], "list_paths": rep["list_paths"],
                                  "not_ingested": rep["not_ingested"], "not_listed": rep["not_listed"],
                                  "session_overlap": lk["group_overlap"]}
    # sessions + pseudo-signers
    groups, prep = sp.include_pseudo_signers(clips, sessions, seed=args.seed)
    paths.include_groups.parent.mkdir(parents=True, exist_ok=True)
    paths.include_groups.write_text(json.dumps({"clips": groups, "report": prep}, indent=1))
    report["pseudo_signers"] = {k: prep.get(k) for k in ("sessions", "blocks", "k", "silhouette", "weak", "sizes")}
    for kind, key in (("session", "session"), ("pseudo_signer", "pseudo_signer")):
        items = {c: g[key] for c, g in groups.items()}
        folds = sp.grouped_kfold(items, k=args.k, seed=args.seed, merge=dups)
        res_f = []
        for i, f in enumerate(folds):
            lkf = sp.leakage_report(f, items, class_of, dups)
            name = f"include-{kind.replace('_', '-')}-cv{args.k}-fold{i}"
            si = "estimated" if kind == "pseudo_signer" else "session"
            sp.write_split(paths, sp.split_doc(name, kind, f, lkf, args.seed, manifest, si))
            res_f.append({"fold": i, "n": lkf["n"], "passed": lkf["passed"],
                          "test_classes_missing": len(lkf["classes_missing"]["test"])})
        report[f"include-{kind}-cv{args.k}"] = res_f
    n_ps = len(prep.get("sizes", {}))
    enough = n_ps >= args.k and max(prep.get("sizes", {0: 0}).values()) <= 0.5 * max(1, len(clips))
    primary = "pseudo_signer" if (not prep.get("weak") and enough) else "session"
    report["primary_split"] = f"include-{primary.replace('_', '-')}-cv{args.k}"
    if primary == "session":
        report["primary_reason"] = ("pseudo-signer clustering is weak (silhouette < 0.15)" if prep.get("weak") else
                                    f"too few / too uneven pseudo-signers for {args.k} folds ({n_ps} groups, sizes "
                                    f"{sorted(prep.get('sizes', {}).values(), reverse=True)}); expected on partial data")
    # one-shot
    if vocab is not None:
        concepts = sp.oneshot_concepts(vocab.entries)
        fold_of = sp.oneshot_folds(concepts, k=args.k, seed=args.seed)
        prot = sp.oneshot_protocol(fold_of, clips, args.k)
        doc = {"name": f"oneshot-concept-cv{args.k}", "kind": "oneshot", "label": sp.LABELS["oneshot"],
               "seed": args.seed, "created": sp.now_iso(), "concepts": concepts, "fold_of": fold_of, "protocol": prot}
        (paths.splits / "oneshot_concepts.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        report["oneshot"] = {"linked_concepts": len(concepts), "usable": sum(c["usable"] for c in concepts),
                             "needs_review": sum(not c["usable"] for c in concepts),
                             "queries_per_fold": [len(f["queries"]) for f in prot["folds"]]}
    rp = paths.splits / "splits_report.json"
    rp.write_text(json.dumps(report, indent=1))
    _print(report)
    _ = include_match_key  # re-exported helper (documented in sources)
    return 0


def cmd_dedup(paths: IslPaths, args) -> int:
    from app.isl.data.dedup import duplicate_clusters, find_duplicates
    from app.isl.data.manifest import write_jsonl
    from app.isl.data.sources import get_adapter

    from app.isl.training.multisource import open_adapter

    clips = []
    for s in args.sources:
        try:
            ad = get_adapter(s, paths)
        except ValueError:  # phase-2 sources (wslp, isl500, mendeley_*) come from the multi-source registry
            ad = open_adapter(s, paths)
        if ad is not None and ad.available():
            clips += list(ad.clips())
    pairs = find_duplicates(clips, threshold=args.threshold,
                            progress=lambda i, n: print(f"  fingerprints {i}/{n}", flush=True))
    clusters = duplicate_clusters(pairs)
    src = {c.clip_id: c.source for c in clips}
    rows = [{"clips": cl, "sources": sorted({src[c] for c in cl})} for cl in clusters]
    write_jsonl(paths.derived / "duplicates.jsonl", rows)
    _print({"clips": len(clips), "pairs": len(pairs), "clusters": len(clusters),
            "cross_source": sum(len(r["sources"]) > 1 for r in rows), "examples": rows[:10]})
    return 0


def cmd_access(paths: IslPaths, args) -> int:
    from app.isl.data.remote import PROBE_FILES, hf_access

    res = {r: hf_access(r) for r in PROBE_FILES}
    _print(res)
    return 0 if all(v == "granted" for v in res.values()) else 3


def cmd_readiness(paths: IslPaths, args) -> int:
    """Data-completeness guard of full phase-2 runs: exit 0 = ready (or --allow-partial: then the work folder is
    marked partial_data), 4 = an ingestion job runs or a manifest is incomplete."""
    from app.isl.training.readiness import check, mark_partial

    rep = check(paths, args.profile or "commercial-safe")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1))
    marker = None
    if not rep["ready"] and args.allow_partial and args.work:
        marker = str(mark_partial(Path(args.work), rep))
    _print({"ready": rep["ready"], "reasons": rep["reasons"], "partial_marker": marker,
            "sources": {k: {x: v.get(x) for x in ("done", "expected", "complete", "note") if v.get(x) is not None}
                        for k, v in rep["sources"].items()}})
    if rep["ready"] or args.allow_partial:
        return 0
    return 4


def cmd_profiles(paths: IslPaths, args) -> int:
    from app.isl.profiles import PROFILES

    _print({k: v.to_dict() for k, v in PROFILES.items()})
    return 0


# --------------------------------------------------------------------------------------------- model commands

def _vocab_path(args) -> Path:
    v = getattr(args, "vocab", None)
    return Path(v) if v else PACK_DIR / "vocabulary.jsonl"


def _record_sources(work: Path, rep: dict, role: str, used: dict | None = None) -> None:
    """Merge a source report into ``<work>/sources.json``. ``used`` = clips of each source that really reached this
    role's data (after caps, sampling and duplicate removal; default: the report's ``clips``). ``build-pack`` names
    in the pack's licence list only the sources whose ``used`` count is > 0 for some role."""
    f = work / "sources.json"
    cur = json.loads(f.read_text()) if f.exists() else {}
    for name, d in rep.items():
        if not isinstance(d, dict):
            continue
        old = cur.get(name, {})
        if not d.get("available"):
            n = 0
        elif used is not None:
            n = int(used.get(name) or 0)  # a source read but not sampled contributed nothing
        else:
            n = int(d.get("clips") or 0)
        per_role = {**(old.get("used") or {}), role: n}
        roles = sorted(r for r, k in per_role.items() if k > 0)
        cur[name] = {**old, **{k: v for k, v in d.items() if k != "clips"}, "available": bool(old.get("available") or
                     d.get("available")), "clips_read": d.get("clips", old.get("clips_read")), "used": per_role,
                     "clips": max(per_role.values(), default=0), "roles": roles, "role": ",".join(roles),
                     "contributed": any(k > 0 for k in per_role.values())}
    f.write_text(json.dumps(cur, indent=1, ensure_ascii=False))


def _model_ctx(paths: IslPaths, args):
    from app.core.config import get_settings
    from app.isl.profiles import get_profile
    from app.isl.training.common import ISL_FEATURES, load_isl_data, load_vocab, set_threads
    from app.isl.training.multisource import GlossLinker, attach_all_sources, write_extra_concepts
    from app.isl.training.pipeline import workdir

    set_threads(args.threads)
    profile = get_profile(args.profile or get_settings().isl_profile).name
    vocab = load_vocab(_vocab_path(args))
    data = load_isl_data(paths, vocab, profile, args.split, max_islrtc=args.max_islrtc or None, seed=args.seed)
    work = workdir(profile)
    data.linker = GlossLinker.from_vocab(vocab)
    if getattr(args, "all_sources", False):
        rep = attach_all_sources(data, paths, vocab, profile, seed=args.seed,
                                 max_per_source=getattr(args, "max_per_source", 0) or None)
        write_extra_concepts(data.linker, work / "extra_concepts.jsonl")
        used = Counter(c.source for c in data.extra if c.concept_id)
        used.update({"include": sum(1 for c in data.include if c.concept_id), "islrtc": len(data.islrtc)})
        _record_sources(work, rep.sources, "words", used)
        _record_sources(work, {"include": {"available": bool(data.include), "licence": "CC-BY-4.0", "commercial": True},
                               "islrtc": {"available": bool(data.islrtc), "licence": "GODL-India", "commercial": True}},
                        "words", used)
        (work / "multisource.json").write_text(json.dumps(rep.to_json(), indent=1, default=str))
    return profile, data, ISL_FEATURES, work


def _train_cfg(args, smoke_defaults: dict):
    from app.training.trainer import TrainConfig

    d = dict(smoke_defaults) if args.smoke else {}
    for k in ("epochs", "hidden", "layers", "batch_size", "lr", "patience"):
        v = getattr(args, k, None)
        if v is not None:
            d[k] = v
    return TrainConfig(**d)


SMOKE_CLS = {"hidden": 64, "layers": 1, "epochs": 8, "batch_size": 32, "patience": 8, "lr": 3e-3}
SMOKE_ENC = {"hidden": 64, "layers": 1, "epochs": 4, "batch_size": 32, "patience": 4, "lr": 2e-3}


def _log(msg: str) -> None:
    print(msg, flush=True)


EX_PARTIAL = 75  # stopped by --max-minutes with the state saved: run the same command again to continue


def cmd_train_classifier(paths: IslPaths, args) -> int:
    from app.isl.training.train_classifier import train_classifier

    profile, data, fcfg, work = _model_ctx(paths, args)
    _print({"data": data.summary()})
    out = Path(args.ckpt) if getattr(args, "ckpt", None) else work / "classifier.pt"
    res = train_classifier(data, out, fcfg, _train_cfg(args, SMOKE_CLS), smoke=args.smoke,
                           threads=args.threads, progress=lambda f, m: _log(f"  classifier {m}"),
                           use_extra=args.all_sources, min_signers=args.min_signers, resume=args.resume,
                           max_minutes=args.max_minutes, steps_per_epoch=args.batches)
    _print(res)
    return 0 if res.get("finished", True) else EX_PARTIAL


def cmd_train_encoder(paths: IslPaths, args) -> int:
    from app.isl.training.train_encoder import train_encoder

    profile, data, fcfg, work = _model_ctx(paths, args)
    _print({"data": data.summary()})
    res = train_encoder(data, work / "encoder.pt", fcfg, _train_cfg(args, SMOKE_ENC),
                        emb_dim=args.emb_dim or (128 if args.smoke else 256), p=args.p, k=args.k, smoke=args.smoke,
                        threads=args.threads, batches_per_epoch=args.batches or (20 if args.smoke else None),
                        progress=lambda f, m: _log(f"  encoder {m}"), init_from=_init_path(args, work),
                        use_extra=args.all_sources, resume=args.resume, max_minutes=args.max_minutes)
    _print(res)
    return 0 if res.get("finished", True) else EX_PARTIAL


def _init_path(args, work: Path) -> Path | None:
    v = getattr(args, "init", None)
    if not v or v == "none":
        return None
    if v == "auto":
        p = work / "pretrained_encoder.pt"
        return p if p.exists() else None
    return Path(v)


def _eval_clip_ids(data) -> set[str]:
    from app.isl.training.multisource import eval_clip_ids

    return eval_clip_ids(data)


def cmd_pretrain(paths: IslPaths, args) -> int:
    """Self-supervised encoder pre-training on every unlabelled / sentence source of the profile."""
    import random as _r

    from app.isl.training.multisource import pretrain_clips, write_pretrain_list
    from app.isl.training.pose_dedup import DEDUP_FILE, apply_dedup, load_dedup
    from app.isl.training.pretrain import PretrainConfig, pretrain_encoder

    args.all_sources = True
    profile, data, fcfg, work = _model_ctx(paths, args)
    rep: dict = {}
    clips = pretrain_clips(paths, None, profile, exclude_ids=_eval_clip_ids(data), report=rep)
    # OpenHands Sign Library: chunks that repeat another chunk or contain an evaluation clip (pretrain-dedup step);
    # without that step's result the whole channel is left out
    clips, sl_rep = apply_dedup(clips, load_dedup(work / DEDUP_FILE))
    if args.max_clips and len(clips) > args.max_clips:
        clips = _r.Random(args.seed).sample(clips, args.max_clips)
    _record_sources(work, rep, "pretraining", Counter(c.source for c in clips))
    lst = write_pretrain_list(work, clips, resume=args.resume)
    d = PretrainConfig()
    cfg = PretrainConfig(**{**d.__dict__, **({"hidden": 64, "layers": 1, "emb_dim": 128, "batch": 16, "epochs": 2,
                                               "steps_per_epoch": 10, "val_batches": 1} if args.smoke else {}),
                            **{k: v for k, v in {"hidden": args.hidden, "layers": args.layers, "emb_dim": args.emb_dim,
                                                 "batch": args.batch_size, "epochs": args.epochs,
                                                 "steps_per_epoch": args.batches, "lr": args.lr,
                                                 "seed": args.seed}.items() if v is not None}})
    _print({"pretrain_clips": len(clips), "by_source": dict(Counter(c.source for c in clips)), "sign_library": sl_rep,
            "clip_list": lst, "config": cfg.__dict__})
    res = pretrain_encoder(clips, work / "pretrained_encoder.pt", fcfg, cfg, profile=profile,
                           max_minutes=args.max_minutes, fresh=not args.resume, threads=args.threads,
                           smoke=args.smoke, progress=lambda m: _log(f"  pretrain {m}"),
                           extra_meta={"clip_list": lst, "sign_library": sl_rep})
    _print(res)
    return 0 if res.get("finished", True) else EX_PARTIAL


def cmd_pretrain_dedup(paths: IslPaths, args) -> int:
    """OpenHands Sign Library: find chunks that repeat another Sign Library chunk (its two files) or contain an
    evaluation clip, by pose content -> <work>/pretrain_dedup.json (read by ``pretrain``)."""
    import random as _r

    from app.isl.training.multisource import PRETRAIN_SOURCES, eval_clips, open_adapter
    from app.isl.training.pose_dedup import DEDUP_FILE, SIGN_LIBRARY, dedupe_sign_library

    args.all_sources = True
    profile, data, fcfg, work = _model_ctx(paths, args)
    out = work / DEDUP_FILE
    ad = open_adapter("openhands_raw", paths, None, profile) if "openhands_raw" in PRETRAIN_SOURCES[profile] else None
    sl = [c for c in ad.clips() if c.category == SIGN_LIBRARY] if ad is not None else []
    if not sl:
        doc = {"skipped": True, "reason": "no OpenHands Sign Library chunks in this profile's pre-training data",
               "reliable": True, "drop_ids": [], "drop_channels": [], "checked_ids": []}
        out.write_text(json.dumps(doc, indent=1))
        _print(doc)
        return 0
    if args.max_chunks and len(sl) > args.max_chunks:  # smoke: every one-minute chunk + a sample of the others
        one = [c for c in sl if "1_min" in str(c.quality.get("member") or "")]
        rest = [c for c in sl if c not in one]
        sl = one + _r.Random(args.seed).sample(rest, min(len(rest), max(0, args.max_chunks - len(one))))
    res = dedupe_sign_library(sl, eval_clips(data), seed=args.seed, n_calib=args.calib, progress=_log)
    doc = {**res.to_json(), "profile": profile, "split": data.split.get("name"), "sign_library_chunks_checked": len(sl)}
    out.write_text(json.dumps(doc, indent=1, default=str))
    _print({k: v for k, v in doc.items() if k not in ("drop_ids", "checked_ids", "within_examples", "eval_matches")}
           | {"drop_ids": len(doc["drop_ids"]), "checked_ids": len(doc["checked_ids"])})
    return 0


def cmd_train_oneshot(paths: IslPaths, args) -> int:
    """Encoders for the held-out-concept one-shot protocol (E2): one per fold, trained without the fold's
    concepts (and every concept sharing a headword / synonym with them) in any supervised loss."""
    from app.isl.evaluation.runners import heldout_set
    from app.isl.training.train_encoder import train_encoder

    profile, data, fcfg, work = _model_ctx(paths, args)
    one = paths.splits / "oneshot_concepts.json"
    if not one.exists():
        _print({"skipped": True, "reason": "no oneshot_concepts.json: run 'splits' with a vocabulary first"})
        return 0
    doc = json.loads(one.read_text())
    out, code = {}, 0
    for k in args.folds:
        fd = next((f for f in doc["protocol"]["folds"] if f["fold"] == k), None)
        if fd is None or not fd["held_out_concepts"]:
            out[k] = {"skipped": True}
            continue
        held = heldout_set(data.linker, fd["held_out_concepts"])
        d = work / "oneshot" / f"fold{k}"
        d.mkdir(parents=True, exist_ok=True)
        res = train_encoder(data, d / "encoder.pt", fcfg, _train_cfg(args, SMOKE_ENC),
                            emb_dim=args.emb_dim or (128 if args.smoke else 256), p=args.p, k=args.k,
                            exclude_concepts=held, smoke=args.smoke, threads=args.threads,
                            batches_per_epoch=args.batches or (20 if args.smoke else None),
                            progress=lambda f, m, _k=k: _log(f"  oneshot fold{_k} {m}"),
                            init_from=_init_path(args, work), use_extra=args.all_sources, resume=args.resume,
                            max_minutes=args.max_minutes, version=f"isl-enc-1-oneshot{k}")
        out[k] = {**res, "held_out": len(fd["held_out_concepts"]), "held_out_expanded": len(held)}
        if not res.get("finished", True):
            code = EX_PARTIAL
            break
    _print(out)
    return code


def cmd_train_fingerspell(paths: IslPaths, args) -> int:
    from app.isl.fingerspell.lexicon import build_spelling_lexicon
    from app.isl.profiles import get_profile
    from app.isl.training.common import set_threads
    from app.isl.training.pipeline import workdir
    from app.isl.training.train_fingerspell import letter_data, train_fingerspell
    from app.packs.vocabulary import Vocabulary

    set_threads(args.threads)
    profile = get_profile(args.profile or "commercial-safe").name
    work = workdir(profile)
    rep: dict = {}
    ls = letter_data(paths, profile, report=rep)
    srcs = {}
    from app.isl.training.multisource import open_adapter

    for n, d in rep.items():
        ad = open_adapter(n, paths, None, profile) if d.get("available") else None
        srcs[n] = {**d, "licence": getattr(ad, "licence", ""), "commercial": bool(getattr(ad, "commercial", False))}
    _record_sources(work, srcs, "letters")
    words = [w for w, _ in build_spelling_lexicon(Vocabulary.load(_vocab_path(args)).rows)]
    res = train_fingerspell(ls, work / "letters.pt", profile=profile, epochs=args.epochs or (5 if args.smoke else 40),
                            seed=args.seed, lexicon_words=words, smoke=args.smoke)
    _print({"sources": rep, **res})
    return 0


def cmd_bench(paths: IslPaths, args) -> int:
    """CPU time per optimisation step for the phase-2 model sizes (synthetic batches, no data needed)."""
    import time as _t

    import torch

    from app.isl.models.encoder import SignEncoder
    from app.isl.models.losses import SubCenterArcFace, SupConLoss
    from app.isl.training.common import ISL_FEATURES, set_threads
    from app.isl.training.pretrain import _Pretrainer, nt_xent
    from app.models.networks import build_model

    set_threads(args.threads)
    fcfg = ISL_FEATURES
    T, D = fcfg.seq_len, fcfg.dim
    out = {"threads": args.threads, "seq_len": T, "dim": D}
    for hidden, layers in [(h, l) for h in args.hidden_sizes for l in (args.layers or 2,)]:
        enc = SignEncoder(D, hidden=hidden, layers=layers, emb_dim=256)
        pre = _Pretrainer(enc, D, hidden)
        opt = torch.optim.AdamW(pre.parameters(), 1e-3)
        B = args.batch_size or 48
        xa, xb = torch.randn(B, T, D), torch.randn(B, T, D)
        times = []
        for i in range(args.steps + 1):
            t0 = _t.perf_counter()
            za, ha = pre.enc.encode(xa)
            zb, _ = pre.enc.encode(xb)
            loss = nt_xent(za, zb, 0.1)[0] + ((pre.recon(ha) - xa) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i:
                times.append(_t.perf_counter() - t0)
        out[f"pretrain_h{hidden}_l{layers}_b{B}x2_s"] = round(float(sum(times) / len(times)), 3)
        arc = SubCenterArcFace(256, 12000, k=3)
        sc = SupConLoss(0.1)
        params = list(enc.parameters()) + list(arc.parameters())
        opt = torch.optim.AdamW(params, 1e-3)
        x = torch.randn(64, T, D)
        y = torch.arange(32).repeat_interleave(2)
        times = []
        for i in range(args.steps + 1):
            t0 = _t.perf_counter()
            z = enc(x)
            loss = sc(z, y) + 0.5 * arc(z, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i:
                times.append(_t.perf_counter() - t0)
        out[f"encoder_h{hidden}_l{layers}_b64_arc12k_s"] = round(float(sum(times) / len(times)), 3)
        m = build_model("gru", D, 600, hidden=hidden, layers=layers, bidirectional=True, dropout=0.3)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        x = torch.randn(32, T, D)
        y = torch.randint(0, 600, (32,))
        times = []
        for i in range(args.steps + 1):
            t0 = _t.perf_counter()
            loss = torch.nn.functional.cross_entropy(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i:
                times.append(_t.perf_counter() - t0)
        out[f"classifier_h{hidden}_l{layers}_b32_s"] = round(float(sum(times) / len(times)), 3)
    # data side: strong augmentation + featurisation of one clip
    from app.dataset.synthetic import make_sequence
    from app.features.build import featurize, prepare_body_frame
    from app.isl.augment import augment_strong

    import numpy as _np

    raw = make_sequence(1, n_frames=60, fps=12.0)
    rng = _np.random.default_rng(0)
    t0 = _t.perf_counter()
    for _ in range(50):
        featurize(augment_strong(prepare_body_frame(raw, fcfg), rng), fcfg)
    out["featurize_augment_per_clip_ms"] = round((_t.perf_counter() - t0) / 50 * 1000, 2)
    out["note"] = "measured on this machine now; other jobs share the CPUs, so times vary"
    _print(out)
    return 0


def cmd_build_index(paths: IslPaths, args) -> int:
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index

    profile, data, fcfg, work = _model_ctx(paths, args)
    enc = EncoderPredictor(work / "encoder.pt")
    include = data.include_part("train") if not args.all_include else [c for c in data.include if c.concept_id]
    # --all-include (final pack) indexes every reference; otherwise only training parts (evaluation stays clean)
    extra = [c for c in data.extra if args.all_include or data.extra_assign.get(c.clip_id) == "train"]
    _print(build_index(enc, data.islrtc, include, work / "index", profile, extra=extra or None))
    return 0


def cmd_fit_fusion(paths: IslPaths, args) -> int:
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import load_recognizer

    profile, data, fcfg, work = _model_ctx(paths, args)
    rec = load_recognizer(work, fcfg, _vocab_path(args), profile=profile)
    params, rep = fit_fusion(rec, data.labelled_part("val"), profile)
    assert set(params.accept) == {"strict", "balanced", "lenient"}, params.accept  # decide() needs all three
    params.save(work / "fusion.json")
    _print(rep)
    return 0


def cmd_evaluate(paths: IslPaths, args) -> int:
    from app.isl.training.pipeline import evaluate

    profile, data, fcfg, work = _model_ctx(paths, args)
    if getattr(args, "all", False):
        from app.isl.evaluation.runners import evaluate_all
        from app.isl.training.pipeline import pack_dir

        outs = [Path(args.out) / "eval"] if args.out else [pack_dir(profile) / "eval"]
        doc = evaluate_all(work, data, fcfg, profile=profile, smoke=args.smoke, vocab_path=_vocab_path(args),
                           paths=paths, out_dirs=outs, progress=lambda m: _log(f"  evaluate {m}"))
        _print({"report": [str(work / "eval" / "report.json")] + [str(o / "report.json") for o in outs],
                **{k: {"n": v.get("n"), "metrics": v.get("metrics"), "run": v.get("run", True)}
                   for k, v in doc["protocols"].items()}})
        return 0
    doc = evaluate(work, data, fcfg, smoke=args.smoke, part=args.part, vocab_path=_vocab_path(args))
    _print({k: {"n": v.get("n"), "metrics": v.get("metrics")} for k, v in doc["protocols"].items()})
    return 0


def cmd_train_letters(paths: IslPaths, args) -> int:
    import string

    from app.isl.training.pipeline import workdir
    from app.isl.training.train_letters import examples_from_sequence, train_letter_model
    from app.landmarks.raw import RawSequence

    letters = list(args.letters or string.ascii_uppercase)
    X, y, noise = [], [], []
    for c in args.clips or []:
        x1, y1, n1 = examples_from_sequence(RawSequence.load(Path(c)), letters)
        X += x1
        y += y1
        noise += n1
    if not X:
        _print({"skipped": True, "reason": "no in-order alphabet recordings given (--clips) or no holds aligned. The "
                "ISLRTC dictionary has no A-Z video; spelling mode uses each user's alphabet calibration instead."})
        return 0
    out = workdir(args.profile or "commercial-safe") / "letters.pt"
    _print(train_letter_model(X, y, noise, out, n_signers=args.signers, source=",".join(args.clips)))
    return 0


def cmd_build_pack(paths: IslPaths, args) -> int:
    from app.isl.profiles import get_profile
    from app.isl.training.common import ISL_FEATURES
    from app.isl.training.pipeline import build_pack, workdir

    profile = get_profile(args.profile or "commercial-safe").name
    _print(build_pack(workdir(profile), profile, ISL_FEATURES, Path(args.out) if args.out else None, smoke=args.smoke,
                      vocab_path=_vocab_path(args)))
    return 0


def cmd_link_sources(paths: IslPaths, args) -> int:
    """Read every labelled source of the profile, link glosses to concepts, split signers, drop duplicates."""
    args.all_sources = True
    profile, data, fcfg, work = _model_ctx(paths, args)
    _print({"data": data.summary(), "multisource": data.multisource, "work": str(work)})
    return 0


def cmd_smoke(paths: IslPaths, args) -> int:
    """End-to-end smoke run on the data on disk: classifier, encoder, index, fusion, evaluation, pack."""
    import time

    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import build_pack, evaluate, load_recognizer
    from app.isl.training.train_classifier import train_classifier
    from app.isl.training.train_encoder import train_encoder

    args.smoke = True
    t0 = time.time()
    # first-sign trims for dictionary clips ingested since the last run (incremental, a few ms per clip)
    cmd_trim(paths, argparse.Namespace(redo=False, limit=0, learned=False, learned_model="auto"))
    profile, data, fcfg, work = _model_ctx(paths, args)
    report: dict = {"data": data.summary(), "work": str(work)}
    _log(f"[smoke] data: {report['data']}")
    report["classifier"] = train_classifier(data, work / "classifier.pt", fcfg, _train_cfg(args, SMOKE_CLS), smoke=True,
                                            threads=args.threads, progress=lambda f, m: _log(f"  classifier {m}"))
    _log(f"[smoke] classifier: {report['classifier']}")
    report["encoder"] = train_encoder(data, work / "encoder.pt", fcfg, _train_cfg(args, SMOKE_ENC), emb_dim=128, p=args.p,
                                      k=args.k, smoke=True, threads=args.threads, batches_per_epoch=args.batches or 20,
                                      progress=lambda f, m: _log(f"  encoder {m}"))
    _log(f"[smoke] encoder: {report['encoder']}")
    report["index"] = build_index(EncoderPredictor(work / "encoder.pt"), data.islrtc, data.include_part("train"),
                                  work / "index", profile)
    _log(f"[smoke] index: {report['index']}")
    rec = load_recognizer(work, fcfg, profile=profile)
    params, report["fusion"] = fit_fusion(rec, data.include_part("val"), profile)
    params.save(work / "fusion.json")
    _log(f"[smoke] fusion: {report['fusion'].get('weights')} {report['fusion'].get('thresholds')}")
    doc = evaluate(work, data, fcfg, smoke=True)
    report["evaluation"] = {k: {"n": v.get("n"), "metrics": v.get("metrics")} for k, v in doc["protocols"].items()}
    report["pack"] = build_pack(work, profile, fcfg, Path(args.out) if args.out else None, smoke=True)
    report["seconds"] = round(time.time() - t0, 1)
    (work / "smoke_report.json").write_text(json.dumps(report, indent=1, default=str))
    _print(report)
    return 0


ASL_PACK_DIR = PACK_DIR.parent / "asl-wlasl100"


def _asl_class_names() -> list[str]:
    """WLASL100 glosses of the bundled ASL model (the ASL pack's concept ids)."""
    from app.core.config import REPO_ROOT

    mj = REPO_ROOT / "models" / "wlasl100-gru-signer-holdout" / "model.json"
    return list(json.loads(mj.read_text()).get("class_names", [])) if mj.exists() else []


def cmd_hindi(paths: IslPaths, args) -> int:
    """Hindi lexicon (docs/ISL_DESIGN.md section 8.1): curated core list + number rules + IndicTrans2 drafts."""
    from app.isl.text import hindi_build as hb
    from app.packs.vocabulary import Vocabulary

    base = Path(args.out) if args.out else PACK_DIR / "lexicon_hi.jsonl"
    asl_out = ASL_PACK_DIR / "lexicon_hi.jsonl"
    from app.core.config import REPO_ROOT, get_settings

    local = get_settings().data_dir / "isl" / "lexicon_hi.local.jsonl" if args.local is None else Path(args.local)
    if args.merge_local:
        _print({"merged": hb.merge_local(base, local), "from": str(local), "into": str(base)})
        return 0
    vocab = Vocabulary.load(PACK_DIR / "vocabulary.jsonl").rows
    asl = _asl_class_names()
    core = hb.load_core()
    cache_path = Path(args.cache) if args.cache else paths.derived / "hindi_mt_cache.jsonl"
    if args.translate:
        from app.isl.text.mt import IndicTrans2, fetch_model

        model_dir = Path(args.model_dir) if args.model_dir else (
            REPO_ROOT / "artifacts" / "hf" / "indictrans2-en-indic-dist-200M")
        fetch_model(model_dir, log=_log)
        cache = hb.read_cache(cache_path)
        asl_rows = hb.asl_vocab_rows(asl)
        core_asl = {k: v for k, v in core.items() if not k.startswith("isl.")}
        need = [i for r in vocab if hb._core_for(r, core) is None and r.get("kind") != "number"
                for i in hb.mt_inputs(r)]
        need += [i for r in asl_rows if hb._core_for(r, core_asl) is None for i in hb.mt_inputs(r)]
        need = [i for i in need if i not in cache]
        _log(f"[hindi] {len(set(need))} inputs to translate ({len(cache)} cached)")
        tr = IndicTrans2(model_dir, threads=args.threads, num_beams=args.beams)
        hb.translate_missing(need, cache_path, tr, batch=args.batch, limit=args.limit or None, log=_log)
    cache = hb.read_cache(cache_path)
    rows, stats = hb.build_rows(vocab, core, cache, hb.read_existing(base))
    hb.write_rows(base, rows)
    out = {"isl": {"file": str(base), **stats}}
    if asl:
        arows, astats = hb.build_asl_rows(asl, core, cache, hb.read_existing(asl_out))
        hb.write_rows(asl_out, arows)
        out["asl"] = {"file": str(asl_out), **astats}
    out["cache"] = {"file": str(cache_path), "entries": len(cache)}
    out["core_list"] = {"file": str(hb.CORE_TSV), "entries": len(core)}
    _print(out)
    return 0


COMMANDS = {
    "status": cmd_status, "wordnet": cmd_wordnet, "vocab": cmd_vocab, "trim": cmd_trim,
    "trim-review": cmd_trim_review, "trim-eval": cmd_trim_eval, "splits": cmd_splits, "dedup": cmd_dedup,
    "access": cmd_access, "profiles": cmd_profiles,
    "train-classifier": cmd_train_classifier, "train-encoder": cmd_train_encoder, "build-index": cmd_build_index,
    "fit-fusion": cmd_fit_fusion, "evaluate": cmd_evaluate, "train-letters": cmd_train_letters,
    "build-pack": cmd_build_pack, "smoke": cmd_smoke, "hindi": cmd_hindi,
    "link-sources": cmd_link_sources, "pretrain": cmd_pretrain, "train-oneshot": cmd_train_oneshot,
    "train-fingerspell": cmd_train_fingerspell, "bench": cmd_bench, "readiness": cmd_readiness,
    "pretrain-dedup": cmd_pretrain_dedup,
}


def _model_args(sp, train: bool = False) -> None:
    sp.add_argument("--profile", choices=["commercial-safe", "research"], default=None)
    sp.add_argument("--vocab", help="vocabulary.jsonl (default models/packs/isl/vocabulary.jsonl)")
    sp.add_argument("--all-sources", action="store_true",
                    help="phase 2: add every other labelled word source of the profile (ISL500, WSLP, CISLR, ...)")
    sp.add_argument("--max-per-source", type=int, default=0, help="cap clips per extra source (smoke runs)")
    sp.add_argument("--split", help="INCLUDE split name (default: the primary split, fold 0)")
    sp.add_argument("--threads", type=int, default=2)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--max-islrtc", type=int, default=0, help="use at most N ISLRTC clips (0 = all)")
    sp.add_argument("--smoke", action="store_true", help="tiny settings; results are labelled as a smoke run")
    sp.add_argument("--out", help="build-pack / smoke: pack output folder")
    if train:
        for k, t in (("epochs", int), ("hidden", int), ("layers", int), ("batch-size", int), ("lr", float),
                     ("patience", int)):
            sp.add_argument(f"--{k}", type=t, default=None)
        sp.add_argument("--emb-dim", type=int, default=None)
        sp.add_argument("--p", type=int, default=32, help="encoder: concepts per batch")
        sp.add_argument("--k", type=int, default=2, help="encoder: clips per concept in a batch")
        sp.add_argument("--batches", type=int, default=None, help="batches (steps) per epoch")
        sp.add_argument("--resume", action="store_true", help="continue from the saved training state")
        sp.add_argument("--max-minutes", type=float, default=0.0,
                        help="stop after this long with the state saved (exit code 75); run again with --resume")
        sp.add_argument("--init", default=None, help="encoder warm start: a checkpoint, 'auto' (pretrained) or 'none'")
        sp.add_argument("--min-signers", type=int, default=2, help="classifier: signer groups a class needs")
        sp.add_argument("--ckpt", help="classifier: checkpoint path (default <work>/classifier.pt)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.isl.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", help="ISL data root (default: <SLR_DATA_DIR>/isl or $SLR_ISL_DIR)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--offline", action="store_true", help="skip the Hugging Face access check")
    sub.add_parser("wordnet")
    v = sub.add_parser("vocab")
    v.add_argument("--out")
    v.add_argument("--no-wordnet", action="store_true")
    v.add_argument("--no-trims", action="store_true", help="do not copy trim info into the vocabulary")
    t = sub.add_parser("trim")
    t.add_argument("--redo", action="store_true")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--learned", action="store_true", help="old stage-2 boundary model (artifacts/isl/boundary.pt)")
    t.add_argument("--learned-model", default="auto",
                   help="learned end model: auto (artifacts/isl/trim_boundary.pt when present), a path, or none")
    t.add_argument("--retrain", action="store_true")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--examples", type=int, default=2000)
    t.add_argument("--threads", type=int, default=2)
    r = sub.add_parser("trim-review")
    r.add_argument("--n", type=int, default=60)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out")
    e = sub.add_parser("trim-eval")
    e.add_argument("--n", type=int, default=300)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--learned", action="store_true", help="also apply the stage-2 boundary model")
    sp_ = sub.add_parser("splits")
    sp_.add_argument("--fetch", action="store_true", help="(re)download the official INCLUDE split lists")
    sp_.add_argument("--no-fetch", action="store_true", help="never download (use what is on disk)")
    sp_.add_argument("--k", type=int, default=5)
    sp_.add_argument("--seed", type=int, default=0)
    sp_.add_argument("--vocab", help="vocabulary.jsonl (default models/packs/isl/vocabulary.jsonl)")
    d = sub.add_parser("dedup")
    d.add_argument("--sources", nargs="+", default=["include", "islrtc", "cislr"])
    d.add_argument("--threshold", type=float, default=0.02)
    sub.add_parser("access")
    sub.add_parser("profiles")
    rd = sub.add_parser("readiness")
    rd.add_argument("--profile", choices=["commercial-safe", "research"], default=None)
    rd.add_argument("--out", help="write the full report (JSON) here")
    rd.add_argument("--work", help="work folder to mark partial_data when --allow-partial is given")
    rd.add_argument("--allow-partial", action="store_true", help="exit 0 even when not ready (marks --work)")
    for name in ("train-classifier", "train-encoder", "smoke", "pretrain", "train-oneshot"):
        spx = sub.add_parser(name)
        _model_args(spx, train=True)
        if name == "pretrain":
            spx.add_argument("--max-clips", type=int, default=0, help="use at most N unlabelled clips (0 = all)")
        if name == "train-oneshot":
            spx.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    lk = sub.add_parser("link-sources")
    _model_args(lk)
    pdd = sub.add_parser("pretrain-dedup")
    _model_args(pdd)
    pdd.add_argument("--max-chunks", type=int, default=0, help="smoke: check at most N Sign Library chunks")
    pdd.add_argument("--calib", type=int, default=300, help="synthetic positive / negative controls")
    fs = sub.add_parser("train-fingerspell")
    _model_args(fs)
    fs.add_argument("--epochs", type=int, default=None)
    bn = sub.add_parser("bench")
    bn.add_argument("--threads", type=int, default=2)
    bn.add_argument("--hidden-sizes", type=int, nargs="+", default=[128, 192])
    bn.add_argument("--layers", type=int, default=2)
    bn.add_argument("--batch-size", type=int, default=48)
    bn.add_argument("--steps", type=int, default=5)
    for name in ("build-index", "fit-fusion", "evaluate", "build-pack"):
        sp2 = sub.add_parser(name)
        _model_args(sp2)
        if name == "build-index":
            sp2.add_argument("--all-include", action="store_true", help="centroids from every INCLUDE clip (final pack)")
        if name == "evaluate":
            sp2.add_argument("--part", default="test", choices=["val", "test"])
            sp2.add_argument("--all", action="store_true",
                             help="every protocol (E1a-E8) -> <work>/eval and <pack>/eval report.json + report.md")
    hi = sub.add_parser("hindi")
    hi.add_argument("--translate", action="store_true", help="run IndicTrans2 on CPU for concepts not in the cache")
    hi.add_argument("--limit", type=int, default=0, help="translate at most N new inputs (0 = all)")
    hi.add_argument("--threads", type=int, default=2)
    hi.add_argument("--beams", type=int, default=4)
    hi.add_argument("--batch", type=int, default=24)
    hi.add_argument("--model-dir", help="IndicTrans2 folder (default artifacts/hf/indictrans2-en-indic-dist-200M)")
    hi.add_argument("--cache", help="translation cache (default data/isl/derived/hindi_mt_cache.jsonl)")
    hi.add_argument("--out", help="lexicon file (default models/packs/isl/lexicon_hi.jsonl)")
    hi.add_argument("--merge-local", action="store_true",
                    help="fold local corrections (data/isl/lexicon_hi.local.jsonl) into the tracked lexicon")
    hi.add_argument("--local", default=None, help="local corrections file for --merge-local")
    tl = sub.add_parser("train-letters")
    tl.add_argument("--clips", nargs="*", help="RawSequence .npz files of the alphabet signed in order")
    tl.add_argument("--letters", help="letters in the recordings, in order (default A-Z)")
    tl.add_argument("--signers", type=int, default=1)
    tl.add_argument("--profile", choices=["commercial-safe", "research"], default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = isl_paths(args.root)
    return COMMANDS[args.cmd](paths, args)


if __name__ == "__main__":
    sys.exit(main())
