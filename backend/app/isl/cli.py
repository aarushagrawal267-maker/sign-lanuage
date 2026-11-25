"""ISL command-line interface (data commands; model commands are added by later phases).

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
    from app.isl.data.manifest import read_manifest

    return [r for r in read_manifest(paths.manifest("islrtc")) if r.get("n_frames")]


def cmd_trim(paths: IslPaths, args) -> int:
    import time

    from app.isl.data.manifest import read_manifest, write_jsonl
    from app.isl.data.trim import TrimConfig, trim_record_for, words_for
    from app.isl.vocab.normalize import parse_islrtc_name
    from app.landmarks.raw import RawSequence

    cfg = TrimConfig()
    existing = {r["id"]: r for r in read_manifest(paths.trim_sidecar, only_ok=False)}
    recs = _islrtc_records(paths)
    todo = [r for r in recs if args.redo or r["id"] not in existing
            or existing[r["id"]].get("trim_version") != cfg.version]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    n_err = 0
    for i, r in enumerate(todo):
        pn = parse_islrtc_name(r["id"].removeprefix("islrtc/"))
        try:
            raw = RawSequence.load(paths.resolve(r["path"]))
            rec = trim_record_for(r["id"], raw, r.get("stop"), pn.kind, cfg, n_words=words_for(pn.kind, pn.text))
        except Exception as e:  # noqa: BLE001
            n_err += 1
            rec = {"id": r["id"], "start": 0, "end": int(r.get("n_frames") or 0), "method": "empty", "confidence": 0.0,
                   "alternatives": [], "notes": [f"{type(e).__name__}: {e}"[:200]], "trim_version": cfg.version}
        rec["stop"] = r.get("stop")
        rec["kind"] = pn.kind
        existing[r["id"]] = rec
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(todo)} trimmed ({time.time() - t0:.0f}s)", flush=True)
    if args.learned:
        existing = _apply_learned(paths, existing, args)
    write_jsonl(paths.trim_sidecar, [existing[k] for k in sorted(existing)])
    methods = Counter(r.get("method") for r in existing.values())
    _print({"trimmed_now": len(todo), "errors": n_err, "total": len(existing), "methods": dict(methods),
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

    clips = []
    for s in args.sources:
        ad = get_adapter(s, paths)
        if ad.available():
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


def cmd_profiles(paths: IslPaths, args) -> int:
    from app.isl.profiles import PROFILES

    _print({k: v.to_dict() for k, v in PROFILES.items()})
    return 0


COMMANDS = {
    "status": cmd_status, "wordnet": cmd_wordnet, "vocab": cmd_vocab, "trim": cmd_trim,
    "trim-review": cmd_trim_review, "trim-eval": cmd_trim_eval, "splits": cmd_splits, "dedup": cmd_dedup,
    "access": cmd_access, "profiles": cmd_profiles,
}


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
    t.add_argument("--learned", action="store_true")
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = isl_paths(args.root)
    return COMMANDS[args.cmd](paths, args)


if __name__ == "__main__":
    sys.exit(main())
