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


# --------------------------------------------------------------------------------------------- model commands

def _model_ctx(paths: IslPaths, args):
    from app.core.config import get_settings
    from app.isl.profiles import get_profile
    from app.isl.training.common import ISL_FEATURES, load_isl_data, load_vocab, set_threads
    from app.isl.training.pipeline import workdir

    set_threads(args.threads)
    profile = get_profile(args.profile or get_settings().isl_profile).name
    vocab = load_vocab()
    data = load_isl_data(paths, vocab, profile, args.split, max_islrtc=args.max_islrtc or None, seed=args.seed)
    return profile, data, ISL_FEATURES, workdir(profile)


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


def cmd_train_classifier(paths: IslPaths, args) -> int:
    from app.isl.training.train_classifier import train_classifier

    profile, data, fcfg, work = _model_ctx(paths, args)
    _print({"data": data.summary()})
    res = train_classifier(data, work / "classifier.pt", fcfg, _train_cfg(args, SMOKE_CLS), smoke=args.smoke,
                           threads=args.threads, progress=lambda f, m: _log(f"  classifier {m}"))
    _print(res)
    return 0


def cmd_train_encoder(paths: IslPaths, args) -> int:
    from app.isl.training.train_encoder import train_encoder

    profile, data, fcfg, work = _model_ctx(paths, args)
    _print({"data": data.summary()})
    res = train_encoder(data, work / "encoder.pt", fcfg, _train_cfg(args, SMOKE_ENC),
                        emb_dim=args.emb_dim or (128 if args.smoke else 256), p=args.p, k=args.k, smoke=args.smoke,
                        threads=args.threads, batches_per_epoch=args.batches or (20 if args.smoke else None),
                        progress=lambda f, m: _log(f"  encoder {m}"))
    _print(res)
    return 0


def cmd_build_index(paths: IslPaths, args) -> int:
    from app.isl.models.encoder import EncoderPredictor
    from app.isl.training.build_index import build_index

    profile, data, fcfg, work = _model_ctx(paths, args)
    enc = EncoderPredictor(work / "encoder.pt")
    include = data.include_part("train") if not args.all_include else [c for c in data.include if c.concept_id]
    _print(build_index(enc, data.islrtc, include, work / "index", profile))
    return 0


def cmd_fit_fusion(paths: IslPaths, args) -> int:
    from app.isl.training.fit_fusion import fit_fusion
    from app.isl.training.pipeline import load_recognizer

    profile, data, fcfg, work = _model_ctx(paths, args)
    rec = load_recognizer(work, fcfg, profile=profile)
    params, rep = fit_fusion(rec, data.include_part("val"), profile)
    params.save(work / "fusion.json")
    _print(rep)
    return 0


def cmd_evaluate(paths: IslPaths, args) -> int:
    from app.isl.training.pipeline import evaluate

    profile, data, fcfg, work = _model_ctx(paths, args)
    doc = evaluate(work, data, fcfg, smoke=args.smoke, part=args.part)
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
    _print(build_pack(workdir(profile), profile, ISL_FEATURES, Path(args.out) if args.out else None, smoke=args.smoke))
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
    cmd_trim(paths, argparse.Namespace(redo=False, limit=0, learned=False))
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
}


def _model_args(sp, train: bool = False) -> None:
    sp.add_argument("--profile", choices=["commercial-safe", "research"], default=None)
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
        sp.add_argument("--batches", type=int, default=None, help="encoder: batches per epoch")


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
    for name in ("train-classifier", "train-encoder", "smoke"):
        _model_args(sub.add_parser(name), train=True)
    for name in ("build-index", "fit-fusion", "evaluate", "build-pack"):
        sp2 = sub.add_parser(name)
        _model_args(sp2)
        if name == "build-index":
            sp2.add_argument("--all-include", action="store_true", help="centroids from every INCLUDE clip (final pack)")
        if name == "evaluate":
            sp2.add_argument("--part", default="test", choices=["val", "test"])
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
