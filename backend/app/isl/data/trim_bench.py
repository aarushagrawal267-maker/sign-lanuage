"""Checks for the ISLRTC first-sign trimmer (docs/DATASETS_ISL.md section 7).

Run from ``backend/`` (the main ISL CLI is owned by other modules, so the trim tooling has its own entry point)::

    python -m app.isl.data.trim_bench sample  --n 64 --seed 2026 > sample.json   # stratified by stop/word count
    python -m app.isl.data.trim_bench sheets  sample.json OUT_DIR [--marked]   # download videos, blind sheets
    python -m app.isl.data.trim_bench labels                                    # trimmer vs by-eye labels
    python -m app.isl.data.trim_bench synthetic [--n 400]                       # synthetic sign+explanation
    python -m app.isl.data.trim_bench include [--n 400]                         # INCLUDE active frames kept
    python -m app.isl.data.trim_bench run [--out FILE] [--learned MODEL]        # trim all ingested clips -> side file

By-eye labels live in ``trim_labels_islrtc.jsonl`` next to this module: one object per clip with the video time
(ms) where the first sign ends (``end_ms``: first contact-sheet tile at 200 ms steps where the hands have left the
sign), ``split`` (``dev`` = used while tuning, ``test`` = held out) and ``status`` (``ok`` | ``unclear`` |
``not_a_sign``). Labels were made on blind sheets (no cut shown).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np

from app.isl.data.trim import TrimConfig, find_first_sign, trim_record_for, words_for

LABELS_PATH = Path(__file__).with_name("trim_labels_islrtc.jsonl")
HF_RESOLVE = "https://huggingface.co/datasets/silentone0725/Indian_Sign_Language_Data.gov_Rencoded/resolve/main/"
SKIP_KINDS = ("phrase", "explanation", "example", "compilation")


# ------------------------------------------------------------------------------------------------ data access

def _paths():
    from app.isl.paths import isl_paths

    return isl_paths()


def islrtc_records(paths) -> list[dict]:
    from app.isl.data.manifest import read_manifest

    return [r for r in read_manifest(paths.manifest("islrtc")) if r.get("n_frames")]


def parsed(clip_id: str):
    from app.isl.vocab.normalize import parse_islrtc_name

    return parse_islrtc_name(clip_id.removeprefix("islrtc/"))


def stop_group(stop: str | None) -> str:
    s = stop or ""
    if s == "max_s":
        return "max_s"
    if s.startswith("max_s"):
        return "max_s+posegap"
    return "bounded"


def word_group(kind: str | None, text: str | None) -> str:
    if kind == "synonym_list":
        return "syn"
    n = words_for(kind, text)
    return "1w" if n <= 1 else ("2w" if n == 2 else "3w+")


# ------------------------------------------------------------------------------------------------ sampling

DEFAULT_ALLOC = {
    ("max_s", "1w"): 10, ("max_s", "2w"): 9, ("max_s", "3w+"): 7, ("max_s", "syn"): 4,
    ("max_s+posegap", "1w"): 5, ("max_s+posegap", "2w"): 3, ("max_s+posegap", "3w+"): 2,
    ("bounded", "1w"): 7, ("bounded", "2w"): 6, ("bounded", "3w+"): 5, ("bounded", "syn"): 2,
    ("max_s+posegap", "syn"): 1, ("bounded", "_card"): 3,
}


def stratified_sample(records: list[dict], seed: int = 2026, alloc: dict | None = None,
                      exclude: set[str] | None = None) -> list[dict]:
    """Random clips per (stop group, word group) stratum; ``_card`` = clips cut at the title card. Within each
    stratum, clips alternate between the ``dev`` and ``test`` split."""
    alloc = alloc or DEFAULT_ALLOC
    exclude = exclude or set()
    rng = random.Random(seed)
    by: dict[tuple[str, str], list[dict]] = {}
    for r in sorted(records, key=lambda x: x["id"]):
        if r["id"] in exclude:
            continue
        pn = parsed(r["id"])
        if pn.kind in SKIP_KINDS:
            continue
        g = stop_group(r.get("stop"))
        by.setdefault((g, word_group(pn.kind, pn.text)), []).append(r)
        if (r.get("stop") or "").startswith("card"):
            by.setdefault(("bounded", "_card"), []).append(r)
    out, seen = [], set()
    for key in sorted(alloc):
        items = [r for r in by.get(key, []) if r["id"] not in seen]
        rng.shuffle(items)
        for k, r in enumerate(items[: alloc[key]]):
            seen.add(r["id"])
            out.append({"id": r["id"], "stratum": "/".join(key), "split": "test" if k % 2 == 0 else "dev",
                        "stop": r.get("stop"), "n_frames": r["n_frames"],
                        "duration_ms": round(1000 * r["duration_s"], 1)})
    return out


# ------------------------------------------------------------------------------------------------ sheets

def _download(url: str, dest: Path) -> None:
    import subprocess

    r = subprocess.run(["curl", "-sSfL", "--retry", "3", "--connect-timeout", "30", "--max-filesize", "200000000",
                        "-o", str(dest), url], capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"download failed ({r.returncode}): {r.stderr.strip()[:200]}")


def render_sheets(sample: list[dict], out_dir: Path, tmp_dir: Path, marked: bool = False, trims: dict | None = None,
                  labels: dict | None = None, full_frame: bool = False) -> list[dict]:
    """Download each video (one at a time, deleted afterwards) and write ``NNN.png`` sheets + ``index.json``."""
    from app.isl.data.trim_sheet import SheetConfig, contact_sheet

    scfg = SheetConfig(crop_x=(0.0, 1.0), crop_y=(0.0, 1.0), tile_w=160) if full_frame else SheetConfig()

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for k, s in enumerate(sample):
        rel = s["id"].removeprefix("islrtc/")
        name = f"{k:03d}.png"
        dest = tmp_dir / f"v{k:03d}.mp4"
        try:
            if not (out_dir / name).exists():
                _download(HF_RESOLVE + urllib.parse.quote(rel), dest)
                kept, marks = None, None
                if marked and trims and s["id"] in trims:
                    t = trims[s["id"]]
                    kept = (t["t_start_ms"], t["t_end_ms"])
                    marks = {"cut": [t["t_end_ms"]], "alt": list(t.get("alt_ms", []))}
                    if labels and s["id"] in labels and labels[s["id"]].get("end_ms") is not None:
                        marks["label"] = [labels[s["id"]]["end_ms"]]
                title = f"{k:03d} {rel}  [{s.get('stop')}]"
                contact_sheet(dest, out_dir / name, title=title, cfg=scfg, duration_ms=s.get("duration_ms"), kept=kept,
                              marks=marks)
            index.append({**s, "sheet": name})
        except Exception as e:  # noqa: BLE001
            index.append({**s, "sheet": None, "error": f"{type(e).__name__}: {e}"[:200]})
        finally:
            dest.unlink(missing_ok=True)
        print(f"  {k + 1}/{len(sample)} {rel}", file=sys.stderr, flush=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False))
    return index


# ------------------------------------------------------------------------------------------------ evaluation

def load_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def score_errors(errs: list[float], tol_ms: float = 300.0, late_ok_ms: float = 600.0) -> dict:
    """``within``: |err| <= tol; ``usable``: -tol <= err <= late_ok (a bit of the hand drop is harmless, a cut
    inside the sign is not)."""
    e = np.asarray(errs, dtype=np.float64)
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "within_300ms": round(float(np.mean(np.abs(e) <= tol_ms)), 3),
            "usable": round(float(np.mean((e >= -tol_ms) & (e <= late_ok_ms))), 3),
            "early": round(float(np.mean(e < -tol_ms)), 3), "late": round(float(np.mean(e > tol_ms)), 3),
            "median_abs_ms": round(float(np.median(np.abs(e))), 1)}


def predict_end_ms(raw, rec: dict, cfg: TrimConfig = TrimConfig()) -> tuple[float, dict]:
    pn = parsed(rec["id"])
    r = trim_record_for(rec["id"], raw, rec.get("stop"), pn.kind, cfg, n_words=words_for(pn.kind, pn.text))
    return float(r["t_end_ms"]), r


def eval_labels(paths, labels: dict[str, dict], cfg: TrimConfig = TrimConfig(), split: str | None = None,
                verbose: bool = False, statuses: tuple[str, ...] = ("ok",)) -> dict:
    from app.landmarks.raw import RawSequence

    recs = {r["id"]: r for r in islrtc_records(paths)}
    errs, per_split, rows = [], {}, []
    for cid, lab in sorted(labels.items()):
        if lab.get("status") not in statuses or cid not in recs or (split and lab.get("split") != split):
            continue
        raw = RawSequence.load(paths.resolve(recs[cid]["path"]))
        pred, r = predict_end_ms(raw, recs[cid], cfg)
        # the label is the first tile after the sign; the trim end is the last kept frame
        err = pred - float(lab["end_ms"])
        errs.append(err)
        per_split.setdefault(lab.get("split", "?"), []).append(err)
        # best of the two cuts (main + ``second``), as used by a two-row index
        sec = r.get("second")
        err2 = err
        if sec:
            e2 = float(raw.t_ms[min(len(raw), int(sec)) - 1]) - float(lab["end_ms"])
            err2 = e2 if abs(e2) < abs(err) else err
        per_split.setdefault(f"{lab.get('split', '?')}_best_of_2", []).append(err2)
        rows.append({"id": cid, "err_ms": round(err), "err_best2_ms": round(err2), "method": r["method"],
                     "pred_ms": round(pred), "label_ms": lab["end_ms"], "split": lab.get("split"),
                     "stratum": lab.get("stratum"), "status": lab.get("status")})
    out = {"all": score_errors(errs), **{k: score_errors(v) for k, v in sorted(per_split.items())},
           "methods": dict(Counter(r["method"] for r in rows))}
    if verbose:
        out["rows"] = rows
    return out


def _half(clip_id: str) -> str:
    """Stable split of clips into a 'tune' half and a 'report' half (crc32 parity)."""
    import zlib

    return "tune" if zlib.crc32(clip_id.encode()) % 2 == 0 else "report"


def synthetic_eval(paths, n: int = 400, seed: int = 0, cfg: TrimConfig = TrimConfig(), trim_fn=None,
                   half: str | None = None, only_ids: set[str] | None = None,
                   parts: tuple[str, ...] = ("spliced", "joined")) -> dict:
    """Two synthetic checks with a known boundary (error = predicted end time - true end time).

    ``spliced`` (the hard, common case: the signer goes straight on into the explanation): an ISLRTC ``max_s``
    clip whose sign ends in a clear rest/dip (both hands down for >= 300 ms, between 1 and 8 s after the sign
    starts; on by-eye labels these ends are right about 90% of the time) has the hands-down frames cut out
    and bridged by 2 interpolated frames, so the same signer runs from the sign straight into the explanation.
    True end = the last frame before the hands went down.

    ``joined``: a bounded ISLRTC clip (``eof``/``card``: the video holds the sign only, true end = its last
    frame with a hand up) followed by 2-6 s from the second half of another signer's ``max_s`` clip, moved into
    A's body frame (the trailing hands-down frames of A are kept, so this tests the rest/dip rules across
    signers).

    ``trim_fn(raw, stop, cfg, n_words=)`` defaults to :func:`find_first_sign`; ``half`` restricts the clips
    to a stable crc32 half ('tune' / 'report'), ``only_ids`` pins the clip set while ingestion is running."""
    from app.isl.data.trim import _runs, activity_signal
    from app.isl.data.trim_synth import align_concat, splice_out
    from app.landmarks.raw import RawSequence

    trim_fn = trim_fn or find_first_sign
    rng = random.Random(seed)
    recs = sorted((r for r in islrtc_records(paths) if (only_ids is None or r["id"] in only_ids)
                   and (half is None or _half(r["id"]) == half) and parsed(r["id"]).kind not in SKIP_KINDS),
                  key=lambda r: r["id"])
    by_key: dict[str, list[float]] = {}

    def add(keys, err, err2):
        for k in keys:
            by_key.setdefault(k, []).append(err)
            by_key.setdefault(f"{k}_best_of_2", []).append(err2)

    def errs_of(x, res, truth_ms):
        err = float(x.t_ms[max(1, min(len(x), res.end)) - 1] - truth_ms)
        err2 = err
        if getattr(res, "second", None):
            e2 = float(x.t_ms[min(len(x), int(res.second)) - 1] - truth_ms)
            err2 = e2 if abs(e2) < abs(err) else err
        return err, err2

    # spliced: same signer, rest removed
    maxs = [r for r in recs if r.get("stop") == "max_s"] if "spliced" in parts else []
    rng.shuffle(maxs)
    for r in maxs[:n]:
        raw = RawSequence.load(paths.resolve(r["path"]))
        pn = parsed(r["id"])
        nw = words_for(pn.kind, pn.text)
        try:
            sig = activity_signal(raw, cfg)
        except Exception:  # noqa: BLE001
            continue
        t = sig["t"]
        act = (sig["up_left"] | sig["up_right"])
        runs = [(a, b) for a, b in _runs(act) if t[b - 1] - t[a] >= 250]
        if not runs:
            continue
        t0 = t[runs[0][0]]
        # the first hands-down run of >= 150 ms after the start must be long (>= 300 ms) and 1-8 s in: then it
        # is an unambiguous boundary (no shorter dips inside the sign that could be the real end)
        dips = [(a, b) for a, b in _runs(sig["dip"]) if a > runs[0][0] and t[b - 1] - t[a] >= 150]
        if not dips:
            continue
        a, b = dips[0]
        if not (t[b - 1] - t[a] >= 300 and 1000 <= t[a] - t0 <= 8000 and b < len(t) - 12):
            continue
        x = splice_out(raw, a, b, bridge=2)
        res = trim_fn(x, "max_s", cfg, n_words=nw)
        err, err2 = errs_of(x, res, float(x.t_ms[a - 1]))
        add(["spliced", f"spliced_{word_group(pn.kind, pn.text)}", f"spliced_{res.method}"], err, err2)

    # joined: bounded clip + another signer's explanation
    clean = [r for r in recs if stop_group(r.get("stop")) == "bounded" and r["n_frames"] <= 120
             and "joined" in parts]
    tails = [r for r in recs if r.get("stop") == "max_s" and r["n_frames"] >= 100]
    rng.shuffle(clean)
    for r in clean[:n]:
        a_full = RawSequence.load(paths.resolve(r["path"]))
        pn = parsed(r["id"])
        nw = words_for(pn.kind, pn.text)
        own = find_first_sign(a_full, r.get("stop"), cfg, n_words=nw)
        if own.method != "source_end" or own.end - own.start < 6 or not tails:
            continue
        b_rec = rng.choice(tails)
        b_full = RawSequence.load(paths.resolve(b_rec["path"]))
        h = len(b_full) // 2
        L = int(rng.uniform(2.0, 6.0) * (b_rec.get("fps") or 12.0))
        b = b_full.slice(h, min(len(b_full), h + L))
        if len(b) < 6:
            continue
        x = align_concat(a_full, b, gap_frames=rng.randint(0, 3))
        res = trim_fn(x, "max_s", cfg, n_words=nw)
        err, err2 = errs_of(x, res, float(a_full.t_ms[own.end - 1]))
        add(["joined", f"joined_{word_group(pn.kind, pn.text)}"], err, err2)
    return {k: score_errors(v) for k, v in sorted(by_key.items())}


def include_eval(paths, n: int = 400, cfg: TrimConfig = TrimConfig()) -> dict:
    """INCLUDE clips are one sign each: the trimmer (bounded) should keep >= 90% of the active frames."""
    from app.isl.data.sources import IncludeAdapter
    from app.isl.data.trim import activity_signal

    keep = []
    for c in list(IncludeAdapter(paths).clips())[:n]:
        raw = c.load()
        try:
            sig = activity_signal(raw, cfg)
        except Exception:  # noqa: BLE001
            continue
        act = ~sig["dip"]
        if act.sum() == 0:
            continue
        res = find_first_sign(raw, "eof", cfg)
        keep.append(float(act[res.start: res.end].sum() / act.sum()))
    k = np.asarray(keep)
    return {"n": int(len(k)), "mean": round(float(k.mean()), 3) if len(k) else None,
            "frac_ge_90pct": round(float((k >= 0.9).mean()), 3) if len(k) else None}


def run_all(paths, out: Path, cfg: TrimConfig = TrimConfig(), model_path: Path | None = None) -> dict:
    """Trim every ingested ISLRTC clip into ``out`` (JSON lines; .npz files are only read). With ``model_path``
    the learned end (``trim_learned.refine_cap``) replaces stage-1 ``cap`` ends."""
    from app.isl.data.manifest import write_jsonl
    from app.landmarks.raw import RawSequence

    model = None
    if model_path is not None:
        from app.isl.data.trim_learned import load

        model, _ = load(model_path)
    rows, methods, n_err, n_learned = [], Counter(), 0, 0
    for r in islrtc_records(paths):
        pn = parsed(r["id"])
        try:
            raw = RawSequence.load(paths.resolve(r["path"]))
            nw = words_for(pn.kind, pn.text)
            if model is not None and pn.kind != "phrase":
                from app.isl.data.trim_learned import refine_cap

                res = refine_cap(raw, find_first_sign(raw, r.get("stop"), cfg, n_words=nw), model, nw, cfg)
                n_learned += any("trim-learned" in x for x in res.notes)
                rec = res.to_record(r["id"], len(raw), cfg.version)
            else:
                rec = trim_record_for(r["id"], raw, r.get("stop"), pn.kind, cfg, n_words=nw)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            rec = {"id": r["id"], "start": 0, "end": int(r.get("n_frames") or 0), "method": "empty", "confidence": 0.0,
                   "alternatives": [], "notes": [f"{type(e).__name__}: {e}"[:200]], "trim_version": cfg.version}
        rec["stop"] = r.get("stop")
        rec["kind"] = pn.kind
        rows.append(rec)
        methods[rec["method"]] += 1
    write_jsonl(out, sorted(rows, key=lambda x: x["id"]))
    conf = [r.get("confidence", 0.0) for r in rows]
    return {"clips": len(rows), "errors": n_err, "methods": dict(methods), "out": str(out), "learned_ends": n_learned,
            "low_conf_frac": round(float(np.mean(np.asarray(conf) < 0.5)), 3) if conf else None}


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.isl.data.trim_bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--exclude", help="JSON file of an earlier sample whose ids are skipped")
    sh = sub.add_parser("sheets")
    sh.add_argument("sample")
    sh.add_argument("out")
    sh.add_argument("--tmp", default=None)
    sh.add_argument("--marked", action="store_true", help="mark the current cut, alternatives and label")
    sh.add_argument("--full-frame", action="store_true", help="whole frame instead of the centre crop")
    sh.add_argument("--only", nargs="*", type=int, default=None, help="indices into the sample")
    lb = sub.add_parser("labels")
    lb.add_argument("--split", default=None)
    lb.add_argument("-v", action="store_true")
    sy = sub.add_parser("synthetic")
    sy.add_argument("--n", type=int, default=400)
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--half", choices=["tune", "report"], default=None)
    sy.add_argument("--ids", default=None, help="file with one clip id per line (pins the clip set)")
    inc = sub.add_parser("include")
    inc.add_argument("--n", type=int, default=400)
    ra = sub.add_parser("run")
    ra.add_argument("--out", default=None)
    ra.add_argument("--learned", default=None, help="model file from trim_learned train (refines cap ends)")
    args = ap.parse_args(argv)
    paths = _paths()
    if args.cmd == "sample":
        exclude = {x["id"] for x in json.loads(Path(args.exclude).read_text())} if args.exclude else set()
        print(json.dumps(stratified_sample(islrtc_records(paths), args.seed, exclude=exclude), indent=1))
    elif args.cmd == "sheets":
        sample = json.loads(Path(args.sample).read_text())
        if args.only is not None:
            sample = [sample[i] for i in args.only]
        trims = None
        if args.marked:
            from app.landmarks.raw import RawSequence

            recs = {r["id"]: r for r in islrtc_records(paths)}
            trims = {}
            for x in sample:
                raw = RawSequence.load(paths.resolve(recs[x["id"]]["path"]))
                _, t = predict_end_ms(raw, recs[x["id"]])
                t["alt_ms"] = [float(raw.t_ms[a - 1]) for a in t.get("alternatives", []) if 0 < a <= len(raw)]
                trims[x["id"]] = t
        out = Path(args.out)
        tmp = Path(args.tmp) if args.tmp else out / "_tmp"
        idx = render_sheets(sample, out, tmp, args.marked, trims, load_labels(), args.full_frame)
        errors = [i for i in idx if i.get("error")]
        print(json.dumps({"sheets": sum(1 for i in idx if i.get("sheet")), "errors": errors}, indent=1))
    elif args.cmd == "labels":
        print(json.dumps(eval_labels(paths, load_labels(), split=args.split, verbose=args.v), indent=1))
    elif args.cmd == "synthetic":
        ids = set(Path(args.ids).read_text().split("\n")) - {""} if args.ids else None
        print(json.dumps(synthetic_eval(paths, args.n, args.seed, half=args.half, only_ids=ids), indent=1))
    elif args.cmd == "include":
        print(json.dumps(include_eval(paths, args.n), indent=1))
    elif args.cmd == "run":
        out = Path(args.out) if args.out else paths.trim_sidecar
        print(json.dumps(run_all(paths, out, model_path=Path(args.learned) if args.learned else None), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
