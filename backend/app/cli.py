"""Command-line interface for the data / training pipeline.

Run from the backend folder:  python -m app.cli <command> [options]

  annotations            fetch official WLASL annotations, select WLASL100, build splits
  acquire [--limit N] [--classes 0,1,2] [--retry-failed]
  scan                   validate files already on disk (incl. data/videos/provided)
  extract [--limit N] [--redo]
  splits                 print partition sizes and leakage-check results
  train [--arch gru] [--split signer-holdout] [--epochs 80] [--seq-len 48] [--hidden 192] [--layers 2] [--activate]
  evaluate-test ID       held-out test evaluation of experiment ID (run once, after model selection)
  activate ID            make experiment ID the live recognition model
  seed-synthetic         TESTS ONLY: create a synthetic landmark dataset in an EMPTY data dir
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict


def _print_progress(frac: float, msg: str) -> None:
    sys.stdout.write(f"\r[{frac * 100:5.1f}%] {msg[:100]:<100}")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotations")
    a.add_argument("--no-fetch", action="store_true", help="use files already in data/annotations")
    q = sub.add_parser("acquire")
    q.add_argument("--limit", type=int)
    q.add_argument("--classes", type=str)
    q.add_argument("--retry-failed", action="store_true")
    sub.add_parser("scan")
    e = sub.add_parser("extract")
    e.add_argument("--limit", type=int)
    e.add_argument("--redo", action="store_true")
    sp = sub.add_parser("splits")
    sp.add_argument("--available", action="store_true", help="(re)build signer-holdout-available over acquired videos")
    t = sub.add_parser("train")
    t.add_argument("--arch", default="gru", choices=["gru", "lstm", "tcn"])
    t.add_argument("--split", default="signer-holdout")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--patience", type=int, default=20)
    t.add_argument("--seq-len", type=int, default=48)
    t.add_argument("--hidden", type=int, default=192)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--no-face", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--name")
    t.add_argument("--activate", action="store_true")
    ev = sub.add_parser("evaluate-test")
    ev.add_argument("id", type=int)
    ac = sub.add_parser("activate")
    ac.add_argument("id", type=int)
    sy = sub.add_parser("seed-synthetic")
    sy.add_argument("--classes", type=int, default=6)
    sy.add_argument("--signers", type=int, default=8)
    args = p.parse_args(argv)

    from app.db.session import session_scope

    if args.cmd == "annotations":
        from app.dataset.annotations import load_and_ingest
        from app.splits.splits import ensure_default_splits

        with session_scope() as db:
            res = load_and_ingest(db, fetch=not args.no_fetch)
            names = [s.name for s in ensure_default_splits(db)]
        print(json.dumps({"added": res["added"], "updated": res["updated"], "checks": res["checks"], "splits": names}, indent=2, default=str))
    elif args.cmd == "acquire":
        from app.dataset.acquisition import acquire

        classes = [int(c) for c in args.classes.split(",")] if args.classes else None
        res = acquire(None, classes, args.retry_failed, args.limit, _print_progress)
        print("\n" + json.dumps(res))
    elif args.cmd == "scan":
        from app.dataset.acquisition import scan_and_validate

        print("\n" + json.dumps(scan_and_validate(_print_progress)))
    elif args.cmd == "extract":
        from app.landmarks.pipeline import extract_all

        print("\n" + json.dumps(extract_all(args.redo, args.limit, None, _print_progress)))
    elif args.cmd == "splits":
        from sqlalchemy import select

        from app.db.models import Split

        if args.available:
            from app.splits.splits import build_available_holdout

            with session_scope() as db:
                build_available_holdout(db)
        with session_scope() as db:
            for s in db.scalars(select(Split).order_by(Split.id)):
                r = s.leakage_report
                print(f"{s.name:22s} videos={r['n_videos']} signers={r['n_signers']} signer_independent={r['signer_independent']} passed={r['passed']}")
    elif args.cmd == "train":
        from sqlalchemy import select

        from app.db.models import Experiment, MetaKV, Split
        from app.features.spec import FeatureConfig
        from app.inference.registry import activate
        from app.training.trainer import TrainConfig, run_training

        cfg = TrainConfig(architecture=args.arch, epochs=args.epochs, patience=args.patience, hidden=args.hidden, layers=args.layers,
                          batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                          features=FeatureConfig(seq_len=args.seq_len, use_face=not args.no_face).to_dict())
        cfg.validate()
        with session_scope() as db:
            split = db.scalar(select(Split).where(Split.name == args.split))
            if split is None:
                print(f"split {args.split!r} not found; run 'annotations' first", file=sys.stderr)
                return 2
            exp = Experiment(name=args.name or f"{args.arch.upper()} · {split.name} · T={args.seq_len} (cli)", architecture=args.arch,
                             config=asdict(cfg), feature_config=cfg.features, split_id=split.id, split_name=split.name,
                             is_synthetic=db.get(MetaKV, "synthetic") is not None)
            db.add(exp)
            db.flush()
            exp_id = exp.id
        try:
            res = run_training(exp_id, _print_progress)
        except Exception as ex:
            with session_scope() as db:
                x = db.get(Experiment, exp_id)
                x.status, x.error = "failed", str(ex)
            raise
        print("\n" + json.dumps({"experiment_id": exp_id, **res}))
        if args.activate:
            activate(exp_id)
            print(f"activated experiment {exp_id}")
    elif args.cmd == "evaluate-test":
        from app.training.trainer import run_test_evaluation

        print(json.dumps(run_test_evaluation(args.id)))
    elif args.cmd == "activate":
        from app.inference.registry import activate

        activate(args.id)
        print(f"activated experiment {args.id}")
    elif args.cmd == "seed-synthetic":
        from sqlalchemy import func, select

        from app.dataset.synthetic import populate_synthetic_db
        from app.db.models import Video
        from app.splits.splits import ensure_default_splits

        with session_scope() as db:
            if (db.scalar(select(func.count()).select_from(Video)) or 0) > 0:
                print("refusing: the data directory already contains videos. Use an empty SLR_DATA_DIR.", file=sys.stderr)
                return 2
        print(json.dumps(populate_synthetic_db(args.classes, args.signers, 3, 1)))
        with session_scope() as db:
            ensure_default_splits(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
