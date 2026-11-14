# Evaluation protocol

## Why signer independence matters

If the same person appears in training and test data, a model can recognise the *person's* appearance and movement style instead of the sign. WLASL's official split does not separate signers: in WLASL100, **253 of 258 test videos and 333 of 338 validation videos** are performed by signers who also appear in the training partition (52 of the 56 test signers are training signers). Results on that split are optimistic and are labelled as such everywhere in the application.

## Splits (`splits/splits.py`)

| Name | Construction |
|---|---|
| `official` | as published; kept for comparison with the literature; flagged *not signer-independent* |
| `signer-holdout` | whole signer groups assigned to train/val/test (target 70/15/15 of videos). 3,000 random signer orders are scored on closeness to the target fractions and on class coverage in val and test; the best is kept (seeded, reproducible) |
| `signer-cv5-fold0…4` | grouped 5-fold CV over signer groups, balancing fold sizes; fold *k* = test, fold *k+1* = val, rest = train. Every signer is tested exactly once |

**Signer groups**: signers that share a raw source file (one YouTube compilation carries signer ids 5 and 64) are merged, so identical footage cannot appear in two partitions.

## Automated leakage checks

Run when a split is built, re-runnable from the API (`POST /api/splits/{id}/check`) and covered by tests:

* signer overlap for every partition pair (must be 0 for signer-grouped splits)
* content-hash overlap (identical files in two partitions)
* source-file overlap (same raw video in two partitions)
* duplicate video assignment
* classes missing from each partition
* number of val/test videos whose signer appears in training

`passed` = signer-independent and no hash/source overlap and no duplicates.

## Selection vs. testing

* Early stopping, model choice and the live threshold use **validation** data only.
* The **test** partition is evaluated only through an explicit action ("Run held-out test evaluation" / `python -m app.cli evaluate-test ID`), stored separately with a timestamp, and shown separately.

## Metrics (`evaluation/metrics.py`)

top-1, top-5, macro F1 (over classes present in the partition — the number of evaluated classes is reported), per-class precision / recall / F1 / support, full confusion matrix, most frequent confusions, per-signer top-1, accuracy-vs-coverage curve, CPU latency (batch 1, mean and p95). All classes are listed, including those with poor performance.

## Caveats

With roughly 20 videos per gloss, a 15% test partition has only 2–4 videos per class; per-class numbers are therefore very noisy. The 5-fold signer CV (mean ± spread across folds) gives a more stable estimate than a single holdout.
