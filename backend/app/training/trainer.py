"""Model training. Progress written to the DB comes straight from the loop."""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.core.config import get_settings, resolve_checkpoint, resolve_device
from app.dataset.annotations import dataset_version
from app.db.models import Experiment, Split, utcnow
from app.db.session import session_scope
from app.evaluation.metrics import compute_metrics, pick_threshold
from app.features.build import AugmentConfig
from app.features.spec import FeatureConfig
from app.models.checkpoint import Predictor, frame_mask, save_checkpoint
from app.models.networks import ARCHITECTURES, build_model, count_params
from app.training.data import LoadedData, SequenceDataset, class_weights, load_samples

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    architecture: str = "gru"
    hidden: int = 192
    layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 32
    epochs: int = 80
    patience: int = 20
    label_smoothing: float = 0.1
    class_weighting: str = "none"  # none | balanced
    augment: bool = True
    seed: int = 42
    device: str = "auto"
    min_hand_rate: float = 0.1
    features: dict = field(default_factory=lambda: FeatureConfig().to_dict())

    def validate(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if not (8 <= int(self.features.get("seq_len", 48)) <= 128):
            raise ValueError("seq_len must be between 8 and 128")
        if not (1 <= self.epochs <= 500):
            raise ValueError("epochs must be between 1 and 500")
        if not (2 <= self.batch_size <= 512):
            raise ValueError("batch_size must be between 2 and 512")
        if self.class_weighting not in ("none", "balanced"):
            raise ValueError("class_weighting must be 'none' or 'balanced'")

    def model_kwargs(self) -> dict:
        return {"hidden": self.hidden, "layers": self.layers, "bidirectional": self.bidirectional, "dropout": self.dropout}


class Cancelled(Exception):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, fcfg: FeatureConfig, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, frame_mask(x, fcfg))
        loss_sum += float(crit(logits, y))
        n += len(y)
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        ys.append(y.cpu().numpy())
    if not probs:
        return np.zeros((0, 0)), np.zeros(0, int), float("nan")
    return np.concatenate(probs), np.concatenate(ys), loss_sum / max(1, n)


def measure_latency(predictor: Predictor, repeats: int = 30) -> dict:
    fcfg = predictor.feature_config
    x = np.random.default_rng(0).normal(size=(fcfg.seq_len, fcfg.dim)).astype(np.float32)
    predictor.predict(x)  # warm-up
    times = [predictor.predict(x)[1] for _ in range(repeats)]
    return {"device": predictor.device, "batch": 1, "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95))}


def evaluate_partition(exp: Experiment, data: LoadedData, partition: str, device: str = "cpu") -> dict:
    predictor = Predictor(resolve_checkpoint(exp.checkpoint_path), device)
    fcfg = predictor.feature_config
    samples = data.samples[partition]
    ds = SequenceDataset(samples, fcfg, train=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs, y, loss = predict_loader(predictor.model, loader, fcfg, device)
    if len(y) == 0:
        return {"n": 0, "error": f"no {partition} samples with landmarks"}
    m = compute_metrics(probs, y, data.class_names, [s.video_id for s in samples], [s.signer_id for s in samples])
    m["loss"] = loss
    m["partition"] = partition
    m["latency"] = measure_latency(predictor)
    m["evaluated_at"] = utcnow().isoformat()
    return m


@dataclass
class FitResult:
    best_epoch: int
    best_metric: float
    history: list[dict]
    checkpoint: Path


@torch.no_grad()
def classification_val_metrics(model: nn.Module, val_loader: DataLoader, fcfg: FeatureConfig, device: str) -> dict:
    """val_loss / val_acc / val_macro_f1 of a classifier (the metrics ``run_training`` has always logged)."""
    from sklearn.metrics import f1_score

    vp, vy, vloss = predict_loader(model, val_loader, fcfg, device)
    if len(vy) == 0:
        return {"val_loss": float("nan"), "val_acc": 0.0, "val_macro_f1": 0.0}
    vpred = vp.argmax(1)
    vf1 = float(f1_score(vy, vpred, labels=sorted(set(vy.tolist())), average="macro", zero_division=0))
    return {"val_loss": vloss, "val_acc": float((vpred == vy).mean()), "val_macro_f1": vf1}


def fit(model: nn.Module, train_loader: DataLoader, val_fn: Callable[[nn.Module], dict], loss_fn: Callable,
        cfg: TrainConfig, ckpt_path: Path, meta: dict, select_key: str = "val_macro_f1", *, device: str = "cpu",
        steps_per_epoch: int | None = None, params=None, save_module: nn.Module | None = None,
        on_epoch: Callable[[list[dict]], None] | None = None,
        epoch_progress: Callable[[int, dict], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None) -> FitResult:
    """The DB-free epoch loop (docs/ISL_DESIGN.md 6.6), shared by ``run_training`` and the ISL trainers.

    ``loss_fn(model, x, y) -> (loss, logits | None)``; ``val_fn(model) -> dict`` with ``select_key`` (and
    optionally ``val_acc`` as the tie-breaker). AdamW + warm-up/cosine schedule, gradient clipping at 1.0, early
    stopping after ``cfg.patience`` epochs without improvement. The best epoch's ``save_module`` (default: the
    model) is saved with ``save_checkpoint(ckpt_path, ..., {**meta, "best_epoch": epoch})``."""
    params = list(params) if params is not None else list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    spe = steps_per_epoch if steps_per_epoch is not None else max(1, len(train_loader))
    steps = max(1, cfg.epochs * spe)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    history: list[dict] = []
    best = (-math.inf, -math.inf)
    best_epoch, bad = -1, 0
    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            raise Cancelled()
        model.train()
        t0 = time.time()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss, logits = loss_fn(model, x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(y)
            if logits is not None:
                tc += int((logits.argmax(-1) == y).sum())
            tn += len(y)
        vm = val_fn(model)
        row = {"epoch": epoch, "train_loss": tl / max(1, tn), "train_acc": tc / max(1, tn), **vm,
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        score = (float(vm.get(select_key, 0.0)), float(vm.get("val_acc", 0.0)))
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            save_checkpoint(ckpt_path, save_module if save_module is not None else model, {**meta, "best_epoch": epoch})
        else:
            bad += 1
        if on_epoch:
            on_epoch(history)
        if epoch_progress:
            epoch_progress(epoch, row)
        elif progress:
            progress(epoch / cfg.epochs, f"epoch {epoch}/{cfg.epochs} - {select_key} {score[0]:.3f}")
        if bad >= cfg.patience:
            break
    return FitResult(best_epoch, best[0], history, ckpt_path)


def run_training(exp_id: int, progress: Callable[[float, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> dict:
    settings = get_settings()
    with session_scope() as db:
        exp = db.get(Experiment, exp_id)
        cfg = TrainConfig(**exp.config)
        cfg.validate()
        fcfg = FeatureConfig.from_dict(cfg.features)
        split = db.get(Split, exp.split_id)
        data = load_samples(db, split.id, fcfg, cfg.min_hand_rate)
        exp.status = "running"
        exp.dataset_version = dataset_version(db)
        exp.data_summary = data.summary()
        exp.feature_config = fcfg.to_dict()

    if len(data.samples["train"]) < 2 or len(data.samples["val"]) < 1:
        raise RuntimeError(
            f"not enough data: {len(data.samples['train'])} train / {len(data.samples['val'])} val sequences "
            "have landmarks. Acquire videos and extract landmarks first."
        )

    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    n_classes = len(data.class_names)
    model = build_model(cfg.architecture, fcfg.dim, n_classes, **cfg.model_kwargs()).to(device)
    n_params = count_params(model)

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds = SequenceDataset(data.samples["train"], fcfg, train=cfg.augment, seed=cfg.seed, aug=AugmentConfig())
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g, drop_last=len(train_ds) > cfg.batch_size)
    val_loader = DataLoader(SequenceDataset(data.samples["val"], fcfg, train=False), batch_size=64)

    weight = class_weights(data.samples["train"], n_classes).to(device) if cfg.class_weighting == "balanced" else None
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    ckpt_path = settings.checkpoints_dir / f"exp{exp_id:04d}_{cfg.architecture}.pt"
    meta = {
        "experiment_id": exp_id,
        "architecture": cfg.architecture,
        "model_kwargs": cfg.model_kwargs(),
        "feature_config": fcfg.to_dict(),
        "class_names": data.class_names,
        "split": split.name,
        "train_config": asdict(cfg),
    }
    t_start = time.time()

    def loss_fn(m: nn.Module, x: torch.Tensor, y: torch.Tensor):
        logits = m(x, frame_mask(x, fcfg))
        return crit(logits, y), logits

    def val_fn(m: nn.Module) -> dict:
        return classification_val_metrics(m, val_loader, fcfg, device)

    def on_epoch(history: list[dict]) -> None:
        with session_scope() as db:
            e = db.get(Experiment, exp_id)
            e.history = list(history)
            e.n_params = n_params
            e.device = device
            e.checkpoint_path = str(ckpt_path)

    def epoch_progress(epoch: int, row: dict) -> None:
        if progress:
            progress(epoch / cfg.epochs,
                     f"epoch {epoch}/{cfg.epochs} - val acc {row['val_acc']:.3f}, macro F1 {row['val_macro_f1']:.3f}")

    res = fit(model, train_loader, val_fn, loss_fn, cfg, ckpt_path, meta, device=device,
              steps_per_epoch=math.ceil(len(train_ds) / cfg.batch_size), on_epoch=on_epoch,
              epoch_progress=epoch_progress, should_stop=should_stop)
    best_epoch = res.best_epoch

    # Final validation metrics with the best checkpoint; derive the rejection threshold.
    with session_scope() as db:
        exp = db.get(Experiment, exp_id)
        vm = evaluate_partition(exp, data, "val", "cpu")
    th = pick_threshold(vm.get("confidence_curve", []))
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck["meta"]["recommended_threshold"] = th
    ck["meta"]["val_top1"] = vm.get("top1")
    torch.save(ck, ckpt_path)
    vm["recommended_threshold"] = th
    vm["best_epoch"] = best_epoch
    vm["train_seconds"] = time.time() - t_start
    with session_scope() as db:
        e = db.get(Experiment, exp_id)
        e.val_metrics = vm
        e.status = "completed"
        e.finished_at = utcnow()
    return {"best_epoch": best_epoch, "val_top1": vm.get("top1"), "val_macro_f1": vm.get("macro_f1")}


def run_test_evaluation(exp_id: int) -> dict:
    """Held-out test evaluation. Run explicitly, after model selection, never during training."""
    with session_scope() as db:
        exp = db.get(Experiment, exp_id)
        if exp.status != "completed" or not exp.checkpoint_path:
            raise RuntimeError("experiment has no completed checkpoint")
        if exp.split_id is None:
            raise RuntimeError("this is an imported pretrained model; its held-out test result was recorded when it was "
                               "trained and the original split is not available here")
        fcfg = FeatureConfig.from_dict(exp.feature_config)
        cfg = TrainConfig(**exp.config)
        data = load_samples(db, exp.split_id, fcfg, cfg.min_hand_rate)
        tm = evaluate_partition(exp, data, "test", "cpu")
        exp.test_metrics = tm
        return {"top1": tm.get("top1"), "n": tm.get("n")}
