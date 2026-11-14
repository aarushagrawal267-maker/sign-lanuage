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

from app.core.config import get_settings, resolve_device
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
    predictor = Predictor(Path(exp.checkpoint_path), device)
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, cfg.epochs * math.ceil(len(train_ds) / cfg.batch_size))
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )

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
    history: list[dict] = []
    best = (-1.0, -1.0)
    best_epoch, bad = -1, 0
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        if should_stop and should_stop():
            raise Cancelled()
        model.train()
        t0 = time.time()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, frame_mask(x, fcfg))
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(y)
            tc += int((logits.argmax(-1) == y).sum())
            tn += len(y)
        vp, vy, vloss = predict_loader(model, val_loader, fcfg, device)
        vpred = vp.argmax(1)
        from sklearn.metrics import f1_score

        vf1 = float(f1_score(vy, vpred, labels=sorted(set(vy.tolist())), average="macro", zero_division=0))
        vacc = float((vpred == vy).mean())
        row = {
            "epoch": epoch, "train_loss": tl / max(1, tn), "train_acc": tc / max(1, tn), "val_loss": vloss,
            "val_acc": vacc, "val_macro_f1": vf1, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0,
        }
        history.append(row)
        improved = (vf1, vacc) > best
        if improved:
            best, best_epoch, bad = (vf1, vacc), epoch, 0
            save_checkpoint(ckpt_path, model, {**meta, "best_epoch": epoch})
        else:
            bad += 1
        with session_scope() as db:
            e = db.get(Experiment, exp_id)
            e.history = list(history)
            e.n_params = n_params
            e.device = device
            e.checkpoint_path = str(ckpt_path)
        if progress:
            progress(epoch / cfg.epochs, f"epoch {epoch}/{cfg.epochs} - val acc {vacc:.3f}, macro F1 {vf1:.3f}")
        if bad >= cfg.patience:
            break

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
        fcfg = FeatureConfig.from_dict(exp.feature_config)
        cfg = TrainConfig(**exp.config)
        data = load_samples(db, exp.split_id, fcfg, cfg.min_hand_rate)
        tm = evaluate_partition(exp, data, "test", "cpu")
        exp.test_metrics = tm
        return {"top1": tm.get("top1"), "n": tm.get("n")}
