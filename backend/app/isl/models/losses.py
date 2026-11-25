"""Metric-learning losses for the sign encoder (docs/ISL_DESIGN.md section 6.3 b)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. 2020) over L2-normalised embeddings.

    Every other sample with the same label is a positive. Anchors without any positive in the batch are
    ignored (returns 0 when no anchor has a positive)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        sim = z @ z.T / self.t
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, -1e9)
        pos = (labels[:, None] == labels[None, :]) & ~eye
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = pos.sum(1)
        has = n_pos > 0
        if not bool(has.any()):
            return z.sum() * 0.0
        mean_log_pos = (log_prob * pos).sum(1)[has] / n_pos[has]
        return -mean_log_pos.mean()


class SubCenterArcFace(nn.Module):
    """Sub-centre ArcFace (Deng et al. 2020): ``k`` learnable centres per concept capture sign variants; the
    class cosine is the max over its centres; an additive angular margin ``m`` is applied to the target."""

    def __init__(self, emb_dim: int, n_classes: int, k: int = 3, s: float = 30.0, m: float = 0.2):
        super().__init__()
        self.n_classes, self.k, self.s, self.m = n_classes, k, s, m
        self.weight = nn.Parameter(torch.empty(n_classes * k, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, z: torch.Tensor) -> torch.Tensor:
        cos = F.normalize(z, dim=-1) @ F.normalize(self.weight, dim=-1).T  # (B, C*k)
        return cos.view(z.shape[0], self.n_classes, self.k).max(dim=2).values

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(z).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos.gather(1, labels[:, None]))
        target = torch.cos(theta + self.m)
        # keep the margin monotonic for angles past pi - m (standard "easy margin" fallback)
        target = torch.where(theta + self.m < math.pi, target, cos.gather(1, labels[:, None]) - self.m * math.sin(self.m))
        logits = cos.scatter(1, labels[:, None], target) * self.s
        return F.cross_entropy(logits, labels)
