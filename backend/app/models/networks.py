"""Sequence classifiers over landmark feature sequences.

All models take ``x`` of shape (B, T, D) and an optional boolean ``mask`` (B, T)
marking frames that contain a detection, and return logits (B, num_classes).
"""

from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPool(nn.Module):
    """Learned attention over time; frames without detections are ignored."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        s = self.score(h).squeeze(-1)  # (B, T)
        if mask is not None:
            s = s.masked_fill(~mask, -1e4)
        w = torch.softmax(s, dim=1)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1)


class InputProjection(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


class RecurrentClassifier(nn.Module):
    """Projection -> (bi)GRU/LSTM stack -> masked attention pooling -> classifier."""

    def __init__(self, in_dim: int, num_classes: int, cell: str = "gru", hidden: int = 192, layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.3, input_dropout: float = 0.1):
        super().__init__()
        self.inp = InputProjection(in_dim, hidden, input_dropout)
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0.0)
        out_dim = hidden * (2 if bidirectional else 1)
        self.pool = MaskedAttentionPool(out_dim)
        self.head = nn.Sequential(nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return self.head(self.pool(h, mask))


class TemporalBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation), nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TCNClassifier(nn.Module):
    """1D temporal CNN baseline with dilated residual blocks."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 192, layers: int = 4, kernel: int = 5,
                 dropout: float = 0.3, input_dropout: float = 0.1, **_):
        super().__init__()
        self.inp = InputProjection(in_dim, hidden, input_dropout)
        self.blocks = nn.Sequential(*[TemporalBlock(hidden, kernel, 2 ** i, dropout) for i in range(layers)])
        self.pool = MaskedAttentionPool(hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.blocks(self.inp(x).transpose(1, 2)).transpose(1, 2)
        return self.head(self.pool(h, mask))


ARCHITECTURES = ("gru", "lstm", "tcn")


def build_model(arch: str, in_dim: int, num_classes: int, hidden: int = 192, layers: int = 2,
                bidirectional: bool = True, dropout: float = 0.3, **_) -> nn.Module:
    arch = arch.lower()
    if arch in ("gru", "lstm"):
        return RecurrentClassifier(in_dim, num_classes, arch, hidden, layers, bidirectional, dropout)
    if arch == "tcn":
        return TCNClassifier(in_dim, num_classes, hidden, max(layers, 2) * 2, dropout=dropout)
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHITECTURES}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
