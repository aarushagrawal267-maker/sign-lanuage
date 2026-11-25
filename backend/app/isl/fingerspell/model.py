"""Letter classifier: a small MLP over hold features (A-Z, 0-9 and "noise")."""

from __future__ import annotations

import string
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

from app.isl.fingerspell.features import LETTER_DIM

LETTERS: list[str] = list(string.ascii_uppercase) + [str(d) for d in range(10)] + ["noise"]
NOISE = LETTERS.index("noise")


class LetterClassifier(nn.Module):
    def __init__(self, in_dim: int = LETTER_DIM, n_classes: int = len(LETTERS), hidden: tuple[int, int] = (128, 64),
                 dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden[0]), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden[1], n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class LetterPredictor:
    """Loaded letter model (checkpoint ``{"state_dict", "meta"}`` with ``meta["kind"] == "isl-letters"``)."""

    def __init__(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = ck["meta"]
        if self.meta.get("kind") != "isl-letters":
            raise ValueError(f"{path} is not a letter-model checkpoint")
        self.labels: list[str] = self.meta.get("labels", LETTERS)
        self.temperature = float(self.meta.get("temperature", 1.0))
        self.model = LetterClassifier(self.meta.get("in_dim", LETTER_DIM), len(self.labels),
                                      tuple(self.meta.get("hidden", (128, 64))))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self._lock = threading.Lock()
        self.trained_letters: list[str] = self.meta.get("trained_letters", [])
        self.n_signers: int = int(self.meta.get("n_signers", 1))

    @torch.inference_mode()
    def log_probs(self, feats: np.ndarray) -> np.ndarray:
        """(N, LETTER_DIM) -> (N, n_labels) log-probabilities."""
        if len(feats) == 0:
            return np.zeros((0, len(self.labels)), np.float32)
        with self._lock:
            z = self.model(torch.from_numpy(np.asarray(feats, np.float32))) / self.temperature
            return torch.log_softmax(z, -1).numpy()
