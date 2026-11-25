"""Lexicon-constrained decoding of fingerspelled letter posteriors (docs/ISL_DESIGN.md section 6.4).

A beam search walks a letter trie of known words, consuming one observed hold per step. Two edit operations
absorb hold-detection errors: an *insertion* (a spurious hold that is not a letter: skip it) and a *deletion*
(a letter whose hold was missed: advance in the trie without consuming a hold). Each costs a fixed log
penalty. Word scores add the word's log prior. The raw best-letter string is always returned too, marked as
"spelled, not in dictionary", so names that are not in the lexicon can still be written out.
"""

from __future__ import annotations

import heapq
import math
import string
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from app.isl.fingerspell.model import LETTERS, NOISE

ALPHA = set(string.ascii_uppercase)


@dataclass
class _Node:
    children: dict[str, _Node] = field(default_factory=dict)
    word: str | None = None
    prior: float = 0.0


class LexiconTrie:
    """Words (letters A-Z only after upper-casing; spaces/hyphens removed) with a log prior each."""

    def __init__(self, words: Iterable[tuple[str, float]]):
        self.root = _Node()
        self.size = 0
        for w, lp in words:
            key = normalize_word(w)
            if len(key) < 1:
                continue
            node = self.root
            for ch in key:
                node = node.children.setdefault(ch, _Node())
            if node.word is None or lp > node.prior:
                if node.word is None:
                    self.size += 1
                node.word, node.prior = w, float(lp)

    def __len__(self) -> int:
        return self.size

    def __contains__(self, w: str) -> bool:
        node = self.root
        for ch in normalize_word(w):
            node = node.children.get(ch)
            if node is None:
                return False
        return node.word is not None

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> LexiconTrie:
        """``word`` or ``word<TAB>log_prior`` per line; '#' comments."""
        rows = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            w, _, lp = ln.partition("\t")
            rows.append((w.strip(), float(lp) if lp else 0.0))
        return cls(rows)


def normalize_word(w: str) -> str:
    return "".join(ch for ch in w.upper() if ch in ALPHA)


class SpellDecoder:
    def __init__(self, trie: LexiconTrie, beam: int = 16, ins_pen: float = -2.5, del_pen: float = -2.5,
                 max_del_run: int = 1, letter_index: dict[str, int] | None = None, oov_pen: float = -4.0):
        self.trie = trie
        self.oov_pen = oov_pen  # a word outside the lexicon must beat one edit of a known word
        self.beam, self.ins_pen, self.del_pen, self.max_del_run = beam, ins_pen, del_pen, max_del_run
        self.idx = letter_index or {ch: i for i, ch in enumerate(LETTERS)}

    def raw_string(self, letter_logprobs: list[np.ndarray] | np.ndarray) -> tuple[str, float]:
        s, score = [], 0.0
        for lp in letter_logprobs:
            j = int(np.argmax(lp))
            if j != NOISE:
                s.append(LETTERS[j])
            score += float(lp[j])
        return "".join(s), score

    def decode(self, letter_logprobs: list[np.ndarray] | np.ndarray, k: int = 5) -> list[tuple[str, float, bool]]:
        """Top-``k`` (word, log score, in_lexicon). The raw string is included (in_lexicon False) unless it is
        itself a lexicon word."""
        obs = [np.asarray(x, np.float64) for x in letter_logprobs]
        T = len(obs)
        if T == 0:
            return []
        # state: (node id) -> (score, node, deletions in a row)
        beams: dict[int, tuple[float, _Node, int]] = {id(self.trie.root): (0.0, self.trie.root, 0)}
        for t in range(T):
            lp = obs[t]
            # epsilon (deletion) closure, bounded
            frontier = list(beams.values())
            for _ in range(self.max_del_run):
                new = []
                for sc, node, dr in frontier:
                    for ch, child in node.children.items():
                        cand = (sc + self.del_pen, child, dr + 1)
                        cur = beams.get(id(child))
                        if cur is None or cand[0] > cur[0]:
                            beams[id(child)] = cand
                            new.append(cand)
                frontier = new
            nxt: dict[int, tuple[float, _Node, int]] = {}

            def put(sc: float, node: _Node) -> None:
                cur = nxt.get(id(node))
                if cur is None or sc > cur[0]:
                    nxt[id(node)] = (sc, node, 0)

            for sc, node, _ in beams.values():
                put(sc + max(self.ins_pen, float(lp[NOISE])), node)  # insertion: spurious hold (cheap if it looks like noise)
                for ch, child in node.children.items():
                    j = self.idx.get(ch)
                    if j is not None:
                        put(sc + float(lp[j]), child)
            beams = dict(heapq.nlargest(self.beam * 4, nxt.items(), key=lambda kv: kv[1][0]))
        # final deletions (missed last letters), then collect words
        finals: dict[str, float] = {}
        frontier = list(beams.values())
        for step in range(self.max_del_run + 1):
            new = []
            for sc, node, dr in frontier:
                if node.word is not None:
                    s = sc + node.prior
                    if s > finals.get(node.word, -math.inf):
                        finals[node.word] = s
                if step < self.max_del_run:
                    for child in node.children.values():
                        new.append((sc + self.del_pen, child, dr + 1))
            frontier = heapq.nlargest(self.beam * 4, new, key=lambda x: x[0])
        out = [(w, s, True) for w, s in finals.items()]
        raw, raw_score = self.raw_string(obs)
        if raw and raw not in {normalize_word(w) for w, _, _ in out}:
            out.append((raw, raw_score + self.oov_pen, raw in self.trie))
        out.sort(key=lambda x: -x[1])
        return out[:k]


def to_probs(scores: list[float]) -> list[float]:
    if not scores:
        return []
    a = np.asarray(scores, np.float64)
    e = np.exp(a - a.max())
    return (e / e.sum()).tolist()
