"""Evidence-only answer selector (IV-C, Eq. 1) and the non-retrieval baselines (IV-D).

    s(o) = sum_k w_k * ( sum_{t in T(o)} idf(t) [t in c_k] ) / ( sum_{t in T(o)} idf(t) )

T(o) = content tokens of option o that do not occur in all four options. The selector has
no parameters fitted to MedQA, so accuracy above the random-context control measures what
retrieval found rather than what a model memorised.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .data import LETTERS
from .text import content_tokens


class IDF:
    def __init__(self, chunks: list[str]):
        df = Counter()
        for c in chunks:
            df.update(set(content_tokens(c)))
        self.n = len(chunks)
        self.df = df
        self._max = math.log(1 + self.n)

    def __call__(self, t: str) -> float:
        return math.log(1 + self.n / (1 + self.df.get(t, 0)))


def distinctive(options: dict[str, str]) -> dict[str, set[str]]:
    toks = {l: set(content_tokens(options[l])) for l in LETTERS}
    shared = set.intersection(*toks.values())
    return {l: toks[l] - shared for l in LETTERS}


class Selector:
    def __init__(self, idf: IDF, chunk_tokens, seed: int = 0):
        self.idf = idf
        self.chunk_tokens = chunk_tokens       # callable: chunk id -> set of tokens (cached)
        self.rng = np.random.default_rng(seed)

    def scores(self, options, chunk_ids, weights=None) -> dict[str, float]:
        T = distinctive(options)
        w = np.ones(len(chunk_ids)) if weights is None else np.asarray(weights, dtype=float)
        sets = [self.chunk_tokens(int(c)) for c in chunk_ids]
        out = {}
        for l in LETTERS:
            denom = sum(self.idf(t) for t in T[l])
            if denom == 0:
                out[l] = 0.0
                continue
            out[l] = float(sum(wk * sum(self.idf(t) for t in T[l] if t in s) for wk, s in zip(w, sets)) / denom)
        return out

    def pick(self, scores: dict[str, float]) -> str:
        best = max(scores.values())
        tied = [l for l in LETTERS if scores[l] == best]
        return tied[int(self.rng.integers(len(tied)))]

    def predict(self, options, chunk_ids, weights=None) -> str:
        return self.pick(self.scores(options, chunk_ids, weights))


# ------------------------------------------------------------------ baselines

def random_guess(questions, seed=0):
    rng = np.random.default_rng(seed)
    return [LETTERS[int(rng.integers(4))] for _ in questions]


def longest_option(questions, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for q in questions:
        L = {l: len(q.options[l]) for l in LETTERS}
        m = max(L.values())
        tied = [l for l in LETTERS if L[l] == m]
        out.append(tied[int(rng.integers(len(tied)))])
    return out


def frequency_prior(questions, idf: IDF, seed=0):
    """Pick the option whose distinctive tokens are most common in the corpus (no retrieval)."""
    rng = np.random.default_rng(seed)
    out = []
    for q in questions:
        T = distinctive(q.options)
        s = {l: (np.mean([math.log(1 + idf.df.get(t, 0)) for t in T[l]]) if T[l] else 0.0) for l in LETTERS}
        m = max(s.values())
        tied = [l for l in LETTERS if s[l] == m]
        out.append(tied[int(rng.integers(len(tied)))])
    return out
