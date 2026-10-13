"""Leakage audit (Section III): does any dev/test stem share a word 13-gram with the corpus?"""
from __future__ import annotations

import numpy as np

from .text import tokens

N = 13


def _grams(words: list[str], n: int = N):
    return (hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1))


def audit(books: dict[str, str], questions) -> dict:
    corpus = np.unique(np.fromiter((h for t in books.values() for h in _grams(tokens(t))), dtype=np.int64))
    hits = []
    for q in questions:
        g = np.fromiter(_grams(tokens(q.stem)), dtype=np.int64)
        if len(g) and np.isin(g, corpus).any():
            hits.append(q.qid)
    return {"corpus_words": int(sum(len(t.split()) for t in books.values())),
            "distinct_13grams": int(len(corpus)), "questions_checked": len(questions),
            "questions_with_overlap": len(hits), "overlapping_ids": hits[:50]}
