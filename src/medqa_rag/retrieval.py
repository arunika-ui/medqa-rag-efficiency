"""Retrievers (IV-A), FAISS indexes and similarity metrics (IV-B), fusion and compression (IV-D)."""
from __future__ import annotations

import time

import numpy as np

from .text import tokens


class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009); k1 = 1.5, b = 0.75 by default."""

    def __init__(self, chunks: list[str], k1: float = 1.5, b: float = 0.75):
        import bm25s

        self.k1, self.b = k1, b
        t0 = time.perf_counter()
        self.model = bm25s.BM25(method="robertson", k1=k1, b=b)
        self.model.index(bm25s.tokenize(chunks, stopwords="en", show_progress=False), show_progress=False)
        self.build_s = time.perf_counter() - t0
        self._tok = bm25s.tokenize
        self.n = len(chunks)

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        ids, scores = self.model.retrieve(self._tok(queries, stopwords="en", show_progress=False),
                                          k=min(k, self.n), show_progress=False, n_threads=0)
        return np.asarray(ids), np.asarray(scores)


class LSA:
    """Dense lexical-semantic retriever: TF-IDF -> 256-d truncated SVD (Deerwester et al., 1990)."""

    def __init__(self, chunks: list[str], dim: int = 256, seed: int = 0):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        t0 = time.perf_counter()
        self.tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2, max_df=0.5,
                                     token_pattern=r"(?u)\b[a-zA-Z0-9]{2,}\b")
        X = self.tfidf.fit_transform(chunks)
        self.svd = TruncatedSVD(n_components=min(dim, X.shape[1] - 1), random_state=seed)
        self.raw = self.svd.fit_transform(X).astype(np.float32)          # un-normalised vectors
        self.vecs = _l2(self.raw)                                       # unit vectors
        self.build_s = time.perf_counter() - t0

    def embed(self, queries: list[str], normalise: bool = True) -> np.ndarray:
        v = self.svd.transform(self.tfidf.transform(queries)).astype(np.float32)
        return _l2(v) if normalise else v

    def search(self, queries: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.embed(queries)
        s = q @ self.vecs.T
        idx = np.argsort(-s, axis=1)[:, :k]
        return idx, np.take_along_axis(s, idx, axis=1)


def _l2(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def rrf(rankings: list[np.ndarray], k: int, c: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Reciprocal rank fusion (Cormack et al., 2009) of several (n_queries, depth) id arrays."""
    n = rankings[0].shape[0]
    ids = np.zeros((n, k), dtype=np.int64)
    scores = np.zeros((n, k), dtype=np.float32)
    for qi in range(n):
        fused: dict[int, float] = {}
        for r in rankings:
            for rank, d in enumerate(r[qi]):
                fused[int(d)] = fused.get(int(d), 0.0) + 1.0 / (c + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        for j, (d, s) in enumerate(top):
            ids[qi, j], scores[qi, j] = d, s
    return ids, scores


def compress(ids: np.ndarray, scores: np.ndarray, tau: float) -> list[np.ndarray]:
    """Relative-score compression: keep chunks scoring at least tau x the best score."""
    out = []
    for i, s in zip(ids, scores):
        keep = s >= tau * s[0] if s[0] > 0 else np.ones_like(s, dtype=bool)
        out.append(i[keep])
    return out


# ---------------------------------------------------------------- FAISS (IV-B)

def faiss_indexes(dim: int, n: int):
    import faiss

    nlist = int(4 * np.sqrt(n))
    yield "Flat (exact)", {}, lambda: faiss.IndexFlatIP(dim)
    for M in (16, 32):
        for ef in (16, 32, 64, 128):
            def mk(M=M):
                ix = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
                ix.hnsw.efConstruction = 200
                return ix
            yield f"HNSW M={M}, ef={ef}", {"efSearch": ef}, mk
    for nprobe in (1, 4, 16, 64):
        yield f"IVF-Flat, nprobe={nprobe}", {"nprobe": nprobe}, \
            lambda: faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    for nprobe in (16, 64):
        yield f"IVF-PQ m=32, nprobe={nprobe}", {"nprobe": nprobe}, \
            lambda: faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, nlist, 32, 8, faiss.METRIC_INNER_PRODUCT)


def index_size_mb(index) -> float:
    import faiss

    return faiss.serialize_index(index).nbytes / 1e6
