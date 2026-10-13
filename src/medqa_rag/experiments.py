"""End-to-end protocol of the paper: dev-only selection, then a single run on the test set.

    PYTHONPATH=src python -m medqa_rag.experiments          # full run, writes results/
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import data, leakage, stats
from .retrieval import BM25, LSA, compress, faiss_indexes, index_size_mb, rrf
from .selector import IDF, Selector, frequency_prior, longest_option, random_guess
from .text import content_tokens, tokens

ROOT = Path(__file__).resolve().parents[2]
SIZES = (192, 384, 768)            # words; about 256 / 512 / 1,024 tokens
DEFAULT = 384
FORMS = ("q", "q+opts", "metamap+opts", "last+opts")
TOPK = (1, 3, 5, 10, 20)
TAUS = tuple(round(x, 1) for x in np.arange(0, 1.0, 0.1))
EQUAL_BUDGET = {192: 10, 384: 5, 768: 3}      # ~1.9-2.3k words of context each
K_DEFAULT = 5


class Corpus:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.bm25 = BM25(chunks)
        self._tok = lru_cache(maxsize=None)(lambda i: frozenset(content_tokens(self.chunks[i])))

    def toks(self, i: int):
        return self._tok(i)

    def words(self, ids) -> int:
        return int(sum(len(self.chunks[int(i)].split()) for i in ids))


# ------------------------------------------------------------------ answer-hit proxy (V-A)

def _phrase(s: str) -> str:
    return " " + " ".join(tokens(s)) + " "


def answer_hit(corpus: Corpus, ids, phrase: str) -> bool:
    p = _phrase(phrase)
    return any(p in _phrase(corpus.chunks[int(i)]) for i in ids)


def proxy_defined(q) -> bool:
    return len(q.options[q.key].split()) <= 5


def mrr_at(corpus, ids_matrix, qs, depth=20):
    rr = []
    for q, ids in zip(qs, ids_matrix):
        if not proxy_defined(q):
            continue
        r = next((j + 1 for j, i in enumerate(ids[:depth]) if answer_hit(corpus, [i], q.options[q.key])), None)
        rr.append(1.0 / r if r else 0.0)
    return float(np.mean(rr)) if rr else 0.0


# ------------------------------------------------------------------ configurations

class Lab:
    def __init__(self, books, sizes=SIZES, lsa_dim=256, seed=0):
        self.corpora = {s: Corpus(data.chunk(books, s)) for s in sizes}
        self.default = DEFAULT if DEFAULT in self.corpora else sizes[0]
        base = self.corpora[self.default]
        self.lsa = LSA(base.chunks, dim=lsa_dim, seed=seed)
        self.idf = IDF(base.chunks)
        self.seed = seed

    def selector(self, size):
        c = self.corpora[size]
        return Selector(self.idf, c.toks, seed=self.seed)

    def bm25(self, qs, form, k, size=None):
        size = size or self.default
        return self.corpora[size].bm25.search([q.query(form) for q in qs], k)

    def lsa_search(self, qs, form, k):
        return self.lsa.search([q.query(form) for q in qs], k)

    def select(self, qs, ids_list, size=None, weights=None):
        size = size or self.default
        sel, corp = self.selector(size), self.corpora[size]
        preds, words = [], []
        for n, (q, ids) in enumerate(zip(qs, ids_list)):
            w = None if weights is None else weights[n]
            preds.append(sel.predict(q.options, ids, w))
            words.append(corp.words(ids))
        return preds, words

    # --- named configurations: each returns (predictions, context words per question)
    def config(self, name: str, qs):
        K, D = K_DEFAULT, self.default
        if name == "Random-context control":
            rng = np.random.default_rng(self.seed)
            n = len(self.corpora[D].chunks)
            return self.select(qs, [rng.choice(n, K, replace=False) for _ in qs])
        if name.startswith("BM25 (") and name.endswith(")"):
            form = name[6:-1]
            ids, _ = self.bm25(qs, form, K)
            return self.select(qs, ids)
        if name == "LSA (q+opts)":
            ids, _ = self.lsa_search(qs, "q+opts", K)
            return self.select(qs, ids)
        if name == "Hybrid BM25+LSA (RRF)":
            a, _ = self.bm25(qs, "q+opts", 50)
            b, _ = self.lsa_search(qs, "q+opts", 50)
            ids, _ = rrf([a, b], K)
            return self.select(qs, ids)
        if name == "Multi-query fusion (RRF)":
            ids, _ = rrf([self.bm25(qs, f, 50)[0] for f in FORMS], K)
            return self.select(qs, ids)
        if name == "BM25 (q+opts), score-weighted":
            ids, sc = self.bm25(qs, "q+opts", K)
            w = sc / np.maximum(sc[:, :1], 1e-9)
            return self.select(qs, ids, weights=w)
        if name.startswith("BM25 (q+opts), compression tau="):
            tau = float(name.split("=")[1])
            ids, sc = self.bm25(qs, "q+opts", 20)
            return self.select(qs, compress(ids, sc, tau))
        if name.startswith("BM25 (q+opts), K="):
            k = int(name.split("=")[1])
            ids, _ = self.bm25(qs, "q+opts", k)
            return self.select(qs, ids)
        if name.startswith("BM25 (q+opts), chunk="):
            size, k = [int(x) for x in name.split("chunk=")[1].split(", K=")]
            ids, _ = self.bm25(qs, "q+opts", k, size)
            return self.select(qs, ids, size)
        if name == "Per-option BM25":
            sel, corp = self.selector(D), self.corpora[D]
            preds, words = [], []
            for q in qs:
                s = {}
                used = 0
                for l in data.LETTERS:
                    ids, _ = corp.bm25.search([f"{q.stem} {q.options[l]}"], K)
                    s[l] = sel.scores(q.options, ids[0])[l]
                    used += corp.words(ids[0])
                preds.append(sel.pick(s))
                words.append(used)
            return preds, words
        raise KeyError(name)


BASELINES = ("Random guess", "Longest option", "Corpus-frequency prior", "Random-context control")
RETRIEVAL = ("BM25 (q)", "BM25 (q+opts)", "BM25 (metamap+opts)", "BM25 (last+opts)", "LSA (q+opts)",
             "Hybrid BM25+LSA (RRF)", "Multi-query fusion (RRF)", "BM25 (q+opts), score-weighted",
             "BM25 (q+opts), compression tau=0.8", "Per-option BM25")


def chunk_configs(lab):
    return tuple(f"BM25 (q+opts), chunk={s}, K={EQUAL_BUDGET[s]}" for s in lab.corpora if s in EQUAL_BUDGET)


def evaluate(lab: Lab, qs, names) -> dict:
    out = {}
    for name in names:
        t0 = time.perf_counter()
        if name == "Random guess":
            preds, words = random_guess(qs, lab.seed), [0] * len(qs)
        elif name == "Longest option":
            preds, words = longest_option(qs, lab.seed), [0] * len(qs)
        elif name == "Corpus-frequency prior":
            preds, words = frequency_prior(qs, lab.idf, lab.seed), [0] * len(qs)
        else:
            preds, words = lab.config(name, qs)
        correct = [p == q.key for p, q in zip(preds, qs)]
        lo, hi = stats.bootstrap_ci(correct)
        out[name] = {"accuracy": float(np.mean(correct)), "ci95": [lo, hi],
                     "context_words": float(np.mean(words)), "seconds": round(time.perf_counter() - t0, 1),
                     "correct": [int(c) for c in correct]}
        print(f"  {name:45s} {out[name]['accuracy'] * 100:5.1f}%")
    return out


def significance(res: dict) -> dict:
    ctrl = res["Random-context control"]["correct"]
    ret = [n for n in res if n not in BASELINES]
    raw = {n: stats.mcnemar_exact(ctrl, res[n]["correct"]) for n in ret}
    adj = stats.holm(raw)
    ref = res.get("BM25 (q+opts)", {}).get("correct")
    return {n: {"p_vs_control": raw[n], "p_vs_control_holm": adj[n],
                "p_vs_bm25_qopts": (stats.mcnemar_exact(ref, res[n]["correct"]) if ref and n != "BM25 (q+opts)" else None)}
            for n in ret}


# ------------------------------------------------------------------ dev-only analyses

def dev_proxies(lab: Lab, dev) -> dict:
    corp = lab.corpora[lab.default]
    out = {}
    for form in ("q", "q+opts"):
        ids, _ = lab.bm25(dev, form, 5)
        key_hits, wrong_hits, n = 0, 0, 0
        for q, r in zip(dev, ids):
            if not proxy_defined(q):
                continue
            n += 1
            key_hits += answer_hit(corp, r, q.options[q.key])
            wrong = [l for l in data.LETTERS if l != q.key]
            wrong_hits += np.mean([answer_hit(corp, r, q.options[l]) for l in wrong])
        out[form] = {"answer_hit@5": key_hits / n, "wrong_option_hit@5": wrong_hits / n,
                     "discriminative_gap": (key_hits - wrong_hits) / n, "n": n}
    long = [q for q in dev if len(q.options[q.key].split()) >= 6]
    out["share_answers_6plus_words"] = len(long) / len(dev)
    return out


def bm25_grid(lab: Lab, dev) -> list[dict]:
    corp = lab.corpora[lab.default]
    rows = []
    for k1 in (0.5, 0.9, 1.2, 1.5, 2.0):
        for b in (0.25, 0.5, 0.75, 1.0):
            m = BM25(corp.chunks, k1=k1, b=b)
            ids, _ = m.search([q.query("q+opts") for q in dev], 20)
            rows.append({"k1": k1, "b": b, "mrr@20": mrr_at(corp, ids, dev)})
    return rows


def faiss_bench(lab: Lab, dev, n_latency: int = 300) -> list[dict]:
    import faiss

    faiss.omp_set_num_threads(1)
    corp = lab.corpora[lab.default]
    X = lab.lsa.vecs
    Q = lab.lsa.embed([q.query("q+opts") for q in dev])
    flat = faiss.IndexFlatIP(X.shape[1]); flat.add(X)
    _, exact = flat.search(Q, 10)
    rows = []
    for name, params, make in faiss_indexes(X.shape[1], len(X)):
        ix = make()
        t0 = time.perf_counter()
        if not ix.is_trained:
            ix.train(X)
        ix.add(X)
        build = time.perf_counter() - t0
        if "efSearch" in params:
            ix.hnsw.efSearch = params["efSearch"]
        if "nprobe" in params:
            ix.nprobe = params["nprobe"]
        _, top = ix.search(Q, 20)
        recall = float(np.mean([len(set(a[:10]) & set(b)) / 10 for a, b in zip(top, exact)]))
        lat = []
        for q in Q[:n_latency]:
            t = time.perf_counter(); ix.search(q[None, :], 10); lat.append((time.perf_counter() - t) * 1000)
        rows.append({"index": name, "recall@10": recall, "p50_ms": float(np.median(lat)),
                     "size_mb": index_size_mb(ix), "build_s": build, "mrr@20": mrr_at(corp, top, dev)})
        print(f"  {name:28s} R@10 {recall:.3f}  p50 {rows[-1]['p50_ms']:.3f} ms")
    return rows


def similarity_metrics(lab: Lab, dev) -> list[dict]:
    corp = lab.corpora[lab.default]
    raw, unit = lab.lsa.raw, lab.lsa.vecs
    qr = lab.lsa.embed([q.query("q+opts") for q in dev], normalise=False)
    qu = lab.lsa.embed([q.query("q+opts") for q in dev], normalise=True)

    def topk(scores, k=20):
        return np.argsort(-scores, axis=1)[:, :k]

    def l2_neg(q, X):
        return -(np.sum(q ** 2, 1)[:, None] - 2 * q @ X.T + np.sum(X ** 2, 1)[None, :])

    cos = topk(qu @ unit.T)
    variants = {"cosine": cos,
                "inner product (normalised)": topk(qu @ unit.T),
                "L2 (normalised)": topk(l2_neg(qu, unit)),
                "inner product (raw)": topk(qr @ raw.T),
                "L2 (raw)": topk(l2_neg(qr, raw))}
    return [{"metric": n, "top10_overlap_with_cosine": float(np.mean([len(set(a[:10]) & set(b[:10])) / 10
                                                                    for a, b in zip(v, cos)])),
             "mrr@20": mrr_at(corp, v, dev)} for n, v in variants.items()]


# ------------------------------------------------------------------ main

def run(books, dev, test, out: Path, sizes=SIZES, lsa_dim=256, do_faiss=True, do_grid=True):
    out.mkdir(parents=True, exist_ok=True)
    report = {"data": {"dev": len(dev), "test": len(test)}}

    print("Leakage audit ...")
    report["leakage"] = leakage.audit(books, dev + test)
    print(" ", {k: v for k, v in report["leakage"].items() if k != "overlapping_ids"})

    print("Building indexes ...")
    lab = Lab(books, sizes, lsa_dim)
    report["chunks"] = {s: len(c.chunks) for s, c in lab.corpora.items()}

    names = BASELINES + RETRIEVAL + chunk_configs(lab)
    names_k = tuple(f"BM25 (q+opts), K={k}" for k in TOPK)
    names_tau = tuple(f"BM25 (q+opts), compression tau={t}" for t in TAUS if t != 0.8)

    print("DEV (selection) ...")
    dev_res = evaluate(lab, dev, names + names_k + names_tau)
    report["dev"] = {n: {k: v for k, v in r.items() if k != "correct"} for n, r in dev_res.items()}
    best_dev = max((n for n in dev_res if n not in BASELINES), key=lambda n: dev_res[n]["accuracy"])
    report["best_dev_config"] = best_dev
    report["dev_proxies"] = dev_proxies(lab, dev)
    if do_grid:
        print("BM25 k1 x b grid ...")
        report["bm25_grid"] = bm25_grid(lab, dev)
    report["similarity_metrics"] = similarity_metrics(lab, dev)
    if do_faiss:
        print("FAISS indexes ...")
        report["faiss"] = faiss_bench(lab, dev)

    print("TEST (single run) ...")
    test_names = names + names_k + ("BM25 (q+opts), compression tau=0.0", "BM25 (q+opts), compression tau=0.9")
    if best_dev not in test_names:
        test_names += (best_dev,)
    test_res = evaluate(lab, test, test_names)
    report["test"] = {n: {k: v for k, v in r.items() if k != "correct"} for n, r in test_res.items()}
    report["test_significance"] = significance({n: test_res[n] for n in names})
    steps = {}
    for n in ("BM25 (q+opts)", "Per-option BM25"):
        c = np.array(test_res[n]["correct"])
        steps[n] = {s: float(c[[q.step == s for q in test]].mean()) for s in ("step1", "step2&3")
                    if any(q.step == s for q in test)}
        long = np.array([len(q.options[q.key].split()) >= 6 for q in test])
        steps[n]["answers_6plus_words"] = float(c[long].mean()) if long.any() else None
    report["test_subgroups"] = steps

    (out / "report.json").write_text(json.dumps(report, indent=2))
    preds = {n: r["correct"] for n, r in test_res.items()}
    (out / "test_correctness.json").write_text(json.dumps(preds))
    return report


def main():
    books = data.load_textbooks()
    dev, test = data.load_split("dev"), data.load_split("test")
    rep = run(books, dev, test, ROOT / "results")
    from .report import build
    build(rep, ROOT / "results")


if __name__ == "__main__":
    main()
