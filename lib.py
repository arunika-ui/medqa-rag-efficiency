import os
"""Shared utilities: data loading, BM25 sparse index, proxy metrics, stats."""
import json, re, os, time, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = os.path.dirname(os.path.abspath(__file__))
MEDQA = os.environ.get("MEDQA_DIR", "data_clean")
QDIR = f"{MEDQA}/questions/US/4_options"
TOKPAT = r"(?u)\b[a-z0-9][a-z0-9]+\b"
STOP = list(ENGLISH_STOP_WORDS)
_tok = re.compile(r"[a-z0-9]+")
norm = lambda s: " " + " ".join(_tok.findall(s.lower())) + " "
LETTERS = "ABCD"


def load_questions(split):
    return [json.loads(l) for l in open(f"{QDIR}/phrases_no_exclude_{split}.jsonl")]


def load_chunks(tag):
    return [json.loads(l) for l in open(f"{ROOT}/data/chunks_{tag}.jsonl")]


def query_forms(q):
    stem = q["question"]
    sents = re.split(r"(?<=[.?!])\s+", stem.strip())
    opts = " ".join(q["options"].values())
    return {
        "q": stem,
        "q+opts": stem + " " + opts,
        "metamap": " ".join(q.get("metamap_phrases", [])),
        "metamap+opts": " ".join(q.get("metamap_phrases", [])) + " " + opts,
        "ask+opts": sents[-1] + " " + opts,  # only the final interrogative sentence
    }


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75, vectorizer=None):
        t0 = time.time()
        self.cv = vectorizer or CountVectorizer(lowercase=True, token_pattern=TOKPAT, stop_words=STOP, dtype=np.float32)
        tf = (self.cv.transform(texts) if vectorizer else self.cv.fit_transform(texts)).tocsr()
        self.raw_tf = tf
        N = tf.shape[0]
        self.dl = np.asarray(tf.sum(1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.set_params(k1, b)
        self.build_s = time.time() - t0

    def set_params(self, k1, b):
        tf = self.raw_tf.copy()
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * self.dl[rows] / self.dl.mean())
        tf.data = (tf.data * (k1 + 1) / denom * self.idf[tf.indices]).astype(np.float32)
        self.W = tf.T.tocsr()

    def qvec(self, qlist):
        qv = self.cv.transform(qlist); qv.data[:] = 1
        return qv.tocsr()

    def search(self, qlist, topk=20, time_it=False):
        qv = self.qvec(qlist)
        idxs, scores, lat = [], [], []
        for i in range(qv.shape[0]):
            t = time.perf_counter()
            s = (qv[i] @ self.W).toarray().ravel()
            k = min(topk, len(s) - 1)
            idx = np.argpartition(-s, k)[:k]; idx = idx[np.argsort(-s[idx])]
            lat.append(time.perf_counter() - t)
            idxs.append(idx); scores.append(s[idx])
        return np.array(idxs), np.array(scores), np.array(lat)


def hit_metrics(qs, idxs, ntexts, ks=(1, 5, 10, 20), max_ans_words=5):
    """Answer-Hit@K / Distractor-Hit@K, restricted to answers <= max_ans_words (proxy is invalid for long answers)."""
    keep = [i for i, q in enumerate(qs) if len(q["answer"].split()) <= max_ans_words]
    out = {}
    first = []
    for i in keep:
        q, idx = qs[i], idxs[i]
        ans = norm(q["answer"])
        wrong = [norm(v) for k, v in q["options"].items() if k != q["answer_idx"]]
        f = next((r for r, j in enumerate(idx) if ans in ntexts[j]), None)
        first.append(f)
        for k in ks:
            out.setdefault(f"AnsHit@{k}", []).append(f is not None and f < k)
            out.setdefault(f"DisHit@{k}", []).append(np.mean([any(w in ntexts[j] for j in idx[:k]) for w in wrong]))
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["MRR@20"] = float(np.mean([1 / (f + 1) if f is not None else 0 for f in first]))
    res["n_eval"] = len(keep)
    return res


def mcnemar(a, b):
    """Exact two-sided McNemar test on paired boolean correctness vectors."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
