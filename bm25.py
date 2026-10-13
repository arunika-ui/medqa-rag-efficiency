import os
"""Stage 1 (sparse baseline): BM25 retrieval over the textbook corpus for MedQA-US 4-option questions.
Sweeps chunk size x query formulation; logs top-20 chunk ids per question plus retrieval latency.
Retrieval quality proxy: Answer-Hit@K (normalised correct-option text appears in a retrieved chunk),
reported alongside Distractor-Hit@K (same test for the 3 wrong options) so the proxy's discriminative
value can be judged."""
import json, re, sys, time, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "dev"
QF = os.environ.get("MEDQA_DIR", "data_clean") + f"/questions/US/4_options/phrases_no_exclude_{SPLIT}.jsonl"
K1, B, TOPK = 1.5, 0.75, 20
norm = lambda s: " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "

qs = [json.loads(l) for l in open(QF)]
queries = {
    "q": [q["question"] for q in qs],
    "q+opts": [q["question"] + " " + " ".join(q["options"].values()) for q in qs],
}

results = {}
for tag in ["c256", "c512", "c1024"]:
    chunks = [json.loads(l) for l in open(os.path.dirname(os.path.abspath(__file__)) + f"/data/chunks_{tag}.jsonl")]
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    cv = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z0-9][a-z0-9]+\b", stop_words=list(ENGLISH_STOP_WORDS), dtype=np.float32)
    tf = cv.fit_transform(texts).tocsc()
    N, dl = tf.shape[0], np.asarray(tf.sum(1)).ravel()
    df = np.diff(tf.indptr)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
    tf = tf.tocsr(); rows = np.repeat(np.arange(N), np.diff(tf.indptr))
    denom = tf.data + K1 * (1 - B + B * dl[rows] / dl.mean())
    tf.data = (tf.data * (K1 + 1) / denom * idf[tf.indices]).astype(np.float32)
    W = tf.T.tocsr()  # vocab x docs, for fast query scoring
    build = time.time() - t0
    ntexts = [norm(t) for t in texts]
    for qname, qlist in queries.items():
        qv = cv.transform(qlist); qv.data[:] = 1  # binary query terms
        top, lat = [], []
        for i in range(qv.shape[0]):
            t = time.perf_counter()
            s = (qv[i] @ W).toarray().ravel()
            idx = np.argpartition(-s, TOPK)[:TOPK]; idx = idx[np.argsort(-s[idx])]
            lat.append(time.perf_counter() - t); top.append(idx)
        hit = {k: [] for k in (1, 5, 10, 20)}; dhit = {k: [] for k in (1, 5, 10, 20)}
        per_q = []
        for q, idx in zip(qs, top):
            ans = norm(q["answer"])
            wrong = [norm(v) for k, v in q["options"].items() if k != q["answer_idx"]]
            first = next((r for r, j in enumerate(idx) if ans in ntexts[j]), None)
            for k in hit:
                hit[k].append(first is not None and first < k)
                dhit[k].append(np.mean([any(w in ntexts[j] for j in idx[:k]) for w in wrong]))
            per_q.append({"rank_first_answer_hit": first, "top": [chunks[j]["id"] for j in idx]})
        key = f"{tag}|{qname}"
        results[key] = {
            "chunks": N, "index_build_s": round(build, 1), "vocab": len(cv.vocabulary_),
            "lat_ms_p50": round(1000 * np.median(lat), 2), "lat_ms_p95": round(1000 * np.percentile(lat, 95), 2),
            **{f"AnswerHit@{k}": round(float(np.mean(v)), 4) for k, v in hit.items()},
            **{f"DistractorHit@{k}": round(float(np.mean(v)), 4) for k, v in dhit.items()},
            "MRR@20_proxy": round(float(np.mean([1 / (r["rank_first_answer_hit"] + 1) if r["rank_first_answer_hit"] is not None else 0 for r in per_q])), 4),
        }
        json.dump(per_q, open(os.path.dirname(os.path.abspath(__file__)) + f"/results/bm25_{SPLIT}_{tag}_{qname}.json", "w"))
        print(key, results[key], flush=True)
    del W, tf
json.dump(results, open(os.path.dirname(os.path.abspath(__file__)) + f"/results/bm25_{SPLIT}_summary.json", "w"), indent=1)
