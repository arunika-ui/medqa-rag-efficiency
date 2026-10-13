"""E2: dense retrieval with a locally-trained LSA embedding (TF-IDF -> truncated SVD, 256-d),
used as a controlled stand-in for neural embeddings (which cannot be downloaded in this environment).
(a) FAISS index types vs exact search: recall@10 overlap, latency (1 thread), build time, memory.
(b) Similarity metric: cosine vs inner product vs L2, with and without L2-normalisation.
(c) Saves exact LSA top-100 for dev/test (for fusion experiments)."""
import json, time, pickle, numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from lib import *

faiss.omp_set_num_threads(1)
DIM = 256
rows, mrows = [], []
for tag in ["c256", "c512"]:
    chunks = load_chunks(tag); texts = [c["text"] for c in chunks]; ntexts = [norm(t) for t in texts]
    t0 = time.time()
    tv = TfidfVectorizer(lowercase=True, token_pattern=TOKPAT, stop_words=STOP, sublinear_tf=True, min_df=3, dtype=np.float32)
    X = tv.fit_transform(texts)
    svd = TruncatedSVD(DIM, algorithm="randomized", n_iter=5, random_state=0)
    D_raw = svd.fit_transform(X).astype(np.float32)
    emb_s = time.time() - t0
    D = D_raw / np.linalg.norm(D_raw, axis=1, keepdims=True).clip(1e-9)
    print(tag, "embedding trained", round(emb_s, 1), "s", flush=True)

    for split in ["dev", "test"]:
        qs = load_questions(split)
        for form in ["q", "q+opts"]:
            Q_raw = svd.transform(tv.transform([query_forms(q)[form] for q in qs])).astype(np.float32)
            Q = Q_raw / np.linalg.norm(Q_raw, axis=1, keepdims=True).clip(1e-9)
            flat = faiss.IndexFlatIP(DIM); flat.add(D)
            s_ex, i_ex = flat.search(Q, 100)
            np.save(f"{ROOT}/results/lsa_{split}_{tag}_{form}_idx.npy", i_ex); np.save(f"{ROOT}/results/lsa_{split}_{tag}_{form}_sc.npy", s_ex)
            if split != "dev" or form != "q+opts":
                continue
            base = {"chunk": tag, "n_vectors": len(D), "embed_train_s": emb_s}
            rows.append({**base, "index": "Flat (exact)", "param": "-", "build_s": 0.0, "bytes": D.nbytes,
                         **{"recall@10_vs_exact": 1.0}, **hit_metrics(qs, i_ex[:, :20], ntexts)})
            gt = [set(r[:10]) for r in i_ex]

            def evaluate(index, name, param, build_s):
                lat = []
                I = np.zeros((len(Q), 20), dtype=np.int64)
                for i in range(len(Q)):
                    t = time.perf_counter(); _, ii = index.search(Q[i:i + 1], 20); lat.append(time.perf_counter() - t); I[i] = ii[0]
                rec = np.mean([len(gt[i] & set(I[i, :10])) / 10 for i in range(len(Q))])
                rows.append({**base, "index": name, "param": param, "build_s": build_s,
                             "bytes": faiss.serialize_index(index).nbytes,
                             "lat_ms_p50": 1000 * np.median(lat), "lat_ms_p95": 1000 * np.percentile(lat, 95),
                             "recall@10_vs_exact": rec, **hit_metrics(qs, I, ntexts)})
                print(tag, name, param, round(rec, 3), round(rows[-1]["lat_ms_p50"], 3), flush=True)

            evaluate(flat, "Flat (exact)", "-", 0.0); rows.pop(-2)
            for M in [16, 32]:
                t = time.time(); h = faiss.IndexHNSWFlat(DIM, M, faiss.METRIC_INNER_PRODUCT); h.hnsw.efConstruction = 80; h.add(D); b = time.time() - t
                for ef in [16, 32, 64, 128]:
                    h.hnsw.efSearch = ef; evaluate(h, f"HNSW M={M}", f"efSearch={ef}", b)
            nlist = int(4 * np.sqrt(len(D)))
            t = time.time(); qz = faiss.IndexFlatIP(DIM); ivf = faiss.IndexIVFFlat(qz, DIM, nlist, faiss.METRIC_INNER_PRODUCT); ivf.train(D); ivf.add(D); b = time.time() - t
            for npb in [1, 4, 16, 64]:
                ivf.nprobe = npb; evaluate(ivf, f"IVF nlist={nlist}", f"nprobe={npb}", b)
            t = time.time(); qz2 = faiss.IndexFlatIP(DIM); pq = faiss.IndexIVFPQ(qz2, DIM, nlist, 32, 8, faiss.METRIC_INNER_PRODUCT); pq.train(D); pq.add(D); b = time.time() - t
            for npb in [4, 16, 64]:
                pq.nprobe = npb; evaluate(pq, f"IVF-PQ nlist={nlist} m=32", f"nprobe={npb}", b)

            # (b) metric comparison, exact search
            def topk(M_, Qm, metric):
                ix = faiss.IndexFlatIP(DIM) if metric == "ip" else faiss.IndexFlatL2(DIM)
                ix.add(M_); return ix.search(Qm, 20)[1]
            variants = {
                "normalised": {"cosine": topk(D, Q, "ip"), "inner_product": topk(D, Q, "ip"), "L2": topk(D, Q, "l2")},
                "unnormalised": {"cosine": topk(D, Q, "ip"), "inner_product": topk(D_raw, Q_raw, "ip"), "L2": topk(D_raw, Q_raw, "l2")},
            }
            for vname, v in variants.items():
                ref = v["cosine"]
                for mname, I in v.items():
                    agree = np.mean([len(set(ref[i, :10]) & set(I[i, :10])) / 10 for i in range(len(Q))])
                    mrows.append({"chunk": tag, "vectors": vname, "metric": mname, "top10_overlap_with_cosine": agree, **hit_metrics(qs, I, ntexts)})
                    print(tag, vname, mname, round(agree, 3), flush=True)
    del X, D, D_raw
pd.DataFrame(rows).to_csv(f"{ROOT}/results/e2_index_dev.csv", index=False)
pd.DataFrame(mrows).to_csv(f"{ROOT}/results/e2_metric_dev.csv", index=False)
