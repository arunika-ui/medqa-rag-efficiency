"""E3: end-to-end MedQA accuracy from retrieval evidence alone (no LLM).
Answer selection = 'evidence reader': IDF-weighted coverage of each option's distinctive terms in the retrieved chunks.
Sweeps: retriever/query form, top-K, relative-score compression, rank fusion (RRF), per-option retrieval, chunk size.
Baselines: random, longest option, corpus-frequency prior, random-context control."""
import sys, json, time, numpy as np, pandas as pd
from lib import *

SPLIT = sys.argv[1]
K1, B = float(sys.argv[2]), float(sys.argv[3])
rng = np.random.default_rng(0)
qs = load_questions(SPLIT)
gold = np.array([LETTERS.index(q["answer_idx"]) for q in qs])
step = np.array([q["meta_info"] for q in qs])
rows, preds = [], {}


def pick(scores):
    s = np.asarray(scores, float); best = np.flatnonzero(s == s.max())
    return int(rng.choice(best)), len(best) > 1


def record(name, pred_ties, **info):
    p = np.array([x[0] for x in pred_ties]); ties = np.mean([x[1] for x in pred_ties])
    c = p == gold; preds[name] = c.tolist()
    m, lo, hi = boot_ci(c)
    rows.append({"config": name, "acc": m, "ci_lo": lo, "ci_hi": hi, "tie_rate": ties,
                 "acc_step1": c[step == "step1"].mean(), "acc_step23": c[step != "step1"].mean(), **info})
    print(f"{name:55s} acc={m:.4f} [{lo:.3f},{hi:.3f}] ties={ties:.2f}", {k: v for k, v in info.items()}, flush=True)


for tag in ["c512", "c256", "c1024"]:
    chunks = load_chunks(tag); texts = [c["text"] for c in chunks]
    words = np.array([len(t.split()) for t in texts])
    bm = BM25(texts, K1, B)
    an = bm.cv.build_analyzer(); vocab = bm.cv.vocabulary_; idf = bm.idf
    chunk_sets = bm.raw_tf  # csr: row = chunk, indices = present vocab ids

    def opt_terms(q):
        toks = [set(vocab[t] for t in an(v) if t in vocab) for v in q["options"].values()]
        common = set.intersection(*toks)
        return [np.array(sorted(t - common), dtype=np.int64) for t in toks]

    OT = [opt_terms(q) for q in qs]

    def read(qi, cidx, weights=None):
        """score each option by IDF-weighted term coverage summed over retrieved chunks."""
        sc = []
        sub = chunk_sets[cidx]
        for t in OT[qi]:
            if len(t) == 0: sc.append(0.0); continue
            pres = (sub[:, t] > 0).toarray().astype(np.float32)  # K x |t|
            cov = pres @ idf[t] / idf[t].sum()
            sc.append(float(cov.sum() if weights is None else (cov * weights).sum()))
        return sc

    if tag == "c512":
        # --- no-retrieval baselines
        record("random", [(int(rng.integers(4)), False) for _ in qs], family="baseline", calls=0, ctx_words=0)
        record("longest option", [pick([len(v) for v in q["options"].values()]) for q in qs], family="baseline", calls=0, ctx_words=0)
        dfreq = np.bincount(bm.raw_tf.indices, minlength=len(idf))
        record("corpus-frequency prior", [pick([dfreq[t].mean() if len(t) else 0 for t in OT[i]]) for i in range(len(qs))], family="baseline", calls=0, ctx_words=0)

    # --- retrieval lists
    t0 = time.perf_counter()
    lists = {}
    for form in ["q", "q+opts", "metamap+opts"]:
        idx, sc, lat = bm.search([query_forms(q)[form] for q in qs], 50)
        lists[f"bm25:{form}"] = (idx, sc, float(np.median(lat)) * 1000)
    if tag in ("c512", "c256"):
        for form in ["q", "q+opts"]:
            lists[f"lsa:{form}"] = (np.load(f"{ROOT}/results/lsa_{SPLIT}_{tag}_{form}_idx.npy")[:, :50],
                                    np.load(f"{ROOT}/results/lsa_{SPLIT}_{tag}_{form}_sc.npy")[:, :50], np.nan)

    def rrf(names, k=60, depth=50):
        out = []
        for i in range(len(qs)):
            s = {}
            for n in names:
                for r, j in enumerate(lists[n][0][i][:depth]):
                    s[j] = s.get(j, 0) + 1 / (k + r + 1)
            top = sorted(s, key=lambda j: -s[j])[:depth]
            out.append(top)
        return np.array(out)

    if tag in ("c512", "c256"):
        lists["rrf:bm25(q+opts)+lsa(q+opts)"] = (rrf(["bm25:q+opts", "lsa:q+opts"]), None, np.nan)
        lists["rrf:bm25(q,q+opts,metamap+opts)"] = (rrf(["bm25:q", "bm25:q+opts", "bm25:metamap+opts"]), None, np.nan)
        lists["rrf:all5"] = (rrf(["bm25:q", "bm25:q+opts", "bm25:metamap+opts", "lsa:q", "lsa:q+opts"]), None, np.nan)

    Ks = [1, 3, 5, 10, 20] if tag == "c512" else [5]
    if tag != "c512":  # equal context budget comparison: ~1,920 words
        Ks = {"c256": [10], "c1024": [3]}[tag] + [5]
    for name, (idx, sc, lat) in lists.items():
        if tag != "c512" and not name.startswith("bm25:q+opts") and name != "rrf:all5":
            continue
        for K in Ks:
            record(f"{tag} | {name} | K={K}", [pick(read(i, idx[i][:K])) for i in range(len(qs))],
                   family="shared-retrieval", chunk=tag, retriever=name, K=K, calls={"rrf:bm25(q+opts)+lsa(q+opts)":2,"rrf:bm25(q,q+opts,metamap+opts)":3,"rrf:all5":5}.get(name,1),
                   ctx_words=float(np.mean([words[idx[i][:K]].sum() for i in range(len(qs))])), retr_ms=lat)

    if tag == "c512":
        # --- random-context control (same K, random chunks)
        for K in [5, 20]:
            record(f"c512 | random-context | K={K}", [pick(read(i, rng.choice(len(texts), K, replace=False))) for i in range(len(qs))],
                   family="control", chunk=tag, K=K, calls=0, ctx_words=float(words.mean() * K))
        # --- compression: keep chunks with score >= tau * top score, from top-20 of bm25:q+opts
        idx, sc, _ = lists["bm25:q+opts"]
        for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
            kept = [idx[i][:20][sc[i][:20] >= tau * sc[i][0]] for i in range(len(qs))]
            record(f"c512 | bm25:q+opts | K=20 | compress tau={tau}", [pick(read(i, kept[i])) for i in range(len(qs))],
                   family="compression", chunk=tag, K=20, tau=tau, calls=1,
                   ctx_words=float(np.mean([words[k].sum() for k in kept])), kept_chunks=float(np.mean([len(k) for k in kept])))
        # --- score-weighted reading (rank-aware)
        for K in [5, 20]:
            record(f"c512 | bm25:q+opts | K={K} | score-weighted", [pick(read(i, idx[i][:K], sc[i][:K] / sc[i][0])) for i in range(len(qs))],
                   family="shared-retrieval", chunk=tag, K=K, calls=1, ctx_words=float(np.mean([words[idx[i][:K]].sum() for i in range(len(qs))])))
        # --- per-option retrieval (4 retrieval calls; Jin et al.-style IR baseline and a 'map'-style method)
        per = {L: bm.search([q["question"] + " " + q["options"][L] for q in qs], 10) for L in LETTERS}
        for agg, f in [("top1 score", lambda s: s[:, 0]), ("mean top3 score", lambda s: s[:, :3].mean(1))]:
            S = np.stack([f(per[L][1]) for L in LETTERS], 1)
            record(f"c512 | per-option BM25 | {agg}", [pick(S[i]) for i in range(len(qs))], family="per-option", chunk=tag, calls=4,
                   ctx_words=0, retr_ms=4 * float(np.median(per["A"][2])) * 1000)
        for K in [3, 5]:
            record(f"c512 | per-option BM25 + reader | K={K}", [pick([read(i, per[L][0][i][:K])[j] for j, L in enumerate(LETTERS)]) for i in range(len(qs))],
                   family="per-option", chunk=tag, K=K, calls=4, ctx_words=float(4 * words.mean() * K))
    del bm

pd.DataFrame(rows).to_csv(f"{ROOT}/results/e3_answer_{SPLIT}.csv", index=False)
json.dump(preds, open(f"{ROOT}/results/e3_preds_{SPLIT}.json", "w"))
