"""Pre-registered test-set comparison (configs chosen on dev before looking at test)."""
import json, numpy as np, pandas as pd
from lib import mcnemar, ROOT

SEL = [
    ("random", "Random guess"),
    ("longest option", "Longest option"),
    ("corpus-frequency prior", "Corpus-frequency prior"),
    ("c512 | random-context | K=5", "Random-context control (K=5)"),
    ("c512 | bm25:q | K=5", "BM25, question only (K=5)"),
    ("c512 | bm25:q+opts | K=5", "BM25, question+options (K=5)"),
    ("c512 | bm25:q+opts | K=5 | score-weighted", "BM25 q+opts, score-weighted (K=5)"),
    ("c512 | bm25:metamap+opts | K=5", "BM25, MetaMap phrases+opts (K=5)"),
    ("c512 | lsa:q+opts | K=5", "LSA dense, q+opts (K=5)"),
    ("c512 | rrf:bm25(q+opts)+lsa(q+opts) | K=5", "Hybrid RRF BM25+LSA (K=5)"),
    ("c512 | rrf:bm25(q,q+opts,metamap+opts) | K=5", "Multi-query RRF (3 queries, K=5)"),
    ("c512 | bm25:q+opts | K=20 | compress tau=0.8", "BM25 K=20 + compression tau=0.8"),
    ("c512 | per-option BM25 | mean top3 score", "Per-option BM25 score (4 queries)"),
    ("c512 | per-option BM25 + reader | K=5", "Per-option retrieval + reader (4 queries, K=5)"),
    ("c256 | bm25:q+opts | K=10", "Chunk 256t, K=10 (~1.9k words)"),
    ("c1024 | bm25:q+opts | K=3", "Chunk 1024t, K=3 (~2.3k words)"),
]
REF = "c512 | bm25:q+opts | K=5"
CTRL = "c512 | random-context | K=5"
out = []
for split in ["dev", "test"]:
    d = pd.read_csv(f"{ROOT}/results/e3_answer_{split}.csv").set_index("config")
    p = json.load(open(f"{ROOT}/results/e3_preds_{split}.json"))
    for key, label in SEL:
        r = d.loc[key]
        row = {"split": split, "config": label, "acc": r.acc, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
               "acc_step1": r.acc_step1, "acc_step23": r.acc_step23, "ctx_words": r.ctx_words, "retrieval_calls": r.calls}
        if split == "test":
            _, _, row["p_vs_randomctx"] = mcnemar(p[CTRL], p[key])
            _, _, row["p_vs_bm25qopts"] = mcnemar(p[REF], p[key])
        out.append(row)
df = pd.DataFrame(out)
# Holm correction over the test-set comparisons against the random-context control
t = df[df.split == "test"].copy()
ps = t["p_vs_randomctx"].values; order = np.argsort(ps); m = len(ps); adj = np.empty(m)
running = 0
for rank, i in enumerate(order):
    running = max(running, (m - rank) * ps[i]); adj[i] = min(1, running)
df.loc[df.split == "test", "p_vs_randomctx_holm"] = adj
df.to_csv(f"{ROOT}/results/main_table.csv", index=False)
wide = df.pivot(index="config", columns="split", values="acc").loc[[l for _, l in SEL]]
wide["test_CI"] = [f"[{a:.3f}, {b:.3f}]" for a, b in df[df.split == "test"][["ci_lo", "ci_hi"]].values]
wide["p_vs_ctrl(Holm)"] = df[df.split == "test"]["p_vs_randomctx_holm"].values
wide["p_vs_BM25"] = df[df.split == "test"]["p_vs_bm25qopts"].values
wide["ctx_words"] = df[df.split == "test"]["ctx_words"].values
print(wide.round(4).to_string())
