import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lib import ROOT

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"],
                     "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "savefig.bbox": "tight",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
OUT = os.path.join(ROOT, "figures")
import os; os.makedirs(OUT, exist_ok=True)

# ---- Fig 1: test accuracy with 95% CI
m = pd.read_csv(f"{ROOT}/results/main_table.csv"); t = m[m.split == "test"].iloc[::-1]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ctrl = {"Random guess", "Longest option", "Corpus-frequency prior", "Random-context control (K=5)"}
for i, r in enumerate(t.itertuples()):
    col = C["gray"] if r.config in ctrl else C["blue"]
    ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(r.acc, i, "o", color=col, ms=6, mec="#fcfcfb", mew=1.5)
ax.axvline(0.25, color=C["ink2"], lw=0.8, ls=":"); ax.text(0.251, len(t) - 0.4, "chance", color=C["ink2"], fontsize=7.5)
ax.set_yticks(range(len(t))); ax.set_yticklabels(t.config, fontsize=7.8); ax.grid(axis="y", visible=False)
ax.set_xlabel("MedQA-USMLE test accuracy (n = 1,273), 95% bootstrap CI")
ax.set_title("Evidence-only answer selection: baselines (gray) vs retrieval configurations (blue)", fontsize=9, loc="left", color=C["ink"])
fig.savefig(f"{OUT}/fig1_test_accuracy.png"); plt.close(fig)

# ---- Fig 2: context budget vs accuracy (K sweep and compression), test split
e = pd.read_csv(f"{ROOT}/results/e3_answer_test.csv")
k = e[e.config.str.match(r"c512 \| bm25:q\+opts \| K=\d+$")].sort_values("ctx_words")
c = e[e.family == "compression"].sort_values("tau", ascending=False)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(k.ctx_words, k.acc, "-o", color=C["blue"], lw=2, ms=6, mec="#fcfcfb", mew=1.5, label="Top-K (K = 1…20)")
ax.plot(c.ctx_words, c.acc, "--s", color=C["orange"], lw=2, ms=6, mec="#fcfcfb", mew=1.5, label="K = 20 + compression (τ = 0…0.9)")
for r in k.itertuples(): ax.annotate(f"K={int(r.K)}", (r.ctx_words, r.acc), xytext=(4, -11), textcoords="offset points", fontsize=7, color=C["ink2"])
for r in c.itertuples():
    if r.tau in (0.7, 0.8, 0.9): ax.annotate(f"τ={r.tau}", (r.ctx_words, r.acc), xytext=(4, 5), textcoords="offset points", fontsize=7, color=C["ink2"])
rc = e[e.config == "c512 | random-context | K=5"].acc.iloc[0]
ax.axhline(rc, color=C["gray"], lw=1, ls=":"); ax.text(380, rc + 0.0015, "random-context control", ha="left", fontsize=7, color=C["ink2"])
ax.set_xscale("log"); ax.set_xlabel("Context size passed to the reader (words, log scale)"); ax.set_ylabel("Test accuracy")
ax.legend(frameon=False, fontsize=7.5, loc="upper left"); ax.set_title("Accuracy vs context budget: gains flatten after ~10 chunks", fontsize=9, loc="left", color=C["ink"])
fig.savefig(f"{OUT}/fig2_context_budget.png"); plt.close(fig)

# ---- Fig 3: ANN index trade-off (c512, 38k vectors, 1 thread)
d = pd.read_csv(f"{ROOT}/results/e2_index_dev.csv"); d = d[d.chunk == "c512"]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
fam = [("HNSW M=32", C["blue"], "o"), ("IVF nlist", C["orange"], "s"), ("IVF-PQ", C["aqua"], "^")]
for pref, col, mk in fam:
    s = d[d["index"].str.startswith(pref)].sort_values("lat_ms_p50")
    ax.plot(s.lat_ms_p50, s["recall@10_vs_exact"], "-" + mk, color=col, lw=2, ms=6, mec="#fcfcfb", mew=1.5, label=pref.replace(" nlist", "Flat").replace("IVFFlat", "IVF-Flat"))
ex = d[d["index"] == "Flat (exact)"].iloc[0]
ax.plot(ex.lat_ms_p50, 1.0, "D", color=C["ink"], ms=6); ax.annotate("Exact (Flat)", (ex.lat_ms_p50, 1.0), xytext=(-60, -12), textcoords="offset points", fontsize=7, color=C["ink2"])
ax.set_xscale("log"); ax.set_xlabel("Median query latency (ms, single thread, log scale)"); ax.set_ylabel("Recall@10 vs exact search")
ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_title("Index choice: speed-recall trade-off (38k × 256-d vectors)", fontsize=9, loc="left", color=C["ink"])
fig.savefig(f"{OUT}/fig3_index_tradeoff.png"); plt.close(fig)
print("ok")
