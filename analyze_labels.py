"""Labelled-subset analysis: graded relevance (0/1/2) for the top-3 chunks of three retrievers on 60 random dev questions.
Labels: 2 = chunk states the fact needed to reach the correct answer; 1 = on-topic/partially supportive; 0 = irrelevant.
Annotation: LLM-assisted (Claude), single annotator; must be spot-checked by a human before publication."""
import json, numpy as np, pandas as pd, os
from lib import load_questions, load_chunks, norm, ROOT, boot_ci

meta = json.load(open(f"{ROOT}/label/meta.json"))
lab = {}
for l in open(f"{ROOT}/label/labels.txt"):
    q, c, g = map(int, l.split()); lab[(q, c)] = g
qs = load_questions("dev"); chunks = load_chunks("c512"); nt = {}

def ntext(j):
    if j not in nt: nt[j] = norm(chunks[j]["text"])
    return nt[j]

rows, per = [], []
for m in meta:
    q = qs[m["qid"]]; ans = norm(q["answer"]); short = len(q["answer"].split()) <= 5
    for r, lst in m["lists"].items():
        g = np.array([lab[(m["qid"], c)] for c in lst])
        dcg = sum((2 ** x - 1) / np.log2(i + 2) for i, x in enumerate(g))
        ideal = sorted([lab[(m["qid"], c)] for c in m["chunks"]], reverse=True)[:3]
        idcg = sum((2 ** x - 1) / np.log2(i + 2) for i, x in enumerate(ideal))
        first2 = next((i for i, x in enumerate(g) if x == 2), None)
        proxy = any(ans in ntext(c) for c in lst)
        per.append({"qid": m["qid"], "retriever": r, "step": q["meta_info"], "short_answer": short,
                    "P@3(>=1)": (g >= 1).mean(), "P@3(=2)": (g == 2).mean(), "Hit@3(=2)": float((g == 2).any()),
                    "Hit@3(>=1)": float((g >= 1).any()), "MRR@3(=2)": 1 / (first2 + 1) if first2 is not None else 0.0,
                    "nDCG@3": dcg / idcg if idcg > 0 else np.nan, "proxy_AnsHit@3": float(proxy)})
p = pd.DataFrame(per)
out = []
for r, d in p.groupby("retriever", sort=False):
    row = {"retriever": r}
    for k in ["P@3(>=1)", "P@3(=2)", "Hit@3(=2)", "Hit@3(>=1)", "MRR@3(=2)", "nDCG@3", "proxy_AnsHit@3"]:
        m_, lo, hi = boot_ci(d[k].dropna().values); row[k] = f"{m_:.3f} [{lo:.2f},{hi:.2f}]"
    out.append(row)
print(pd.DataFrame(out).to_string(index=False))
print("\nquestions with >=1 grade-2 chunk in union of all retrievers:",
      np.mean([any(lab[(m['qid'], c)] == 2 for c in m['chunks']) for m in meta]))
print("questions where nothing relevant (all 0) in union:", np.mean([all(lab[(m['qid'], c)] == 0 for c in m['chunks']) for m in meta]))
# proxy validity: agreement of proxy AnsHit@3 with gold Hit@3(=2), short-answer questions only
s = p[p.short_answer]
tp = ((s.proxy_AnsHit@3 if False else s["proxy_AnsHit@3"]) == 1) & (s["Hit@3(=2)"] == 1)
prec = tp.sum() / max((s["proxy_AnsHit@3"] == 1).sum(), 1); rec = tp.sum() / max((s["Hit@3(=2)"] == 1).sum(), 1)
agree = (s["proxy_AnsHit@3"] == s["Hit@3(=2)"]).mean()
print(f"\nproxy vs gold (short answers, n={len(s)} question-retriever pairs): agreement={agree:.3f} precision={prec:.3f} recall={rec:.3f}")
from scipy.stats import spearmanr
print("per-question Spearman(proxy, nDCG):", spearmanr(p["proxy_AnsHit@3"], p["nDCG@3"].fillna(0)))
print("\nby step:\n", p.groupby(["retriever", "step"])["Hit@3(=2)"].mean().unstack().round(3))
# paired comparison bm25:q+opts vs others on Hit@3(=2)
from lib import mcnemar
w = p.pivot(index="qid", columns="retriever", values="Hit@3(=2)").astype(bool)
for a, b in [("bm25:q+opts", "bm25:q"), ("bm25:q+opts", "lsa:q+opts")]:
    print(a, "vs", b, "McNemar (n01,n10,p):", mcnemar(w[a], w[b]))
pd.DataFrame(out).to_csv(f"{ROOT}/results/labelled_subset_metrics.csv", index=False)
p.to_csv(f"{ROOT}/results/labelled_subset_perquestion.csv", index=False)
