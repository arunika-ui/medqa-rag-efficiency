"""Summarise Stage-2 (LLM) runs: accuracy + 95% CI, McNemar vs closed-book and vs random-context,
tokens, latency, LLM calls, answer-in-context 2x2 (retrieval vs generation failure), harmful-retrieval rate.
Usage: python analyze_stage2.py runs/test"""
import sys, glob, json, os, numpy as np, pandas as pd
from lib import mcnemar, boot_ci

d = sys.argv[1]
runs = {os.path.basename(p)[:-6]: pd.DataFrame([json.loads(l) for l in open(p)]).sort_values("qid") for p in glob.glob(f"{d}/*.jsonl")}
closed = runs.get("closed"); rand = runs.get("random5")
rows = []
for name, r in sorted(runs.items()):
    c = r.correct.values.astype(bool)
    m, lo, hi = boot_ci(c)
    row = {"config": name, "n": len(r), "acc": m, "ci_lo": lo, "ci_hi": hi, "invalid": r.invalid.mean(),
           "acc_step1": r[r.step == "step1"].correct.mean(), "acc_step23": r[r.step != "step1"].correct.mean(),
           "in_tok": r.in_tok.mean(), "out_tok": r.out_tok.mean(), "llm_calls": r.calls.mean(),
           "total_ms_p50": r.total_ms.median(), "total_ms_p95": r.total_ms.quantile(.95),
           "retr_ms_p50": r.get("retr_ms", pd.Series([np.nan])).median()}
    if closed is not None and len(closed) == len(r) and name != "closed":
        cc = closed.correct.values.astype(bool)
        n01, n10, p = mcnemar(cc, c)
        row.update(p_vs_closed=p, fixed_by_rag=n01, broken_by_rag=n10, harmful_rate=n10 / max(cc.sum(), 1))
    if rand is not None and len(rand) == len(r) and name not in ("random5", "closed"):
        row["p_vs_random_ctx"] = mcnemar(rand.correct.values.astype(bool), c)[2]
    if "ans_in_ctx" in r and r.ctx.map(len).sum() > 0:
        a = r.ans_in_ctx.values.astype(bool)
        row.update(ctx_hit=a.mean(), acc_given_hit=c[a].mean() if a.any() else np.nan,
                   acc_given_miss=c[~a].mean() if (~a).any() else np.nan)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(f"{d}/summary.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string(index=False))
