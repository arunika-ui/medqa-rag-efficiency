"""Figures and README tables from results/report.json (mirrors Figs. 1-3 and Tables 2 of the paper)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GREY, BLUE, ORANGE = "#9aa0a6", "#2a6fb5", "#d9662b"
BASE = ("Random guess", "Longest option", "Corpus-frequency prior", "Random-context control")


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, axis="x")


def fig_accuracy(rep, out):
    sig = rep["test_significance"]
    names = list(BASE) + [n for n in sig]
    names = [n for n in names if n in rep["test"]]
    acc = [rep["test"][n]["accuracy"] * 100 for n in names]
    lo = [acc[i] - rep["test"][n]["ci95"][0] * 100 for i, n in enumerate(names)]
    hi = [rep["test"][n]["ci95"][1] * 100 - acc[i] for i, n in enumerate(names)]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, acc, xerr=[lo, hi], color=[GREY if n in BASE else BLUE for n in names],
            error_kw={"lw": 1, "capsize": 2})
    ax.axvline(25, ls=":", color="#555", lw=1)
    ax.set_yticks(y, names, fontsize=8)
    ax.set(xlabel=f"Test accuracy (%), n = {rep['data']['test']:,}, 95% bootstrap CI", xlim=(15, None))
    _style(ax)
    fig.tight_layout(); fig.savefig(out / "fig_test_accuracy.png", dpi=160); plt.close(fig)


def fig_context(rep, out):
    t = rep["test"]
    ks = sorted((int(n.split("=")[1]), n) for n in t if n.startswith("BM25 (q+opts), K="))
    taus = sorted((float(n.split("=")[1]), n) for n in t if "compression tau=" in n)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([t[n]["context_words"] for _, n in ks], [t[n]["accuracy"] * 100 for _, n in ks], "o-",
            color=BLUE, label="top-K")
    for k, n in ks:
        ax.annotate(f"K={k}", (t[n]["context_words"], t[n]["accuracy"] * 100), fontsize=7,
                    xytext=(3, -10), textcoords="offset points")
    if taus:
        ax.plot([t[n]["context_words"] for _, n in taus], [t[n]["accuracy"] * 100 for _, n in taus], "s--",
                color=ORANGE, label="K=20 + compression")
        for tau, n in taus:
            ax.annotate(f"τ={tau}", (t[n]["context_words"], t[n]["accuracy"] * 100), fontsize=7,
                        xytext=(3, 4), textcoords="offset points", color=ORANGE)
    ax.set(xscale="log", xlabel="Context words per question (log)", ylabel="Test accuracy (%)",
           title="Accuracy vs. context size (BM25, q+opts)")
    ax.legend(frameon=False); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_context.png", dpi=160); plt.close(fig)


def fig_faiss(rep, out):
    if "faiss" not in rep:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    fam = {"Flat": "#111", "HNSW": BLUE, "IVF-Flat": ORANGE, "IVF-PQ": GREY}
    for f, col in fam.items():
        rows = [r for r in rep["faiss"] if r["index"].startswith(f)]
        ax.scatter([r["p50_ms"] for r in rows], [r["recall@10"] for r in rows], color=col, label=f, s=30)
    ax.set(xscale="log", xlabel="Median latency per query, single thread (ms, log)",
           ylabel="Recall@10 vs. exact search", title="Speed–recall frontier of FAISS indexes")
    ax.legend(frameon=False); ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_faiss.png", dpi=160); plt.close(fig)


def _pct(x):
    return f"{x * 100:.1f}%"


def markdown(rep) -> str:
    t, s = rep["test"], rep["test_significance"]
    L = rep["leakage"]
    lines = [f"_Reproduction run: {rep['data']['dev']:,} dev / {rep['data']['test']:,} test questions, "
             f"{L['corpus_words'] / 1e6:.2f}M-word corpus, chunks: "
             + ", ".join(f"{k} words → {v:,}" for k, v in rep["chunks"].items()) + "._", "",
             f"**Leakage audit:** {L['distinct_13grams'] / 1e6:.1f}M distinct 13-grams; "
             f"{L['questions_with_overlap']} of {L['questions_checked']:,} dev/test stems share a 13-gram with the corpus.",
             "", "**Test accuracy** (selected on dev, single test run)", "",
             "| Configuration | Accuracy | 95% CI | p vs. control (Holm) | p vs. BM25 (q+opts) | Context words |",
             "| --- | --- | --- | --- | --- | --- |"]
    order = list(BASE) + list(s)
    for n in order:
        if n not in t:
            continue
        r = t[n]
        sg = s.get(n, {})
        ph = f"{sg['p_vs_control_holm']:.3f}" if sg else "—"
        pb = f"{sg['p_vs_bm25_qopts']:.3f}" if sg.get("p_vs_bm25_qopts") is not None else "—"
        lines.append(f"| {n} | {_pct(r['accuracy'])} | {_pct(r['ci95'][0])}–{_pct(r['ci95'][1])} | {ph} | {pb} | "
                     f"{r['context_words']:.0f} |")
    lines += ["", f"Best configuration on dev: **{rep['best_dev_config']}** "
              f"({_pct(rep['dev'][rep['best_dev_config']]['accuracy'])} dev → "
              f"{_pct(t[rep['best_dev_config']]['accuracy'])} test)."]
    p = rep["dev_proxies"]
    lines += ["", "**Answer-Hit@5 proxy on dev** (answers ≤ 5 words)", "",
              "| Query | Key hit | Wrong-option hit | Discriminative gap |", "| --- | --- | --- | --- |"]
    for f in ("q", "q+opts"):
        lines.append(f"| {f} | {p[f]['answer_hit@5']:.3f} | {p[f]['wrong_option_hit@5']:.3f} | "
                     f"{p[f]['discriminative_gap']:.3f} |")
    if "faiss" in rep:
        lines += ["", "**FAISS indexes** (LSA vectors, dev queries, one thread)", "",
                  "| Index | Recall@10 | p50 latency | Size | MRR@20 |", "| --- | --- | --- | --- | --- |"]
        for r in rep["faiss"]:
            lines.append(f"| {r['index']} | {r['recall@10']:.3f} | {r['p50_ms']:.3f} ms | {r['size_mb']:.1f} MB | "
                         f"{r['mrr@20']:.3f} |")
    if "bm25_grid" in rep:
        m = [r["mrr@20"] for r in rep["bm25_grid"]]
        lines += ["", f"**BM25 k1 × b grid** (20 settings, dev): MRR@20 ranges {min(m):.3f}–{max(m):.3f}."]
    sm = rep["similarity_metrics"]
    lines += ["", "**Similarity metrics** (LSA, dev): " + "; ".join(
        f"{r['metric']}: top-10 overlap with cosine {r['top10_overlap_with_cosine']:.2f}, MRR@20 {r['mrr@20']:.3f}"
        for r in sm) + "."]
    return "\n".join(lines)


def build(rep, out: Path, readme: Path = ROOT / "README.md"):
    out.mkdir(parents=True, exist_ok=True)
    fig_accuracy(rep, out); fig_context(rep, out); fig_faiss(rep, out)
    md = markdown(rep)
    (out / "RESULTS.md").write_text(md + "\n")
    if readme and readme.exists():
        text = readme.read_text()
        readme.write_text(re.sub(r"(<!-- RESULTS:START -->).*?(<!-- RESULTS:END -->)",
                                 lambda m: f"{m.group(1)}\n{md}\n{m.group(2)}", text, flags=re.S))
    print(md)


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    build(json.loads((path / "report.json").read_text()), path)
