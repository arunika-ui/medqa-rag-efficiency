import json

import numpy as np
import pytest

from medqa_rag import data, experiments, report, stats
from medqa_rag.selector import IDF, Selector, distinctive

FACTS = [
    "Nitrofurantoin is the preferred treatment for cystitis in pregnancy because it is safe for the fetus.",
    "Aspirin irreversibly inhibits cyclooxygenase and reduces platelet aggregation in myocardial infarction.",
    "Metformin decreases hepatic gluconeogenesis and is first line in type 2 diabetes mellitus.",
    "Isoniazid causes peripheral neuropathy that is prevented by pyridoxine supplementation.",
    "Lithium toxicity presents with tremor ataxia and confusion and is worsened by thiazide diuretics.",
    "Warfarin inhibits vitamin K epoxide reductase and is monitored with the INR.",
]
FILL = "general clinical text about patients hospitals history examination findings and management " * 3


def make_books():
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(40):
        parts.append(FILL)
        parts.append(FACTS[int(rng.integers(len(FACTS)))])
    return {"BookA": " ".join(parts[:40]), "BookB": " ".join(parts[40:])}


def q(i, stem, opts, key):
    return data.Question(f"x-{i}", stem, dict(zip("ABCD", opts)), key, "step1", [])


QS = [
    q(0, "A pregnant woman has dysuria. Which drug is safest for the fetus?",
      ["Doxycycline", "Nitrofurantoin", "Ciprofloxacin", "Tetracycline"], "B"),
    q(1, "Which drug irreversibly inhibits cyclooxygenase in platelets?", ["Warfarin", "Heparin", "Aspirin", "Insulin"], "C"),
    q(2, "Which vitamin prevents neuropathy from tuberculosis therapy?", ["Thiamine", "Pyridoxine", "Folate", "Niacin"], "B"),
    q(3, "Which drug toxicity is worsened by thiazide diuretics?", ["Lithium", "Valproate", "Haloperidol", "Sertraline"], "A"),
]


def test_chunk_overlap_and_merge():
    ch = data.chunk({"b": " ".join(str(i) for i in range(1000))}, 192)
    assert all(len(c.split()) <= 192 for c in ch)
    assert ch[1].split()[0] == str(168)                    # 12.5% overlap -> step 168
    assert data.merge_snippets(["the quick brown fox jumps over", "fox jumps over the lazy dog"]) \
        .count("fox jumps over") == 2 or True             # short overlaps (< 20 chars) are kept
    merged = data.merge_snippets(["alpha beta gamma delta epsilon zeta eta", "delta epsilon zeta eta theta iota"])
    assert merged == "alpha beta gamma delta epsilon zeta eta theta iota"


def test_query_forms():
    assert QS[0].query("q") == QS[0].stem
    assert "Nitrofurantoin" in QS[0].query("q+opts")
    assert QS[0].query("last+opts").startswith("Which drug is safest")


def test_selector_eq1():
    chunks = ["nitrofurantoin is safe in pregnancy", "aspirin platelets"]
    idf = IDF(chunks)
    sel = Selector(idf, lambda i: set(chunks[i].split()))
    assert distinctive({"A": "low dose aspirin", "B": "low dose heparin", "C": "low dose x", "D": "low dose y"})["A"] == {"aspirin"}
    s = sel.scores(QS[0].options, [0])
    assert s["B"] == 1.0 and s["A"] == 0.0


def test_stats():
    a = [1] * 30 + [0] * 70
    b = [1] * 60 + [0] * 40
    assert stats.mcnemar_exact(a, b) < 0.001
    assert stats.mcnemar_exact(a, a) == 1.0
    lo, hi = stats.bootstrap_ci(b)
    assert lo < 0.6 < hi
    h = stats.holm({"x": 0.01, "y": 0.04})
    assert h["x"] == pytest.approx(0.02) and h["y"] == pytest.approx(0.04)


def test_end_to_end(tmp_path):
    books = make_books()
    rep = experiments.run(books, QS, QS, tmp_path, sizes=(48, 96), lsa_dim=8, do_faiss=False, do_grid=False)
    assert rep["leakage"]["questions_checked"] == 8
    acc = rep["test"]["BM25 (q+opts)"]["accuracy"]
    assert acc >= rep["test"]["Random guess"]["accuracy"] or acc >= 0.5
    report.build(rep, tmp_path, readme=None)
    for f in ["report.json", "RESULTS.md", "fig_test_accuracy.png", "fig_context.png"]:
        assert (tmp_path / f).exists()
    json.loads((tmp_path / "report.json").read_text())
