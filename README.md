# Does Retrieval Configuration Matter for Medical QA?

**A controlled retrieval-stage study of RAG on MedQA-USMLE**

Code for the paper by **Arunika Jain** (Manipal University Jaipur), submitted to **IEEE NICE–TEAS ASIA 2026**.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![tests](https://github.com/arunika-ui/medqa-rag-efficiency/actions/workflows/tests.yml/badge.svg)

## Summary

Studies that compare retrieval-augmented generation (RAG) setups usually score answers by *semantic
similarity* to a reference, and rarely test whether retrieval found useful evidence. This project
**isolates the retrieval stage** of medical RAG and measures it with objective ground truth on
MedQA-USMLE (4-option, chance = 25%).

Across the 18 English MedQA textbooks, it varies **chunk size, query formulation, BM25 parameters, a dense
LSA retriever, FAISS index type, similarity metric, top-K, score-based compression, rank fusion and
per-option retrieval**. Every configuration is scored by a **parameter-free evidence-only answer selector**,
compared with a **random-context control**, selected on the dev set only, and run **once** on the 1,273
test questions.

### Key findings (as reported in the paper)

| Finding | Number |
| --- | --- |
| Retrieval beats random context **only when the answer options enter the query** | 31.2% vs. 25.6% (Holm p = 0.017) |
| Question-only retrieval | 27.1%, not better than the control |
| Hybrid fusion, compression, score weighting vs. plain BM25 | no significant gain; multi-query fusion is worse (p = 0.03) |
| Score-based compression | removes **60%** of context at unchanged accuracy |
| HNSW vs. exact search | same quality at **~18× lower latency** |
| Graded relevance (398 chunks) | a directly supporting passage exists for only **47%** of questions |
| String-match retrieval proxies vs. graded labels | agree only **75%** of the time |

**Conclusion:** retrieval *coverage*, not retriever sophistication, is the binding constraint.

## Method

| Paper section | What | Code |
| --- | --- | --- |
| III | MedQA-USMLE 4-option (1,272 dev / 1,273 test), 18 textbooks, **13-gram leakage audit** | `data.py`, `leakage.py` |
| IV-A | Sliding windows of 192 / 384 / 768 words (12.5% overlap); Okapi BM25 (k1 = 1.5, b = 0.75) and LSA (TF-IDF → 256-d SVD); queries: q, q+opts, MetaMap+opts, last sentence+opts | `data.py`, `retrieval.py` |
| IV-B | FAISS Flat, HNSW, IVF-Flat, IVF-PQ; cosine vs. inner product vs. L2, with and without normalisation | `retrieval.py`, `experiments.py` |
| IV-C | Evidence-only answer selector (Eq. 1) with seeded tie-breaking | `selector.py` |
| IV-D | Baselines (random, longest option, corpus-frequency prior, random-context control); top-K, compression, RRF fusion, per-option retrieval, equal-budget chunk sizes; 2,000-sample bootstrap CIs, exact McNemar tests, Holm correction | `selector.py`, `stats.py`, `experiments.py` |

The answer selector (Eq. 1): for each option *o*, with *T(o)* its tokens not shared by all four options,

```
s(o) = Σ_k  w_k · Σ_{t ∈ T(o)} idf(t)·[t ∈ c_k]  /  Σ_{t ∈ T(o)} idf(t)
```

It has no parameters fitted to MedQA, so accuracy above the random-context control reflects what retrieval found.

## Reproduction run

The workflow in `.github/workflows/reproduce.yml` downloads the data, runs the full protocol and commits
the results below. Data sources: questions from the BigBio mirror of MedQA (original US 4-option split
with MetaMap phrases); textbooks rebuilt from the `MedRAG/textbooks` snippets of the same 18 books, with
the splitter's overlaps removed. Small differences from the paper's numbers are expected from
corpus-preparation details; the paper remains the reference.

<!-- RESULTS:START -->
_Results appear here after the first reproduction run._
<!-- RESULTS:END -->

| Test accuracy with 95% CIs | Accuracy vs. context size | FAISS speed–recall |
| --- | --- | --- |
| ![](results/fig_test_accuracy.png) | ![](results/fig_context.png) | ![](results/fig_faiss.png) |

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m medqa_rag.experiments     # full protocol → results/
pytest -q                                          # unit + end-to-end tests on a toy corpus
```

## Limitations

The selector is lexical, so accuracies are a lower bound on the evidential value of retrieval, not RAG
accuracy with an LLM. LSA stands in for neural encoders such as MedCPT. The graded relevance labels
(60 dev questions, 398 chunks) come from one AI-assisted annotator and are not yet included in this
repository. Latencies depend on hardware.

## Next stage

Already specified in the paper: an LLM reader under the same protocol, with closed-book and random-context
baselines, BM25 and neural dense retrievers (bge-small, bge-base, MedCPT) at K ∈ {1, 3, 5, 10},
cross-encoder reranking, compression, Map Re-rank and Reciprocal RAG, with accuracy, token and latency
accounting.

## Citation

```bibtex
@inproceedings{jain2026retrieval,
  title     = {Does Retrieval Configuration Matter for Medical {QA}? A Controlled Retrieval-Stage Study of {RAG} on {MedQA-USMLE}},
  author    = {Jain, Arunika},
  booktitle = {IEEE NICE--TEAS ASIA 2026 (submitted)},
  year      = {2026}
}
```
