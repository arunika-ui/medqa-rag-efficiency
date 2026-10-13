# Stage 1 results: retrieval-stage study (no LLM)

**Setup**
- **Corpus:** the 18 English MedQA textbooks, 12.85M words, split into word-window chunks of about 256, 512 or 1,024 tokens with 12.5% overlap. That gives 76,507, 38,255 and 19,131 chunks.
- **Questions:** MedQA-US, 4-option version. Dev has 1,272 questions and is used for all selection. Test has 1,273 and was run once, on configurations fixed on dev.
- **Default settings:** 512-token chunks and BM25 with k1 = 1.5, b = 0.75.
- **Dense retriever:** a locally trained LSA model (TF-IDF → 256-d SVD). It stands in for neural embeddings, which couldn't be downloaded in this environment.
- **Answering:** an *evidence reader* picks the option whose distinctive terms (IDF-weighted) are most covered by the retrieved chunks. This gives an objective accuracy for the retrieval stage without an LLM.
- **Statistics:** 95% bootstrap confidence intervals and exact McNemar tests, Holm-corrected against the random-context control.

## R0. Benchmark leakage
None of the 2,545 dev and test question stems shares a single word 13-gram with the textbook corpus.

## R1. Retrieval proxy metrics (dev)
Answer-Hit@K is valid only for answers of 5 words or fewer (n = 1,073). For answers of 6 or more words, Answer-Hit@5 was 0% in the question + options run (512-token chunks).

| Query (512-token chunks) | Answer-Hit@5 | Distractor-Hit@5 | Gap | MRR@20 |
|---|---|---|---|---|
| question | 0.160 | 0.094 | 0.066 | 0.104 |
| question + options | 0.359 | 0.282 | 0.076 | 0.266 |
| MetaMap phrases + options | 0.374 | 0.292 | 0.081 | 0.267 |
| final sentence + options | 0.502 | 0.447 | 0.055 | 0.404 |

- **Adding options to the query inflates hits for wrong options almost as much as for the correct one.** A raw hit rate therefore overstates retrieval quality, and the *gap* between the two is the better signal.
- **BM25 parameters barely matter.** Across a 20-point k1 × b grid, MRR@20 only ranges from 0.257 to 0.271.

## R2. Similarity metric (dev, exact search, LSA vectors)

| Vectors | Cosine | Inner product | L2 |
|---|---|---|---|
| L2-normalised: top-10 overlap with cosine | 1.00 | 1.00 | 1.00 |
| Unnormalised: top-10 overlap with cosine | 1.00 | 0.47 | 0.18 |
| Unnormalised: MRR@20 | 0.156 | 0.155 | 0.071 |

**With normalised vectors the three metrics give identical results.** They diverge only on unnormalised vectors, and there L2 roughly halves retrieval quality.

## R3. Index type (dev, 38,255 × 256-d, one thread)

| Index | Recall@10 vs exact | p50 latency (ms) | Size (MB) | MRR@20 (proxy) |
|---|---|---|---|---|
| Flat (exact) | 1.000 | 1.43 | 39.2 | 0.156 |
| HNSW M=32, ef=64 | 0.972 | 0.081 | 49.6 | 0.154 |
| IVF-Flat, nprobe=16 | 0.918 | 0.071 | 40.3 | 0.154 |
| IVF-PQ (m=32), nprobe=64 | 0.608 | 0.099 | 2.6 | 0.141 |

- **HNSW gives about 97% recall at roughly 18× lower latency than exact search.**
- **IVF-PQ is 15× smaller but caps at about 61% recall.** Its downstream proxy quality drops only about 10%, so recall against exact search overstates the harm.
- **At textbook scale, every index is under 2 ms.** That is negligible next to LLM latency.

## R4. End-to-end accuracy from retrieval evidence (test, n = 1,273)

| Configuration | Test accuracy [95% CI] | p vs random-context (Holm) |
|---|---|---|
| Random-context control (K=5) | 0.256 [0.232, 0.280] | reference |
| BM25, question only (K=5) | 0.271 [0.247, 0.297] | 1.00 |
| BM25, question + options (K=5) | 0.312 [0.288, 0.338] | 0.017 |
| BM25, MetaMap + options (K=5) | 0.324 [0.300, 0.350] | 0.002 |
| LSA dense (K=5) | 0.304 [0.280, 0.330] | 0.045 |
| Hybrid RRF BM25 + LSA (K=5) | 0.321 [0.295, 0.346] | 0.004 |
| Multi-query RRF (3 queries) | 0.287 [0.262, 0.310] | 0.46 |
| K=20 + compression τ=0.8 | 0.319 [0.294, 0.344] | 0.006 |
| Per-option BM25 (4 queries) | 0.339 [0.313, 0.364] | <0.001 |
| 1,024-token chunks, K=3 | 0.344 [0.318, 0.369] | <0.001 |

Other baselines: random guessing 0.280, longest option 0.271, corpus-frequency prior 0.232.

**Findings**
1. **Retrieved evidence helps only when the options are in the query.** Retrieving with the question alone is no better than random textbook chunks.
2. **Most "advanced" additions are not significantly better than plain BM25 with question + options.** This covers hybrid retrieval, score weighting, compression and multi-query fusion (all p > 0.2 vs BM25 on test). Multi-query fusion is significantly *worse* (p = 0.03).
3. **Compression saves context without losing accuracy.** At τ = 0.8, K=20 keeps about 8 of 20 chunks and uses 60% fewer words, with the same accuracy (0.319 vs 0.316).
4. **More context helps only up to about 10 chunks.** Test accuracy rises from K=1 (0.294) to K=10 (0.329), then flattens (0.324 at K=20).
5. **Winner's curse.** The best dev configuration (per-option retrieval + reader, 0.354 on dev) dropped to 0.309 on test. Pre-registering choices on dev and testing once is necessary.

## Limitations
- The evidence reader is deliberately simple, so these numbers are a lower bound on retrieval usefulness, not LLM-RAG accuracy.
- LSA stands in for neural embedding models.
- Latencies are from a 2-core cloud VM.
- Comparisons against BM25 are not multiplicity-corrected.
