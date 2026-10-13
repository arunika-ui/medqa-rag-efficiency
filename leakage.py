import os
"""Benchmark-leakage check: do MedQA dev/test question stems appear verbatim in the textbook corpus?
Measures word 13-gram overlap (standard contamination heuristic) between each question and the corpus."""
import json, glob, re, os, numpy as np, hashlib

SRC = os.environ.get("MEDQA_DIR", "data_clean")
N = 13
norm = lambda s: re.findall(r"[a-z0-9]+", s.lower())

def grams(tokens):
    return [hash(" ".join(tokens[i:i + N])) for i in range(len(tokens) - N + 1)]

corpus = []
for p in sorted(glob.glob(f"{SRC}/textbooks/en/*.txt")):
    corpus.extend(grams(norm(open(p, errors="ignore").read())))
corpus = np.unique(np.array(corpus, dtype=np.int64))
print("corpus 13-grams:", len(corpus))

out = {}
for split in ["dev", "test"]:
    qs = [json.loads(l) for l in open(f"{SRC}/questions/US/4_options/phrases_no_exclude_{split}.jsonl")]
    fracs = []
    for q in qs:
        g = np.array(grams(norm(q["question"])), dtype=np.int64)
        pos=np.searchsorted(corpus,g).clip(0,len(corpus)-1); fracs.append(float((corpus[pos]==g).mean()) if len(g) else 0.0)
    fracs = np.array(fracs)
    out[split] = {"n": len(qs), "any_13gram_hit": int((fracs > 0).sum()),
                  "over_20pct": int((fracs > 0.2).sum()), "over_50pct": int((fracs > 0.5).sum()),
                  "mean_overlap": float(fracs.mean())}
    np.save(os.path.dirname(os.path.abspath(__file__)) + f"/results/leak_{split}.npy", fracs)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/results/leakage.json", "w"), indent=1)
