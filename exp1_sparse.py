"""E1: sparse retrieval on dev. (a) query formulation x chunk size, (b) BM25 k1/b grid on c512."""
import json, numpy as np, pandas as pd
from lib import *

qs = load_questions("dev")
rows = []
for tag in ["c256", "c512", "c1024"]:
    chunks = load_chunks(tag); texts = [c["text"] for c in chunks]; ntexts = [norm(t) for t in texts]
    bm = BM25(texts)
    forms = {f: [query_forms(q)[f] for q in qs] for f in query_forms(qs[0])}
    for f, ql in forms.items():
        idx, sc, lat = bm.search(ql, 20)
        m = hit_metrics(qs, idx, ntexts)
        rows.append({"exp": "form", "chunk": tag, "query": f, "k1": 1.5, "b": 0.75,
                     "lat_ms_p50": 1000 * np.median(lat), "lat_ms_p95": 1000 * np.percentile(lat, 95),
                     "build_s": bm.build_s, **m})
        print(rows[-1], flush=True)
    if tag == "c512":
        ql = forms["q+opts"]
        for k1 in [0.6, 0.9, 1.2, 1.5, 2.0]:
            for b in [0.3, 0.5, 0.75, 0.9]:
                bm.set_params(k1, b)
                idx, sc, lat = bm.search(ql, 20)
                rows.append({"exp": "grid", "chunk": tag, "query": "q+opts", "k1": k1, "b": b, **hit_metrics(qs, idx, ntexts)})
                print(k1, b, rows[-1]["AnsHit@5"], rows[-1]["DisHit@5"], flush=True)
    del bm
pd.DataFrame(rows).to_csv(f"{ROOT}/results/e1_sparse_dev.csv", index=False)
