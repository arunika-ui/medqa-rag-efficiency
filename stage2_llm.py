"""Stage 2: dense retrieval + LLM generation experiments on MedQA-US (4-option).

Designed to run on Google Colab (GPU) or any machine that can reach Hugging Face and an LLM API.
Reuses the chunk files produced by prep.py and the same question splits as the retrieval-stage study.

Usage (examples):
  python stage2_llm.py --data /content/data_clean --split dev500 --configs closed,random5,bm25_k5
  python stage2_llm.py --data /content/data_clean --split test --configs all \
      --llm-model gpt-4o-mini --llm-base-url https://api.openai.com/v1      # key in env LLM_API_KEY
  # any OpenAI-compatible endpoint works (OpenAI, Groq, OpenRouter, Together, local vLLM/Ollama)

Every question x config is logged as one JSONL row in runs/<split>/<config>.jsonl with:
  prediction, correct, retrieved chunk ids + scores, answer-in-context proxy, input/output tokens
  (from the API's usage field), retrieval / rerank / LLM latency, number of LLM calls.
"""
import argparse, hashlib, json, os, random, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BM25, norm, query_forms, LETTERS  # noqa: E402

# ----------------------------------------------------------------------------- args
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to MedQA data_clean folder")
ap.add_argument("--chunks", default="data/chunks_c512.jsonl")
ap.add_argument("--split", default="dev500", help="dev500 (fixed random 500 dev questions) | dev | test")
ap.add_argument("--configs", default="all")
ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5", help="dense model: BAAI/bge-small-en-v1.5 | BAAI/bge-base-en-v1.5 | medcpt")
ap.add_argument("--reranker", default="BAAI/bge-reranker-base")
ap.add_argument("--llm-model", default="gpt-4o-mini")
ap.add_argument("--llm-base-url", default="https://api.openai.com/v1")
ap.add_argument("--backend", default="openai", choices=["openai", "hf"],
                help="openai = any OpenAI-compatible API (needs LLM_API_KEY); hf = run --llm-model locally with transformers on the GPU (no key)")
ap.add_argument("--load-4bit", action="store_true", help="hf backend: load the model in 4-bit (bitsandbytes), e.g. for 7-8B models on a T4")
ap.add_argument("--out", default="runs")
ap.add_argument("--limit", type=int, default=0, help="debug: only first N questions")
args = ap.parse_args()

random.seed(0); np.random.seed(0)
QDIR = f"{args.data}/questions/US/4_options"


def load_split(name):
    if name == "dev500":
        qs = [json.loads(l) for l in open(f"{QDIR}/phrases_no_exclude_dev.jsonl")]
        idx = sorted(np.random.default_rng(12345).choice(len(qs), 500, replace=False))
        return [qs[i] for i in idx]
    return [json.loads(l) for l in open(f"{QDIR}/phrases_no_exclude_{name}.jsonl")]


QS = load_split(args.split)
if args.limit: QS = QS[: args.limit]
CHUNKS = [json.loads(l) for l in open(args.chunks)]
TEXTS = [c["text"] for c in CHUNKS]
NTEXTS = [norm(t) for t in TEXTS]

# ----------------------------------------------------------------------------- retrievers
print("building BM25 ...", flush=True)
bm25 = BM25(TEXTS)

_dense = {}


def dense():
    """Lazy-load the dense encoder + exact FAISS index (cached embeddings on disk)."""
    if _dense: return _dense
    import faiss, torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.embed.replace("/", "_")
    cache = f"{args.out}/emb_{tag}_{os.path.basename(args.chunks)}.npy"
    if args.embed == "medcpt":
        from transformers import AutoTokenizer, AutoModel
        tok_a = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder"); enc_a = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(dev).eval()
        tok_q = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder"); enc_q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(dev).eval()

        def _enc(tok, enc, texts, maxlen, bs=64):
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    b = tok(texts[i:i + bs], truncation=True, padding=True, max_length=maxlen, return_tensors="pt").to(dev)
                    out.append(enc(**b).last_hidden_state[:, 0].float().cpu().numpy())
            e = np.concatenate(out); return e / np.linalg.norm(e, axis=1, keepdims=True)
        enc_docs = lambda t: _enc(tok_a, enc_a, t, 512)
        enc_query = lambda t: _enc(tok_q, enc_q, t, 512)
    else:
        m = SentenceTransformer(args.embed, device=dev)
        m.max_seq_length = 512
        prefix = "Represent this sentence for searching relevant passages: " if "bge" in args.embed else ""
        enc_docs = lambda t: m.encode(t, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        enc_query = lambda t: m.encode([prefix + x for x in t], batch_size=64, normalize_embeddings=True)
    if os.path.exists(cache):
        D = np.load(cache)
    else:
        t0 = time.time(); D = np.asarray(enc_docs(TEXTS), dtype=np.float32); np.save(cache, D)
        print(f"embedded corpus in {time.time()-t0:.0f}s", flush=True)
    index = faiss.IndexFlatIP(D.shape[1]); index.add(D)
    _dense.update(index=index, enc_query=enc_query, D=D)
    return _dense


def retrieve(kind, query, k):
    t = time.perf_counter()
    if kind == "bm25":
        idx, sc, _ = bm25.search([query], k); idx, sc = idx[0], sc[0]
    else:
        d = dense(); qv = np.asarray(d["enc_query"]([query]), dtype=np.float32)
        sc, idx = d["index"].search(qv, k); idx, sc = idx[0], sc[0]
    return list(map(int, idx)), list(map(float, sc)), (time.perf_counter() - t) * 1000


_rr = {}


def rerank(query, idx, k):
    if not _rr:
        from sentence_transformers import CrossEncoder
        _rr["m"] = CrossEncoder(args.reranker, max_length=512)
    t = time.perf_counter()
    s = _rr["m"].predict([(query, TEXTS[j]) for j in idx], batch_size=32)
    order = np.argsort(-s)[:k]
    return [idx[i] for i in order], [float(s[i]) for i in order], (time.perf_counter() - t) * 1000


def rrf(lists, k=60, depth=50):
    s = {}
    for L in lists:
        for r, j in enumerate(L[:depth]): s[j] = s.get(j, 0) + 1 / (k + r + 1)
    return sorted(s, key=lambda j: -s[j])

# ----------------------------------------------------------------------------- LLM
if args.backend == "openai":
    from openai import OpenAI  # noqa: E402
    client = OpenAI(base_url=args.llm_base_url, api_key=os.environ.get("LLM_API_KEY", "none"))
else:  # local open-weights model, greedy decoding (deterministic, reproducible, free)
    import torch  # noqa: E402
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
    TOK = AutoTokenizer.from_pretrained(args.llm_model)
    _kw = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        _kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    else:
        _kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    MODEL = AutoModelForCausalLM.from_pretrained(args.llm_model, **_kw).eval()


def _hf_generate(user, max_tokens):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        prompt = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # models without a system role in their template
        prompt = TOK.apply_chat_template([{"role": "user", "content": SYSTEM + "\n\n" + user}], tokenize=False, add_generation_prompt=True)
    enc = TOK(prompt, return_tensors="pt", add_special_tokens=False).to(MODEL.device)
    with torch.no_grad():
        out = MODEL.generate(**enc, max_new_tokens=max_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=TOK.pad_token_id if TOK.pad_token_id is not None else TOK.eos_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    return TOK.decode(gen, skip_special_tokens=True), int(enc["input_ids"].shape[1]), int(gen.shape[0])
CACHE_F = f"{args.out}/llm_cache.jsonl"; os.makedirs(args.out, exist_ok=True)
CACHE = {}
if os.path.exists(CACHE_F):
    for l in open(CACHE_F):
        r = json.loads(l); CACHE[r["key"]] = r
SYSTEM = "You are a medical expert answering USMLE-style multiple-choice questions."


def llm(user, max_tokens=5):
    key = hashlib.sha256(json.dumps([args.llm_model, SYSTEM, user, max_tokens]).encode()).hexdigest()
    if key in CACHE: return CACHE[key]
    for attempt in range(6):
        try:
            t = time.perf_counter()
            if args.backend == "hf":
                text, it, ot = _hf_generate(user, max_tokens)
                out = {"key": key, "text": text, "in_tok": it, "out_tok": ot, "ms": (time.perf_counter() - t) * 1000}
            else:
                r = client.chat.completions.create(model=args.llm_model, temperature=0, max_tokens=max_tokens,
                                                   messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
                out = {"key": key, "text": r.choices[0].message.content or "", "in_tok": r.usage.prompt_tokens,
                       "out_tok": r.usage.completion_tokens, "ms": (time.perf_counter() - t) * 1000}
            break
        except Exception as e:  # rate limits etc.
            if args.backend == "hf": raise
            time.sleep(2 ** attempt); err = e
    else:
        raise err
    CACHE[key] = out
    with open(CACHE_F, "a") as f: f.write(json.dumps(out) + "\n")
    return out


def fmt_q(q):
    return q["question"] + "\n\n" + "\n".join(f"{k}. {v}" for k, v in q["options"].items())


def ask(q, ctx_idx=None):
    ctx = ""
    if ctx_idx:
        ctx = "Use the following textbook excerpts if they are relevant.\n\n" + "\n\n".join(f"[{i+1}] {TEXTS[j]}" for i, j in enumerate(ctx_idx)) + "\n\n"
    return llm(ctx + "Question:\n" + fmt_q(q) + "\n\nAnswer with the single letter (A, B, C or D) of the correct option only.")


def parse(text):
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    return m.group(1) if m else None

# ----------------------------------------------------------------------------- methods
def m_closed(q, **_):
    r = ask(q); return dict(pred=parse(r["text"]), calls=1, in_tok=r["in_tok"], out_tok=r["out_tok"], llm_ms=r["ms"], ctx=[])


def m_random(q, k=5, qi=0, **_):
    ctx = list(np.random.default_rng(qi).choice(len(TEXTS), k, replace=False))
    r = ask(q, ctx); return dict(pred=parse(r["text"]), calls=1, in_tok=r["in_tok"], out_tok=r["out_tok"], llm_ms=r["ms"], ctx=ctx)


def m_stuff(q, kind="bm25", form="q+opts", k=5, rerank_from=0, tau=None, **_):
    query = query_forms(q)[form]
    idx, sc, rms = retrieve(kind, query, rerank_from or k)
    rrms = 0
    if rerank_from:
        idx, sc, rrms = rerank(q["question"], idx, k)
    if tau is not None:  # contextual compression: keep chunks with score >= tau * best score
        keep = [i for i, s in zip(idx, sc) if s >= tau * sc[0]]; idx = keep or idx[:1]
    r = ask(q, idx)
    return dict(pred=parse(r["text"]), calls=1, in_tok=r["in_tok"], out_tok=r["out_tok"], llm_ms=r["ms"], retr_ms=rms, rerank_ms=rrms, ctx=idx, scores=sc)


def m_map_rerank(q, kind="bm25", k=5, **_):
    """Map Re-rank: answer from each chunk separately with a 0-100 confidence; keep the most confident answer."""
    idx, sc, rms = retrieve(kind, query_forms(q)["q+opts"], k)
    best, calls, it, ot, ms = (-1, None), 0, 0, 0, 0
    for j in idx:
        ctx = f"Textbook excerpt:\n{TEXTS[j]}\n\nQuestion:\n{fmt_q(q)}\n\nUsing the excerpt, reply exactly as: <letter> <confidence 0-100>"
        r = llm(ctx, max_tokens=8); calls += 1; it += r["in_tok"]; ot += r["out_tok"]; ms += r["ms"]
        m = re.search(r"([ABCD])\D*(\d{1,3})", r["text"].upper())
        if m and int(m.group(2)) > best[0]: best = (int(m.group(2)), m.group(1))
    return dict(pred=best[1], calls=calls, in_tok=it, out_tok=ot, llm_ms=ms, retr_ms=rms, ctx=idx)


def m_reciprocal(q, kind="bm25", k=5, n_sub=3, **_):
    """Reciprocal RAG / RAG-Fusion: LLM writes n_sub focused sub-queries; retrieve for each + original; RRF-fuse; answer once."""
    g = llm(f"Write {n_sub} short, specific search queries (one per line, no numbering) that would retrieve textbook passages needed to answer:\n\n{fmt_q(q)}", max_tokens=120)
    subs = [s.strip("-• ").strip() for s in g["text"].split("\n") if s.strip()][:n_sub]
    lists, rms = [], 0
    for s in [query_forms(q)["q+opts"]] + subs:
        idx, _, ms = retrieve(kind, s, 20); lists.append(idx); rms += ms
    ctx = rrf(lists)[:k]
    r = ask(q, ctx)
    return dict(pred=parse(r["text"]), calls=2, in_tok=g["in_tok"] + r["in_tok"], out_tok=g["out_tok"] + r["out_tok"],
                llm_ms=g["ms"] + r["ms"], retr_ms=rms, ctx=ctx, subqueries=subs)


CONFIGS = {
    "closed": (m_closed, {}),
    "random5": (m_random, {"k": 5}),
    **{f"bm25_k{k}": (m_stuff, {"kind": "bm25", "k": k}) for k in (1, 3, 5, 10)},
    **{f"dense_k{k}": (m_stuff, {"kind": "dense", "k": k}) for k in (1, 3, 5, 10)},
    "dense_q_k5": (m_stuff, {"kind": "dense", "form": "q", "k": 5}),
    "bm25_rerank20_k5": (m_stuff, {"kind": "bm25", "k": 5, "rerank_from": 20}),
    "dense_rerank20_k5": (m_stuff, {"kind": "dense", "k": 5, "rerank_from": 20}),
    **{f"bm25_k10_compress{t}": (m_stuff, {"kind": "bm25", "k": 10, "tau": t}) for t in (0.7, 0.8, 0.9)},
    "bm25_maprerank_k5": (m_map_rerank, {"kind": "bm25", "k": 5}),
    "bm25_reciprocal_k5": (m_reciprocal, {"kind": "bm25", "k": 5}),
}
todo = list(CONFIGS) if args.configs == "all" else args.configs.split(",")

# ----------------------------------------------------------------------------- run
os.makedirs(f"{args.out}/{args.split}", exist_ok=True)
for name in todo:
    fn, kw = CONFIGS[name]
    path = f"{args.out}/{args.split}/{name}.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as f:
        for qi, q in enumerate(QS):
            if qi in done: continue
            t = time.perf_counter()
            r = fn(q, qi=qi, **kw)
            r["total_ms"] = (time.perf_counter() - t) * 1000
            ans = norm(q["answer"])
            r.update(qid=qi, gold=q["answer_idx"], correct=r["pred"] == q["answer_idx"], invalid=r["pred"] is None,
                     step=q["meta_info"], ans_in_ctx=any(ans in NTEXTS[j] for j in r.get("ctx", [])))
            f.write(json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    rows = [json.loads(l) for l in open(path)]
    print(f"{name:24s} acc={np.mean([x['correct'] for x in rows]):.4f} invalid={np.mean([x['invalid'] for x in rows]):.3f} "
          f"in_tok={np.mean([x['in_tok'] for x in rows]):.0f} ms={np.median([x['total_ms'] for x in rows]):.0f}", flush=True)
