import os
"""Build word-window chunks of the MedQA English textbooks at several sizes.
Chunk sizes are in words (~0.75 words/token): 192w≈256t, 384w≈512t, 768w≈1024t.
Overlap is 12.5% of chunk size (64 tokens at 512)."""
import json, os, re, glob, statistics as st

SRC = os.environ.get("MEDQA_DIR", "data_clean") + "/textbooks/en"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SIZES = {"c256": 192, "c512": 384, "c1024": 768}
os.makedirs(OUT, exist_ok=True)

books = {}
for p in sorted(glob.glob(f"{SRC}/*.txt")):
    name = os.path.basename(p)[:-4]
    words = open(p, encoding="utf-8", errors="ignore").read().split()
    books[name] = words

stats = {"books": {b: len(w) for b, w in books.items()}, "total_words": sum(len(w) for w in books.values())}
for tag, size in SIZES.items():
    step = size - size // 8
    n = 0
    with open(f"{OUT}/chunks_{tag}.jsonl", "w") as f:
        for b, words in books.items():
            for i in range(0, max(1, len(words) - size // 8), step):
                f.write(json.dumps({"id": f"{b}:{i}", "book": b, "text": " ".join(words[i:i + size])}) + "\n")
                n += 1
    stats[tag] = n
json.dump(stats, open(f"{OUT}/corpus_stats.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
