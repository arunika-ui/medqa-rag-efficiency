"""MedQA-USMLE (4-option) questions, the 18 English MedQA textbooks, chunking and query formulations.

Section III and IV-A of the paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LETTERS = "ABCD"


@dataclass
class Question:
    qid: str
    stem: str
    options: dict[str, str]
    key: str                      # gold letter
    step: str = ""                # "step1" or "step2&3"
    metamap: list[str] = field(default_factory=list)

    def query(self, form: str) -> str:
        opts = " ".join(self.options[l] for l in LETTERS)
        if form == "q":
            return self.stem
        if form == "q+opts":
            return f"{self.stem} {opts}"
        if form == "metamap+opts":
            return f"{' '.join(self.metamap) or self.stem} {opts}"
        if form == "last+opts":
            return f"{last_sentence(self.stem)} {opts}"
        raise ValueError(form)


def last_sentence(stem: str) -> str:
    parts = [p for p in re.split(r"(?<=[.?!])\s+", stem.strip()) if p]
    return parts[-1] if parts else stem


# ---- loading (sources verified to carry dev split + MetaMap phrases; see data/SOURCES.md) ----

def _norm_row(i, r, split) -> Question:
    opts = r["options"]
    if isinstance(opts, list):  # [{"key": "A", "value": "..."}]
        opts = {o["key"]: o["value"] for o in opts}
    return Question(qid=f"{split}-{i}", stem=r["question"], options={k: opts[k] for k in LETTERS},
                    key=r["answer_idx"], step=r.get("meta_info", ""), metamap=list(r.get("metamap_phrases") or []))


def load_split(split: str) -> list[Question]:
    """split in {"dev", "test"}."""
    from datasets import load_dataset

    from .sources import QUESTIONS
    ds = load_dataset(QUESTIONS["name"], QUESTIONS.get("config"), split=QUESTIONS["splits"][split])
    return [_norm_row(i, r, split) for i, r in enumerate(ds)]


def load_textbooks() -> dict[str, str]:
    """{book: full text}, rebuilt in order from the MedRAG/textbooks snippets of the same 18 books."""
    from datasets import load_dataset

    ds = load_dataset("MedRAG/textbooks", split="train")
    books: dict[str, list[tuple[int, str]]] = {}
    for r in ds:
        book, _, idx = r["id"].rpartition("_")
        books.setdefault(book, []).append((int(idx), r["content"]))
    return {b: merge_snippets([t for _, t in sorted(p)]) for b, p in sorted(books.items())}


def merge_snippets(snippets: list[str], max_overlap: int = 400) -> str:
    """Join consecutive snippets, dropping the character overlap the splitter introduced,
    so no passage of the original book is counted twice."""
    out = snippets[0] if snippets else ""
    for nxt in snippets[1:]:
        cut = 0
        for n in range(min(max_overlap, len(out), len(nxt)), 20, -1):
            if out.endswith(nxt[:n]):
                cut = n
                break
        out = out + (nxt[cut:] if cut else " " + nxt)
    return out


def chunk(books: dict[str, str], size: int, overlap: float = 0.125) -> list[str]:
    """Sliding word windows of `size` words with 12.5% overlap (192 / 384 / 768 words in the paper)."""
    step = int(round(size * (1 - overlap)))
    out = []
    for text in books.values():
        w = text.split()
        for s in range(0, max(len(w) - size, 0) + step, step):
            piece = w[s:s + size]
            if len(piece) >= size // 4:
                out.append(" ".join(piece))
    return out
