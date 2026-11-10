"""Hybrid retrieval: vector search (meaning) + BM25 keyword search (exact words), merged with
Reciprocal Rank Fusion.

Vector search understands paraphrases and works across languages, but can miss exact terms such as
names or numbers ("273,15", "Kelvin"). BM25 finds those exact words. Fusing both rankings keeps
the strengths of each.
"""
from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_STOPWORDS = {
    # French
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "en", "dans", "sur", "pour", "par",
    "avec", "que", "qui", "quoi", "est", "sont", "ce", "cette", "ces", "il", "elle", "ils", "elles",
    "on", "ne", "pas", "plus", "au", "aux", "se", "sa", "son", "ses", "leur", "leurs", "qu", "comment",
    "pourquoi", "quel", "quelle", "quels", "quelles", "c", "d", "l", "j", "n", "s", "y", "a",
    # English
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "is", "are", "was", "what", "why",
    "how", "which", "who", "does", "do", "it", "this", "that", "these", "with", "about", "can",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words without accents or stopwords: "Qu'est-ce que l'Été ?" -> ["ete"]."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [word for word in re.findall(r"[a-z0-9]+", text) if word not in _STOPWORDS]


def doc_key(doc: Document) -> str:
    """Identifies a chunk across the two searches."""
    return doc.id or f"{doc.metadata.get('video_id')}@{doc.metadata.get('start')}"


class KeywordIndex:
    """BM25 index over every chunk of the vector store. It is rebuilt automatically when
    videos are added or removed (the chunk count changes)."""

    def __init__(self, store):
        self.store = store
        self.count = -1
        self.docs: list[Document] = []
        self.bm25: BM25Okapi | None = None

    def _refresh(self) -> None:
        count = self.store._collection.count()
        if count == self.count:
            return
        data = self.store.get(include=["documents", "metadatas"])
        self.docs = [Document(id=i, page_content=text, metadata=meta)
                     for i, text, meta in zip(data["ids"], data["documents"], data["metadatas"])]
        self.tokens = [set(tokenize(doc.page_content)) for doc in self.docs]
        self.bm25 = BM25Okapi([tokenize(doc.page_content) for doc in self.docs]) if self.docs else None
        self.count = count

    def search(self, query: str, k: int, video_id: str | None = None) -> list[Document]:
        self._refresh()
        words = tokenize(query)
        if not self.bm25 or not words:
            return []
        scores = self.bm25.get_scores(words)
        # Only chunks that contain at least one word of the query. (BM25 scores alone are not enough:
        # a word present in most chunks can get a zero or negative weight in a small collection.)
        candidates = [i for i, tokens in enumerate(self.tokens)
                      if tokens & set(words)
                      and (video_id is None or self.docs[i].metadata.get("video_id") == video_id)]
        candidates.sort(key=lambda i: (len(self.tokens[i] & set(words)), scores[i]), reverse=True)
        return [self.docs[i] for i in candidates[:k]]


def reciprocal_rank_fusion(rankings: list[list[Document]], k: int, constant: int = 60) -> list[Document]:
    """Merges several rankings: each document scores sum(1 / (60 + rank)) over the lists it appears in.
    Documents found by both searches rise to the top."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank + 1)
            docs.setdefault(key, doc)
    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[key] for key in best]
