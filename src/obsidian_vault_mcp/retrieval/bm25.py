"""BM25 index wrapper with serialization support."""

import json
import logging
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Thread-safe BM25 index over text chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._lock = threading.Lock()

    def build(self, corpus: list[tuple[str, str]]) -> None:
        """Build the BM25 index from (chunk_id, text) pairs."""
        if not corpus:
            with self._lock:
                self._bm25 = None
                self._chunk_ids = []
                self._corpus_texts = []
            return

        chunk_ids = [cid for cid, _ in corpus]
        texts = [text for _, text in corpus]
        tokenized = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized)

        with self._lock:
            self._bm25 = bm25
            self._chunk_ids = chunk_ids
            self._corpus_texts = texts

    def query(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Query the index. Returns list of (chunk_id, score) sorted by score desc."""
        with self._lock:
            if self._bm25 is None or not self._chunk_ids:
                return []
            bm25 = self._bm25
            chunk_ids = self._chunk_ids

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored = list(zip(chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(cid, float(s)) for cid, s in scored[:top_k] if s > 0]

    def save(self, path: Path) -> None:
        """Serialize corpus to disk as JSON (safe, no pickle)."""
        with self._lock:
            data = {
                "chunk_ids": self._chunk_ids,
                "corpus_texts": self._corpus_texts,
            }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Deserialize corpus from disk and rebuild BM25 index."""
        with open(path, "r") as f:
            data = json.load(f)
        index = cls()
        corpus = list(zip(data["chunk_ids"], data["corpus_texts"]))
        index.build(corpus)
        return index

    @property
    def corpus_texts(self) -> list[str]:
        with self._lock:
            return list(self._corpus_texts)

    @property
    def chunk_ids(self) -> list[str]:
        with self._lock:
            return list(self._chunk_ids)
