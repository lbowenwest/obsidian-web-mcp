"""SentenceTransformer wrapper for embedding generation."""

import logging
from typing import Protocol

from .. import config

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """Protocol for embedding functions (allows mocking in tests)."""

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Lazy-loading wrapper around SentenceTransformer."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded")

    def encode(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Encode texts into embedding vectors."""
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()


def build_context_prefix(title: str, tags: list[str], section: str | None) -> str:
    """Build context prefix for a chunk to improve embedding quality."""
    tags_str = ", ".join(tags) if tags else ""
    section_str = section or ""
    return f"Title: {title} | Tags: {tags_str} | Section: {section_str}\n\n"
