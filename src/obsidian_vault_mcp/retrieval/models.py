"""Pydantic schemas for the retrieval engine."""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    """Metadata stored alongside each chunk in the index."""

    path: str
    section: str | None
    chunk_index: int
    total_chunks: int
    tags: list[str]
    content_hash: str


class Chunk(BaseModel):
    """A text chunk with its metadata."""

    text: str
    metadata: ChunkMetadata


class SearchResult(BaseModel):
    """A single search result returned to the caller."""

    path: str
    section: str | None
    content: str
    score: float
    tags: list[str]


class ReindexResult(BaseModel):
    """Stats returned after a reindex operation."""

    files_indexed: int
    chunks_created: int
    files_skipped: int
    duration_seconds: float
