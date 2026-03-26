"""Tests for retrieval Pydantic models."""

import pytest
from obsidian_vault_mcp.retrieval.models import Chunk, ChunkMetadata, SearchResult, ReindexResult


def test_chunk_metadata_validates():
    meta = ChunkMetadata(
        path="notes/test.md",
        section="Introduction",
        chunk_index=0,
        total_chunks=3,
        tags=["python", "testing"],
        content_hash="abc123",
    )
    assert meta.path == "notes/test.md"
    assert meta.tags == ["python", "testing"]


def test_chunk_metadata_allows_none_section():
    meta = ChunkMetadata(
        path="test.md",
        section=None,
        chunk_index=0,
        total_chunks=1,
        tags=[],
        content_hash="abc123",
    )
    assert meta.section is None


def test_chunk_includes_text_and_metadata():
    chunk = Chunk(
        text="Some content here",
        metadata=ChunkMetadata(
            path="test.md",
            section=None,
            chunk_index=0,
            total_chunks=1,
            tags=[],
            content_hash="abc",
        ),
    )
    assert chunk.text == "Some content here"
    assert chunk.metadata.path == "test.md"


def test_search_result_validates():
    result = SearchResult(
        path="notes/test.md",
        section="Introduction",
        content="Some matching content",
        score=0.85,
        tags=["python"],
    )
    assert result.score == 0.85


def test_reindex_result_validates():
    result = ReindexResult(
        files_indexed=100,
        chunks_created=450,
        files_skipped=2,
        duration_seconds=3.14,
    )
    assert result.files_indexed == 100
    assert result.files_skipped == 2
