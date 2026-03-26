"""Tests for BM25 index wrapper."""

import tempfile
from pathlib import Path

import pytest
from obsidian_vault_mcp.retrieval.bm25 import BM25Index


def test_build_and_query():
    """Build an index and query it."""
    index = BM25Index()
    corpus = [
        ("chunk_0", "python programming language"),
        ("chunk_1", "javascript web development"),
        ("chunk_2", "python data science machine learning"),
    ]
    index.build(corpus)
    results = index.query("python", top_k=2)
    assert len(results) == 2
    chunk_ids = [r[0] for r in results]
    assert "chunk_0" in chunk_ids or "chunk_2" in chunk_ids


def test_query_returns_scores():
    """Results include positive scores."""
    index = BM25Index()
    index.build([("a", "hello there world"), ("b", "goodbye forever world"), ("c", "goodbye again")])
    results = index.query("hello", top_k=2)
    assert len(results) >= 1
    assert results[0][1] > 0


def test_empty_index_returns_empty():
    """Querying empty index returns empty list."""
    index = BM25Index()
    index.build([])
    results = index.query("anything", top_k=5)
    assert results == []


def test_serialize_deserialize(tmp_path):
    """Index can be saved and loaded from disk."""
    index = BM25Index()
    index.build([
        ("a", "python programming testing language"),
        ("b", "java development testing framework"),
        ("c", "rust systems programming")
    ])
    pkl_path = tmp_path / "bm25.pkl"

    index.save(pkl_path)
    assert pkl_path.exists()

    loaded = BM25Index.load(pkl_path)
    results = loaded.query("python", top_k=1)
    assert len(results) == 1
    assert results[0][0] == "a"


def test_rebuild_replaces_index():
    """Rebuilding with new corpus replaces old results."""
    index = BM25Index()
    index.build([
        ("a", "python programming language"),
        ("b", "java development tools"),
        ("c", "go systems programming")
    ])
    results1 = index.query("python", top_k=1)
    assert results1[0][0] == "a"

    index.build([
        ("c", "rust systems language"),
        ("d", "python is great language"),
        ("e", "go programming")
    ])
    results2 = index.query("python", top_k=1)
    assert results2[0][0] == "d"


def test_query_no_match():
    """Query with no matching terms returns empty or low scores."""
    index = BM25Index()
    index.build([("a", "python programming"), ("b", "java development")])
    results = index.query("zzznonexistent", top_k=5)
    non_zero = [(cid, s) for cid, s in results if s > 0]
    assert len(non_zero) == 0
