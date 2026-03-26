"""Tests for Reciprocal Rank Fusion and search pipeline logic."""

import pytest
from obsidian_vault_mcp.retrieval.search import reciprocal_rank_fusion, deduplicate_by_path


def test_rrf_merges_two_lists():
    """RRF merges results from vector and BM25."""
    vector_results = [("chunk_a", 0.9), ("chunk_b", 0.7), ("chunk_c", 0.5)]
    bm25_results = [("chunk_b", 5.0), ("chunk_d", 3.0), ("chunk_a", 1.0)]
    merged = reciprocal_rank_fusion(
        vector_results, bm25_results,
        vector_weight=0.6, bm25_weight=0.4, k=60,
    )
    ids = [cid for cid, _ in merged]
    assert "chunk_a" in ids
    assert "chunk_b" in ids
    assert "chunk_d" in ids
    assert "chunk_c" in ids


def test_rrf_scores_normalized_0_to_1():
    """All RRF scores should be between 0 and 1."""
    vector_results = [("a", 0.9), ("b", 0.5)]
    bm25_results = [("a", 5.0), ("b", 2.0)]
    merged = reciprocal_rank_fusion(
        vector_results, bm25_results,
        vector_weight=0.6, bm25_weight=0.4, k=60,
    )
    for _, score in merged:
        assert 0.0 <= score <= 1.0


def test_rrf_sorted_by_score_desc():
    """Results are sorted by descending score."""
    vector_results = [("a", 0.9), ("b", 0.5)]
    bm25_results = [("b", 5.0), ("a", 1.0)]
    merged = reciprocal_rank_fusion(
        vector_results, bm25_results,
        vector_weight=0.6, bm25_weight=0.4, k=60,
    )
    scores = [s for _, s in merged]
    assert scores == sorted(scores, reverse=True)


def test_rrf_empty_inputs():
    """Empty inputs return empty results."""
    assert reciprocal_rank_fusion([], [], 0.6, 0.4, 60) == []


def test_rrf_one_source_empty():
    """Works with one empty source."""
    vector_results = [("a", 0.9)]
    merged = reciprocal_rank_fusion(vector_results, [], 0.6, 0.4, 60)
    assert len(merged) == 1
    assert merged[0][0] == "a"


def test_deduplicate_keeps_highest_score():
    """Deduplication keeps highest-scoring chunk per path."""
    results = [
        ("chunk_0", "notes/a.md", 0.9),
        ("chunk_1", "notes/a.md", 0.7),
        ("chunk_2", "notes/b.md", 0.8),
    ]
    deduped = deduplicate_by_path(results)
    assert len(deduped) == 2
    paths = {r[1] for r in deduped}
    assert paths == {"notes/a.md", "notes/b.md"}
    a_result = [r for r in deduped if r[1] == "notes/a.md"][0]
    assert a_result[2] == 0.9
