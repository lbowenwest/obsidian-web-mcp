"""Hybrid search with Reciprocal Rank Fusion."""

import logging

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge vector and BM25 results using RRF.

    Returns list of (chunk_id, normalized_score) sorted by score descending.
    Scores normalized to 0.0-1.0 range.
    """
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight / (k + rank + 1)

    if not scores:
        return []

    max_possible = vector_weight / (k + 1) + bm25_weight / (k + 1)
    normalized = [(cid, score / max_possible) for cid, score in scores.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)

    return normalized


def deduplicate_by_path(
    results: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    """Keep only the highest-scoring chunk per note path."""
    best: dict[str, tuple[str, str, float]] = {}
    for chunk_id, path, score in results:
        if path not in best or score > best[path][2]:
            best[path] = (chunk_id, path, score)

    deduped = list(best.values())
    deduped.sort(key=lambda x: x[2], reverse=True)
    return deduped
