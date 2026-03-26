"""Semantic search tool for the vault MCP server."""

import json
import logging

logger = logging.getLogger(__name__)

_engine = None


def set_engine(engine) -> None:
    """Set the retrieval engine reference. Called by server.py."""
    global _engine
    _engine = engine


def vault_semantic_search_impl(
    query: str,
    max_results: int = 10,
    min_score: float = 0.3,
    filter_tags: list[str] | None = None,
    filter_folder: str = "",
    return_full_notes: bool = False,
) -> str:
    """Run hybrid semantic + keyword search. Returns JSON string."""
    if _engine is None:
        return json.dumps({"error": "Semantic search is not available"})

    return _engine.search(
        query=query,
        max_results=max_results,
        min_score=min_score,
        filter_tags=filter_tags,
        filter_folder=filter_folder,
        return_full_notes=return_full_notes,
    )
