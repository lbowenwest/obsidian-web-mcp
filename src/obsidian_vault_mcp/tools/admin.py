"""Admin tools for the vault MCP server."""

import json
import logging

logger = logging.getLogger(__name__)

_engine = None


def set_engine(engine) -> None:
    """Set the retrieval engine reference. Called by server.py."""
    global _engine
    _engine = engine


def vault_reindex_impl(full: bool = False) -> str:
    """Trigger reindexing. Returns JSON string with stats."""
    if _engine is None:
        return json.dumps({"error": "Semantic search is not available"})

    return _engine.reindex(full=full)
