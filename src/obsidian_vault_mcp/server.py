"""Obsidian Vault MCP Server.

Exposes read/write access to an Obsidian vault over Streamable HTTP.
Designed to run behind Cloudflare Tunnel for secure remote access.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from . import config
from .config import VAULT_MCP_PORT, VAULT_MCP_TOKEN, VAULT_PATH
from .frontmatter_index import FrontmatterIndex

logger = logging.getLogger(__name__)

# Global frontmatter index instance
frontmatter_index = FrontmatterIndex()

# Global retrieval engine — survives across stateless HTTP sessions.
# With stateless_http=True, lifespan runs per-session, so the engine
# must live at module scope to avoid re-loading the embedding model
# on every request.
_retrieval_engine = None


def _get_or_create_engine():
    """Get or lazily create the global retrieval engine."""
    global _retrieval_engine
    if _retrieval_engine is not None:
        return _retrieval_engine

    if not config.SEMANTIC_SEARCH_ENABLED:
        return None

    try:
        from .retrieval import RetrievalEngine
        _retrieval_engine = RetrievalEngine()
        logger.info("Retrieval engine created (will lazy-init on first use)")
        return _retrieval_engine
    except ImportError:
        logger.error(
            "Semantic search enabled but dependencies not installed. "
            "Run: pip install obsidian-vault-mcp[semantic]"
        )
        return None
    except Exception as e:
        logger.error("Failed to create retrieval engine: %s", e)
        return None


@asynccontextmanager
async def lifespan(server):
    """Start frontmatter index and optional retrieval engine on startup."""
    logger.info("Lifespan starting (vault: %s)", VAULT_PATH)
    frontmatter_index.start()
    logger.info("Frontmatter index built: %d files indexed", frontmatter_index.file_count)

    engine = _get_or_create_engine()
    if engine is not None:
        from .tools.semantic_search import set_engine as set_search_engine
        from .tools.admin import set_engine as set_admin_engine

        set_search_engine(engine)
        set_admin_engine(engine)
        frontmatter_index.on_change(engine.handle_file_change)
        logger.info("Semantic search enabled")

    yield {"frontmatter_index": frontmatter_index, "retrieval_engine": engine}

    # Don't shut down the engine on session end — it's module-level
    # and needs to survive across stateless HTTP sessions.
    frontmatter_index.stop()
    logger.info("Lifespan ended")


# Create the MCP server
mcp = FastMCP(
    "obsidian_web_mcp",
    stateless_http=True,
    # json_response=True,
    streamable_http_path="/",
    lifespan=lifespan,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            # Add your tunnel hostname here, e.g.:
            "vault-mcp.tweakyllama.uk",
        ],
    ),
)


# --- Register all tools ---

from .tools.read import vault_read as _vault_read, vault_batch_read as _vault_batch_read
from .tools.write import vault_write as _vault_write, vault_batch_frontmatter_update as _vault_batch_frontmatter_update
from .tools.search import vault_search as _vault_search, vault_search_frontmatter as _vault_search_frontmatter
from .tools.manage import vault_list as _vault_list, vault_move as _vault_move, vault_delete as _vault_delete
from .models import (
    VaultReadInput,
    VaultWriteInput,
    VaultBatchReadInput,
    VaultBatchFrontmatterUpdateInput,
    VaultSearchInput,
    VaultSearchFrontmatterInput,
    VaultListInput,
    VaultMoveInput,
    VaultDeleteInput,
)


@mcp.tool(
    name="vault_read",
    description="Read a file from the Obsidian vault, returning content, metadata, and parsed YAML frontmatter.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_read(path: str) -> str:
    """Read a file from the vault."""
    logger.debug("vault_read: path=%s", path)
    inp = VaultReadInput(path=path)
    return _vault_read(inp.path)


@mcp.tool(
    name="vault_batch_read",
    description="Read multiple files from the vault in one call. Handles missing files gracefully.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_batch_read(paths: list[str], include_content: bool = True) -> str:
    """Read multiple files at once."""
    logger.debug("vault_batch_read: %d paths", len(paths))
    inp = VaultBatchReadInput(paths=paths, include_content=include_content)
    return _vault_batch_read(inp.paths, inp.include_content)


@mcp.tool(
    name="vault_write",
    description="Write a file to the Obsidian vault. Supports frontmatter merging with existing files. Creates parent directories by default.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_write(path: str, content: str, create_dirs: bool = True, merge_frontmatter: bool = False) -> str:
    """Write a file to the vault."""
    logger.debug("vault_write: path=%s", path)
    inp = VaultWriteInput(path=path, content=content, create_dirs=create_dirs, merge_frontmatter=merge_frontmatter)
    return _vault_write(inp.path, inp.content, inp.create_dirs, inp.merge_frontmatter)


@mcp.tool(
    name="vault_batch_frontmatter_update",
    description="Update YAML frontmatter fields on multiple files without changing body content. Each update merges new fields into existing frontmatter.",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_batch_frontmatter_update(updates: list[dict]) -> str:
    """Batch update frontmatter fields."""
    logger.debug("vault_batch_frontmatter_update: %d updates", len(updates))
    inp = VaultBatchFrontmatterUpdateInput(updates=updates)
    return _vault_batch_frontmatter_update(inp.updates)


@mcp.tool(
    name="vault_search",
    description="Search for text across vault files. Uses ripgrep if available, falls back to Python. Returns matching lines with context and frontmatter excerpts.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_search(
    query: str,
    path_prefix: str | None = None,
    file_pattern: str = "*.md",
    max_results: int = 20,
    context_lines: int = 2,
) -> str:
    """Search vault file contents."""
    logger.debug("vault_search: query=%r", query)
    inp = VaultSearchInput(query=query, path_prefix=path_prefix, file_pattern=file_pattern, max_results=max_results, context_lines=context_lines)
    return _vault_search(inp.query, inp.path_prefix, inp.file_pattern, inp.max_results, inp.context_lines)


@mcp.tool(
    name="vault_search_frontmatter",
    description="Search vault files by YAML frontmatter field values. Queries an in-memory index for fast results. Supports exact match, contains, and field-exists queries.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_search_frontmatter(
    field: str,
    value: str = "",
    match_type: str = "exact",
    path_prefix: str | None = None,
    max_results: int = 20,
) -> str:
    """Search by frontmatter fields."""
    logger.debug("vault_search_frontmatter: field=%s match=%s", field, match_type)
    inp = VaultSearchFrontmatterInput(field=field, value=value, match_type=match_type, path_prefix=path_prefix, max_results=max_results)
    return _vault_search_frontmatter(inp.field, inp.value, inp.match_type, inp.path_prefix, inp.max_results)


@mcp.tool(
    name="vault_list",
    description="List directory contents in the vault. Supports recursion depth, file/dir filtering, and glob patterns. Excludes .obsidian, .trash, .git directories.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
def vault_list(
    path: str = "",
    depth: int = 1,
    include_files: bool = True,
    include_dirs: bool = True,
    pattern: str | None = None,
) -> str:
    """List vault directory contents."""
    logger.debug("vault_list: path=%s depth=%d", path, depth)
    inp = VaultListInput(path=path, depth=depth, include_files=include_files, include_dirs=include_dirs, pattern=pattern)
    return _vault_list(inp.path, inp.depth, inp.include_files, inp.include_dirs, inp.pattern)


@mcp.tool(
    name="vault_move",
    description="Move a file or directory within the vault. Validates both source and destination paths.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_move(source: str, destination: str, create_dirs: bool = True) -> str:
    """Move a file or directory."""
    logger.debug("vault_move: %s -> %s", source, destination)
    inp = VaultMoveInput(source=source, destination=destination, create_dirs=create_dirs)
    return _vault_move(inp.source, inp.destination, inp.create_dirs)


@mcp.tool(
    name="vault_delete",
    description="Delete a file by moving it to .trash/ in the vault root. Requires confirm=true as a safety gate. Does NOT hard delete.",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": False},
)
def vault_delete(path: str, confirm: bool = False) -> str:
    """Delete a file (move to .trash/)."""
    logger.debug("vault_delete: path=%s confirm=%s", path, confirm)
    inp = VaultDeleteInput(path=path, confirm=confirm)
    return _vault_delete(inp.path, inp.confirm)


# --- Conditional semantic search tools ---

if config.SEMANTIC_SEARCH_ENABLED:
    try:
        from .tools.semantic_search import vault_semantic_search_impl as _vault_semantic_search
        from .tools.admin import vault_reindex_impl as _vault_reindex
        from .models import VaultSemanticSearchInput, VaultReindexInput

        @mcp.tool(
            name="vault_semantic_search",
            description="Hybrid semantic + keyword search across the vault. Combines vector similarity with BM25 keyword matching using Reciprocal Rank Fusion. Returns ranked results with relevance scores. Use this for natural language queries instead of vault_search.",
            annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
        )
        def vault_semantic_search(
            query: str,
            max_results: int = 10,
            min_score: float = 0.3,
            filter_tags: list[str] | None = None,
            filter_folder: str = "",
            return_full_notes: bool = False,
        ) -> str:
            """Hybrid semantic + keyword search."""
            logger.info("vault_semantic_search: query=%r max_results=%d", query, max_results)
            inp = VaultSemanticSearchInput(
                query=query, max_results=max_results, min_score=min_score,
                filter_tags=filter_tags, filter_folder=filter_folder,
                return_full_notes=return_full_notes,
            )
            result = _vault_semantic_search(
                inp.query, inp.max_results, inp.min_score,
                inp.filter_tags, inp.filter_folder, inp.return_full_notes,
            )
            logger.info("vault_semantic_search: completed")
            return result

        @mcp.tool(
            name="vault_reindex",
            description="Rebuild the semantic search index. Use full=true to rebuild from scratch, or full=false for incremental sync.",
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
        )
        def vault_reindex(full: bool = False) -> str:
            """Trigger reindexing."""
            logger.info("vault_reindex: full=%s", full)
            inp = VaultReindexInput(full=full)
            result = _vault_reindex(inp.full)
            logger.info("vault_reindex: completed")
            return result

        logger.info("Semantic search tools registered")

    except ImportError:
        logger.warning(
            "Semantic search enabled but dependencies not installed. "
            "Run: pip install obsidian-vault-mcp[semantic]"
        )


def main():
    """Entry point. Run with streamable HTTP transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not VAULT_PATH.is_dir():
        logger.error(f"Vault path does not exist: {VAULT_PATH}")
        sys.exit(1)

    if not VAULT_MCP_TOKEN:
        logger.warning("VAULT_MCP_TOKEN is not set -- auth will reject all requests")

    # Build the Starlette app with auth middleware and OAuth endpoints
    try:
        from .auth import BearerAuthMiddleware
        from .oauth import oauth_routes

        app = mcp.streamable_http_app()

        # Mount OAuth routes (these are excluded from bearer auth via the middleware)
        for route in oauth_routes:
            app.routes.insert(0, route)

        app.add_middleware(BearerAuthMiddleware)
        logger.info(f"Starting server on port {VAULT_MCP_PORT} with bearer auth + OAuth")

        import uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=VAULT_MCP_PORT,
            log_level="info",
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
    except Exception as e:
        logger.warning(f"Could not build app ({e}), falling back to mcp.run()")
        logger.warning("Auth will NOT be enforced in this mode")
        mcp.run(transport="streamable-http", port=VAULT_MCP_PORT)


if __name__ == "__main__":
    main()
