# Semantic Search for obsidian-web-mcp

**Date:** 2026-03-26
**Status:** Approved

## Overview

Add hybrid semantic + keyword search to obsidian-web-mcp by embedding a retrieval engine directly in the server process. Two new MCP tools: `vault_semantic_search` (primary search) and `vault_reindex` (safety valve for index rebuilds). The engine combines vector similarity via ChromaDB with BM25 keyword matching using Reciprocal Rank Fusion.

### Scope

**In scope:**
- Hybrid vector + BM25 search with RRF fusion
- Automatic indexing via existing watchdog + manual reindex tool
- Lazy model loading, feature-gated behind env var
- Optional dependency group (`[semantic]`)
- Chunk-based and full-note return modes

**Out of scope (future enhancements):**
- CrossEncoder reranking
- Wikilink graph expansion
- `vault_get_context` (token-budgeted context assembly)
- `vault_find_connections` (graph traversal)

### Design Decisions

- **Embedded engine, not sidecar** — single process, shares the existing watchdog, no IPC. Lazy-loads models on first use for zero overhead when disabled.
- **No LangChain** — use ChromaDB, sentence-transformers, and rank-bm25 directly. Smaller dependency tree, more control over fusion logic.
- **ChromaDB over pgvector** — embedded in-process, no database dependency for the MCP server.
- **CPU-only embeddings** — `paraphrase-multilingual-mpnet-base-v2` via sentence-transformers. No GPU passthrough needed.
- **No reranker** — skip CrossEncoder to save ~2.3 GB model footprint. Hybrid fusion provides sufficient quality.
- **Cache outside vault by default** — `~/.cache/obsidian-web-mcp/` avoids Obsidian Sync interference. Configurable via env var.
- **Optional dependency group** — base install stays light; `pip install obsidian-vault-mcp[semantic]` pulls in the heavy deps.

---

## Module Structure

```
src/obsidian_vault_mcp/
├── retrieval/
│   ├── __init__.py        # RetrievalEngine public interface
│   ├── embeddings.py      # SentenceTransformer init, encode helper
│   ├── chunker.py         # Markdown -> chunks with metadata
│   ├── indexer.py         # ChromaDB + BM25 population, incremental updates
│   ├── bm25.py            # BM25Okapi index wrapper, serialize/deserialize
│   ├── search.py          # Hybrid search + RRF fusion logic
│   └── models.py          # Pydantic schemas (SearchResult, ReindexResult, etc.)
├── tools/
│   ├── semantic_search.py # vault_semantic_search tool
│   ├── admin.py           # vault_reindex tool
│   ├── ...existing...
```

**Boundaries:**
- `chunker.py` knows about Markdown structure but nothing about embeddings or ChromaDB
- `indexer.py` orchestrates chunking -> embedding -> storage, but doesn't know about search
- `search.py` queries both stores and fuses results, but doesn't know about indexing
- Tool modules are thin wrappers that validate input and call the engine

---

## Chunking Strategy

### Splitting Logic

1. Parse frontmatter via `python-frontmatter` (already a project dependency — no optional import guard needed)
2. Split on Markdown headers (H1/H2/H3) to create section-based chunks
3. If a section exceeds ~800 tokens, split further on paragraph breaks (double newline)
4. Keep code blocks intact — never split mid-fence
5. Target 400-800 tokens per chunk, ~80 token overlap between consecutive chunks within the same section
6. Token counting uses a rough chars/4 heuristic (no tokenizer dependency needed — chunk boundaries are approximate by nature)

### Chunk Metadata

```python
class ChunkMetadata(BaseModel):
    path: str              # Relative vault path
    section: str | None    # Nearest heading above the chunk
    chunk_index: int       # Position within the note
    total_chunks: int      # How many chunks this note produced
    tags: list[str]        # From frontmatter
    content_hash: str      # Hash of source file (for change detection)
```

### Context Prefix

Prepended to each chunk before embedding (not stored, used for embedding quality only):

```
"Title: {note_title} | Tags: {tags} | Section: {heading}\n\n{chunk_text}"
```

---

## Indexing & Storage

### Storage Location

- Default: `~/.cache/obsidian-web-mcp/`
- Configurable via `SEMANTIC_CACHE_PATH` env var
- Contents:
  - `chroma/` — ChromaDB persistent store
  - `bm25.pkl` — serialized BM25 index
  - `manifest.json` — `{path: content_hash}` for change detection

### Full Index Build

Triggered on first startup (no existing index) or `vault_reindex(full=True)`:

1. Walk all `.md` files via `vault.py`'s safe path resolution (respects `EXCLUDED_DIRS`)
2. Chunk each file, compute `hashlib.sha256` hash of file content
3. Embed all chunks via sentence-transformers (batched, batch size 64)
4. Store in ChromaDB with metadata, build BM25 index from chunk texts
5. Write `manifest.json`
6. Serialize BM25 to `bm25.pkl`

### Incremental Updates

Triggered by watchdog file change events:

1. Existing `FrontmatterIndex` watchdog detects `.md` file change
2. Debounce (same 5-second window as frontmatter index)
3. Compare file's current hash against `manifest.json`
4. If changed: delete old chunks from ChromaDB (filter by path), re-chunk, re-embed, re-insert
5. If deleted: remove from ChromaDB and BM25
6. Rebuild BM25 index from ChromaDB's stored texts (<100ms for 5K notes)

### Cold Start Optimization

- If `chroma/` and `manifest.json` exist, load them instead of full rebuild
- Diff `manifest.json` against current vault files to catch changes while server was down
- Only re-index the delta

### Watchdog Integration

**Prerequisite:** `FrontmatterIndex` currently has no callback mechanism. Add a change callback list to `FrontmatterIndex`:

1. Add `self._change_callbacks: list[Callable[[list[str]], None]] = []` to `__init__`
2. Add public method `on_change(callback: Callable[[list[str]], None])` that appends to the list
3. At the end of `_flush_pending()`, after processing all paths, call each callback with the list of changed relative paths

This is a minimal, backward-compatible change — existing behavior is unaffected when no callbacks are registered. The retrieval engine's `handle_file_change` callback receives the list of changed paths and performs incremental re-indexing.

### Thread Safety

`FrontmatterIndex._flush_pending()` runs on a `threading.Timer` thread, so `engine.handle_file_change` will be called from a background thread while search queries arrive on the asyncio event loop. ChromaDB handles its own internal thread safety. The BM25 index rebuild must be protected with a `threading.Lock` in the engine to prevent concurrent read/write during rebuilds.

---

## Search Pipeline

### Pipeline Steps

1. **Vector search** — embed the query, query ChromaDB for top `3 * max_results` candidates with cosine similarity. Apply `filter_tags` and `filter_folder` as ChromaDB metadata filters (pre-filtering).
2. **BM25 search** — tokenize the query, score against BM25 index, take top `6 * max_results` candidates (over-fetch to compensate for post-filtering). Apply `filter_tags` and `filter_folder` as post-filters on the results.
3. **Reciprocal Rank Fusion** — merge both result sets:
   ```
   score(doc) = (vector_weight / (k + vector_rank)) + (bm25_weight / (k + bm25_rank))
   ```
   Default weights: 0.6 vector / 0.4 BM25, k=60. Normalize scores to 0.0-1.0 range by dividing by the theoretical max score (vector_weight/k+1 + bm25_weight/k+1) so that `min_score` thresholds are meaningful.
4. **Score threshold** — drop results below `min_score` (default 0.3)
5. **Deduplicate** — if multiple chunks from same note, keep highest-scoring (unless `return_full_notes=True`)
6. **Return** — top `max_results` as `SearchResult` objects

### `return_full_notes` Behavior

- `False` (default): return individual chunks with scores
- `True`: for each unique note in results, read full file via `vault.py`. Score is the max chunk score from that note. `max_results` acts as a note count limit.

### Tool Schema

Follows the existing codebase pattern: synchronous `def`, returns `str` (JSON-serialized), uses Pydantic input model for validation.

```python
# In tools/semantic_search.py
def vault_semantic_search_impl(
    query: str,
    max_results: int = 10,
    min_score: float = 0.3,
    filter_tags: list[str] | None = None,
    filter_folder: str = "",
    return_full_notes: bool = False,
) -> str:
    """Returns JSON string with results list or error."""

# In server.py — registered with @mcp.tool() like existing tools
@mcp.tool(
    name="vault_semantic_search",
    description="Hybrid semantic + keyword search across the vault. Combines vector similarity with BM25 keyword matching. Returns ranked results with relevance scores.",
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
    inp = VaultSemanticSearchInput(...)
    return _vault_semantic_search(inp.query, inp.max_results, ...)
```

### Return Schema

Serialized to JSON string (consistent with all existing tools):

```python
class SearchResult(BaseModel):
    path: str              # Relative vault path
    section: str | None    # Section heading (None if return_full_notes)
    content: str           # Chunk text or full note content
    score: float           # 0.0-1.0 normalized fused relevance score
    tags: list[str]        # Frontmatter tags
```

### `vault_reindex` Tool

```python
# Same sync def + str return pattern
def vault_reindex(full: bool = False) -> str:
    """Returns JSON string with reindex stats."""

class ReindexResult(BaseModel):
    files_indexed: int
    chunks_created: int
    files_skipped: int     # Files that failed to parse/chunk/embed
    duration_seconds: float
```

---

## Server Integration

### Feature Gate

- `SEMANTIC_SEARCH_ENABLED` env var (default `false`)
- When `false`: semantic tools not registered, no imports, zero overhead
- When `true`: tools registered, but model/index loading deferred to first use

### Tool Registration

Conditional registration in `server.py`, following the existing import + wrapper pattern:

```python
# At module level, after existing tool registrations
if config.SEMANTIC_SEARCH_ENABLED:
    from .tools.semantic_search import vault_semantic_search_impl as _vault_semantic_search
    from .tools.admin import vault_reindex_impl as _vault_reindex
    from .models import VaultSemanticSearchInput, VaultReindexInput

    @mcp.tool(
        name="vault_semantic_search",
        description="...",
        annotations={...},
    )
    def vault_semantic_search(...) -> str:
        inp = VaultSemanticSearchInput(...)
        return _vault_semantic_search(...)

    @mcp.tool(
        name="vault_reindex",
        description="...",
        annotations={...},
    )
    def vault_reindex(...) -> str:
        inp = VaultReindexInput(...)
        return _vault_reindex(...)
```

### Lazy Initialization

`RetrievalEngine` instantiated at startup (lightweight config only). On first call to any semantic tool, `_ensure_initialized()`:

1. Loads sentence-transformers model (~5s first time, ~1s from cache)
2. Opens ChromaDB persistent store
3. Loads or builds BM25 index
4. If no existing index, triggers full index build
5. Hooks into watchdog for incremental updates

### Lifespan Changes

Matches the existing `start()`/`stop()` + yield state dict pattern in `server.py`:

```python
@asynccontextmanager
async def lifespan(server):
    logger.info(f"Starting vault MCP server. Vault: {VAULT_PATH}")
    frontmatter_index.start()
    logger.info(f"Frontmatter index built: {frontmatter_index.file_count} files indexed")

    if config.SEMANTIC_SEARCH_ENABLED:
        engine = RetrievalEngine()
        frontmatter_index.on_change(engine.handle_file_change)
    else:
        engine = None

    yield {"frontmatter_index": frontmatter_index, "retrieval_engine": engine}

    if engine is not None:
        engine.shutdown()
    frontmatter_index.stop()
    logger.info("Vault MCP server shut down.")
```

### Configuration

New module-level constants added to `config.py` (matching existing pattern of bare constants, not a settings object):

```python
# Semantic search (optional — requires [semantic] extras)
SEMANTIC_SEARCH_ENABLED = os.environ.get("SEMANTIC_SEARCH_ENABLED", "false").lower() == "true"
SEMANTIC_CACHE_PATH = Path(os.environ.get("SEMANTIC_CACHE_PATH", os.path.expanduser("~/.cache/obsidian-web-mcp")))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "0.6"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))        # Approximate tokens (chars/4)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))    # Approximate tokens (chars/4)
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.3"))
```

---

## Error Handling & Graceful Degradation

### Model Loading Failures

If sentence-transformers fails to load (OOM, missing model, etc.), log the error and mark the engine as `unavailable`. Subsequent calls return: `{"error": "Semantic search unavailable: <reason>. Use vault_search for keyword search."}`. Server stays up, existing tools unaffected.

### Missing Dependencies

If `SEMANTIC_SEARCH_ENABLED=true` but the `[semantic]` extras are not installed, catch `ImportError` at registration time, log a clear error ("semantic search enabled but dependencies not installed — run pip install obsidian-vault-mcp[semantic]"), and skip tool registration. Server starts normally with only the base tools.

### Index Corruption

If ChromaDB or BM25 pickle fails to load, delete the cache and trigger a full rebuild. If rebuild also fails, mark engine as `unavailable`.

### Partial Indexing Failures

Skip individual files that fail to parse/chunk/embed, log warnings. Track skip count and surface in `vault_reindex` results via `files_skipped` field.

### Search-Time Fallbacks

- ChromaDB query fails -> fall back to BM25-only (log warning, include `"fallback": "bm25_only"` in response)
- BM25 query fails -> fall back to vector-only
- Both fail -> return error

### Resource Constraints

- Embedding batched (batch size 64) to limit peak memory
- ChromaDB uses persistent storage, not in-memory
- BM25 index is in-memory but lightweight (~50 bytes/chunk, ~500 KB for 10K chunks)

---

## Testing Strategy

### Unit Tests (`tests/test_retrieval.py`)

- **Chunker**: markdown splitting respects headers, code fences intact, overlap correct, metadata extraction
- **BM25**: index build, serialize/deserialize, query returns ranked results
- **Search fusion**: RRF merging with known scores, deduplication, score threshold filtering, normalization produces 0-1 range
- **Models**: Pydantic schema validation

### Integration Tests (`tests/test_semantic_search.py`)

- Temp vault with 5-10 sample notes (varying content, tags)
- Full index build, semantic search, verify relevance
- `return_full_notes` returns complete note content
- `filter_tags` and `filter_folder` narrow results
- Incremental re-index after file modification
- `vault_reindex(full=True)` rebuilds from scratch

### Edge Cases

- Empty vault — index builds, search returns empty
- Binary / non-UTF8 files — skipped gracefully
- Very large note (>100 KB) — chunks correctly
- Search before index ready — returns error, not crash

### Test Performance

- Mock embedding function in unit tests (deterministic vectors)
- Real model only in integration tests marked `@pytest.mark.slow`
- Extend existing `conftest.py` vault fixture

---

## Dependencies

### New (optional dependency group)

```toml
[project.optional-dependencies]
semantic = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
]
```

Note: `xxhash` dropped in favor of `hashlib.sha256` (stdlib) — files are typically <100 KB, so the performance difference is negligible and it avoids an extra compiled dependency.

Install: `pip install obsidian-vault-mcp[semantic]`

If `SEMANTIC_SEARCH_ENABLED=true` but packages not installed, engine logs error and skips tool registration.

### Disk Footprint

- PyPI packages: ~300 MB (mostly PyTorch CPU)
- Embedding model (first run download): ~420 MB in `~/.cache/huggingface/`
- Total: ~720 MB
