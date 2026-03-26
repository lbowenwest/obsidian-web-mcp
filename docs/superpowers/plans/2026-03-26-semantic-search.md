# Semantic Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid semantic + BM25 search to obsidian-web-mcp as two new MCP tools (`vault_semantic_search`, `vault_reindex`), feature-gated behind `SEMANTIC_SEARCH_ENABLED`.

**Architecture:** Embedded retrieval engine in the server process. ChromaDB for vector storage, sentence-transformers for embeddings, rank-bm25 for keyword search, Reciprocal Rank Fusion to merge results. Lazy-loads models on first use. Hooks into existing watchdog for incremental index updates.

**Tech Stack:** Python 3.12+, ChromaDB, sentence-transformers (`paraphrase-multilingual-mpnet-base-v2`), rank-bm25, Pydantic, hashlib (stdlib)

**Spec:** `docs/superpowers/specs/2026-03-26-semantic-search-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/obsidian_vault_mcp/retrieval/__init__.py` | `RetrievalEngine` class — public interface, lazy init, shutdown |
| `src/obsidian_vault_mcp/retrieval/models.py` | Pydantic schemas: `Chunk`, `ChunkMetadata`, `SearchResult`, `ReindexResult` |
| `src/obsidian_vault_mcp/retrieval/chunker.py` | Markdown splitting into chunks with metadata |
| `src/obsidian_vault_mcp/retrieval/embeddings.py` | SentenceTransformer wrapper — init, encode, context prefix |
| `src/obsidian_vault_mcp/retrieval/bm25.py` | BM25Okapi wrapper — build, query, serialize/deserialize |
| `src/obsidian_vault_mcp/retrieval/indexer.py` | Full/incremental indexing: chunking + embedding + ChromaDB + BM25 + manifest |
| `src/obsidian_vault_mcp/retrieval/search.py` | Hybrid search: vector + BM25 + RRF fusion + filtering |
| `src/obsidian_vault_mcp/tools/semantic_search.py` | `vault_semantic_search_impl` tool function |
| `src/obsidian_vault_mcp/tools/admin.py` | `vault_reindex_impl` tool function |
| `tests/test_chunker.py` | Unit tests for chunker |
| `tests/test_bm25.py` | Unit tests for BM25 wrapper |
| `tests/test_search_fusion.py` | Unit tests for RRF fusion logic |
| `tests/test_semantic_search.py` | Integration tests for full search pipeline |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `[semantic]` optional dependency group |
| `src/obsidian_vault_mcp/config.py` | Add semantic search config constants |
| `src/obsidian_vault_mcp/frontmatter_index.py` | Add `on_change` callback mechanism |
| `src/obsidian_vault_mcp/models.py` | Add `VaultSemanticSearchInput`, `VaultReindexInput` |
| `src/obsidian_vault_mcp/server.py` | Conditional tool registration, lifespan changes |

---

### Task 1: Dependencies and Configuration

**Files:**
- Modify: `pyproject.toml:14-18`
- Modify: `src/obsidian_vault_mcp/config.py:1-30`
- Test: manual — `uv run python -c "from obsidian_vault_mcp import config; print(config.SEMANTIC_SEARCH_ENABLED)"`

- [ ] **Step 1: Add semantic optional dependency group to pyproject.toml**

In `pyproject.toml`, add the `semantic` group after the existing `dev` group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
semantic = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
]
```

- [ ] **Step 2: Add config constants**

In `src/obsidian_vault_mcp/config.py`, add after the `RATE_LIMIT_WRITE` line:

```python
# Semantic search (optional -- requires [semantic] extras)
SEMANTIC_SEARCH_ENABLED = os.environ.get("SEMANTIC_SEARCH_ENABLED", "false").lower() == "true"
SEMANTIC_CACHE_PATH = Path(os.environ.get("SEMANTIC_CACHE_PATH", os.path.expanduser("~/.cache/obsidian-web-mcp")))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", "0.6"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.3"))
```

- [ ] **Step 3: Add pytest slow marker config**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 4: Install semantic extras and verify**

Run: `uv pip install -e ".[semantic]"`
Expected: packages install successfully (chromadb, sentence-transformers, rank-bm25)

Run: `uv run python -c "from obsidian_vault_mcp import config; print(config.SEMANTIC_SEARCH_ENABLED, config.SEMANTIC_CACHE_PATH)"`
Expected: `False /Users/<user>/.cache/obsidian-web-mcp`

- [ ] **Step 5: Run existing tests to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: all existing tests pass

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/obsidian_vault_mcp/config.py
git commit -m "feat: add semantic search config, optional deps, and pytest markers"
```

---

### Task 2: Retrieval Models

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/__init__.py`
- Create: `src/obsidian_vault_mcp/retrieval/models.py`
- Test: `tests/test_retrieval_models.py` (inline validation)

- [ ] **Step 1: Create retrieval package**

Create `src/obsidian_vault_mcp/retrieval/__init__.py` as an empty file (will be populated in Task 9).

- [ ] **Step 2: Write model tests**

Create `tests/test_retrieval_models.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'obsidian_vault_mcp.retrieval.models'`

- [ ] **Step 4: Write the models**

Create `src/obsidian_vault_mcp/retrieval/models.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval_models.py -v`
Expected: all 5 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/ tests/test_retrieval_models.py
git commit -m "feat: add retrieval Pydantic models"
```

---

### Task 3: Markdown Chunker

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write chunker tests**

Create `tests/test_chunker.py`:

```python
"""Tests for Markdown chunker."""

import pytest
from obsidian_vault_mcp.retrieval.chunker import chunk_markdown


def test_simple_note_single_chunk():
    """Short note with frontmatter becomes one chunk."""
    content = "---\ntags:\n  - python\n---\n\nA short note about Python."
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    assert len(chunks) == 1
    assert chunks[0].text == "A short note about Python."
    assert chunks[0].metadata.path == "test.md"
    assert chunks[0].metadata.tags == ["python"]
    assert chunks[0].metadata.section is None
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.total_chunks == 1


def test_splits_on_headers():
    """Note with multiple sections splits on headers."""
    content = (
        "---\ntags: []\n---\n\n"
        "# Title\n\nIntro paragraph.\n\n"
        "## Section A\n\nContent for section A.\n\n"
        "## Section B\n\nContent for section B.\n"
    )
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    assert len(chunks) == 3
    assert chunks[0].metadata.section == "Title"
    assert chunks[1].metadata.section == "Section A"
    assert chunks[2].metadata.section == "Section B"
    assert "Intro paragraph" in chunks[0].text
    assert "Content for section A" in chunks[1].text
    assert "Content for section B" in chunks[2].text


def test_preserves_code_blocks():
    """Code fences are never split mid-block."""
    code_block = "```python\ndef hello():\n    print('world')\n```"
    content = f"---\ntags: []\n---\n\n## Code\n\n{code_block}\n"
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    # The code block must appear intact in exactly one chunk
    found = [c for c in chunks if "def hello():" in c.text]
    assert len(found) == 1
    assert "```python" in found[0].text
    assert "```" in found[0].text.split("```python")[1]


def test_large_section_splits_on_paragraphs():
    """Section exceeding chunk_size splits on paragraph breaks."""
    # Each paragraph ~100 chars = ~25 tokens. 10 paragraphs ~= 250 tokens.
    # With chunk_size=50 (tokens), should split into multiple chunks.
    paragraphs = "\n\n".join([f"Paragraph {i} with some filler content here." for i in range(10)])
    content = f"---\ntags: []\n---\n\n## Big Section\n\n{paragraphs}\n"
    chunks = chunk_markdown(content, path="test.md", chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
    # All chunks should reference the same section
    for chunk in chunks:
        assert chunk.metadata.section == "Big Section"


def test_no_frontmatter():
    """Note without frontmatter still chunks correctly."""
    content = "Just plain text, no frontmatter.\n\n## Section\n\nMore text."
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    assert len(chunks) >= 1
    assert chunks[0].metadata.tags == []


def test_empty_content():
    """Empty or whitespace-only content returns no chunks."""
    chunks = chunk_markdown("", path="test.md", chunk_size=600, chunk_overlap=80)
    assert chunks == []
    chunks = chunk_markdown("---\ntags: []\n---\n", path="test.md", chunk_size=600, chunk_overlap=80)
    assert chunks == []


def test_content_hash_set():
    """All chunks from same file share the same content hash."""
    content = "---\ntags: []\n---\n\n## A\n\nText A.\n\n## B\n\nText B.\n"
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    assert len(chunks) == 2
    assert chunks[0].metadata.content_hash == chunks[1].metadata.content_hash
    assert len(chunks[0].metadata.content_hash) > 0


def test_total_chunks_set_correctly():
    """All chunks report correct total_chunks."""
    content = "---\ntags: []\n---\n\n## A\n\nA.\n\n## B\n\nB.\n\n## C\n\nC.\n"
    chunks = chunk_markdown(content, path="test.md", chunk_size=600, chunk_overlap=80)
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.metadata.total_chunks == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'obsidian_vault_mcp.retrieval.chunker'`

- [ ] **Step 3: Implement the chunker**

Create `src/obsidian_vault_mcp/retrieval/chunker.py`:

```python
"""Markdown chunker that splits vault notes into embeddable chunks."""

import hashlib
import re

import frontmatter

from .models import Chunk, ChunkMetadata


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def _split_on_paragraphs(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text on paragraph breaks to fit within chunk_size tokens."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)
        if current and (current_tokens + para_tokens) > chunk_size:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraphs that fit within overlap budget
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for p in reversed(current):
                pt = _estimate_tokens(p)
                if overlap_tokens + pt > chunk_overlap:
                    break
                overlap_parts.insert(0, p)
                overlap_tokens += pt
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        text_joined = "\n\n".join(current)
        if text_joined.strip():
            chunks.append(text_joined)

    return chunks


def _extract_sections(body: str) -> list[tuple[str | None, str]]:
    """Split markdown body into (heading, content) pairs.

    Returns a list of (section_heading, section_text) tuples.
    Text before any heading has heading=None.
    """
    # Match H1, H2, H3 at start of line
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str | None, str]] = []
    last_end = 0
    last_heading: str | None = None

    for match in header_pattern.finditer(body):
        # Capture text before this header
        text_before = body[last_end:match.start()].strip()
        if text_before or last_end == 0:
            if text_before:
                sections.append((last_heading, text_before))

        last_heading = match.group(2).strip()
        last_end = match.end()

    # Capture remaining text after last header
    remaining = body[last_end:].strip()
    if remaining:
        sections.append((last_heading, remaining))

    return sections


def chunk_markdown(
    content: str,
    path: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Split a markdown note into chunks with metadata.

    Args:
        content: Raw file content (may include frontmatter).
        path: Relative vault path for metadata.
        chunk_size: Target tokens per chunk (approximate, chars/4).
        chunk_overlap: Overlap tokens between consecutive chunks in same section.

    Returns:
        List of Chunk objects. Empty list if no meaningful content.
    """
    if not content or not content.strip():
        return []

    # Parse frontmatter
    try:
        post = frontmatter.loads(content)
        tags = post.metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        body = post.content
    except Exception:
        tags = []
        body = content

    if not body or not body.strip():
        return []

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # Split into sections by headers
    sections = _extract_sections(body)
    if not sections:
        return []

    # Build raw chunks from sections
    raw_chunks: list[tuple[str | None, str]] = []
    for heading, text in sections:
        if _estimate_tokens(text) <= chunk_size:
            raw_chunks.append((heading, text))
        else:
            # Split large sections on paragraphs
            sub_chunks = _split_on_paragraphs(text, chunk_size, chunk_overlap)
            for sub in sub_chunks:
                raw_chunks.append((heading, sub))

    # Build Chunk objects
    total = len(raw_chunks)
    chunks: list[Chunk] = []
    for i, (heading, text) in enumerate(raw_chunks):
        chunks.append(Chunk(
            text=text,
            metadata=ChunkMetadata(
                path=path,
                section=heading,
                chunk_index=i,
                total_chunks=total,
                tags=tags,
                content_hash=content_hash,
            ),
        ))

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: all 8 tests pass

- [ ] **Step 5: Run all tests to check for regressions**

Run: `uv run pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/chunker.py tests/test_chunker.py
git commit -m "feat: add Markdown chunker for semantic search"
```

---

### Task 4: BM25 Wrapper

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write BM25 tests**

Create `tests/test_bm25.py`:

```python
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
    index.build([("a", "hello world"), ("b", "goodbye world")])
    results = index.query("hello", top_k=2)
    assert len(results) >= 1
    assert results[0][1] > 0  # score > 0


def test_empty_index_returns_empty():
    """Querying empty index returns empty list."""
    index = BM25Index()
    index.build([])
    results = index.query("anything", top_k=5)
    assert results == []


def test_serialize_deserialize(tmp_path):
    """Index can be saved and loaded from disk."""
    index = BM25Index()
    index.build([("a", "python testing"), ("b", "java testing")])
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
    index.build([("a", "python"), ("b", "java")])
    results1 = index.query("python", top_k=1)
    assert results1[0][0] == "a"

    index.build([("c", "rust"), ("d", "python is great")])
    results2 = index.query("python", top_k=1)
    assert results2[0][0] == "d"


def test_query_no_match():
    """Query with no matching terms returns empty or low scores."""
    index = BM25Index()
    index.build([("a", "python programming"), ("b", "java development")])
    results = index.query("zzznonexistent", top_k=5)
    # BM25 may return results with 0 scores; filter them
    non_zero = [(cid, s) for cid, s in results if s > 0]
    assert len(non_zero) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement BM25 wrapper**

Create `src/obsidian_vault_mcp/retrieval/bm25.py`:

```python
"""BM25 index wrapper with serialization support."""

import logging
import pickle
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Thread-safe BM25 index over text chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._lock = threading.Lock()

    def build(self, corpus: list[tuple[str, str]]) -> None:
        """Build the BM25 index from (chunk_id, text) pairs."""
        if not corpus:
            with self._lock:
                self._bm25 = None
                self._chunk_ids = []
                self._corpus_texts = []
            return

        chunk_ids = [cid for cid, _ in corpus]
        texts = [text for _, text in corpus]
        tokenized = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized)

        with self._lock:
            self._bm25 = bm25
            self._chunk_ids = chunk_ids
            self._corpus_texts = texts

    def query(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Query the index. Returns list of (chunk_id, score) sorted by score desc."""
        with self._lock:
            if self._bm25 is None or not self._chunk_ids:
                return []
            bm25 = self._bm25
            chunk_ids = self._chunk_ids

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        scored = list(zip(chunk_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filter out zero scores and limit
        return [(cid, s) for cid, s in scored[:top_k] if s > 0]

    def save(self, path: Path) -> None:
        """Serialize index to disk."""
        with self._lock:
            data = {
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "corpus_texts": self._corpus_texts,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Deserialize index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        index = cls()
        index._bm25 = data["bm25"]
        index._chunk_ids = data["chunk_ids"]
        index._corpus_texts = data["corpus_texts"]
        return index

    @property
    def corpus_texts(self) -> list[str]:
        """Access corpus texts (for rebuilding from stored data)."""
        with self._lock:
            return list(self._corpus_texts)

    @property
    def chunk_ids(self) -> list[str]:
        with self._lock:
            return list(self._chunk_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bm25.py -v`
Expected: all 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/bm25.py tests/test_bm25.py
git commit -m "feat: add BM25 index wrapper with serialization"
```

---

### Task 5: Embeddings Wrapper

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/embeddings.py`

No dedicated test file — this is a thin wrapper around sentence-transformers. Tested via integration tests in Task 10.

- [ ] **Step 1: Implement embeddings wrapper**

Create `src/obsidian_vault_mcp/retrieval/embeddings.py`:

```python
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
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from obsidian_vault_mcp.retrieval.embeddings import SentenceTransformerEmbedder, build_context_prefix; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/embeddings.py
git commit -m "feat: add SentenceTransformer embeddings wrapper"
```

---

### Task 6: Search Fusion Logic

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/search.py`
- Test: `tests/test_search_fusion.py`

- [ ] **Step 1: Write RRF fusion tests**

Create `tests/test_search_fusion.py`:

```python
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
    # chunk_b and chunk_a appear in both, should have higher scores
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
    # a.md should have score 0.9
    a_result = [r for r in deduped if r[1] == "notes/a.md"][0]
    assert a_result[2] == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement search fusion**

Create `src/obsidian_vault_mcp/retrieval/search.py`:

```python
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

    Args:
        vector_results: List of (chunk_id, score) from vector search.
        bm25_results: List of (chunk_id, score) from BM25 search.
        vector_weight: Weight for vector source in fusion.
        bm25_weight: Weight for BM25 source in fusion.
        k: RRF smoothing constant.

    Returns:
        List of (chunk_id, normalized_score) sorted by score descending.
        Scores are normalized to 0.0-1.0 range.
    """
    scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(vector_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(bm25_results):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight / (k + rank + 1)

    if not scores:
        return []

    # Normalize by theoretical max (rank 1 in both sources)
    max_possible = vector_weight / (k + 1) + bm25_weight / (k + 1)
    normalized = [(cid, score / max_possible) for cid, score in scores.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)

    return normalized


def deduplicate_by_path(
    results: list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    """Keep only the highest-scoring chunk per note path.

    Args:
        results: List of (chunk_id, path, score).

    Returns:
        Deduplicated list, one entry per unique path, highest score wins.
    """
    best: dict[str, tuple[str, str, float]] = {}
    for chunk_id, path, score in results:
        if path not in best or score > best[path][2]:
            best[path] = (chunk_id, path, score)

    deduped = list(best.values())
    deduped.sort(key=lambda x: x[2], reverse=True)
    return deduped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search_fusion.py -v`
Expected: all 7 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/search.py tests/test_search_fusion.py
git commit -m "feat: add RRF fusion and deduplication for hybrid search"
```

---

### Task 7: FrontmatterIndex Callback Hook

**Files:**
- Modify: `src/obsidian_vault_mcp/frontmatter_index.py:17-26,126-145`
- Test: `tests/test_frontmatter.py` (extend)

- [ ] **Step 1: Write callback test**

Add to the end of `tests/test_frontmatter.py`:

```python
def test_on_change_callback_fires(vault_dir):
    """Registered callback is called with changed paths after flush."""
    from obsidian_vault_mcp.frontmatter_index import FrontmatterIndex

    index = FrontmatterIndex()
    index.start()

    changed_paths: list[list[str]] = []
    index.on_change(lambda paths: changed_paths.append(paths))

    # Create a new file to trigger the watcher
    new_file = vault_dir / "callback-test.md"
    new_file.write_text("---\nstatus: test\n---\n\nCallback test.\n")

    # Wait for debounce (5s) + processing time
    import time
    time.sleep(7)

    index.stop()

    assert len(changed_paths) >= 1
    # The callback should have received the relative path
    all_paths = [p for batch in changed_paths for p in batch]
    assert "callback-test.md" in all_paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_frontmatter.py::test_on_change_callback_fires -v`
Expected: FAIL — `AttributeError: 'FrontmatterIndex' has no attribute 'on_change'`

- [ ] **Step 3: Add callback mechanism to FrontmatterIndex**

In `src/obsidian_vault_mcp/frontmatter_index.py`, modify `__init__` to add:

```python
def __init__(self) -> None:
    self._index: dict[str, dict] = {}
    self._lock = threading.Lock()
    self._observer: Observer | None = None
    self._debounce_timer: threading.Timer | None = None
    self._pending_paths: set[str] = set()
    self._change_callbacks: list = []
```

Add the `on_change` method after `file_count` property:

```python
def on_change(self, callback) -> None:
    """Register a callback to be called with changed relative paths after flush.

    Callback signature: callback(changed_paths: list[str]) -> None
    Called from the debounce timer thread.
    """
    self._change_callbacks.append(callback)
```

At the end of `_flush_pending`, after processing all paths, add callback invocation:

```python
def _flush_pending(self) -> None:
    """Process all pending file changes."""
    with self._lock:
        paths = self._pending_paths.copy()
        self._pending_paths.clear()
        self._debounce_timer = None

    changed_rel_paths: list[str] = []
    for abs_path_str in paths:
        abs_path = Path(abs_path_str)
        rel = str(abs_path.relative_to(config.VAULT_PATH))
        changed_rel_paths.append(rel)
        if abs_path.exists():
            fm = self._parse_frontmatter(abs_path)
            with self._lock:
                if fm is not None:
                    self._index[rel] = fm
                else:
                    self._index.pop(rel, None)
        else:
            with self._lock:
                self._index.pop(rel, None)

    # Notify registered callbacks
    for callback in self._change_callbacks:
        try:
            callback(changed_rel_paths)
        except Exception:
            logger.warning("Change callback failed", exc_info=True)
```

- [ ] **Step 4: Run the callback test**

Run: `uv run pytest tests/test_frontmatter.py::test_on_change_callback_fires -v`
Expected: PASS

- [ ] **Step 5: Run all frontmatter tests to check regressions**

Run: `uv run pytest tests/test_frontmatter.py -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/obsidian_vault_mcp/frontmatter_index.py tests/test_frontmatter.py
git commit -m "feat: add on_change callback to FrontmatterIndex"
```

---

### Task 8: Indexer (ChromaDB + BM25 + Manifest)

**Files:**
- Create: `src/obsidian_vault_mcp/retrieval/indexer.py`

No dedicated unit tests — the indexer orchestrates chunker + embeddings + ChromaDB + BM25. Tested via integration tests in Task 10. The individual components (chunker, BM25, fusion) are already unit-tested.

- [ ] **Step 1: Implement the indexer**

Create `src/obsidian_vault_mcp/retrieval/indexer.py`:

```python
"""Vault indexer: builds and maintains ChromaDB + BM25 indices."""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import chromadb

from .. import config
from ..vault import resolve_vault_path
from .bm25 import BM25Index
from .chunker import chunk_markdown
from .embeddings import Embedder, build_context_prefix
from .models import ReindexResult

logger = logging.getLogger(__name__)


class VaultIndexer:
    """Manages full and incremental indexing of vault markdown files."""

    def __init__(self, embedder: Embedder, cache_path: Path | None = None) -> None:
        self._embedder = embedder
        self._cache_path = cache_path or config.SEMANTIC_CACHE_PATH
        self._cache_path.mkdir(parents=True, exist_ok=True)

        self._manifest_path = self._cache_path / "manifest.json"
        self._bm25_path = self._cache_path / "bm25.pkl"
        self._chroma_path = self._cache_path / "chroma"

        self._manifest: dict[str, str] = {}
        self._bm25 = BM25Index()
        self._chroma_client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def initialize(self) -> None:
        """Load existing indices or build from scratch."""
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=str(self._chroma_path)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="vault_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            logger.warning("ChromaDB failed to load, rebuilding", exc_info=True)
            shutil.rmtree(self._chroma_path, ignore_errors=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(self._chroma_path)
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="vault_chunks",
                metadata={"hnsw:space": "cosine"},
            )

        # Load manifest
        if self._manifest_path.exists():
            try:
                self._manifest = json.loads(self._manifest_path.read_text())
            except Exception:
                logger.warning("Manifest corrupted, will rebuild")
                self._manifest = {}

        # Load BM25
        if self._bm25_path.exists():
            try:
                self._bm25 = BM25Index.load(self._bm25_path)
            except Exception:
                logger.warning("BM25 index corrupted, will rebuild")
                self._bm25 = BM25Index()

        # Check if we need a full build or just a delta
        if self._collection.count() == 0:
            logger.info("No existing index found, building from scratch")
            self.full_index()
        else:
            self._sync_delta()

    def full_index(self) -> ReindexResult:
        """Build the full index from scratch."""
        t0 = time.monotonic()

        # Clear existing data
        if self._collection is not None:
            self._chroma_client.delete_collection("vault_chunks")
            self._collection = self._chroma_client.get_or_create_collection(
                name="vault_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        self._manifest = {}

        files_indexed = 0
        files_skipped = 0
        all_chunks_for_bm25: list[tuple[str, str]] = []
        batch_ids: list[str] = []
        batch_documents: list[str] = []
        batch_metadatas: list[dict] = []
        batch_texts_for_embed: list[str] = []

        for md_path in config.VAULT_PATH.rglob("*.md"):
            if config.EXCLUDED_DIRS & set(md_path.relative_to(config.VAULT_PATH).parts):
                continue

            rel_path = str(md_path.relative_to(config.VAULT_PATH))

            try:
                content = md_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                files_skipped += 1
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            chunks = chunk_markdown(
                content, rel_path,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            if not chunks:
                files_skipped += 1
                continue

            title = Path(rel_path).stem
            for chunk in chunks:
                chunk_id = f"{rel_path}::{chunk.metadata.chunk_index}"
                prefix = build_context_prefix(
                    title, chunk.metadata.tags, chunk.metadata.section
                )
                text_for_embedding = prefix + chunk.text

                batch_ids.append(chunk_id)
                batch_documents.append(chunk.text)
                batch_texts_for_embed.append(text_for_embedding)
                batch_metadatas.append({
                    "path": chunk.metadata.path,
                    "section": chunk.metadata.section or "",
                    "chunk_index": chunk.metadata.chunk_index,
                    "total_chunks": chunk.metadata.total_chunks,
                    "tags": json.dumps(chunk.metadata.tags),
                    "content_hash": chunk.metadata.content_hash,
                })
                all_chunks_for_bm25.append((chunk_id, chunk.text))

                # Batch embed when we hit batch size
                if len(batch_ids) >= 64:
                    self._flush_batch(
                        batch_ids, batch_documents, batch_metadatas,
                        batch_texts_for_embed,
                    )
                    batch_ids, batch_documents, batch_metadatas, batch_texts_for_embed = [], [], [], []

            self._manifest[rel_path] = content_hash
            files_indexed += 1

        # Flush remaining batch
        if batch_ids:
            self._flush_batch(batch_ids, batch_documents, batch_metadatas, batch_texts_for_embed)

        # Build BM25
        self._bm25.build(all_chunks_for_bm25)

        # Save to disk
        self._save_manifest()
        self._bm25.save(self._bm25_path)

        duration = time.monotonic() - t0
        total_chunks = self._collection.count() if self._collection else 0
        logger.info(
            "Full index complete: %d files, %d chunks in %.1fs (%d skipped)",
            files_indexed, total_chunks, duration, files_skipped,
        )

        return ReindexResult(
            files_indexed=files_indexed,
            chunks_created=total_chunks,
            files_skipped=files_skipped,
            duration_seconds=round(duration, 2),
        )

    def update_files(self, rel_paths: list[str]) -> None:
        """Incrementally update index for changed/deleted files."""
        if not self._collection:
            return

        bm25_needs_rebuild = False

        for rel_path in rel_paths:
            abs_path = config.VAULT_PATH / rel_path

            # Remove old chunks for this file
            try:
                self._collection.delete(where={"path": rel_path})
                bm25_needs_rebuild = True
            except Exception:
                pass

            if not abs_path.exists():
                # File was deleted
                self._manifest.pop(rel_path, None)
                continue

            try:
                content = abs_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                self._manifest.pop(rel_path, None)
                continue

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

            # Skip if unchanged
            if self._manifest.get(rel_path) == content_hash:
                continue

            chunks = chunk_markdown(
                content, rel_path,
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            if chunks:
                title = Path(rel_path).stem
                ids = []
                documents = []
                metadatas = []
                texts_for_embed = []

                for chunk in chunks:
                    chunk_id = f"{rel_path}::{chunk.metadata.chunk_index}"
                    prefix = build_context_prefix(
                        title, chunk.metadata.tags, chunk.metadata.section
                    )
                    ids.append(chunk_id)
                    documents.append(chunk.text)
                    metadatas.append({
                        "path": chunk.metadata.path,
                        "section": chunk.metadata.section or "",
                        "chunk_index": chunk.metadata.chunk_index,
                        "total_chunks": chunk.metadata.total_chunks,
                        "tags": json.dumps(chunk.metadata.tags),
                        "content_hash": chunk.metadata.content_hash,
                    })
                    texts_for_embed.append(prefix + chunk.text)

                embeddings = self._embedder.encode(texts_for_embed)
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )

            self._manifest[rel_path] = content_hash

        # Rebuild BM25 from ChromaDB stored texts
        if bm25_needs_rebuild:
            self._rebuild_bm25()

        self._save_manifest()
        self._bm25.save(self._bm25_path)

    def vector_search(
        self, query_embedding: list[float], n_results: int,
        filter_tags: list[str] | None = None,
        filter_folder: str = "",
    ) -> list[tuple[str, float, dict]]:
        """Query ChromaDB. Returns list of (chunk_id, similarity, metadata).

        Folder and tag filtering is done post-query since ChromaDB's
        where filters don't support prefix matching cleanly.
        Over-fetches to compensate for post-filtering.
        """
        if not self._collection or self._collection.count() == 0:
            return []

        # Over-fetch if we'll be post-filtering
        fetch_n = n_results
        if filter_folder or filter_tags:
            fetch_n = min(n_results * 3, self._collection.count())

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(fetch_n, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]

                # Post-filter by folder prefix
                if filter_folder and not metadata.get("path", "").startswith(filter_folder):
                    continue

                # Post-filter by tags
                if filter_tags:
                    chunk_tags = json.loads(metadata.get("tags", "[]"))
                    if not set(filter_tags) <= set(chunk_tags):
                        continue

                distance = results["distances"][0][i]
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity: 1 - (distance / 2)
                similarity = 1.0 - (distance / 2.0)
                output.append((chunk_id, similarity, metadata))

                if len(output) >= n_results:
                    break

        return output

    @property
    def bm25(self) -> BM25Index:
        return self._bm25

    def _flush_batch(
        self, ids: list[str], documents: list[str],
        metadatas: list[dict], texts_for_embed: list[str],
    ) -> None:
        """Embed and add a batch of chunks to ChromaDB."""
        embeddings = self._embedder.encode(texts_for_embed)
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 from all documents in ChromaDB."""
        if not self._collection or self._collection.count() == 0:
            self._bm25.build([])
            return

        all_data = self._collection.get(include=["documents"])
        corpus = list(zip(all_data["ids"], all_data["documents"]))
        self._bm25.build(corpus)

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _sync_delta(self) -> None:
        """Sync index with vault state on cold start."""
        current_files: dict[str, str] = {}
        for md_path in config.VAULT_PATH.rglob("*.md"):
            if config.EXCLUDED_DIRS & set(md_path.relative_to(config.VAULT_PATH).parts):
                continue
            rel_path = str(md_path.relative_to(config.VAULT_PATH))
            try:
                content = md_path.read_text(encoding="utf-8")
                current_files[rel_path] = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            except (UnicodeDecodeError, PermissionError):
                continue

        changed = []
        for path, hash_val in current_files.items():
            if self._manifest.get(path) != hash_val:
                changed.append(path)

        # Files removed since last run
        for path in set(self._manifest.keys()) - set(current_files.keys()):
            changed.append(path)

        if changed:
            logger.info("Cold start delta: %d files changed", len(changed))
            self.update_files(changed)
        else:
            logger.info("Index up to date, no changes detected")
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from obsidian_vault_mcp.retrieval.indexer import VaultIndexer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/indexer.py
git commit -m "feat: add vault indexer with ChromaDB and BM25"
```

---

### Task 9: RetrievalEngine (Public Interface)

**Files:**
- Modify: `src/obsidian_vault_mcp/retrieval/__init__.py`

- [ ] **Step 1: Implement RetrievalEngine**

Write `src/obsidian_vault_mcp/retrieval/__init__.py`:

```python
"""Retrieval engine for semantic search over Obsidian vaults."""

import json
import logging
import threading

from .. import config
from ..vault import read_file
from .models import SearchResult, ReindexResult
from .search import reciprocal_rank_fusion, deduplicate_by_path

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Main interface for semantic search. Lazy-initializes on first use."""

    def __init__(self) -> None:
        self._initialized = False
        self._available = True
        self._error_message: str | None = None
        self._indexer = None
        self._embedder = None
        self._init_lock = threading.Lock()

    def _ensure_initialized(self) -> None:
        """Lazy-load models and index on first use."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            try:
                from .embeddings import SentenceTransformerEmbedder
                from .indexer import VaultIndexer

                self._embedder = SentenceTransformerEmbedder()
                self._indexer = VaultIndexer(self._embedder)
                self._indexer.initialize()
                self._initialized = True
                logger.info("Retrieval engine initialized")
            except Exception as e:
                self._available = False
                self._error_message = str(e)
                logger.error("Failed to initialize retrieval engine: %s", e)

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.3,
        filter_tags: list[str] | None = None,
        filter_folder: str = "",
        return_full_notes: bool = False,
    ) -> str:
        """Run hybrid semantic + BM25 search. Returns JSON string."""
        self._ensure_initialized()

        if not self._available:
            return json.dumps({
                "error": f"Semantic search unavailable: {self._error_message}. Use vault_search for keyword search."
            })

        try:
            # 1. Vector search
            query_embedding = self._embedder.encode([query])[0]
            vector_results = self._indexer.vector_search(
                query_embedding, n_results=3 * max_results,
                filter_tags=filter_tags, filter_folder=filter_folder,
            )
            vector_ranked = [(cid, sim) for cid, sim, _ in vector_results]

            # 2. BM25 search
            bm25_results = self._indexer.bm25.query(query, top_k=6 * max_results)

            # Post-filter BM25 by folder if specified
            if filter_folder:
                bm25_results = [
                    (cid, s) for cid, s in bm25_results
                    if cid.split("::")[0].startswith(filter_folder)
                ]

            # 3. RRF fusion
            fused = reciprocal_rank_fusion(
                vector_ranked, bm25_results,
                vector_weight=config.VECTOR_WEIGHT,
                bm25_weight=config.BM25_WEIGHT,
                k=60,
            )

            # 4. Score threshold
            fused = [(cid, s) for cid, s in fused if s >= min_score]

            # 5. Look up metadata and build results
            results_with_meta: list[tuple[str, str, float, dict]] = []
            for chunk_id, score in fused:
                # Extract path from chunk_id (format: "path::index")
                path = chunk_id.rsplit("::", 1)[0]
                meta = self._get_chunk_metadata(chunk_id)
                if meta:
                    results_with_meta.append((chunk_id, path, score, meta))

            # 6. Deduplicate
            if not return_full_notes:
                deduped = deduplicate_by_path(
                    [(cid, path, score) for cid, path, score, _ in results_with_meta]
                )
                # Re-attach metadata
                meta_lookup = {cid: meta for cid, _, _, meta in results_with_meta}
                results_with_meta = [
                    (cid, path, score, meta_lookup.get(cid, {}))
                    for cid, path, score in deduped
                ]

            # Limit results
            results_with_meta = results_with_meta[:max_results]

            # 7. Build output
            search_results = []
            for chunk_id, path, score, meta in results_with_meta:
                if return_full_notes:
                    try:
                        content, _ = read_file(path)
                        section = None
                    except Exception:
                        continue
                else:
                    content = self._get_chunk_document(chunk_id) or ""
                    section = meta.get("section", None)
                    if section == "":
                        section = None

                tags = json.loads(meta.get("tags", "[]"))

                search_results.append(SearchResult(
                    path=path,
                    section=section,
                    content=content,
                    score=round(score, 4),
                    tags=tags,
                ).model_dump())

            return json.dumps({"results": search_results, "total": len(search_results)})

        except Exception as e:
            logger.error("Search error: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

    def reindex(self, full: bool = False) -> str:
        """Trigger reindex. Returns JSON string with stats."""
        self._ensure_initialized()

        if not self._available:
            return json.dumps({
                "error": f"Semantic search unavailable: {self._error_message}"
            })

        try:
            if full:
                result = self._indexer.full_index()
            else:
                # Incremental: re-sync with current vault state
                self._indexer._sync_delta()
                result = ReindexResult(
                    files_indexed=0,
                    chunks_created=self._indexer._collection.count() if self._indexer._collection else 0,
                    files_skipped=0,
                    duration_seconds=0,
                )
            return json.dumps(result.model_dump())
        except Exception as e:
            logger.error("Reindex error: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})

    def handle_file_change(self, changed_paths: list[str]) -> None:
        """Callback for FrontmatterIndex watchdog. Called from timer thread."""
        if not self._initialized or not self._available:
            return
        try:
            self._indexer.update_files(changed_paths)
        except Exception:
            logger.warning("Incremental index update failed", exc_info=True)

    def shutdown(self) -> None:
        """Clean up resources."""
        logger.info("Retrieval engine shutting down")
        self._initialized = False

    def _get_chunk_metadata(self, chunk_id: str) -> dict | None:
        """Look up chunk metadata from ChromaDB."""
        try:
            result = self._indexer._collection.get(ids=[chunk_id], include=["metadatas"])
            if result["metadatas"]:
                return result["metadatas"][0]
        except Exception:
            pass
        return None

    def _get_chunk_document(self, chunk_id: str) -> str | None:
        """Look up chunk text from ChromaDB."""
        try:
            result = self._indexer._collection.get(ids=[chunk_id], include=["documents"])
            if result["documents"]:
                return result["documents"][0]
        except Exception:
            pass
        return None
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from obsidian_vault_mcp.retrieval import RetrievalEngine; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/obsidian_vault_mcp/retrieval/__init__.py
git commit -m "feat: add RetrievalEngine public interface"
```

---

### Task 10: Tool Functions and Input Models

**Files:**
- Create: `src/obsidian_vault_mcp/tools/semantic_search.py`
- Create: `src/obsidian_vault_mcp/tools/admin.py`
- Modify: `src/obsidian_vault_mcp/models.py`

- [ ] **Step 1: Add input models**

Add to the end of `src/obsidian_vault_mcp/models.py`:

```python
class VaultSemanticSearchInput(BaseModel):
    """Hybrid semantic + keyword search across the vault."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        description="Natural language search query",
        min_length=1,
        max_length=500,
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Maximum number of results to return",
    )
    min_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score threshold (0.0-1.0)",
    )
    filter_tags: list[str] | None = Field(
        default=None,
        description="Only include notes with all of these tags",
    )
    filter_folder: str = Field(
        default="",
        description="Restrict search to notes under this folder prefix",
        max_length=500,
    )
    return_full_notes: bool = Field(
        default=False,
        description="Return full note content instead of matching chunks",
    )


class VaultReindexInput(BaseModel):
    """Trigger reindexing of the semantic search index."""

    model_config = ConfigDict(extra="forbid")

    full: bool = Field(
        default=False,
        description="Full rebuild (true) or incremental sync (false)",
    )
```

- [ ] **Step 2: Create semantic search tool**

Create `src/obsidian_vault_mcp/tools/semantic_search.py`:

```python
"""Semantic search tool for the vault MCP server."""

import json
import logging

logger = logging.getLogger(__name__)

# Engine reference set by server.py during lifespan
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
```

- [ ] **Step 3: Create admin tool**

Create `src/obsidian_vault_mcp/tools/admin.py`:

```python
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
```

- [ ] **Step 4: Verify imports**

Run: `uv run python -c "from obsidian_vault_mcp.models import VaultSemanticSearchInput, VaultReindexInput; print('OK')"`
Expected: `OK`

Run: `uv run python -c "from obsidian_vault_mcp.tools.semantic_search import vault_semantic_search_impl; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add src/obsidian_vault_mcp/models.py src/obsidian_vault_mcp/tools/semantic_search.py src/obsidian_vault_mcp/tools/admin.py
git commit -m "feat: add semantic search and reindex tool functions"
```

---

### Task 11: Server Integration

**Files:**
- Modify: `src/obsidian_vault_mcp/server.py`

- [ ] **Step 1: Add conditional tool registration and lifespan changes**

In `src/obsidian_vault_mcp/server.py`, update the import section to include `config`:

After line 15 (`from .frontmatter_index import FrontmatterIndex`), add:

```python
from . import config
```

Replace the `lifespan` function (lines 24-32) with:

```python
@asynccontextmanager
async def lifespan(server):
    """Start frontmatter index and optional retrieval engine on startup."""
    logger.info(f"Starting vault MCP server. Vault: {VAULT_PATH}")
    frontmatter_index.start()
    logger.info(f"Frontmatter index built: {frontmatter_index.file_count} files indexed")

    engine = None
    if config.SEMANTIC_SEARCH_ENABLED:
        try:
            from .retrieval import RetrievalEngine
            from .tools.semantic_search import set_engine as set_search_engine
            from .tools.admin import set_engine as set_admin_engine

            engine = RetrievalEngine()
            set_search_engine(engine)
            set_admin_engine(engine)
            frontmatter_index.on_change(engine.handle_file_change)
            logger.info("Semantic search enabled")
        except ImportError:
            logger.error(
                "Semantic search enabled but dependencies not installed. "
                "Run: pip install obsidian-vault-mcp[semantic]"
            )
        except Exception as e:
            logger.error("Failed to set up semantic search: %s", e)

    yield {"frontmatter_index": frontmatter_index, "retrieval_engine": engine}

    if engine is not None:
        engine.shutdown()
    frontmatter_index.stop()
    logger.info("Vault MCP server shut down.")
```

After the existing tool registrations (after line 188, the `vault_delete` tool), add the conditional semantic tool registration:

```python
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
            inp = VaultSemanticSearchInput(
                query=query, max_results=max_results, min_score=min_score,
                filter_tags=filter_tags, filter_folder=filter_folder,
                return_full_notes=return_full_notes,
            )
            return _vault_semantic_search(
                inp.query, inp.max_results, inp.min_score,
                inp.filter_tags, inp.filter_folder, inp.return_full_notes,
            )

        @mcp.tool(
            name="vault_reindex",
            description="Rebuild the semantic search index. Use full=true to rebuild from scratch, or full=false for incremental sync.",
            annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
        )
        def vault_reindex(full: bool = False) -> str:
            """Trigger reindexing."""
            inp = VaultReindexInput(full=full)
            return _vault_reindex(inp.full)

        logger.info("Semantic search tools registered")

    except ImportError:
        logger.warning(
            "Semantic search enabled but dependencies not installed. "
            "Run: pip install obsidian-vault-mcp[semantic]"
        )
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (semantic tools only register when `SEMANTIC_SEARCH_ENABLED=true`)

- [ ] **Step 3: Commit**

```bash
git add src/obsidian_vault_mcp/server.py
git commit -m "feat: integrate semantic search into server lifecycle"
```

---

### Task 12: Integration Tests

**Files:**
- Create: `tests/test_semantic_search.py`

These tests use the real embedding model and ChromaDB. Mark them with `@pytest.mark.slow`.

- [ ] **Step 1: Write integration tests**

Create `tests/test_semantic_search.py`:

```python
"""Integration tests for semantic search pipeline.

These tests use real models and are slow. Run with: pytest -m slow
"""

import json
import os
import time
from pathlib import Path

import pytest

# Skip all tests if semantic deps not installed
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")
pytest.importorskip("rank_bm25")

pytestmark = pytest.mark.slow


@pytest.fixture
def semantic_vault(tmp_path, monkeypatch):
    """Create a vault with varied content for semantic search testing."""
    vault = tmp_path / "semantic-vault"
    vault.mkdir()

    # Note about Python
    (vault / "python-guide.md").write_text(
        "---\ntags:\n  - python\n  - programming\n---\n\n"
        "# Python Programming Guide\n\n"
        "## Getting Started\n\n"
        "Python is a versatile programming language used for web development, "
        "data science, machine learning, and automation.\n\n"
        "## Data Structures\n\n"
        "Python provides lists, dictionaries, sets, and tuples as built-in data structures.\n"
    )

    # Note about cooking
    (vault / "recipes.md").write_text(
        "---\ntags:\n  - cooking\n  - food\n---\n\n"
        "# Favorite Recipes\n\n"
        "## Pasta Carbonara\n\n"
        "A classic Italian pasta dish made with eggs, cheese, pancetta, and pepper.\n\n"
        "## Chocolate Cake\n\n"
        "Rich chocolate cake with ganache frosting.\n"
    )

    # Note in a subfolder
    projects = vault / "Projects"
    projects.mkdir()
    (projects / "web-app.md").write_text(
        "---\ntags:\n  - python\n  - web\n---\n\n"
        "# Web Application Project\n\n"
        "Building a web application using FastAPI and React.\n"
    )

    # Note about machine learning
    (vault / "ml-notes.md").write_text(
        "---\ntags:\n  - python\n  - ml\n---\n\n"
        "# Machine Learning Notes\n\n"
        "## Neural Networks\n\n"
        "Neural networks are the foundation of deep learning.\n\n"
        "## Training\n\n"
        "Training involves forward propagation, loss calculation, and backpropagation.\n"
    )

    # .obsidian dir (should be excluded)
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "config.json").write_text("{}")

    monkeypatch.setenv("VAULT_PATH", str(vault))
    monkeypatch.setenv("SEMANTIC_SEARCH_ENABLED", "true")
    monkeypatch.setenv("SEMANTIC_CACHE_PATH", str(tmp_path / "cache"))

    import obsidian_vault_mcp.config as cfg
    cfg.VAULT_PATH = vault
    cfg.SEMANTIC_SEARCH_ENABLED = True
    cfg.SEMANTIC_CACHE_PATH = tmp_path / "cache"

    yield vault


@pytest.fixture
def engine(semantic_vault):
    """Create and initialize a retrieval engine."""
    from obsidian_vault_mcp.retrieval import RetrievalEngine

    eng = RetrievalEngine()
    eng._ensure_initialized()
    yield eng
    eng.shutdown()


def test_search_returns_relevant_results(engine):
    """Searching for 'python programming' returns python-related notes."""
    result = json.loads(engine.search("python programming", max_results=5))
    assert "error" not in result
    assert result["total"] >= 1
    paths = [r["path"] for r in result["results"]]
    assert "python-guide.md" in paths


def test_search_scores_in_range(engine):
    """All scores are between 0 and 1."""
    result = json.loads(engine.search("programming", max_results=10, min_score=0.0))
    for r in result["results"]:
        assert 0.0 <= r["score"] <= 1.0


def test_search_min_score_filters(engine):
    """Results below min_score are excluded."""
    low = json.loads(engine.search("programming", max_results=50, min_score=0.0))
    high = json.loads(engine.search("programming", max_results=50, min_score=0.8))
    assert high["total"] <= low["total"]


def test_search_filter_folder(engine):
    """filter_folder restricts results to a folder prefix."""
    result = json.loads(engine.search("python", max_results=10, min_score=0.0, filter_folder="Projects/"))
    for r in result["results"]:
        assert r["path"].startswith("Projects/")


def test_search_return_full_notes(engine):
    """return_full_notes returns complete file content."""
    result = json.loads(engine.search("python", max_results=5, min_score=0.0, return_full_notes=True))
    assert result["total"] >= 1
    # Full note content should include the frontmatter delimiter or full body
    for r in result["results"]:
        assert len(r["content"]) > 50  # Full note, not just a chunk
        assert r["section"] is None


def test_search_empty_query_returns_error(engine):
    """Engine handles edge cases gracefully."""
    # Empty vault search for nonsense should return few/no results
    result = json.loads(engine.search("xyzzy_nonexistent_term_12345", max_results=5))
    assert result["total"] == 0 or all(r["score"] < 0.5 for r in result["results"])


def test_reindex_full(engine):
    """Full reindex rebuilds the index."""
    result = json.loads(engine.reindex(full=True))
    assert "error" not in result
    assert result["files_indexed"] >= 4
    assert result["chunks_created"] > 0


def test_incremental_update(engine, semantic_vault):
    """Adding a new file and triggering update includes it in results."""
    # Write a new note
    (semantic_vault / "rust-guide.md").write_text(
        "---\ntags:\n  - rust\n---\n\n"
        "# Rust Programming\n\n"
        "Rust is a systems programming language focused on safety and performance.\n"
    )

    # Trigger incremental update
    engine.handle_file_change(["rust-guide.md"])

    # Search should find the new note
    result = json.loads(engine.search("rust systems programming", max_results=5, min_score=0.0))
    paths = [r["path"] for r in result["results"]]
    assert "rust-guide.md" in paths
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_semantic_search.py -v -m slow`
Expected: all tests pass (first run will download the embedding model, ~420 MB)

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v -m "not slow"`
Expected: all non-slow tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_semantic_search.py
git commit -m "test: add integration tests for semantic search"
```

---

### Task 13: Final Verification

- [ ] **Step 1: Run full test suite including slow tests**

Run: `uv run pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 2: Manual smoke test with semantic search disabled**

Run: `VAULT_PATH=/tmp/test-vault VAULT_MCP_TOKEN=test uv run python -c "from obsidian_vault_mcp.server import mcp; tools = mcp.list_tools(); print([t.name for t in tools])"`
Expected: list of 9 original tools, no `vault_semantic_search` or `vault_reindex`

- [ ] **Step 3: Manual smoke test with semantic search enabled**

Run: `VAULT_PATH=/tmp/test-vault VAULT_MCP_TOKEN=test SEMANTIC_SEARCH_ENABLED=true uv run python -c "from obsidian_vault_mcp.server import mcp; tools = mcp.list_tools(); print([t.name for t in tools])"`
Expected: list of 11 tools including `vault_semantic_search` and `vault_reindex`

- [ ] **Step 4: Commit any final fixes**

If any fixes were needed, commit them.

- [ ] **Step 5: Final commit — update README (optional, only if user requests)**

Only if requested: add a semantic search section to README.md describing the feature, configuration, and usage.
