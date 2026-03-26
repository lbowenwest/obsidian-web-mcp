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
    for r in result["results"]:
        assert len(r["content"]) > 50
        assert r["section"] is None


def test_search_empty_query_returns_few_results(engine):
    """Search for nonsense term returns no/few results above a high threshold."""
    result = json.loads(engine.search("xyzzy_nonexistent_term_12345", max_results=5, min_score=0.9))
    assert result["total"] == 0


def test_reindex_full(engine):
    """Full reindex rebuilds the index."""
    result = json.loads(engine.reindex(full=True))
    assert "error" not in result
    assert result["files_indexed"] >= 4
    assert result["chunks_created"] > 0


def test_incremental_update(engine, semantic_vault):
    """Adding a new file and triggering update includes it in results."""
    (semantic_vault / "rust-guide.md").write_text(
        "---\ntags:\n  - rust\n---\n\n"
        "# Rust Programming\n\n"
        "Rust is a systems programming language focused on safety and performance.\n"
    )

    engine.handle_file_change(["rust-guide.md"])

    result = json.loads(engine.search("rust systems programming", max_results=5, min_score=0.0))
    paths = [r["path"] for r in result["results"]]
    assert "rust-guide.md" in paths
