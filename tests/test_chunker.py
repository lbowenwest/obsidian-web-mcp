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
    found = [c for c in chunks if "def hello():" in c.text]
    assert len(found) == 1
    assert "```python" in found[0].text
    assert "```" in found[0].text.split("```python")[1]


def test_large_section_splits_on_paragraphs():
    """Section exceeding chunk_size splits on paragraph breaks."""
    paragraphs = "\n\n".join([f"Paragraph {i} with some filler content here." for i in range(10)])
    content = f"---\ntags: []\n---\n\n## Big Section\n\n{paragraphs}\n"
    chunks = chunk_markdown(content, path="test.md", chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1
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
