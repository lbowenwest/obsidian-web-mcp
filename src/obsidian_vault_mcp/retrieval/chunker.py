"""Markdown chunker that splits vault notes into embeddable chunks."""

import hashlib
import re

import frontmatter

from .models import Chunk, ChunkMetadata

# Shared with indexer — keep in sync
_HASH_PREFIX_LEN = 16


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
    """Split markdown body into (heading, content) pairs."""
    header_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str | None, str]] = []
    last_end = 0
    last_heading: str | None = None

    for match in header_pattern.finditer(body):
        text_before = body[last_end:match.start()].strip()
        if text_before or last_end == 0:
            if text_before:
                sections.append((last_heading, text_before))

        last_heading = match.group(2).strip()
        last_end = match.end()

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
    """Split a markdown note into chunks with metadata."""
    if not content or not content.strip():
        return []

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

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:_HASH_PREFIX_LEN]

    sections = _extract_sections(body)
    if not sections:
        return []

    raw_chunks: list[tuple[str | None, str]] = []
    for heading, text in sections:
        if _estimate_tokens(text) <= chunk_size:
            raw_chunks.append((heading, text))
        else:
            sub_chunks = _split_on_paragraphs(text, chunk_size, chunk_overlap)
            for sub in sub_chunks:
                raw_chunks.append((heading, sub))

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
