"""Vault indexer: builds and maintains ChromaDB + BM25 indices."""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

import chromadb

from .. import config
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

        for path in set(self._manifest.keys()) - set(current_files.keys()):
            changed.append(path)

        if changed:
            logger.info("Cold start delta: %d files changed", len(changed))
            self.update_files(changed)
        else:
            logger.info("Index up to date, no changes detected")
