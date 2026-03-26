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
                path = chunk_id.rsplit("::", 1)[0]
                meta = self._get_chunk_metadata(chunk_id)
                if meta:
                    results_with_meta.append((chunk_id, path, score, meta))

            # 6. Deduplicate
            if not return_full_notes:
                deduped = deduplicate_by_path(
                    [(cid, path, score) for cid, path, score, _ in results_with_meta]
                )
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
