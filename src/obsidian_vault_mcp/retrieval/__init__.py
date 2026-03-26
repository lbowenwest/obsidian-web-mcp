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
            from .indexer import CHUNK_ID_SEPARATOR

            query_embedding = self._embedder.encode([query])[0]
            vector_results = self._indexer.vector_search(
                query_embedding, n_results=3 * max_results,
                filter_tags=filter_tags, filter_folder=filter_folder,
            )
            vector_ranked = [(cid, sim) for cid, sim, _ in vector_results]

            bm25_results = self._indexer.bm25.query(query, top_k=6 * max_results)
            if filter_folder:
                bm25_results = [
                    (cid, s) for cid, s in bm25_results
                    if cid.split(CHUNK_ID_SEPARATOR)[0].startswith(filter_folder)
                ]

            fused = reciprocal_rank_fusion(
                vector_ranked, bm25_results,
                vector_weight=config.VECTOR_WEIGHT,
                bm25_weight=config.BM25_WEIGHT,
                k=60,
            )

            fused = [(cid, s) for cid, s in fused if s >= min_score]

            # Attach path from chunk_id
            results_with_path = [
                (cid, cid.rsplit(CHUNK_ID_SEPARATOR, 1)[0], score)
                for cid, score in fused
            ]

            if not return_full_notes:
                results_with_path = deduplicate_by_path(results_with_path)

            results_with_path = results_with_path[:max_results]

            if not results_with_path:
                return json.dumps({"results": [], "total": 0})

            # Batch retrieve all chunk data at once (avoids N+1)
            chunk_ids = [cid for cid, _, _ in results_with_path]
            chunk_data = self._indexer.get_chunks(chunk_ids)

            search_results = []
            file_cache: dict[str, str] = {}
            for chunk_id, path, score in results_with_path:
                data = chunk_data.get(chunk_id)
                if not data:
                    continue
                doc_text, meta = data

                if return_full_notes:
                    if path not in file_cache:
                        try:
                            file_cache[path], _ = read_file(path)
                        except Exception:
                            continue
                    content = file_cache[path]
                    section = None
                else:
                    content = doc_text
                    section = meta.get("section") or None

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
                self._indexer.sync_delta()
                result = ReindexResult(
                    files_indexed=0,
                    chunks_created=self._indexer.chunk_count(),
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
