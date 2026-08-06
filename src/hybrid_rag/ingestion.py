from __future__ import annotations

import threading
import time
from pathlib import Path

from hybrid_rag.chunking import build_chunker
from hybrid_rag.config import Settings
from hybrid_rag.embeddings import EmbeddingProvider, cosine_similarity
from hybrid_rag.loaders import DocumentLoader
from hybrid_rag.models import Chunk, IngestionResult
from hybrid_rag.sparse_index import BM25Index
from hybrid_rag.vector_store import VectorStore


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        loader: DocumentLoader,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_index: BM25Index,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self._lock = threading.RLock()

    def ingest(self, path: Path, *, strategy: str | None = None) -> IngestionResult:
        started = time.perf_counter()
        document = self.loader.load(path)
        chunk_settings = (
            self.settings.model_copy(update={"chunk_strategy": strategy})
            if strategy
            else self.settings
        )
        chunker = build_chunker(chunk_settings, self.embeddings)
        proposed = chunker.split(document)

        with self._lock:
            existing_chunks, existing_embeddings = self.vector_store.all(include_embeddings=True)
            same_document = [chunk for chunk in existing_chunks if chunk.document_id == document.id]
            if same_document:
                return IngestionResult(
                    document_id=document.id,
                    source=document.source,
                    chunks_created=0,
                    duplicates_skipped=len(proposed),
                    strategy=chunker.strategy,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )

            # A source filename represents one logical document. Replace an older revision.
            previous_document_ids = {
                chunk.document_id for chunk in existing_chunks if chunk.source == document.source
            }
            previous_pairs = [
                (chunk, embedding)
                for chunk, embedding in zip(existing_chunks, existing_embeddings, strict=True)
                if chunk.document_id in previous_document_ids
            ]
            for document_id in previous_document_ids:
                self.vector_store.delete_document(document_id)
            if previous_document_ids:
                existing_chunks, existing_embeddings = self.vector_store.all(
                    include_embeddings=True
                )

            proposed_embeddings = self.embeddings.embed([chunk.text for chunk in proposed])
            accepted_chunks: list[Chunk] = []
            accepted_embeddings: list[list[float]] = []
            duplicates = 0
            known_hashes = {chunk.content_hash for chunk in existing_chunks}
            comparison_vectors = list(existing_embeddings)

            for chunk, embedding in zip(proposed, proposed_embeddings, strict=True):
                is_exact = chunk.content_hash in known_hashes
                is_near = any(
                    cosine_similarity(embedding, existing)
                    >= self.settings.duplicate_similarity_threshold
                    for existing in comparison_vectors
                )
                if is_exact or is_near:
                    duplicates += 1
                    continue
                accepted_chunks.append(chunk)
                accepted_embeddings.append(embedding)
                known_hashes.add(chunk.content_hash)
                comparison_vectors.append(embedding)

            try:
                self.vector_store.add(accepted_chunks, accepted_embeddings)
                all_chunks, _ = self.vector_store.all()
                self.sparse_index.rebuild(all_chunks)
            except Exception:
                # Restore the replaced revision and rebuild sparse state from vector truth.
                self.vector_store.delete_document(document.id)
                if previous_pairs:
                    self.vector_store.add(
                        [pair[0] for pair in previous_pairs],
                        [pair[1] for pair in previous_pairs],
                    )
                remaining, _ = self.vector_store.all()
                self.sparse_index.rebuild(remaining)
                raise

        return IngestionResult(
            document_id=document.id,
            source=document.source,
            chunks_created=len(accepted_chunks),
            duplicates_skipped=duplicates,
            strategy=chunker.strategy,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def synchronize_indexes(self) -> int:
        """Rebuild BM25 from vector-store truth. Safe to call on every process start."""
        with self._lock:
            chunks, _ = self.vector_store.all()
            self.sparse_index.rebuild(chunks)
            return len(chunks)
