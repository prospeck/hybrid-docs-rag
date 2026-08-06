from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.embeddings import EmbeddingProvider, build_embedding_provider
from hybrid_rag.generation import TextGenerator, build_generator
from hybrid_rag.ingestion import IngestionPipeline
from hybrid_rag.loaders import DocumentLoader
from hybrid_rag.retrieval import CrossEncoderReranker, HybridRetriever, LexicalReranker, Reranker
from hybrid_rag.service import RAGService
from hybrid_rag.sparse_index import BM25Index
from hybrid_rag.vector_store import VectorStore, build_vector_store


@dataclass
class Components:
    settings: Settings
    embeddings: EmbeddingProvider
    vector_store: VectorStore
    sparse_index: BM25Index
    ingestion: IngestionPipeline
    retriever: HybridRetriever
    generator: TextGenerator
    service: RAGService


def build_components(settings: Settings | None = None) -> Components:
    settings = settings or get_settings()
    settings.ensure_directories()
    embeddings = build_embedding_provider(settings)
    vector_store = build_vector_store(settings)
    sparse_index = BM25Index(settings.index_dir / "bm25.json")
    loader = DocumentLoader(settings)
    ingestion = IngestionPipeline(settings, loader, embeddings, vector_store, sparse_index)
    ingestion.synchronize_indexes()
    reranker: Reranker = LexicalReranker()
    if settings.reranker_enabled:
        try:
            reranker = CrossEncoderReranker(settings.reranker_model)
        except (ImportError, OSError, RuntimeError):
            reranker = LexicalReranker()
    retriever = HybridRetriever(settings, embeddings, vector_store, sparse_index, reranker)
    generator = build_generator(settings)
    service = RAGService(settings, retriever, generator, vector_store, loader)
    return Components(
        settings=settings,
        embeddings=embeddings,
        vector_store=vector_store,
        sparse_index=sparse_index,
        ingestion=ingestion,
        retriever=retriever,
        generator=generator,
        service=service,
    )
