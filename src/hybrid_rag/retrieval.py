from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from hybrid_rag.config import Settings
from hybrid_rag.embeddings import EmbeddingProvider
from hybrid_rag.models import SearchHit
from hybrid_rag.sparse_index import BM25Index
from hybrid_rag.text import clamp, tokenize
from hybrid_rag.vector_store import VectorStore


class Reranker(Protocol):
    name: str

    def rerank(self, question: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]: ...


class LexicalReranker:
    name = "lexical"

    def rerank(self, question: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        query_tokens = set(tokenize(question))
        for candidate in candidates:
            chunk_tokens = set(tokenize(candidate.chunk.text))
            overlap = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
            candidate.rerank_score = 0.65 * overlap + 0.35 * (candidate.fusion_score or 0.0)
            candidate.score = candidate.rerank_score
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not candidates:
            return []
        raw_scores = self.model.predict([(question, item.chunk.text) for item in candidates])
        for candidate, raw_score in zip(candidates, raw_scores, strict=True):
            score = 1.0 / (1.0 + pow(2.718281828, -float(raw_score)))
            candidate.rerank_score = score
            candidate.score = score
        return sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]


def reciprocal_rank_fusion(
    dense: Sequence[SearchHit],
    sparse: Sequence[SearchHit],
    *,
    dense_weight: float,
    sparse_weight: float,
    rrf_k: int,
    limit: int,
) -> list[SearchHit]:
    total = dense_weight + sparse_weight
    dense_weight, sparse_weight = dense_weight / total, sparse_weight / total
    merged: dict[str, SearchHit] = {}
    for rank, item in enumerate(dense, start=1):
        hit = item.model_copy(deep=True)
        hit.dense_rank = rank
        hit.fusion_score = dense_weight / (rrf_k + rank)
        merged[item.chunk.id] = hit
    for rank, item in enumerate(sparse, start=1):
        contribution = sparse_weight / (rrf_k + rank)
        if item.chunk.id in merged:
            hit = merged[item.chunk.id]
            hit.sparse_rank = rank
            hit.sparse_score = item.sparse_score
            hit.fusion_score = (hit.fusion_score or 0.0) + contribution
        else:
            hit = item.model_copy(deep=True)
            hit.sparse_rank = rank
            hit.fusion_score = contribution
            merged[item.chunk.id] = hit
    maximum = max((hit.fusion_score or 0.0 for hit in merged.values()), default=1.0)
    for hit in merged.values():
        hit.fusion_score = (hit.fusion_score or 0.0) / maximum
        hit.score = hit.fusion_score
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_index: BM25Index,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.reranker = reranker or LexicalReranker()

    def retrieve(self, question: str, *, mode: str = "hybrid") -> list[SearchHit]:
        # A seed/worker process may have updated Chroma while this API process stayed live.
        if mode == "hybrid" and self.sparse_index.size != self.vector_store.count():
            chunks, _ = self.vector_store.all()
            self.sparse_index.rebuild(chunks)
        query_embedding = self.embeddings.embed([question])[0]
        dense = self.vector_store.query(query_embedding, self.settings.dense_top_k)
        sparse = (
            self.sparse_index.search(question, self.settings.sparse_top_k)
            if mode == "hybrid"
            else []
        )
        candidates = reciprocal_rank_fusion(
            dense,
            sparse,
            dense_weight=self.settings.dense_weight if mode == "hybrid" else 1.0,
            sparse_weight=self.settings.sparse_weight if mode == "hybrid" else 0,
            rrf_k=self.settings.rrf_k,
            limit=self.settings.fusion_top_k,
        )
        return self.reranker.rerank(question, candidates, self.settings.final_top_k)

    @staticmethod
    def retrieval_confidence(hits: Sequence[SearchHit]) -> float:
        if not hits:
            return 0.0
        weights = [1.0 / (index + 1) for index in range(len(hits))]
        weighted = sum(clamp(hit.score) * weight for hit, weight in zip(hits, weights, strict=True))
        return clamp(weighted / sum(weights))
