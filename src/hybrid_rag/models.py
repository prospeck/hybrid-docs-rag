from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    source: str
    checksum: str
    text: str
    media_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    text: str
    chunk_index: int
    strategy: Literal["fixed", "recursive", "semantic"]
    section_heading: str | None = None
    page_number: int | None = None
    character_count: int
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None


class Citation(BaseModel):
    reference: int
    chunk_id: str
    source: str
    section_heading: str | None = None
    page_number: int | None = None
    quote: str
    verified: bool
    verification_score: float


class ConfidenceScores(BaseModel):
    retrieval: float
    citation_coverage: float
    completeness: float
    composite: float


class AnswerResult(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    confidence: ConfidenceScores
    retrieved_chunks: list[SearchHit]
    status: Literal["answered", "insufficient_context"]
    provider_used: str
    degraded_mode: bool = False
    latency_ms: float


class IngestionResult(BaseModel):
    document_id: str
    source: str
    chunks_created: int
    duplicates_skipped: int
    strategy: str
    elapsed_ms: float


class DocumentSummary(BaseModel):
    id: str
    source: str
    media_type: str
    checksum: str
    chunk_count: int
    created_at: datetime | None = None
