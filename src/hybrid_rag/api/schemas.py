from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    retrieval_mode: Literal["hybrid", "dense"] = "hybrid"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    indexed_chunks: int
    embedding_provider: str
    generation_provider: str
    vector_backend: str


class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
