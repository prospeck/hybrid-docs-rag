from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Secrets are read from the process environment only."""

    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    app_name: str = "Hybrid Docs RAG"
    environment: Literal["development", "test", "production"] = "development"
    data_dir: Path = Path("data")
    max_upload_mb: int = Field(default=25, ge=1, le=250)
    allowed_extensions: tuple[str, ...] = (".md", ".txt", ".html", ".htm", ".pdf")

    chunk_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = Field(default=900, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    semantic_breakpoint_percentile: float = Field(default=80.0, ge=50, le=99)

    embedding_provider: Literal["fastembed", "sentence_transformers", "openai", "hash"] = (
        "fastembed"
    )
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )

    vector_backend: Literal["chroma", "memory"] = "chroma"
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_collection: str = "internal_docs"

    dense_top_k: int = Field(default=10, ge=1, le=100)
    sparse_top_k: int = Field(default=10, ge=1, le=100)
    fusion_top_k: int = Field(default=20, ge=1, le=100)
    final_top_k: int = Field(default=5, ge=1, le=20)
    dense_weight: float = Field(default=0.7, ge=0, le=1)
    sparse_weight: float = Field(default=0.3, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1, le=500)
    reranker_enabled: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    minimum_retrieval_confidence: float = Field(default=0.25, ge=0, le=1)
    duplicate_similarity_threshold: float = Field(default=0.95, ge=0.8, le=1)

    generation_provider: Literal["ollama", "openai", "extractive"] = "ollama"
    generation_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    request_timeout_seconds: float = Field(default=90, ge=5, le=300)

    @model_validator(mode="after")
    def validate_weights_and_overlap(self) -> Settings:
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("At least one retrieval weight must be greater than zero")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    def ensure_directories(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.index_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
