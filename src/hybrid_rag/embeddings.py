from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol, cast

import httpx
import numpy as np

from hybrid_rag.config import Settings
from hybrid_rag.text import tokenize


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashEmbeddingProvider:
    """Deterministic, dependency-light provider for tests and degraded local operation."""

    name = "hash"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimensions, dtype=np.float32)
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                index = value % self.dimensions
                vector[index] += -1.0 if value & 1 else 1.0
            norm = float(np.linalg.norm(vector))
            if norm:
                vector /= norm
            vectors.append(vector.tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FastEmbedEmbeddingProvider:
    name = "fastembed"

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name, lazy_load=True)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.astype(np.float32).tolist() for vector in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self.model.query_embed(text)))
        return cast(list[float], vector.astype(np.float32).tolist())


class SentenceTransformerEmbeddingProvider:
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return cast(list[list[float]], vectors.tolist())

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("RAG_OPENAI_API_KEY is required for the OpenAI embedding provider")
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url.rstrip("/")
        self.model = settings.embedding_model
        self.timeout = settings.request_timeout_seconds

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": list(texts)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [item["embedding"] for item in sorted(payload["data"], key=lambda x: x["index"])]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.embedding_provider == "hash":
        return HashEmbeddingProvider()
    if settings.embedding_provider == "fastembed":
        return FastEmbedEmbeddingProvider(settings.embedding_model)
    return SentenceTransformerEmbeddingProvider(settings.embedding_model)
