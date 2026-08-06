from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from hybrid_rag.config import Settings
from hybrid_rag.embeddings import cosine_similarity
from hybrid_rag.models import Chunk, SearchHit


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchHit]: ...

    def all(self, *, include_embeddings: bool = False) -> tuple[list[Chunk], list[list[float]]]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...


class MemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._items[chunk.id] = (chunk, list(embedding))

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=chunk, score=score, dense_score=score)
            for chunk, vector in self._items.values()
            if (score := cosine_similarity(embedding, vector)) >= -1
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def all(self, *, include_embeddings: bool = False) -> tuple[list[Chunk], list[list[float]]]:
        chunks = [item[0] for item in self._items.values()]
        embeddings = [item[1] for item in self._items.values()] if include_embeddings else []
        return chunks, embeddings

    def delete_document(self, document_id: str) -> int:
        ids = [key for key, (chunk, _) in self._items.items() if chunk.document_id == document_id]
        for chunk_id in ids:
            del self._items[chunk_id]
        return len(ids)

    def count(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    def __init__(self, settings: Settings) -> None:
        import chromadb

        if settings.chroma_host:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            client = chromadb.PersistentClient(path=str(settings.index_dir / "chroma"))
        self.collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
        return {
            "chunk_json": chunk.model_dump_json(),
            "document_id": chunk.document_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
            "strategy": chunk.strategy,
            "section_heading": chunk.section_heading or "",
            "page_number": chunk.page_number or 0,
            "character_count": chunk.character_count,
            "content_hash": chunk.content_hash,
        }

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=cast(Any, [list(vector) for vector in embeddings]),
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    def query(self, embedding: Sequence[float], top_k: int) -> list[SearchHit]:
        if self.collection.count() == 0:
            return []
        response = self.collection.query(
            query_embeddings=cast(Any, [list(embedding)]),
            n_results=min(top_k, self.collection.count()),
            include=["metadatas", "distances"],
        )
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        output = []
        for metadata, distance in zip(metadatas, distances, strict=False):
            if not metadata:
                continue
            chunk = Chunk.model_validate_json(str(metadata["chunk_json"]))
            score = max(-1.0, min(1.0, 1.0 - float(distance)))
            output.append(SearchHit(chunk=chunk, score=score, dense_score=score))
        return output

    def all(self, *, include_embeddings: bool = False) -> tuple[list[Chunk], list[list[float]]]:
        include: list[str] = ["metadatas"]
        if include_embeddings:
            include.append("embeddings")
        result = self.collection.get(include=cast(Any, include))
        chunks = [
            Chunk.model_validate_json(str(metadata["chunk_json"]))
            for metadata in result.get("metadatas") or []
            if metadata and metadata.get("chunk_json")
        ]
        raw_embeddings = result.get("embeddings") if include_embeddings else None
        embeddings = (
            [list(map(float, vector)) for vector in raw_embeddings]
            if raw_embeddings is not None
            else []
        )
        return chunks, embeddings

    def delete_document(self, document_id: str) -> int:
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self.collection.count())


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "memory":
        return MemoryVectorStore()
    return ChromaVectorStore(settings)
