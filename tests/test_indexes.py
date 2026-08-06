from __future__ import annotations

from pathlib import Path

from hybrid_rag.models import Chunk
from hybrid_rag.sparse_index import BM25Index
from hybrid_rag.vector_store import MemoryVectorStore


def chunk(identifier: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        document_id=document_id,
        source=f"{document_id}.md",
        text=text,
        chunk_index=0,
        strategy="fixed",
        character_count=len(text),
        content_hash=identifier,
    )


def test_bm25_persists_and_reloads_portable_corpus(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    first = BM25Index(path)
    first.rebuild([chunk("one", "doc", "OBS-1042 expired token")])
    second = BM25Index(path)
    results = second.search("OBS-1042", top_k=2)
    assert second.size == 1
    assert results[0].chunk.id == "one"


def test_memory_vector_store_queries_and_deletes_by_document() -> None:
    store = MemoryVectorStore()
    store.add(
        [chunk("one", "doc-a", "alpha"), chunk("two", "doc-b", "beta")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    assert store.query([1.0, 0.0], 1)[0].chunk.id == "one"
    assert store.delete_document("doc-a") == 1
    assert store.count() == 1
    chunks, embeddings = store.all(include_embeddings=True)
    assert [item.id for item in chunks] == ["two"]
    assert embeddings == [[0.0, 1.0]]
